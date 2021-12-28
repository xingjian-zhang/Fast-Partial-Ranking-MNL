import random
from typing import List

import networkx as nx
import numpy as np
import torch
from pyclustering.cluster.kmeans import kmeans
from pyclustering.utils.metric import distance_metric, type_metric
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import dataloader

import config
from model import LinearModel, PartitionLoss


def mse(x: np.array, y: np.array) -> float:
    """Compute mean square error(MSE).

    If dimension of y is 2, return the smallest possible mse between x and y[m].
    """
    if len(y.shape) == 1 and len(x.shape) == 1:
        return np.mean((x - y)**2)
    elif len(y.shape) == 2 and len(x.shape) == 1:
        return np.min(np.mean((x[None, :] - y)**2, axis=1))


def linear2dag(linear_order, sample_rate=0.5, rs=np.random.RandomState(0)) -> nx.DiGraph:
    """Sample DAG from a given linear order."""
    pairs = []
    for i in range(len(linear_order)):
        for j in range(i + 1, len(linear_order)):
            if rs.rand() < sample_rate:
                pairs.append((linear_order[i], linear_order[j]))
    G = nx.DiGraph()
    G.add_edges_from(pairs)
    return G


def rankbreak(G: nx.DiGraph, device="cuda:0"):
    """Break the DAG into partitioned preferences."""
    P = []
    while(G.number_of_nodes()):
        sink_nodes = [node for node, outdegree in dict(
            G.out_degree(G.nodes())).items() if outdegree == 0]
        ancestors = [nx.ancestors(G, node) for node in sink_nodes]
        common_ancestors = set.intersection(*ancestors)
        P.append(set(G.nodes()) - common_ancestors)
        G.remove_nodes_from(P[-1])
    P.reverse()
    return [torch.tensor(list(item), device=device) for item in P]


def generate_linear_orders(num_classes: int = 256,
                           num_observed: int = 32,
                           num_samples: int = 10000,
                           alphas: list = [1],
                           rs: np.random.RandomState = np.random.RandomState(0)):
    """Generate linear orders randomly.

    For single PL:
        The ground truth theta are generated uniformly and deterministically from [-2, 2].

    For K-PL:
        The ground truth theta are generated uniformly and stochastically from [-2, 2] for each component.
    """
    if len(alphas) == 1:
        true_theta = np.linspace(-2, 2, num_classes)
        true_p = np.exp(true_theta) / np.sum(np.exp(true_theta))
        linear_orders = []
        for _ in range(num_samples):
            observed = rs.choice(num_classes, size=num_observed, replace=False)
            linear_orders.append(
                rs.choice(observed, size=num_observed, replace=False,
                          p=true_p[observed] / np.sum(true_p[observed]))
            )
    else:
        k = len(alphas)
        if k == 2:
            true_theta = np.stack(
                (np.linspace(0, 1, num_classes), np.linspace(0, -1, num_classes)))
        else:
            true_theta = np.random.rand(k, num_classes)
            # standardize the true_theta onto interval [-2,2]
        true_theta = true_theta * 4
        true_theta -= np.mean(true_theta)
        true_p = np.exp(true_theta) / \
            np.sum(np.exp(true_theta), axis=1)[:, None]
        linear_orders = []
        true_categories = []
        for _ in range(num_samples):
            observed = rs.choice(
                num_classes, size=num_observed, replace=False)
            category = rs.choice(k, p=alphas)
            linear_orders.append(
                rs.choice(observed, size=num_observed, replace=False,
                          p=true_p[category, observed] / np.sum(true_p[category, observed]))
            )
            true_categories.append(category)
    linear_orders = np.array(linear_orders, dtype=int)
    return linear_orders, true_p, true_theta


def get_trainset(partitions_list: list, requires_idx: bool = False):
    """Apply padding to the trainset.

    After the padding, a batch can be fully converted to a GPU-friendly tensor.
    """
    trainset = []
    for j, partitions in enumerate(partitions_list):
        for i in range(len(partitions) - 1):
            with torch.no_grad():
                T = partitions[i]
                B = torch.cat(partitions[i + 1:])
                # spare 0, 1 for padding index T, B
                if requires_idx:
                    trainset.append((T + 2, B + 2, j))
                else:
                    trainset.append((T + 2, B + 2))
    return trainset


def get_collate_fn(requires_idx: bool = False):
    if requires_idx:
        def collate_padding(samples):
            list_T, list_B, list_idx = zip(*samples)
            batch_T = pad_sequence(list_T, batch_first=True, padding_value=0)
            batch_B = pad_sequence(list_B, batch_first=True, padding_value=1)
            batch_idx = torch.tensor(
                list_idx, dtype=torch.long, device=batch_B.device)
            return (batch_T, batch_B, batch_idx)
    else:
        def collate_padding(samples):
            list_T, list_B = zip(*samples)
            batch_T = pad_sequence(list_T, batch_first=True, padding_value=0)
            batch_B = pad_sequence(list_B, batch_first=True, padding_value=1)
            return (batch_T, batch_B)
    return collate_padding


def rank_embedding(par_list: List[torch.Tensor], num_classes: int) -> np.array:
    """Generate the embeddings of a given partial order.

    The embedding of each index (item) is its relative order in the partitions. Missing indexs are filled with 1.
    """
    embedding = - np.ones(num_classes)
    for i, par in enumerate(par_list):
        v = i / len(par_list)
        embedding[par.cpu().detach().numpy()] = v
    return embedding


def rank_dist(embed1: np.array, embed2: np.array) -> float:
    mask1 = embed1 != -1.
    mask2 = embed2 != -1.
    mask = np.logical_and(mask1, mask2)
    masked_embed1 = embed1[mask]
    masked_embed2 = embed2[mask]
    return np.mean((masked_embed1 - masked_embed2)**2)


rank_dist_metric = distance_metric(type_metric.USER_DEFINED, func=rank_dist)


def init_by_cluster(models: List[LinearModel], partial_orders: List[List[torch.Tensor]]):
    """Initialize the models by the clustering-based intialization."""
    num_classes = models[0].num_classes
    embeddings = [rank_embedding(par_list, num_classes)
                  for par_list in partial_orders]
    embeddings = np.array(embeddings)
    k = kmeans(embeddings, np.random.random(
        (len(models), num_classes)), metric=rank_dist_metric)
    k.process()
    clusters = k.get_clusters()
    criterion = PartitionLoss()
    lr = config.DAG_LR
    step_size = config.DAG_LR_STEP
    lr_gamma = config.DAG_LR_GAMMA

    for model, indices in zip(models, clusters):
        training_data = [partial_orders[idx] for idx in indices]
        optimizer = optim.Adagrad(
            [{'params': model.parameters(), 'initial_lr': lr}], lr=lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=lr_gamma)
        trainloader = dataloader.DataLoader(get_trainset(
            training_data), 128, False, collate_fn=get_collate_fn())
        for epoch in range(config.DAG_EPOCHS):
            for i, batch in enumerate(trainloader):
                cur_batch_size = len(batch[0])
                outputs = model.forward(cur_batch_size)
                outputs.retain_grad()
                B, T = batch
                loss = criterion(outputs, (B, T))
                loss.backward()
                cur_loss = loss.item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()


def make_graph(r=0.5, p=0.5, alpha=1., m=5, n_max=100, rs=42):
    """Generate synthetic (r,p)-model data."""
    G = nx.erdos_renyi_graph(1000, 0.005, directed=True, seed=rs)
    high_degree_nodes = random.sample(G.nodes(), 20)
    for i in high_degree_nodes:  # manually add high-degree nodes
        friends = set(nx.ego_graph(G, i, 1).nodes())
        choice_set_all = set(G.nodes()) - friends - set([i])
        n_edges = random.randint(50, 80)
        edges = [(i, j) for j in random.sample(choice_set_all, n_edges)]
        G.add_edges_from(edges)

    # determine node pattern
    patterns = {}
    samples = []
    for i in G.nodes():
        (P, R) = (random.random(), random.random())
        patterns[i] = (P < p, R < r)
    for idx in range(n_max):
        i = random.sample(G.nodes(), 1)[0]

        # determine candidate sets
        friends = set(nx.ego_graph(G, i, 1).nodes())
        choice_set_all = set(G.nodes()) - friends - set([i])
        choice_set_fof = set(nx.ego_graph(G, i, 2).nodes())-friends - set([i])
        if patterns[i][1]:
            # consider all
            choice_set = choice_set_all
        else:
            # consider fof only
            choice_set = choice_set_fof
        choice_set = list(choice_set)
        if len(choice_set) == 0:
            continue
        if patterns[i][0]:
            # uniform attachment
            choice = np.random.choice(choice_set, size=min(
                m, len(choice_set)), replace=False)
        else:
            # preferential attachment with alpha
            deg = list(dict(G.degree(choice_set)).values())
            p = np.array(deg, dtype=float) ** alpha
            p /= sum(p)
            p[-1] = 1 - sum(p[:-1])
            choice = np.random.choice(choice_set, size=min(
                m, len(choice_set)), replace=False, p=p)
        candidates = list(choice_set_all)
        deg = list(dict(G.degree(candidates)).values())
        for c, d in zip(candidates, deg):
            samples.append({
                "deg": d,
                "is_fof": int(c in choice_set_fof),
                "alt": int(c in choice),
                "index": idx
            })
    return samples
