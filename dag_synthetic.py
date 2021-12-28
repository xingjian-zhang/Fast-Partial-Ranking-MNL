"""Synthetic experiments for mix MNL.

- 4.2 Experiments on Single MNL
- 4.3 Experiments on Mixture of MNL

The data are generel partial orders generated from full linear orders. For
single MNL, we fix the ground truth utility scores `w` uniformly from -2 to 2.
Then we draw `n` full orders from an MNL parametrized by `w`. To obtain partial
orders, we sample from the `n(n-1)/2` pairwise comparisons determined by the full
rankings, and keep each pairwise relation with probability `p` independently. For
3-mixture of MNL, we first generated the ground truth utility score `w` for each
MNL model. Then we draw `n` samples in the same way as the single MNL (each
sample is generated from one of 3 random MNL models according to `alpha`).
"""

import argparse
import json
import os
from time import time

import numpy as np
import torch
import tqdm
from torch import optim
from torch.utils.data.dataloader import DataLoader

import config
import utils
from model import LinearModel, PartitionLoss
from utils import (generate_linear_orders, get_collate_fn, get_trainset,
                   init_by_cluster, linear2dag, mse, rankbreak)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", type=int, default=100,
                    help="Number of items/classes `N` in the sample space.")
parser.add_argument("--num_observed", type=int, default=None,
                    help="Max number of observed items/classes in each sample.")
parser.add_argument("--num_samples", type=int, default=5000,
                    help="Number of samples `n`.")
parser.add_argument("--alphas", type=str, default="[1,]",
                    help="weights of k-mixture of MNL, e.g., [0.5,0.5].")
parser.add_argument("--init_by_cluster", action="store_true",
                    help="Whether to use clustering-based initialization.")
parser.add_argument("--rs", type=int, default=42,
                    help="Random seed.")
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

if args.num_observed is None:
    args.num_observed = args.num_classes

if not os.path.exists(config.DAG_OUTPUT_DIR):
    os.makedirs(config.DAG_OUTPUT_DIR)

output_file = "{}_{}_{}_{}_{}_{}.json".format(*vars(args).values())
output_file = os.path.join(config.DAG_OUTPUT_DIR, output_file)

args.alphas = np.array([
    float(alpha) for alpha in args.alphas.strip("[,]").split(",")
])

# Generate synthetic data
k = len(args.alphas)
rs = np.random.RandomState(args.rs)
device = args.device

linear_orders, true_p, true_theta = generate_linear_orders(
    num_classes=args.num_classes,
    num_observed=args.num_observed,
    num_samples=args.num_samples,
    rs=rs
)

partial_orders = []
for linear_order in linear_orders:
    dag = linear2dag(linear_order, config.SAMPLE_RATE, rs)
    partitions = rankbreak(dag, device=device)
    if len(partitions) > 1:
        partial_orders.append(partitions)

trainset = get_trainset(partial_orders, requires_idx=True)
trainloader = DataLoader(trainset,
                         config.DAG_BATCH_SIZE,
                         shuffle=True,
                         collate_fn=get_collate_fn(requires_idx=True))

# Models and model weights
models = [LinearModel(args.num_classes).to(device) for _ in range(k)]
alphas = torch.ones(k) / k
global_time = 0
shift = 1  # max alpha shift

tic = time()
# Apply clustering-based initialization
if args.init_by_cluster:
    init_by_cluster(models, partial_orders, args.num_classes)

# Training
iters = 1 if k == 1 else config.DAG_ITERATIONS
for itr in range(iters):  # the outmost loop - EM

    # check alphas shift is small enough to stop EM
    if shift < config.DAG_TOL:
        break

    # standard expectation maximization (E step)
    respon = [torch.log(alpha) + model.eval_prob(partial_orders)
              for alpha, model in zip(alphas, models)]
    respon = torch.stack(respon)  # (k, num_samples)
    gammas = respon - torch.logsumexp(respon, axis=0, keepdims=True)
    gammas = torch.exp(gammas)
    alphas_new = torch.mean(gammas, axis=1)
    shift = torch.max(alphas_new - alphas).item() if itr > 0 else 1
    alphas = alphas_new

    # standard expectation maximization (M step)
    for m in range(k):  # train individual model by Adagrad
        model = models[m]
        model.train()
        optimizer = optim.Adagrad(
            [{'params': model.parameters(), 'initial_lr': config.DAG_LR}], lr=config.DAG_LR)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=config.DAG_LR_STEP, gamma=config.DAG_LR_GAMMA
        )
        criterion = PartitionLoss(device=device)
        for epoch in range(config.DAG_EPOCHS):
            bar = tqdm.tqdm(trainloader, desc="MSE: N/A")
            for i, batch in enumerate(bar):
                B, T, idx = batch
                cur_batch_size = len(B)
                w_estimate = model.forward(cur_batch_size)
                loss = criterion(w_estimate, (B, T), gamma=gammas[m][idx])
                loss.backward()
                mse_score = mse(model.get_numpy_prob(), true_p)
                if i % 100 == 0:
                    bar.set_description(desc="MSE: {}".format(mse_score))
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

# Collect statistics
toc = time()
global_time = toc - tic
mses = [mse(model.get_numpy_prob(), true_p) for model in models]
tot_mse = np.mean(np.array(mses))
alphas_mse = np.mean((alphas.cpu().numpy()-args.alphas)**2)
statistics = {
    "model_mse": tot_mse,
    "alphas_mse": alphas_mse,
    "time": global_time
}
statistics.update(vars(args))
statistics.update({k: v.tolist()
                  for k, v in statistics.items() if isinstance(v, np.ndarray)})

# Save results
with open(output_file, "w") as fp:
    json.dump(statistics, fp)
