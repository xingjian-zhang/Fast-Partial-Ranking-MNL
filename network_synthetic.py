"""Synthetic experiments for network modeling.

- 5.2 Experiments on Synthetic Network Data

We simulate the growth of a directed network with the synthetic (r,p)-model
formed by both preferential attachment and uniform attachment. When a new
edge is formed, with probability p, it is formed by uniform attachment, and
with probability 1-p, it is formed by preferential attachment with alpha=1.
After choosing the attachment pattern, we choose the candidate set V to
fully determine the mixture component: with probability r, the candidate
set is all nodes in the network that have not been connected by the source
node, while with probability 1-r, the candidate set is restricted to the
friends-of-friends (FoF) of the source node.
"""

import argparse
import json
import os
import random
import numpy as np

import pandas as pd
import torch
from more_itertools import chunked
from torch import optim

import config
import utils
from model import (PreferentialAttachModel, PartitionLossFeatures,
                   TopOneLossFeatures, UniformAttachModel)
from utils import make_graph

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("-r", type=float, default=1,
                    help="The (nodewise) probability of choosing in all avaliable nodes")
parser.add_argument("-p", type=float, default=0,
                    help="The (nodewise) probability to form by uniform attachment.")
parser.add_argument("--fof", action="store_true",
                    help="Whether to include fof features in the model.")
parser.add_argument("--ua", action="store_true",
                    help="Whether to include the uniform attachment model.")
parser.add_argument("--pa", action="store_true",
                    help="Whether to include the preferential attachment model.")
parser.add_argument("--loss", type=str, default="topk", choices=["topk", "topone"],
                    help="Which of the loss functions to use.")
parser.add_argument("--rs", type=int, default=42,
                    help="Random seed.")
parser.add_argument("--gpu", default=0, type=int)
args = parser.parse_args()

if not os.path.exists(config.NETWORK_OUTPUT_DIR):
    os.makedirs(config.NETWORK_OUTPUT_DIR)

output_file = "{}_{}_{}_{}_{}_{}_{}.json".format(*vars(args).values())
output_file = os.path.join(config.NETWORK_OUTPUT_DIR, output_file)


# Generate synthetic data
rs = np.random.RandomState(args.rs)
samples = make_graph(r=args.r, p=args.p, n_max=1000, rs=rs)
data = pd.DataFrame(samples).set_index("index")
training_set = []
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

for idx in list(data.index.unique()):
    partial = data.loc[idx].reset_index()
    top = partial[partial["alt"] == 1][["deg", "is_fof"]].values
    bot = partial[partial["alt"] == 0][["deg", "is_fof"]].values
    training_set.append((
        torch.tensor(top, dtype=float, device=device),
        torch.tensor(bot, dtype=float, device=device)
    ))

# Models and model weights
models = {
    "ua": UniformAttachModel().to(device) if args.ua else None,
    "pa": PreferentialAttachModel(alpha=0.5 + random.random()).to(device) if args.pa else None
}

n_models = (int(args.ua) + int(args.pa)) * (1 + int(args.fof))
weights = torch.ones(n_models) / n_models
shift = 1

# Training
for itr in range(config.NETWORK_ITERATIONS):
    
    for fof in [False, True]:
        # check alphas shift is small enough to stop EM
        if shift < config.NETWORK_TOL:
            break

        # standard expectation maximization (E step)
        respon = []
        for comp in ["ua", "pa"]:
            model = models[comp]
            if model:
                respon.append(model.eval_prob(training_set, fof=False))
                if args.fof:
                    respon.append(model.eval_prob(training_set, fof=True))
        respon = torch.stack(respon)
        for i in range(n_models):
            respon[i] += torch.log(weights[i])

        gammas = respon - torch.logsumexp(respon, axis=0, keepdim=True)
        gammas = torch.exp(gammas)
        weights_new = torch.mean(gammas, axis=1)
        shift = torch.max(torch.abs(weights_new - weights)).item()
        weights = weights_new
        print("shift:", shift)
        if args.loss == "topk":
            criterion = PartitionLossFeatures(c=0, device=device)
        elif args.loss == "topone":
            criterion = TopOneLossFeatures()

        # only optimize pa
        model = models["pa"]
        # optimizer = optim.Adagrad(
        #     [{'params': model.parameters(), 'initial_lr': config.NETWORK_LR}], lr=config.NETWORK_LR)
        optimizer = optim.Adam(model.parameters(), lr=config.NETWORK_LR, betas=(0.9, 0.9), eps=0.01)
        batches = list(chunked(training_set, config.NETWORK_BATCH_SIZE))
        if not args.fof and fof:
            continue
        k = int(args.ua) * (int(args.fof) + 1) + int(fof)
        gamma = gammas[k]
        if n_models == 1:
            config.NETWORK_EPOCHS *= 5
        for e in range(config.NETWORK_EPOCHS):
            idx = 0
            for batch in batches:
                loss = 0
                for top, bot in batch:
                    top_w, bot_w = model(top, fof=fof), model(bot, fof=fof)
                    if gamma[idx] > config.NETWORK_TOL:
                        loss += criterion((top_w, bot_w)) * gamma[idx]
                    idx += 1
                if not isinstance(loss, int):
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                print(model.alpha.item())

# saving results
exp_type = 0
if args.fof and args.ua and args.pa:
    exp_type = 1
elif args.fof and not args.ua and args.pa:
    exp_type = 2
elif not args.fof and not args.ua and args.pa:
    exp_type = 3
with open(output_file, "w+") as fp:
    json.dump({
        "r": args.r,
        "p": args.p,
        "param": weights.cpu().detach().numpy().tolist(),
        "alpha": model.alpha.item(),
        "exp": exp_type
    }, fp)
