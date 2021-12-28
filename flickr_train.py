"""Train Flickr model for analysis.

- 5.3 Experiments on Real-World Network Data

We sample 50000 users in a period (20 days) that have following events, and
record all the new edges formed by them. Edges formed earlier are assumed to be more
preferable than later ones. We then sample 100 edges uniformly at random from
the dataset as the negative samples for each user. The testing set is sampled
in the same way, with the sampling period later for 20 days.

The data should be downloaded first with:
wget -4 http://socialnetworks.mpi-sws.mpg.de/data/flickr-growth.txt.gz ../data/
input : ../data/flickr-growth.txt.gz
output: ../data/flickr-growth_choices.csv
"""

import random
from math import log

import numpy as np
import pandas as pd
import torch
from more_itertools import chunked
from torch import nn, optim
from tqdm import tqdm


def log1mexp(x):
    """Computes log(1-exp(-|x|)).

    See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    """
    x = -x.abs()
    return torch.where(x > -0.693,
                       torch.log(-torch.expm1(x)),
                       torch.log1p(-torch.exp(x)))


def gumbel_log_survival(x):
    """Computes log P(g > x) = log(1 - P(g < x)) = log(1 - exp(-exp(-x))) for a standard Gumbel"""
    y = torch.exp(-x)
    return log1mexp(y)


def test(mlogit, k=5):
    """Precision@k"""
    mlogit.eval()
    total = len(labels) * k
    cnt = 0
    for t, label in zip(testing_alts, labels):
        try:
            w = mlogit.forward(t).reshape(-1)
        except RuntimeError:
            continue
        w += torch.rand_like(w)*0.0001
        _, predict = torch.topk(w, min(k, len(label)))
        for p in predict.tolist():
            if p in label:
                cnt += 1
    mlogit.train()
    return cnt / total


def test_m(mlogit, ks=[1, 3, 5]):
    ret = []
    for k in ks:
        ret.append(test(mlogit, k))
    ret = np.around(np.array(ret), 4)
    return ret


class PartitionLossFeatures(object):
    def __init__(self, c=5., T=10000, device="cuda:0"):
        self.c = c
        self.T = T
        v = torch.arange(100, T + 100, dtype=torch.float32,
                         device=device) / (T + 100)
        self.logv = torch.log(v)
        self.loglogv = torch.log(-self.logv)[:, None, None]

    def __call__(self, partitions):
        T_w, B_w = partitions
        T_w = T_w.reshape(1, -1)
        B_w = B_w.reshape(1, -1)
        return self.nll_partition_loss(T_w, B_w)

    def nll_partition_loss(self, w_T_set, w_B_set):
        w_B = torch.logsumexp(w_B_set + self.c, dim=-1)
        _q = gumbel_log_survival(
            -((w_T_set + self.c)[None, :, :] + self.loglogv)
        )

        # mask
        q = _q.sum(-1) + (torch.expm1(w_B)[None, :] * self.logv[:, None])
        sum_q = torch.logsumexp(q, 0)

        return -sum_q - w_B


class MultiLogit(nn.Module):
    def __init__(self, in_features=3):
        """Multi-Logit Model.

        Compute the utility score w.
        Model list corresponding to `Table 3: Conditional logit model fits for Flickr data.` in
        `choose2grow`:
        - `#1` - 3
        - `#2` - 2
        - `#3` - 4
        - `#4` - 8
        """
        super().__init__()
        self.features = in_features
        self.fc1 = nn.Linear(in_features, 1, bias=False)
        torch.nn.init.normal_(self.fc1.weight, mean=1, std=0.01)

    def forward(self, x):
        if type(x) is not torch.Tensor:
            x = np.array(x, dtype=np.float32)
            x = torch.from_numpy(x)
        x = self.fc1(x)
        return x

    def print_param(self):
        param = self.fc1.weight
        param = param.detach().numpy().reshape(-1).copy()
        print(list(np.around(param, 3)))


def censored_log(x):
    if x == 0:
        return 0
    return log(x)


model_list = [
    ["log_deg", "has_deg", "recip"],
    ["recip", "fof"],
    ["log_deg", "has_deg", "recip", "fof"],
    ["log_deg", "has_deg", "recip", "hops_2",
        "hops_3", "hops_4", "hops_5", "hops_6"]
]

# create features

train_df: pd.DataFrame = pd.read_csv("flickr_train.csv")
test_df: pd.DataFrame = pd.read_csv("flickr_test.csv")

for df in [train_df, test_df]:
    df["log_deg"] = np.vectorize(censored_log)(df["deg"])
    df["has_deg"] = np.where(df["deg"] > 0, 1, 0)
    df["fof"] = np.where(df["hops"] == 2, 1, 0)
    for i in [2, 3, 4, 5]:
        col_name = "hops_" + str(i)
        df[col_name] = np.where(df["hops"] == i, 1, 0)
    df["hops_6"] = np.where(df["hops"] >= 6, 1, 0)

train_df.set_index("index", inplace=True)

# prepare training sets
print("Prepare training batches")
training_sets = []
for model_params in model_list:
    training_partitions = []
    for idx in tqdm(list(train_df.index.unique()), disable=False):
        order = []
        partial = train_df.loc[idx]
        order.append(partial[(partial["alt"] > 0) & (partial["alt"]<=10)]
                         [model_params].values.tolist())
        order.append(partial[(partial["alt"] > 10)]
                         [model_params].values.tolist())
        order.append(partial[partial["alt"] == 0]
                         [model_params].values.tolist())
        order = [x for x in order if len(x) > 0]
        training_partitions.append(order)
    training_sets.append(training_partitions)

# prepare training batchs
tbs = []
for training_partitions in training_sets:
    training_batches = []
    for order in training_partitions:
        training_batches.append((
            [item for sublist in order[:-1] for item in sublist],
            order[-1]
        ))
    tbs.append(training_batches)
print("Prepared %d training batches" % len(training_batches))

# prepare test sets
ta = []
for model_params in model_list:
    testing_alts = []
    labels = []
    num_tests = len(test_df["index"].unique())
    for idx in test_df["index"].unique():
        partial = test_df[test_df["index"] == idx].reset_index()
        label = list(partial.index[partial["alt"] == 1])
        testing_alts.append(
            partial[model_params].values.tolist()
        )
        labels.append(label)
    ta.append(testing_alts)

# baseline
print("baseline")
baseline_params = [[1.149, -0.580, 8.419],
                   [8.347, 6.12],
                   [0.715, -0.631, 8.197, 3.955],
                   [0.536, -1.745, 7.903, 6.290, 2.851, 0.583, -0.585, -1.122]]
for model in [1, 2, 3, 4]:
    model_params = model_list[model-1]
    mlogit = MultiLogit(in_features=len(model_params))
    testing_alts = ta[model - 1]
    mlogit = MultiLogit(len(model_params))
    mlogit.fc1.weight = nn.Parameter(torch.tensor(baseline_params[model - 1]))
    print(test_m(mlogit))

for model in [1, 2, 3, 4]:

    model_params = model_list[model-1]
    mlogit = MultiLogit(in_features=len(model_params))
    optimizer = optim.Adam(mlogit.parameters(), lr=0.1,
                           betas=(0.25, 0.9), eps=0.01)
    criterion = PartitionLossFeatures(device="cpu")

    # training
    epochs = 2
    batch_size = 512
    training_batches = tbs[model - 1]
    testing_alts = ta[model - 1]

    for epoch in range(epochs):
        random.shuffle(training_batches)
        batches = chunked(training_batches, batch_size)
        for step, batch in enumerate(batches):
            loss = 0
            print("Epoch {} Step {} Test {}".format(
                epoch + 1, step + 1, test_m(mlogit)), end=" ")
            mlogit.print_param()
            for T, B in batch:
                T_w = mlogit(T).reshape(1, -1)
                B_w = mlogit(B).reshape(1, -1)
                loss += criterion((T_w, B_w))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print("Final Parameters:")
    mlogit.print_param()
    print("Final Metrics:", test_m(mlogit))
