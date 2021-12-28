"""Train MAG for analysis.

- 5.3 Experiments on Real-World Network Data

"""

import json
import torch
import numpy as np
import random
from torch import nn
from torch import optim
from math import log

with open("./mag.json", "r") as fp:
    mag_data = json.load(fp)


def censored_log(x: torch.Tensor):
    x = torch.where(x < 1, torch.zeros_like(x), torch.log(x))
    return x


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
        """
        super().__init__()
        self.features = in_features
        self.fc1 = nn.Linear(in_features, 1, bias=False)
        torch.nn.init.normal_(self.fc1.weight, mean=0, std=0.01)


    def forward(self, x):
        if type(x) is not torch.Tensor:
            x = np.array(x, dtype=np.float32)
            x = torch.from_numpy(x)
        x = self.fc1(x[..., :self.features])
        return x

    def set_param(self, x):
        self.fc1.weight = nn.Parameter(torch.tensor(x, requires_grad=True))

    def print_param(self):
        param = self.fc1.weight
        param = param.detach().numpy().reshape(-1).copy()
        print(list(np.around(param, 3)))

# collate batches
random.shuffle(mag_data)
dataset = [(torch.tensor(t), torch.tensor(b)) for (t,b) in mag_data]

# apply censored_log

for t, b in dataset:
    t[..., -1] = censored_log(t[..., -1])
    b[..., -1] = censored_log(b[..., -1])


# split dataset
num_test = 2000
trainset = dataset[:-num_test]
testset = dataset[-num_test:]

# collate testset
testset_ = []
for t, b in testset:
    labels = list(range(len(t)))
    options = torch.cat((t, b))
    testset_.append(
        (options, labels)
    )
    
# collate top-1 testset
top1_testset = []
for t, b in testset:
    top_one = t[random.randint(0, len(t)-1)]
    options = torch.cat((top_one.reshape(1, -1), b[:24]))
    top1_testset.append(
        (options, labels)
    )

testset = testset_

# test methods
def test(mlogit, k=5):
    """Precision@k"""
    mlogit.eval()
    total = len(testset) * k
    cnt = 0
    for t, label in testset:
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

def test_m(mlogit, ks=[1, 3, 5, 10]):
    ret = []
    for k in ks:
        ret.append(test(mlogit, k))
    ret = np.around(np.array(ret),4)
    return ret

# baseline
print("baseline")
# 1
mlogit = MultiLogit(2)
mlogit.set_param([0.717, 1.684])
print(test_m(mlogit))
# 2
mlogit = MultiLogit(3)
mlogit.set_param([0.794, 1.677, 6.523])
print(test_m(mlogit))
# 3
mlogit = MultiLogit(4)
mlogit.set_param([1.052,1.862,5.928,-1.096])
print(test_m(mlogit))
# 4
mlogit = MultiLogit(5)
mlogit.set_param([1.044,1.830,5.913, -1.069, 0.029])
print(test_m(mlogit))

from more_itertools import chunked

for n_params in [2, 3, 4, 5]:

    mlogit = MultiLogit(n_params)
    optimizer = optim.Adam(mlogit.parameters(), lr=0.1, betas=(0.75, 0.9), eps=0.01)
    criterion = PartitionLossFeatures(device="cpu",c=2)

    # training
    epochs = 5
    batch_size = 512

    for epoch in range(epochs):
        random.shuffle(trainset)
        batches = chunked(trainset, batch_size)
        for step, batch in enumerate(batches):
            loss = 0
            print("Epoch {} Step {} Test {}".format(
                epoch + 1, step + 1, test_m(mlogit)), end=" ")
            mlogit.print_param()
            for T, B in batch:
                try:
                    # T = T.to("cuda:0")
                    # B = B.to("cuda:0")
                    T_w = mlogit(T).reshape(1, -1)
                except:
                    exit()
                B_w = mlogit(B).reshape(1, -1)
                _, len_T = T_w.shape
                if len_T > 0:
                    loss += criterion((T_w, B_w))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print("Final Parameters:", list(mlogit.parameters())[0])
    print("Final Metrics:", test_m(mlogit))