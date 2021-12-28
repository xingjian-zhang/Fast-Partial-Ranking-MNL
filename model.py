import numpy as np
import torch
from torch._C import dtype
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List
import utils


class LinearModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.w = nn.Parameter(torch.rand(
            (1, num_classes), dtype=torch.float32) * 4)
        self.pad = nn.Parameter(torch.tensor(
            [[float("inf"), -float("inf")]]), requires_grad=False)

    def forward(self, batch_size=1):
        output = torch.cat((self.pad, self.w), 1)
        return output.expand(batch_size, -1)

    def get_numpy_prob(self):
        return F.softmax(self.w.detach().squeeze(), dim=-1).cpu().numpy()

    def get_torch_param(self):
        return self.w.detach().squeeze().cpu()

    def init_param(self, param: np.array):
        self.w = nn.Parameter(torch.Tensor(param.reshape(1, -1)))

    def eval_prob(self, partial_orders: List[List[torch.Tensor]]) -> torch.Tensor:
        """Evaluate the probability of every data point in the partial orders.

        Returns an array of log-probability corresponding to each data point.
        """
        self.eval()
        prob = []
        collate_padding = utils.get_collate_fn()
        device = partial_orders[0][0].device
        with torch.no_grad():
            criterion = PartitionLoss(device=device)
            for partitions in partial_orders:
                evalset = []
                batch_size = len(partitions) - 1
                for i in range(batch_size):
                    T = partitions[i]
                    B = torch.cat(partitions[i + 1:])
                    evalset.append((T + 2, B + 2))

                if batch_size == 0:
                    prob.append(0)
                    continue
                batch_T, batch_B = collate_padding(evalset)
                w = self.forward(batch_size)
                log_p = -criterion.nll_partition_loss(w, batch_T, batch_B)
                prob.append(torch.sum(log_p).item())
            return torch.tensor(prob, device=w.device)


class PartitionLoss(object):
    def __init__(self, c=5., T=10000, device="cuda:0"):
        self.c = c
        self.T = T
        v = torch.arange(100, T + 100, dtype=torch.float32,
                         device=device) / (T + 100)
        self.logv = torch.log(v)
        self.loglogv = torch.log(-self.logv)[:, None, None]

    def __call__(self, outputs, partitions, gamma=None):
        w = outputs
        T, B = partitions
        loss = self.nll_partition_loss(w, T, B, gamma=gamma)
        return torch.mean(loss)

    def nll_partition_loss(self, w, Top, Bot, gamma=None):
        """
        w: (BatchSize, NumPadding + NumClass).
        Top: (BatchSize, TopSize).
        Bot: (BatchSize, BotSize).
        gamma: (BatchSize,). Weights, used only in the EM algorithm.
        """
        w_B_set = torch.gather(w, 1, Bot)
        w_T_set = torch.gather(w, 1, Top)

        w_B = torch.logsumexp(w_B_set + self.c, dim=-1)
        _q = gumbel_log_survival(
            -((w_T_set + self.c)[None, :, :] + self.loglogv)
        )

        q = _q.sum(-1) + (torch.expm1(w_B)[None, :] * self.logv[:, None])
        sum_q = torch.logsumexp(q, 0)

        loss = -sum_q - w_B
        if gamma is not None:
            loss *= gamma
        return loss


class TopOneLossFeatures(object):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, partitions):
        T_w, B_w = partitions
        ret = 0
        for i in range(len(T_w)):
            ret += self.neg_log_likelihood(T_w[[i]], B_w)
        return ret

    def neg_log_likelihood(self, T_w, B_w):
        if T_w < 0:
            return torch.tensor([float("Inf")])
        B_w = torch.cat((T_w, B_w))
        return -T_w + torch.logsumexp(B_w, dim=-1)


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
        T_w = T_w.reshape(1, -1).to(self.loglogv.device)
        B_w = B_w.reshape(1, -1).to(self.loglogv.device)
        # print(T_w.device)
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


class AttachModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, fof=True):
        pass

    def eval_prob(self, partial_orders: List[List[List[torch.Tensor]]], fof=True, loss="topk"):
        """Return the log likelihood."""
        self.eval()
        prob = []
        if loss == "topk":
            criterion = PartitionLossFeatures(
                c=0, device=partial_orders[0][0][0].device)
        else:
            criterion = TopOneLossFeatures()
        for partitions in partial_orders:
            top, bot = partitions
            t_w = self.forward(top, fof=fof)
            b_w = self.forward(bot, fof=fof)
            prob.append(criterion((t_w, b_w)))
        self.train()
        return -torch.tensor(prob)


class UniformAttachModel(AttachModel):
    def __init__(self):
        super().__init__()

    def forward(self, x, fof=True):
        if fof:
            # m = torch.mean(y[x[..., 1] == 1.])
            return torch.where(x[..., 1] == 1., 1., -np.inf)
        else:
            return torch.ones(size=x.shape[:-1])


class PreferentialAttachModel(AttachModel):
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha], dtype=float))

    def forward(self, x, fof=True):
        y = torch.log(x[..., 0]) * self.alpha
        if fof:
            return torch.where(x[..., 1] == 1., y, -np.inf)
        else:
            return y


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
