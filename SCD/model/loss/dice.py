import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from kornia.losses import dice_loss
from torch.autograd import Variable
from typing import Optional, List
from functools import partial
from torch.nn.modules.loss import _Loss
import numpy as np

class BinaryDICELoss(nn.Module):
    def __init__(self):
        super(BinaryDICELoss, self).__init__()

    def forward(self, input, target):
        target = target.squeeze(1)
        loss = dice_loss(input, target)

        return loss


### version 2
"""
class BinaryDICELoss(nn.Module):
    def __init__(self, eps=1e-5):
        super(BinaryDICELoss, self).__init__()
        self.eps = eps

    def to_one_hot(self, target):
        N, C, H, W = target.size()
        assert C == 1
        target = torch.zeros(N, 2, H, W).to(target.device).scatter_(1, target, 1)
        return target

    def forward(self, input, target):
        N, C, _, _ = input.size()
        input = F.softmax(input, dim=1)

        #target = self.to_one_hot(target)
        target = torch.eye(2)[target.squeeze(1)]
        target = target.permute(0, 3, 1, 2).type_as(input)

        dims = tuple(range(1, target.ndimension()))
        inter = torch.sum(input * target, dims)
        cardinality = torch.sum(input + target, dims)
        loss = ((2. * inter) / (cardinality + self.eps)).mean()

        return 1 - loss
"""

class DICELoss(_Loss):
    def __init__(
        self,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        super(DICELoss, self).__init__()
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)
        y_pred = y_pred.log_softmax(dim=1).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(bs, -1)
        y_pred = y_pred.view(bs, num_classes, -1)

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask.unsqueeze(1)

            y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W
        else:
            y_true = F.one_hot(y_true.to(torch.long), num_classes)  # N,H*W -> N,H*W, C
            y_true = y_true.permute(0, 2, 1) # N, C, H*W

        scores = self.compute_score(y_pred, y_true.type_as(y_pred), smooth=self.smooth, eps=self.eps, dims=dims)
        loss = 1.0 - scores

        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)


def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x


def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)

    return dice_score


if __name__ == '__main__':
    x = torch.randn(3, 7, 256, 256)
    y = torch.zeros(3, 256, 256).long()
    model = DICELoss()
    output = model(x, y)
    print(output)
