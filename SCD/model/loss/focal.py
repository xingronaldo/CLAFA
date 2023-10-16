"""
https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
"""

import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from einops import rearrange
from typing import Optional, List
from functools import partial
from torch.nn.modules.loss import _Loss
import numpy as np


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=4.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        if isinstance(alpha, (float, int)):
            self.alpha = torch.as_tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.as_tensor(alpha)

    def forward(self, input, target):
        N, C, H, W = input.size()
        assert C == 2
        # input = input.view(N, C, -1)
        # input = input.transpose(1, 2)
        # input = input.contiguous().view(-1, C)
        input = rearrange(input, 'b c h w -> (b h w) c')
        # input = input.contiguous().view(-1)

        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)
        loss = -1 * (1-pt)**self.gamma * logpt

        return loss.mean()


class FocalLoss(_Loss):
    def __init__(
        self,
        alpha: Optional[float] = 0.25,
        gamma: Optional[float] = 2.0,
        ignore_index: Optional[int] = None,
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
    ):
        super(FocalLoss, self).__init__()
        self.ignore_index = ignore_index
        self.focal_loss_fn = partial(
            focal_loss_with_logits,
            alpha=alpha,
            gamma=gamma,
            reduced_threshold=reduced_threshold,
            reduction='mean',
            normalized=normalized
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        num_classes = y_pred.size(1)
        loss = 0

        # Filter anchors with -1 label from loss computation
        if self.ignore_index is not None:
            not_ignored = y_true != self.ignore_index

        for cls in range(num_classes):
            cls_y_true = (y_true == cls).long()
            cls_y_pred = y_pred[:, cls, ...]

            if self.ignore_index is not None:
                cls_y_true = cls_y_true[not_ignored]
                cls_y_pred = cls_y_pred[not_ignored]

            loss += self.focal_loss_fn(cls_y_pred, cls_y_true)

        return loss


def focal_loss_with_logits(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    reduction: str = 'mean',
    normalized: bool = True,
    reduced_threshold: Optional[float] = None,
    eps: float = 1e-6
) -> torch.Tensor:

    target = target.type(output.type())
    logpt = F.binary_cross_entropy_with_logits(output, target, reduction='none')
    pt = torch.exp(-logpt)

    # compute the loss
    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term[pt < reduced_threshold] = 1

    loss = focal_term * logpt

    if alpha is not None:
        loss *= alpha * target + (1 - alpha) * (1 - target)

    if normalized:
        norm_factor = focal_term.sum().clamp_min(eps)
        loss /= norm_factor

    if reduction == 'mean':
        loss = loss.mean()
    if reduction == 'sum':
        loss = loss.sum()
    if reduction == 'batchwise_mean':
        loss = loss.sum(0)

    return loss



if __name__ == '__main__':
    x = torch.randn(3, 7, 256, 256)
    y = torch.ones(3, 256, 256).long()
    model = FocalLoss(alpha=0.25, gamma=2)
    out = model(x, y)
    print(out)
