"""Loss functions for scMMT."""

from __future__ import annotations

import torch


class no_loss:
    """A differentiable zero loss used when a task is disabled."""

    def __init__(self, device):
        self.device = device

    def __call__(self, outputs, target, **kwargs):
        if isinstance(outputs, torch.Tensor):
            return outputs.sum() * 0.0
        return torch.tensor(0.0, device=self.device)


class mse_loss:
    """Mean squared error with optional censoring for unmeasured proteins.

    The mask is expected to be broadcastable to ``yhat``.  Only observed
    entries contribute to the reduced loss, so reference datasets with
    different protein panels receive comparable weighting.
    """

    def __init__(self, reduce: bool = True):
        self.reduce = reduce

    def __call__(self, yhat, y, bools=None):
        squared_errors = (yhat - y) ** 2
        if bools is None:
            return squared_errors.mean() if self.reduce else squared_errors

        mask = bools.to(device=squared_errors.device, dtype=squared_errors.dtype)
        masked_errors = squared_errors * mask
        if not self.reduce:
            return masked_errors

        observed = mask.expand_as(squared_errors).sum()
        if observed.item() == 0:
            return yhat.sum() * 0.0
        return masked_errors.sum() / observed
