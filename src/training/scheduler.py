"""Learning rate schedulers with Warmup and Cosine Annealing."""

import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def create_cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_epochs: int,
    max_epochs: int,
    min_lr: float = 1e-6,
    base_lr: float | None = None,
) -> LambdaLR:
    r"""Create a learning rate scheduler with Linear Warmup followed by Cosine Annealing.

    During the first `warmup_epochs`, learning rate increases linearly from
    `min_lr` to `base_lr`. Afterwards, it decays following a cosine curve down to `min_lr`.

    Args:
        optimizer: PyTorch optimizer instance.
        warmup_epochs: Number of initial epochs for linear warmup.
        max_epochs: Total number of training epochs.
        min_lr: Minimum target learning rate at the end of training. Default: 1e-6.
        base_lr: Initial peak learning rate. If None, uses `param_groups[0]['lr']`.

    Returns:
        torch.optim.lr_scheduler.LambdaLR instance.
    """
    if base_lr is None:
        base_lr = optimizer.param_groups[0]["lr"]

    min_ratio = min_lr / base_lr

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            # Linear warmup from min_ratio to 1.0
            alpha = (epoch + 1) / max(1, warmup_epochs)
            return min_ratio + (1.0 - min_ratio) * alpha

        # Cosine annealing decay from 1.0 down to min_ratio
        progress = (epoch - warmup_epochs) / max(1, max_epochs - warmup_epochs)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
