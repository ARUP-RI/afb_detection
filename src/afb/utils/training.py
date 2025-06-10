import numpy as np
from torch.optim.lr_scheduler import LRScheduler


class WarmupCosineLRScheduler(LRScheduler):
    """A learning rate schedule that increases linearly from 0 to max_lr over warmup_iters, then
    decreases using cosine decay back down to min_lr for lr_decay_iters, then stays there forever
    """

    def __init__(self, optimizer, max_lr, min_lr, warmup_iters, lr_decay_iters):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.last_lr = float("NaN")
        super().__init__(optimizer)

    def set_iters(self, iters):
        self._step_count = iters

    def get_last_lr(self):
        return self.last_lr

    def get_lr(self):
        """
        Sets the LR for every param_group to be the same value
        """
        # 1) linear warmup for warmup_iters steps
        if self._step_count < self.warmup_iters:
            lr = self.max_lr * (self._step_count + 1) / (self.warmup_iters)
        # 2) if it > lr_decay_iters, return min learning rate
        elif self._step_count > self.lr_decay_iters:
            lr = self.min_lr
        else:
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (self._step_count - self.warmup_iters) / (
                self.lr_decay_iters - self.warmup_iters
            )
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))  # coeff ranges 0..1
            lr = self.min_lr + coeff * (self.max_lr - self.min_lr)
        lr = [lr for _ in self.optimizer.param_groups]
        self.last_lr = lr
        return lr
