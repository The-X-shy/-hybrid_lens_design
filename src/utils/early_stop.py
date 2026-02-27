"""
Early stopping utility.
早停工具
"""

from typing import Optional


class EarlyStopper:
    """
    Monitor a scalar metric and decide whether to stop training.
    监控标量指标并决定是否早停
    """

    def __init__(
        self,
        enabled: bool = False,
        patience: int = 500,
        min_delta: float = 1e-6,
        mode: str = "min",
    ):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.enabled = enabled
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode

        self.best: Optional[float] = None
        self.num_bad_steps = 0
        self.stop_step: Optional[int] = None

    def _is_improvement(self, current: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return current < (self.best - self.min_delta)
        return current > (self.best + self.min_delta)

    def update(self, current: float, step: int) -> bool:
        """
        Returns True when training should stop.
        返回 True 表示应停止训练
        """
        if not self.enabled:
            return False

        if self._is_improvement(current):
            self.best = float(current)
            self.num_bad_steps = 0
            return False

        self.num_bad_steps += 1
        if self.num_bad_steps >= self.patience:
            self.stop_step = int(step)
            return True
        return False
