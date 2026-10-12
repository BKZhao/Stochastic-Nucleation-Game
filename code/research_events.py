"""Sweep-resolution joint events; t=0 is not part of the persistence window."""
from dataclasses import dataclass
import math


@dataclass
class CollapseDetector:
    rho_max: float
    resource_max: float
    confirm_sweeps: int
    last_time: int = 0
    streak: int = 0
    entry_time: int | None = None
    confirm_time: int | None = None

    def __post_init__(self) -> None:
        if (type(self.confirm_sweeps) is not int or self.confirm_sweeps < 1
                or not 0 <= self.rho_max <= 1 or not 0 <= self.resource_max <= 1):
            raise ValueError('invalid collapse threshold or persistence')

    def update(self, time: int, cooperation: float, resource: float) -> None:
        if time != self.last_time + 1:
            raise ValueError('event detection requires consecutive sweeps beginning at t=1')
        if not math.isfinite(cooperation) or not math.isfinite(resource):
            raise ValueError('nonfinite observable is a technical failure')
        self.last_time = time
        if self.confirm_time is not None:
            return
        self.streak = self.streak + 1 if cooperation <= self.rho_max and resource <= self.resource_max else 0
        if self.streak == self.confirm_sweeps:
            self.entry_time = time - self.confirm_sweeps + 1
            self.confirm_time = time
