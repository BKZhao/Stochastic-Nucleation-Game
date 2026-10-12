"""Exact population-growth clocks and finite-window outcomes, without nucleus labels."""
from dataclasses import dataclass, field
import math

from nucleation_survival import survival_summary, wilson_interval


@dataclass
class GrowthTracker:
    """Observe every sweep, including t=0; first crossings may subsequently reverse."""
    node_count: int
    fractions: tuple[float, ...]
    first_crossings: dict[str, int | None] = field(init=False)
    extinction_time: int | None = field(default=None, init=False)
    last_time: int = field(default=-1, init=False)

    def __post_init__(self) -> None:
        if type(self.node_count) is not int or self.node_count < 1:
            raise ValueError('positive integer population required')
        if (not self.fractions or any(type(value) not in (int, float) or not math.isfinite(value)
                                     or not 0 < value <= 1 for value in self.fractions)
                or any(a >= b for a, b in zip(self.fractions, self.fractions[1:]))):
            raise ValueError('strictly increasing finite fractions in (0,1] required')
        self.first_crossings = {str(float(value)): None for value in self.fractions}

    def update(self, time: int, defectors: int) -> tuple[str, ...]:
        if type(time) is not int or time != self.last_time + 1:
            raise ValueError('growth tracker requires consecutive sweeps starting at zero')
        if type(defectors) is not int or not 0 <= defectors <= self.node_count:
            raise ValueError('integer defector count within population required')
        self.last_time = time
        if defectors == 0 and self.extinction_time is None:
            self.extinction_time = time
        crossed = []
        for fraction in self.fractions:
            key = str(float(fraction))
            if self.first_crossings[key] is None and defectors >= math.ceil(fraction * self.node_count):
                self.first_crossings[key] = time
                crossed.append(key)
        return tuple(crossed)

    def record(self) -> dict:
        return dict(definition='first_defector_fraction_crossing_each_sweep_v1',
                    first_crossings=dict(self.first_crossings), extinction_time=self.extinction_time,
                    observed_through=self.last_time,
                    interpretation='population milestones, reversible; not nucleation or commitment times')


def outcome_summary(runs: list[dict], time: int) -> dict:
    """Partition original mu=0 cohort at the requested time, retaining future outcomes as unresolved."""
    if not runs or type(time) is not int or time <= 0:
        raise ValueError('nonempty cohort and positive integer observation time required')
    if any(run['technical_status'] != 'complete' or run['parameters']['mutation'] != 0
           or run['protocol']['t_max_sweeps'] < time for run in runs):
        raise ValueError('complete mutation-free trajectories with full administrative follow-up required')
    times, events = [], []
    recovered = 0
    for run in runs:
        event_time = run['t_confirm_B']
        recovery_time = run['recovered_time']
        if event_time is not None and recovery_time is not None:
            raise ValueError('first-confirmation-stopped mu=0 run cannot also be recovered')
        event = event_time is not None and event_time <= time
        times.append(event_time if event else time)
        events.append(event)
        recovered += int(recovery_time is not None and recovery_time <= time)
    result = survival_summary(times, events, time)
    return dict(result, recovered=recovered, unresolved=len(runs)-sum(events)-recovered,
                collapse_risk=sum(events)/len(runs), wilson95=wilson_interval(sum(events), len(runs)))
