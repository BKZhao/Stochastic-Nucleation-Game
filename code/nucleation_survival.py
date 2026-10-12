"""Administrative censoring, right-continuous Kaplan-Meier and finite-horizon risk."""
import math
from statistics import NormalDist
import numpy as np


def wilson_interval(events: int, count: int, confidence: float = .95) -> list[float]:
    if count < 1 or not 0 <= events <= count:
        raise ValueError('invalid binomial counts')
    z=NormalDist().inv_cdf((1+confidence)/2)
    p=events/count
    den=1+z*z/count
    mid=(p+z*z/(2*count))/den
    half=z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
    return [max(0.,mid-half),min(1.,mid+half)]


def survival_summary(times: list[int], observed: list[bool], horizon: int) -> dict:
    time=np.asarray(times,dtype=float)
    event=np.asarray(observed,dtype=bool)
    if time.ndim != 1 or event.shape != time.shape or len(time)==0:
        raise ValueError('nonempty aligned survival observations required')
    if horizon <= 0 or not np.isfinite(time).all() or np.any(time<0):
        raise ValueError('invalid time or horizon')
    if time.max()<horizon:
        # Beyond the final event, S=0 is known; beyond final censoring it is not.
        final_risk=np.sum(time>=time.max())
        if np.sum(event & (time==time.max())) != final_risk:
            raise ValueError('RMST horizon extends beyond supported follow-up')
    survival,area,last=1.,0.,0.
    median=None
    table=[dict(time=0.,survival=1.,at_risk=len(time),events=0,censored=0)]
    for value in np.unique(time[time<=horizon]):
        area += survival*(value-last)
        at_risk=int(np.sum(time>=value))
        deaths=int(np.sum((time==value)&event))
        censored=int(np.sum((time==value)&~event))
        survival *= 1-deaths/at_risk
        if median is None and survival<=.5:
            median=float(value)
        table.append(dict(time=float(value),survival=float(survival),at_risk=at_risk,
                          events=deaths,censored=censored))
        last=value
    area += survival*(horizon-last)
    return dict(n=len(time),events=int(np.sum(event&(time<=horizon))),horizon=horizon,
                survival=float(survival),rmst=float(area),median=median,km=table)


def fisher_two_sided(events_a: int, count_a: int, events_b: int, count_b: int) -> float:
    """Exact 2x2 probability-ordering p-value, comparing integer table weights."""
    for events,count in [(events_a,count_a),(events_b,count_b)]:
        if type(events) is not int or type(count) is not int or count<1 or not 0<=events<=count:
            raise ValueError('integer binomial counts required')
    successes=events_a+events_b
    observed=math.comb(count_a,events_a)*math.comb(count_b,events_b)
    weights=[math.comb(count_a,k)*math.comb(count_b,successes-k)
             for k in range(max(0,successes-count_b),min(count_a,successes)+1)]
    return sum(weight for weight in weights if weight<=observed)/math.comb(count_a+count_b,successes)


def risk_difference_interval(events_a: int, count_a: int, events_b: int, count_b: int,
                             confidence: float=.95) -> list[float]:
    """Newcombe interval from independent Wilson limits, without continuity correction."""
    low_a,high_a=wilson_interval(events_a,count_a,confidence)
    low_b,high_b=wilson_interval(events_b,count_b,confidence)
    a,b=events_a/count_a,events_b/count_b
    difference=a-b
    return [max(-1.,difference-math.hypot(a-low_a,high_b-b)),
            min(1.,difference+math.hypot(high_a-a,b-low_b))]
