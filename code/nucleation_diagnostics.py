"""Descriptive survivor diagnostics; these do not establish a quasi-stationary state."""
from dataclasses import dataclass
import numpy as np
from nucleation_survival import survival_summary


@dataclass(frozen=True)
class RunTrace:
    run_id: str
    event_time: int | None
    horizon: int
    times: np.ndarray
    cooperation: np.ndarray
    resource: np.ndarray
    defector_fraction: np.ndarray
    streak: np.ndarray


def _sample_index(run: RunTrace, time: int) -> int:
    indices=np.flatnonzero(run.times==time)
    if len(indices)!=1:
        raise ValueError(f'exactly one sample at t={time} required: {run.run_id}')
    return int(indices[0])


def _distribution(values: np.ndarray) -> dict | None:
    if not len(values):
        return None
    return dict(mean=float(values.mean()),q25=float(np.quantile(values,.25)),
                median=float(np.median(values)),q75=float(np.quantile(values,.75)))


def landmark_summary(runs: list[RunTrace], landmark: int, window: int) -> dict:
    """Condition on tau_B > landmark; retain failures and pending confirmation counts."""
    if not runs or landmark<0 or window<=0 or any(run.horizon<landmark+window for run in runs):
        raise ValueError('nonempty cohort and full administrative follow-up required')
    survivors=[run for run in runs if run.event_time is None or run.event_time>landmark]
    rows=[]
    times,events=[],[]
    for run in survivors:
        index=_sample_index(run,landmark)
        rows.append([run.cooperation[index],run.resource[index],run.defector_fraction[index],run.streak[index]])
        end=run.horizon if run.event_time is None else run.event_time
        times.append(min(end-landmark,window))
        events.append(run.event_time is not None and run.event_time<=landmark+window)
    array=np.asarray(rows,dtype=float).reshape(-1,4)
    active=(array[:,0]>=.8)&(array[:,1]>=.6)&(array[:,2]<=.05)&(array[:,3]==0)
    return dict(landmark=landmark,window=window,total=len(runs),
                confirmed_before_or_at=len(runs)-len(survivors),at_risk=len(survivors),
                pending_confirmation=int(np.sum(array[:,3]>0)),screen_active=int(np.sum(active)),
                cooperation=_distribution(array[:,0]),resource=_distribution(array[:,1]),
                max_defector_fraction=_distribution(array[:,2]),
                conditional=survival_summary(times,events,window) if times else None)


def paired_window_drift(runs: list[RunTrace], early: tuple[int,int], late: tuple[int,int],
                        bootstrap_samples: int=2000, seed: int=202610053) -> dict:
    """Later minus earlier trajectory means; a retrospective common-survivor cohort."""
    if not 0<=early[0]<early[1]<late[0]<late[1] or bootstrap_samples<1:
        raise ValueError('nonoverlapping ordered windows and positive bootstrap count required')
    eligible=[run for run in runs if run.horizon>=late[1] and (run.event_time is None or run.event_time>late[1])]
    result=dict(n=len(eligible),total=len(runs),early=list(early),late=list(late),
                conditioning=f'confirmed event strictly after {late[1]} or censored; retrospective selection')
    if not eligible:
        return result
    rng=np.random.default_rng(seed)
    selections=rng.integers(0,len(eligible),size=(bootstrap_samples,len(eligible)))
    for name in ['cooperation','resource','defector_fraction']:
        pairs=[]
        for run in eligible:
            for endpoint in (*early,*late):
                _sample_index(run,endpoint)
            values=getattr(run,name)
            pairs.append([float(values[(run.times>=start)&(run.times<=end)].mean()) for start,end in (early,late)])
        pairs=np.asarray(pairs)
        difference=pairs[:,1]-pairs[:,0]
        estimate=difference[selections].mean(axis=1)
        result[name]=dict(early_mean=float(pairs[:,0].mean()),late_mean=float(pairs[:,1].mean()),
                          difference=float(difference.mean()),bootstrap95=np.quantile(estimate,[.025,.975]).tolist())
    return result
