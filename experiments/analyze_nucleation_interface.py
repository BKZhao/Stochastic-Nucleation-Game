"""Early-window interface diagnostics: J trajectories, per-sweep drift calibration and prediction."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_nucleation_followup import load_campaign, read_trace
from nucleation_interface import resource_substep
from research_common import PROJECT_ROOT, config_hash, file_hash, write_json

def validate_interface_jobs(jobs: list, reference: dict) -> None:
    """Event-clock and initial-intervention validation tolerant of seeded resource fields."""
    from copy import deepcopy
    expected = deepcopy(reference)
    for key in ['damage', 'experiment_id']:
        expected.pop(key)
    for config, summary, directory in jobs:
        comparable = deepcopy(config)
        for key in ['damage', 'experiment_id']:
            comparable.pop(key)
        if comparable != expected:
            raise ValueError('interface comparisons require identical model, horizon and endpoint')
        read_trace(config, summary, directory)
        damage = config['damage']
        with np.load(directory / 'states' / 'initial.npz', allow_pickle=False) as stored:
            strategy, resource = stored['strategy'], stored['resource']
            if (np.count_nonzero(strategy == 0) != damage['count']
                    or not np.isin(strategy, [0, 1]).all()):
                raise ValueError(f'initial strategy intervention mismatch: {directory}')
            if damage.get('resource_field', 'uniform') == 'equilibrium':
                model = config['model']
                again = resource_substep(strategy, resource, model['size'],
                                         model['growth'], model['depletion'], model['diffusion'])
                if not np.allclose(again, resource, atol=1e-9):
                    raise ValueError(f'initial field is not the fixed-strategy equilibrium: {directory}')
            else:
                value = damage.get('resource_value', 0.)
                if (not np.all(resource[strategy == 0] == value)
                        or not np.all(resource[strategy == 1] == 1)):
                    raise ValueError(f'initial strategy/resource intervention mismatch: {directory}')
        with np.load(directory / 'events.npz', allow_pickle=False) as stored:
            confirm = summary['t_confirm_B']
            if (bool(stored['event_observed']) != summary['event_observed']
                    or int(stored['t_confirm_B']) != (-1 if confirm is None else confirm)):
                raise ValueError(f'event file and summary disagree: {directory}')


TRAJECTORY_TIMES = [0, 1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500]
THRESHOLD_FRACTION = 0.05


def _stats(values):
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return None
    return dict(n=int(array.size), mean=float(array.mean()), median=float(np.median(array)),
                q25=float(np.quantile(array, .25)), q75=float(np.quantile(array, .75)),
                min=float(array.min()), max=float(array.max()))


def _pearson(x, y):
    if len(x) < 2:
        return None
    return float(np.corrcoef(np.asarray(x, float), np.asarray(y, float))[0, 1])


def _rankdata(values):
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind='mergesort')
    ranks = np.empty(array.size, dtype=float)
    ranks[order] = np.arange(1, array.size + 1)
    # average ranks for ties
    unique, inverse = np.unique(array, return_inverse=True)
    if unique.size != array.size:
        counts = np.bincount(inverse)
        sums = np.bincount(inverse, weights=ranks)
        ranks = sums[inverse] / counts[inverse]
    return ranks


def _spearman(x, y):
    if len(x) < 3:
        return None
    return _pearson(_rankdata(x), _rankdata(y))


def _auc(labels, scores):
    """Mann-Whitney AUC with average ranks for ties."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    ranks = _rankdata(scores)
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * neg.sum()))


def _ols(x, y):
    if len(x) < 3:
        return None
    slope, intercept = np.polyfit(np.asarray(x, float), np.asarray(y, float), 1)
    return dict(slope=float(slope), intercept=float(intercept), n=int(len(x)))


def _binned_calibration(predictor, response, bins=10):
    """Non-overlapping quantile bins; counts must sum to the number of observations.

    Bin i is half-open [e_i, e_{i+1}) with the final bin closed at the maximum, and
    quantile edges are de-duplicated so tied values cannot be counted twice.
    """
    predictor = np.asarray(predictor, dtype=float)
    response = np.asarray(response, dtype=float)
    if predictor.size == 0:
        return None
    edges = np.unique(np.quantile(predictor, np.linspace(0, 1, bins + 1)))
    if edges.size == 1:
        return [dict(count=int(predictor.size),
                     J_pre_mean=float(predictor.mean()),
                     delta_ND_mean=float(response.mean()),
                     delta_ND_median=float(np.median(response)))]
    assignment = np.clip(np.searchsorted(edges, predictor, side='right') - 1,
                         0, edges.size - 2)
    cells = []
    for index in range(edges.size - 1):
        inside = assignment == index
        if not inside.any():
            continue
        cells.append(dict(count=int(inside.sum()),
                          J_pre_mean=float(predictor[inside].mean()),
                          delta_ND_mean=float(response[inside].mean()),
                          delta_ND_median=float(np.median(response[inside]))))
    if sum(cell['count'] for cell in cells) != predictor.size:
        raise ValueError('binned calibration must count every observation exactly once')
    return cells


def read_interface_run(config, summary, directory):
    horizon = config['protocol']['t_max_sweeps']
    with np.load(directory / 'interface.npz', allow_pickle=False) as stored:
        values = stored['values'].copy()
        equilibrium = stored['equilibrium_values'].copy()
    if values.shape[0] != horizon + 1 or values[0, 0] != 0.0:
        raise ValueError(f'interface series must cover every sweep from zero: {directory}')
    recovered = summary['recovered_time']
    active = np.ones(values.shape[0], dtype=bool)
    if recovered is not None and recovered <= horizon:
        active[int(recovered):] = False
    return dict(seed=summary['seed_id'], values=values, equilibrium=equilibrium,
                recovered_time=recovered, active=active,
                event=summary['event_observed'], confirm=summary['t_confirm_B'])


def calibration_pairs(runs, t_min=0, t_max=None):
    """Pair the J_pre predictor at sweep t with the realised delta over sweep t+1.

    A pair is kept whenever the transition STARTS in a not-yet-recovered state
    (defectors present); the final transition that enters recovery is therefore
    retained, while trivial all-cooperator transitions after recovery are dropped.
    """
    predictor, response = [], []
    for run in runs:
        values = run['values']
        active = run['active']
        last = values.shape[0] - 1 if t_max is None else min(t_max, values.shape[0] - 1)
        for t in range(max(t_min, 0), last):
            if not active[t]:
                continue
            predictor.append(values[t, 4])   # J_pre_next at sweep t
            response.append(values[t + 1, 1] - values[t, 1])  # realised delta D over sweep t+1
    return np.asarray(predictor), np.asarray(response)


def first_crossing(values, threshold):
    defectors = values[:, 1]
    hits = np.nonzero(defectors >= threshold)[0]
    return int(hits[0]) if hits.size else None


def group_report(runs, node_count):
    values = [run['values'] for run in runs]
    threshold = int(np.ceil(THRESHOLD_FRACTION * node_count))
    trajectory = {}
    for t in TRAJECTORY_TIMES:
        index = min(t, values[0].shape[0] - 1)
        trajectory[str(t)] = dict(
            defectors=_stats([v[index, 1] for v in values]),
            J=_stats([v[index, 3] for v in values]),
            J_pre=_stats([v[index, 4] for v in values]),
            interface_edges=_stats([v[index, 2] for v in values]),
            boundary_c_mean=_stats([v[index, 5] for v in values if np.isfinite(v[index, 5])]))
    predictor, response = calibration_pairs(runs)
    sign_mask = np.abs(predictor) > 0.05
    calibration = dict(
        n_pairs=int(predictor.size),
        pearson=_pearson(predictor, response),
        spearman=_spearman(predictor, response),
        ols=_ols(predictor, response),
        binned=_binned_calibration(predictor, response),
        sign_agreement=(float(np.mean(np.sign(predictor[sign_mask])
                                      == np.sign(response[sign_mask])))
                        if sign_mask.any() else None),
        response_mean_given_positive=(float(response[predictor > 0.05].mean())
                                      if (predictor > 0.05).any() else None),
        response_mean_given_negative=(float(response[predictor < -0.05].mean())
                                      if (predictor < -0.05).any() else None))
    # early (first 50 transitions) versus late (sweeps 200+) calibration
    early_predictor, early_response = calibration_pairs(runs, t_min=0, t_max=50)
    late_predictor, late_response = calibration_pairs(runs, t_min=200, t_max=None)
    calibration['early_ols'] = _ols(early_predictor, early_response)
    calibration['early_pearson'] = _pearson(early_predictor, early_response)
    calibration['late_ols'] = _ols(late_predictor, late_response)
    calibration['late_pearson'] = _pearson(late_predictor, late_response)
    crossings = [first_crossing(v, threshold) for v in values]
    reached = [c is not None for c in crossings]
    times = [c for c in crossings if c is not None]
    recovered = [run['recovered_time'] for run in runs if run['recovered_time'] is not None]
    first_positive = []
    for run in runs:
        v = run['values']
        hits = np.nonzero(v[:, 4] > 0)[0]
        if hits.size:
            first_positive.append(int(hits[0]))
    # resource adaptation gap at the recorded equilibrium times
    gap = defaultdict(list)
    for run in runs:
        for row in run['equilibrium']:
            t = int(row[0])
            live = run['values'][t]
            if np.isfinite(live[5]) and np.isfinite(row[4]):
                gap[t].append((live[5] - row[4], live[3] - row[3]))
    equilibrium_trajectory = []
    for t in sorted(gap):
        boundary_gap = [g[0] for g in gap[t]]
        j_gap = [g[1] for g in gap[t]]
        eq_rows = [run['equilibrium'][run['equilibrium'][:, 0] == t] for run in runs]
        j_eq = [float(r[0, 3]) for r in eq_rows if r.size]
        equilibrium_trajectory.append(dict(
            time=t, boundary_resource_gap=_stats(boundary_gap), J_minus_J_eq=_stats(j_gap),
            J_eq=_stats(j_eq)))
    initial = dict(
        J_pre_t0=_stats([v[0, 4] for v in values]),
        J_t0=_stats([v[0, 3] for v in values]),
        delta_ND_sweep1=_stats([v[1, 1] - v[0, 1] for v in values]),
        J_pre_t1=_stats([v[1, 4] for v in values]))
    return dict(
        n=len(runs), threshold=threshold, initial=initial, trajectory=trajectory,
        calibration=calibration, equilibrium_trajectory=equilibrium_trajectory,
        crossing=dict(reached=sum(reached), n=len(runs),
                      first_time=_stats(times) if times else None,
                      first_J_positive_time=_stats(first_positive) if first_positive else None),
        recovered_by_horizon=len(recovered),
        recovered_time=_stats(recovered) if recovered else None)


def within_prediction(runs, node_count):
    """Replica-level prediction for protocols with placement-driven initial variance."""
    threshold = int(np.ceil(THRESHOLD_FRACTION * node_count))
    initial_j = np.asarray([run['values'][0, 4] for run in runs])
    early_j = np.asarray([np.mean(run['values'][0:3, 4]) for run in runs])
    final_nd = np.asarray([run['values'][-1, 1] for run in runs])
    labels = np.asarray([1 if (first_crossing(run['values'], threshold) is not None) else 0
                         for run in runs])
    crossing_times = [first_crossing(run['values'], threshold) for run in runs]
    finite_times = [t for t in crossing_times if t is not None]
    return dict(
        n=len(runs),
        spearman_initialJ_finalND=_spearman(initial_j, final_nd),
        spearman_earlyJ_finalND=_spearman(early_j, final_nd),
        pearson_initialJ_finalND=_pearson(initial_j, final_nd),
        auc_initialJ_reach05=_auc(labels, initial_j),
        auc_earlyJ_reach05=_auc(labels, early_j),
        spearman_initialJ_crossing_time=_spearman(
            [j for j, t in zip(initial_j, crossing_times) if t is not None],
            finite_times) if len(finite_times) > 2 else None,
        reached_05=int(labels.sum()))


def interface_report(config_path: Path, root: Path) -> dict:
    campaign = json.loads(config_path.read_text())
    jobs = load_campaign(config_path, root)
    if len({summary['seed_id'] for _, summary, _ in jobs}) != len(jobs):
        raise ValueError('independent dynamic seeds required')
    validate_interface_jobs(jobs, jobs[0][0])
    groups = defaultdict(list)
    for config, summary, directory in jobs:
        damage = config['damage']
        groups[f"{damage['shape']}_{damage['count']}"].append(
            read_interface_run(config, summary, directory))
    node_count = jobs[0][0]['model']['size'] ** 2
    results = {}
    for name, runs in sorted(groups.items()):
        report = group_report(runs, node_count)
        if name.startswith('random'):
            report['within_prediction'] = within_prediction(runs, node_count)
        results[name] = report
    pooled_predictor, pooled_response = [], []
    for runs in groups.values():
        predictor, response = calibration_pairs(runs)
        pooled_predictor.extend(predictor.tolist())
        pooled_response.extend(response.tolist())
    files = sorted([*PROJECT_ROOT.glob('code/*.py'), *PROJECT_ROOT.glob('experiments/*.py')])
    identity = dict(campaign=config_hash(campaign),
                    manifests={str(directory.relative_to(root)): file_hash(directory / 'COMPLETE')
                               for _, _, directory in jobs},
                    analysis_sources={str(path.relative_to(PROJECT_ROOT)): file_hash(path)
                                      for path in files})
    return dict(identity, groups=results,
                pooled_calibration=dict(
                    n_pairs=len(pooled_predictor),
                    pearson=_pearson(pooled_predictor, pooled_response),
                    ols=_ols(pooled_predictor, pooled_response),
                    binned=_binned_calibration(pooled_predictor, pooled_response)),
                input_versions=sorted({(s['run_commit'], s['source_hash']) for _, s, _ in jobs}),
                interpretation=('descriptive early-window interface diagnostics and within-protocol '
                                'prediction screens; J is an instantaneous first-visit diagnostic; '
                                'no causal, metastable, or infinite-time conclusion'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT / 'data' / 'nucleation')
    args = parser.parse_args()
    report = interface_report(args.config, args.root)
    target = args.root / 'analysis' / config_hash(
        {key: report[key] for key in ['campaign', 'manifests', 'analysis_sources']})
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'interface.json', report)
    print(target / 'interface.json')
