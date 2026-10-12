"""Read and validate exact growth events and observed spatial states from complete jobs."""
import json
import math
from pathlib import Path

import numpy as np

from analyze_nucleation_followup import read_trace
from research_clusters import analyze_clusters


def distribution(values: list) -> dict | None:
    available = np.asarray([value for value in values if value is not None], dtype=float)
    if not available.size:
        return None
    return dict(n=len(available), mean=float(available.mean()),
                q25=float(np.quantile(available, .25)), median=float(np.median(available)),
                q75=float(np.quantile(available, .75)))


def state_metrics(strategy: np.ndarray, resource: np.ndarray, size: int, cutoff: float) -> dict:
    defectors = strategy.reshape(size, size) == 0
    clusters = analyze_clusters(defectors)
    count = int(defectors.sum())
    return dict(defector_fraction=count / size**2,
                largest_share=None if count == 0 else int(clusters['areas'].max()) / count,
                clusters=len(clusters['areas']),
                depleted_fraction=float(np.mean(defectors & (resource.reshape(size, size) < cutoff))),
                winding=bool(clusters['winding'].any()), resource=float(resource.mean()))


def read_growth_run(config: dict, summary: dict, directory: Path, spatial_times: list[int]) -> dict:
    trace = read_trace(config, summary, directory)
    growth = summary['growth']
    keys = [str(float(value)) for value in config['growth']['defector_fractions']]
    if (growth['definition'] != 'first_defector_fraction_crossing_each_sweep_v1'
            or set(growth['first_crossings']) != set(keys)
            or growth['observed_through'] != summary['executed_sweeps']):
        raise ValueError(f'invalid growth event definition/clock: {directory}')
    node_count = config['model']['size'] ** 2
    counts = np.rint((1 - trace.cooperation) * node_count).astype(int)
    crossed_states = {}
    previous = 0
    missing = False
    for key in keys:
        time = growth['first_crossings'][key]
        threshold = math.ceil(float(key) * node_count)
        if time is None:
            missing = True
            if np.any(counts >= threshold) or (directory / 'states' / f'growth-{key}.npz').exists():
                raise ValueError(f'missing crossing has contradictory evidence: {directory}')
            continue
        if missing or type(time) is not int or not previous <= time <= summary['executed_sweeps']:
            raise ValueError(f'growth crossings must be ordered: {directory}')
        if np.any(counts[trace.times < time] >= threshold):
            raise ValueError(f'growth time follows an already observed crossing: {directory}')
        with np.load(directory / 'states' / f'growth-{key}.npz', allow_pickle=False) as stored:
            metadata = json.loads(str(stored['metadata_json']))
            if metadata['local_time'] != time or np.count_nonzero(stored['strategy'] == 0) < threshold:
                raise ValueError(f'crossing snapshot mismatch: {directory}')
            crossed_states[key] = dict(time=time, **state_metrics(stored['strategy'], stored['resource'],
                                        config['model']['size'], config['clusters']['resource_cutoff']))
        previous = time
    extinct = growth['extinction_time']
    if extinct is not None:
        if (type(extinct) is not int or not 0 <= extinct <= summary['executed_sweeps']
                or np.any(counts[trace.times < extinct] == 0)
                or np.any(counts[trace.times >= extinct] != 0)):
            raise ValueError(f'inconsistent absorbing extinction clock: {directory}')
    elif np.any(counts == 0):
        raise ValueError(f'extinction clock missing: {directory}')
    with np.load(directory / 'trajectory.npz', allow_pickle=False) as stored:
        values, columns = stored['values'], stored['columns'].tolist()
    with np.load(directory / 'clusters.npz', allow_pickle=False) as stored:
        cluster_rows, cluster_columns = stored['values'], stored['columns'].tolist()
    spatial = {}
    for time in spatial_times:
        if trace.event_time is not None and trace.event_time <= time:
            spatial[str(time)] = None
            continue
        positions = np.flatnonzero(trace.times == time)
        if len(positions) != 1:
            raise ValueError(f'missing spatial sample t={time}: {directory}')
        index = positions[0]
        rows = cluster_rows[(cluster_rows[:, cluster_columns.index('time')] == time)
                            & (cluster_rows[:, cluster_columns.index('kind')] == 1)]
        count = int(counts[index])
        n_clusters = int(values[index, columns.index('n_clusters_1')])
        max_cluster = int(values[index, columns.index('n_max_1')])
        areas = rows[:, cluster_columns.index('area')]
        if (len(rows) != n_clusters or int(areas.sum()) != count
                or (int(areas.max()) if len(rows) else 0) != max_cluster):
            raise ValueError(f'cluster records do not match trajectory: {directory}')
        spatial[str(time)] = dict(
            defector_fraction=count / node_count, largest_share=None if count == 0 else max_cluster / count,
            clusters=n_clusters, depleted_fraction=float(values[index, columns.index('depleted_area')] / node_count),
            winding=bool(rows[:, [cluster_columns.index('winding_row'), cluster_columns.index('winding_col')]].any()),
            resource=float(trace.resource[index]), pending_confirmation=bool(trace.streak[index] > 0),
        )
    return dict(seed=summary['seed_id'], path=str(directory), growth=growth,
                crossing_states=crossed_states, spatial=spatial, summary=summary)


def spatial_summary(rows: list[dict], time: int) -> dict:
    observed = [run['spatial'][str(time)] for run in rows if run['spatial'][str(time)] is not None]
    return dict(time=time, total=len(rows), observed_nonB=len(observed), confirmed_before_or_at=len(rows)-len(observed),
                with_defectors=sum(row['defector_fraction'] > 0 for row in observed),
                pending_confirmation=sum(row['pending_confirmation'] for row in observed),
                **{key: distribution([row[key] for row in observed])
                   for key in ['defector_fraction', 'largest_share', 'clusters', 'depleted_fraction', 'winding', 'resource']})


def progression_summary(rows: list[dict], times: list[int], keys: list[str]) -> dict:
    crossings = {key: dict(
        reached_by_time={str(time): sum(run['growth']['first_crossings'][key] is not None
                                       and run['growth']['first_crossings'][key] <= time for run in rows) for time in times},
        first_time=distribution([run['growth']['first_crossings'][key] for run in rows]),
        spatial={name: distribution([run['crossing_states'].get(key, {}).get(name) for run in rows])
                 for name in ['largest_share', 'clusters', 'depleted_fraction', 'resource', 'winding']},
    ) for key in keys}
    segments = []
    for start, end in zip(keys, keys[1:] + ['B_confirm']):
        durations = []
        for run in rows:
            left = run['growth']['first_crossings'][start]
            right = run['summary']['t_confirm_B'] if end == 'B_confirm' else run['growth']['first_crossings'][end]
            if left is not None and right is not None:
                if right < left:
                    raise ValueError('negative growth-segment time')
                durations.append(right - left)
        segments.append(dict(start=start, end=end, original_n=len(rows), reached_both=len(durations),
                             not_reached_both=len(rows)-len(durations), same_sweep=durations.count(0),
                             duration=distribution(durations)))
    return dict(original_n=len(rows), crossings=crossings, segments=segments,
                scope='conditional first-crossing descriptions; reversible population thresholds, not nucleus times')
