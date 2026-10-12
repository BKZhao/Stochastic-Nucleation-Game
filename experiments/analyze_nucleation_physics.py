"""Physical time observables from complete independent growth/interface cohorts."""
import argparse
import json
from pathlib import Path
import numpy as np

from nucleation_paper_inputs import ANALYSIS, REPORTS
from research_common import PROJECT_ROOT, config_hash, file_hash, verify_complete, write_json
from nucleation_survival import wilson_interval

GROWTH = REPORTS['growth']
INTERFACE = REPORTS['early_interface']


def cohort_rows(report: dict) -> tuple[dict, list]:
    groups = {}
    checked = []
    for relative, digest in report['manifests'].items():
        directory = PROJECT_ROOT/'data/nucleation'/relative
        if file_hash(directory/'COMPLETE') != digest:
            raise ValueError(f'input manifest mismatch: {directory}')
        row = verify_complete(directory)
        key = f"{row['damage']['shape']}_{row['damage']['count']}"
        groups.setdefault(key, []).append((directory, row))
        checked.append({'path': relative, 'complete_sha256': digest})
    return groups, checked


def kinetics(rows: list, horizon: int) -> dict:
    times = sorted({0, horizon, *[r[k] for _, r in rows
                   for k in ['t_confirm_B', 'recovered_time'] if r[k] is not None]})
    a = np.array([sum(r['recovered_time'] is not None and r['recovered_time'] <= t
                      for _, r in rows) for t in times])
    b = np.array([sum(r['t_confirm_B'] is not None and r['t_confirm_B'] <= t
                      for _, r in rows) for t in times])
    u = len(rows)-a-b
    if np.any(u < 0):
        raise ValueError('overlapping A/B events in zero-mutation cohort')
    return {'n': len(rows), 'time': times,
            'probability': {k: (v/len(rows)).tolist() for k, v in [('A', a), ('B', b), ('U', u)]},
            'interval95': {k: [wilson_interval(int(x), len(rows)) for x in v]
                           for k, v in [('A', a), ('B', b), ('U', u)]}}


def early_dynamics(rows: list) -> dict:
    arrays = []
    for directory, _ in rows:
        with np.load(directory/'interface.npz', allow_pickle=False) as source:
            columns = source['columns'].tolist()
            values = source['values']
            expected = np.arange(501)
            if not np.array_equal(values[:, 0], expected):
                raise ValueError(f'500-sweep interface trajectory is incomplete: {directory}')
            arrays.append(values[:, [columns.index(k) for k in
                                      ['defectors', 'interface_edges', 'J_pre_next']]])
    array = np.stack(arrays)
    mean = array.mean(axis=0)
    sem = array.std(axis=0, ddof=1)/np.sqrt(len(array))
    return {'n': len(array), 'time': expected.tolist(),
            'columns': ['defectors', 'interface_edges', 'J_pre'],
            'mean': mean.tolist(), 'sem': sem.tolist(),
            'uncertainty': 'Across independent trajectories; plus/minus one standard error, not 95% CI.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT/'paper/physics_inputs.json')
    args = parser.parse_args()
    growth = json.loads(GROWTH.read_text())
    interface = json.loads(INTERFACE.read_text())
    long_rows, long_manifest = cohort_rows(growth)
    early_rows, early_manifest = cohort_rows(interface)
    groups = {}
    for key in ['compact_1', 'compact_16', 'random_16']:
        if len(long_rows[key]) != 64 or len(early_rows[key]) != 64:
            raise ValueError('physical trajectory cohort must contain 64 independent runs')
        groups[key] = {'kinetics': kinetics(long_rows[key], 40000),
                       'early': early_dynamics(early_rows[key]),
                       'crossings': growth['groups'][key]['progression']['crossings']}
        candidates = sorted((d, r) for d, r in long_rows[key] if r['event_observed'])
        chosen, _ = candidates[0]
        groups[key]['snapshot_run'] = str(chosen.relative_to(PROJECT_ROOT))
        groups[key]['snapshot_selection'] = 'Smallest dynamic seed among confirmed-B trajectories; outcome-conditioned example.'
    identity = {'growth_sha256': file_hash(GROWTH), 'interface_sha256': file_hash(INTERFACE),
                'analyzer_sha256': file_hash(Path(__file__))}
    report = {'identity': identity, 'groups': groups, 'long_manifest': long_manifest,
              'early_manifest': early_manifest,
              'scope': 'Finite-time fate and conditional spatial progression. No critical-nucleus, power-law or metastable-rate inference.'}
    target = ANALYSIS/config_hash(identity)/'physics.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, report)
    write_json(args.output, {'analysis': str(target.relative_to(PROJECT_ROOT)),
                            'sha256': file_hash(target)})
    print(target)


if __name__ == '__main__':
    main()
