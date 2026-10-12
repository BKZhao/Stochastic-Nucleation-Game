"""Short-horizon conditional moments a_delta and b_delta per mother and resource arm."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_nucleation_interface import _pearson, _spearman
from analyze_nucleation_fate import mother_descriptors
from research_common import PROJECT_ROOT, config_hash, file_hash, verify_complete, write_json
from run_nucleation_fate import expand_fate_campaign

DELTAS = (1, 2, 5, 10, 20)


def defector_series(directory: Path, node_count: int, horizon: int) -> np.ndarray:
    with np.load(directory / 'trajectory.npz', allow_pickle=False) as stored:
        values = stored['values']
        columns = stored['columns'].tolist()
        times = values[:, columns.index('time')]
        cooperation = values[:, columns.index('cooperation')]
    if times.size != horizon + 1 or times[0] != 0 or times[-1] != horizon:
        raise ValueError(f'trajectory must record every sweep 0..{horizon}: {directory}')
    return np.rint((1.0 - cooperation) * node_count).astype(int)


def moments_report(config_path: Path, root: Path) -> dict:
    campaign = json.loads(config_path.read_text())
    horizon = campaign['base']['protocol']['t_max_sweeps']
    node_count = campaign['base']['model']['size'] ** 2
    cells = expand_fate_campaign(campaign, root)
    mothers = defaultdict(list)
    manifests = {}
    model = campaign['base']['model']
    descriptors = {}
    for config, seed in cells:
        directory = root / 'N06' / config_hash(config) / f'seed-{seed:010d}'
        summary = verify_complete(directory)
        fate = config['fate']
        if fate['mother_path'] not in descriptors:
            descriptors[fate['mother_path']] = mother_descriptors(
                PROJECT_ROOT / fate['mother_path'], model)
        mothers[(fate['mother_path'], fate['arm'])].append(
            (summary['seed_id'], defector_series(directory, node_count, horizon)))
        manifests[str(directory.relative_to(root))] = file_hash(directory / 'COMPLETE')
    results = {}
    for (mother_path, arm), runs in sorted(mothers.items()):
        runs.sort()
        series = np.asarray([run[1] for run in runs], dtype=float)
        base = series[:, 0]
        moments = {}
        for delta in DELTAS:
            change = series[:, delta] - base
            moments[str(delta)] = dict(
                mean_change=float(change.mean()),
                a_delta=float(change.mean() / delta),
                var_change=float(change.var(ddof=1)),
                b_delta=float(change.var(ddof=1) / delta))
        results.setdefault(mother_path, {})[arm] = dict(
            n=len(runs), moments=moments, initial_defectors=float(base[0]))
    damage_of = {}
    from run_nucleation_fate import select_mothers
    for row in select_mothers(root, campaign['mothers']['campaign'],
                              campaign['mothers']['rules'],
                              bind_identity=campaign.get('bind_mother_identity', False)):
        damage_of[row['relative_path']] = row['name']
    grouped = {}
    for mother_path, arms in results.items():
        name = damage_of.get(mother_path, 'unknown')
        grouped.setdefault(name, []).append((mother_path, arms))
    screen = {}
    for name, entries in grouped.items():
        for delta in DELTAS:
            a_values = [arms['original']['moments'][str(delta)]['a_delta'] for _, arms in entries]
            b_values = [arms['original']['moments'][str(delta)]['b_delta'] for _, arms in entries]
            j_pre = [descriptors[path]['J_pre'] for path, _ in entries]
            edges = [descriptors[path]['interface_edges'] for path, _ in entries]
            eta = [descriptors[path]['eta_mean'] for path, _ in entries]
            screen[f'{name}/delta{delta}'] = dict(
                n=len(entries),
                spearman_Jpre_a=_spearman(j_pre, a_values),
                spearman_edges_b=_spearman(edges, b_values),
                spearman_Jpre_b=_spearman(j_pre, b_values),
                spearman_edges_a=_spearman(edges, a_values),
                spearman_eta_a=_spearman(eta, a_values),
                mean_a=float(np.mean(a_values)), mean_b=float(np.mean(b_values)))
    # arm contrasts of moments
    arm_moments = {}
    for name, entries in grouped.items():
        rows = {}
        for arm in ('original', 'adapted', 'shifted'):
            a1 = [arms[arm]['moments']['1']['a_delta'] for _, arms in entries if arm in arms]
            b1 = [arms[arm]['moments']['1']['b_delta'] for _, arms in entries if arm in arms]
            a20 = [arms[arm]['moments']['20']['a_delta'] for _, arms in entries if arm in arms]
            b20 = [arms[arm]['moments']['20']['b_delta'] for _, arms in entries if arm in arms]
            rows[arm] = dict(mean_a1=float(np.mean(a1)), mean_b1=float(np.mean(b1)),
                            mean_a20=float(np.mean(a20)), mean_b20=float(np.mean(b20)),
                            n=len(a1))
        arm_moments[name] = rows
    files = sorted([*PROJECT_ROOT.glob('code/*.py'), *PROJECT_ROOT.glob('experiments/*.py')])
    identity = dict(campaign=config_hash(campaign), manifests=manifests,
                    analysis_sources={str(path.relative_to(PROJECT_ROOT)): file_hash(path)
                                      for path in files})
    return dict(identity, groups={name: [dict(mother_path=mother_path,
                                              **{arm: dict(n=arms[arm]['n'],
                                                          moments=arms[arm]['moments'])
                                                 for arm in arms})
                                     for mother_path, arms in entries]
                                  for name, entries in grouped.items()},
                prediction_screen=screen, arm_moments=arm_moments,
                interpretation=('short-horizon conditional moments per mother and arm; '
                                'identification screen for drift versus activity strength; '
                                'no endpoint or causal claims'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT / 'data' / 'nucleation')
    args = parser.parse_args()
    report = moments_report(args.config, args.root)
    target = args.root / 'analysis' / config_hash(
        {key: report[key] for key in ['campaign', 'manifests', 'analysis_sources']})
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'moments.json', report)
    print(target / 'moments.json')
