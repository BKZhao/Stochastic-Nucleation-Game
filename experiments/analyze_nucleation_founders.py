"""Compare a measured single-founder risk with a finite-horizon independent-channel benchmark."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from analyze_nucleation_followup import load_campaign, read_trace
from nucleation_campaign import summarize
from research_common import PROJECT_ROOT, config_hash, file_hash, write_json

EARLY_TIMES = (0, 10, 100, 500, 1000, 2000)


def independent_risk(probability: float, founders: int) -> float:
    """OR probability under an explicit independent, identical channel assumption."""
    if not 0 <= probability <= 1 or type(founders) is not int or founders < 1:
        raise ValueError('probability in [0,1] and a positive integer count required')
    return 1 - (1 - probability) ** founders


def validate_jobs(jobs: list, reference: dict) -> None:
    expected = deepcopy(reference)
    for key in ['damage', 'experiment_id']:
        expected.pop(key)
    for config, summary, directory in jobs:
        comparable = deepcopy(config)
        for key in ['damage', 'experiment_id']:
            comparable.pop(key)
        if comparable != expected:
            raise ValueError('founder comparisons require identical model, horizon and endpoint')
        read_trace(config, summary, directory)
        count = config['damage']['count']
        resource_value = config['damage'].get('resource_value', 0.)
        with np.load(directory / 'states' / 'initial.npz', allow_pickle=False) as stored:
            strategy, resource = stored['strategy'], stored['resource']
            if (np.count_nonzero(strategy == 0) != count
                    or not np.all(resource[strategy == 0] == resource_value)
                    or not np.all(resource[strategy == 1] == 1)
                    or not np.isin(strategy, [0, 1]).all()):
                raise ValueError(f'initial strategy/resource intervention mismatch: {directory}')
        with np.load(directory / 'events.npz', allow_pickle=False) as stored:
            confirm = summary['t_confirm_B']
            if (bool(stored['event_observed']) != summary['event_observed']
                    or int(stored['t_confirm_B']) != (-1 if confirm is None else confirm)):
                raise ValueError(f'event file and summary disagree: {directory}')


def early_description(jobs: list) -> list[dict]:
    """Post-hoc full-cohort summaries, refusing to silently condition on survival."""
    records = {time: [] for time in EARLY_TIMES}
    for config, _, directory in jobs:
        with np.load(directory / 'trajectory.npz', allow_pickle=False) as stored:
            values, columns = stored['values'], stored['columns'].tolist()
        for time in EARLY_TIMES:
            row = values[values[:, columns.index('time')] == time]
            if len(row) != 1:
                raise ValueError(f'early description requires every trajectory at time {time}: {directory}')
            values_at_time = row[0]
            defectors = int(round((1 - values_at_time[columns.index('cooperation')]) * config['model']['size'] ** 2))
            records[time].append([defectors, values_at_time[columns.index('n_max_1')],
                                  values_at_time[columns.index('n_clusters_1')]])
    return [dict(time=time, n=len(rows), mean_defectors=float(np.mean(rows, axis=0)[0]),
                 median_defectors=float(np.median(rows, axis=0)[0]),
                 median_max_cluster=float(np.median(rows, axis=0)[1]),
                 median_clusters=float(np.median(rows, axis=0)[2]),
                 runs_with_defectors=int(np.count_nonzero(np.asarray(rows)[:, 0])))
            for time, rows in records.items()]


def founder_report(single: list, comparison: list) -> dict:
    if not single or not comparison:
        raise ValueError('both complete campaigns are required')
    if any(config['damage'] != {'count': 1, 'shape': 'compact'} for config, _, _ in single):
        raise ValueError('single-founder campaign must plant exactly one compact D/r0 site')
    all_jobs = single + comparison
    seeds = [summary['seed_id'] for _, summary, _ in all_jobs]
    if len(seeds) != len(set(seeds)):
        raise ValueError('independent campaigns must use distinct dynamic seeds')
    validate_jobs(all_jobs, single[0][0])
    one = summarize([summary for _, summary, _ in single], len(single))['cells']
    many = summarize([summary for _, summary, _ in comparison], len(comparison))['cells']
    if len(one) != 1:
        raise ValueError('one fixed single-founder condition required')
    groups = {cell['damage']['shape']: cell for cell in many}
    if len(many) != 2 or set(groups) != {'compact', 'random'}:
        raise ValueError('comparison must contain exactly compact and random layouts')
    founders = groups['random']['damage']['count']
    if founders < 2 or groups['compact']['damage']['count'] != founders:
        raise ValueError('comparison requires equal multi-founder budgets')
    cell = one[0]
    random = groups['random']
    benchmark = independent_risk(cell['collapse_risk'], founders)
    early = {'single': early_description(single)}
    early.update({shape: early_description([job for job in comparison if job[0]['damage']['shape'] == shape])
                  for shape in groups})
    return dict(
        single=cell, comparison=groups,
        benchmark=dict(founders=founders, risk=benchmark,
                       transformed_wilson95=[independent_risk(p, founders) for p in cell['wilson95']],
                       observed_minus_benchmark=random['collapse_risk'] - benchmark),
        equivalent_single_risk=dict(
            risk=1 - (1 - random['collapse_risk']) ** (1 / founders),
            transformed_wilson95=[1 - (1 - p) ** (1 / founders) for p in random['wilson95']]),
        early_posthoc=dict(
            groups=early,
            additive_mean_benchmark=[dict(time=row['time'], mean_defectors=founders * row['mean_defectors'])
                                     for row in early['single']],
            scope=('post-hoc descriptive full-cohort means and medians; all runs present; multiplying '
                   'mean single-founder burden ignores overlap and interactions and is an early-time '
                   'reference only, not a collapse probability model or equivalence test')),
        interpretation=('descriptive finite-horizon independent-channel benchmark; marginal intervals '
                        'do not constitute a formal model test; discrepancies do not identify causal '
                        'interactions, growth-time effects, resource feedback or nucleation'),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--single', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT / 'data' / 'nucleation')
    args = parser.parse_args()
    single = load_campaign(args.single, args.root)
    comparison = load_campaign(args.comparison, args.root)
    report = founder_report(single, comparison)
    jobs = single + comparison
    files = sorted([*PROJECT_ROOT.glob('code/*.py'), *PROJECT_ROOT.glob('experiments/*.py')])
    identity = dict(
        manifests={str(directory.relative_to(args.root)): file_hash(directory / 'COMPLETE')
                   for _, _, directory in jobs},
        analysis_sources={str(path.relative_to(PROJECT_ROOT)): file_hash(path) for path in files},
        campaigns={name: config_hash(json.loads(path.read_text()))
                   for name, path in [('single', args.single), ('comparison', args.comparison)]},
    )
    report.update(identity)
    report['input_versions'] = {name: sorted({(s['run_commit'], s['source_hash']) for _, s, _ in group})
                               for name, group in [('single', single), ('comparison', comparison)]}
    target = args.root / 'analysis' / config_hash(identity)
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'founders.json', report)
    print(target / 'founders.json')


if __name__ == '__main__':
    main()
