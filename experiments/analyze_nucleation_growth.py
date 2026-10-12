"""Audit a fixed growth campaign and assemble historical risks and spatial progression."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from analyze_nucleation_followup import load_campaign
from analyze_nucleation_founders import validate_jobs
from nucleation_growth_analysis import read_growth_run, spatial_summary, progression_summary
from nucleation_progress import outcome_summary
from nucleation_survival import risk_difference_interval
from research_common import PROJECT_ROOT, config_hash, file_hash, write_json


def resource_contrasts(groups: dict) -> dict | None:
    if any('resource_value' not in group['damage'] for group in groups.values()):
        return None
    cells = {(group['damage']['shape'], group['damage']['resource_value']): group['outcomes'][-1]
             for group in groups.values()}
    if set(cells) != {(shape, value) for shape in ['compact', 'random'] for value in [0., 1.]}:
        return None
    comparisons = []
    for value in [0., 1.]:
        random, compact = cells['random', value], cells['compact', value]
        comparisons.append(dict(
            seed_resource=value, random_minus_compact=random['collapse_risk']-compact['collapse_risk'],
            newcombe95=risk_difference_interval(random['events'], random['n'], compact['events'], compact['n'])))
    return dict(comparisons=comparisons,
                interaction_point_estimate=comparisons[0]['random_minus_compact']-comparisons[1]['random_minus_compact'],
                scope='pre-specified descriptive differences; interaction point only, no interaction or equivalence test')


def growth_report(config_path: Path, root: Path) -> dict:
    campaign = json.loads(config_path.read_text())
    jobs = load_campaign(config_path, root)
    if len({summary['seed_id'] for _, summary, _ in jobs}) != len(jobs):
        raise ValueError('independent dynamic seeds required')
    validate_jobs(jobs, jobs[0][0])
    analysis = campaign['analysis']
    spatial_times = analysis['spatial_times']
    times = analysis['outcome_times']
    landmark = analysis['unresolved_landmark']
    keys = [str(float(f)) for f in campaign['base']['growth']['defector_fractions']]
    groups = defaultdict(list)
    for config, summary, directory in jobs:
        damage = config['damage']
        name = f"{damage['shape']}_{damage['count']}"
        if 'resource_value' in damage:
            name += f"_r{damage['resource_value']:g}"
        groups[name].append(read_growth_run(config, summary, directory, spatial_times))
    results = {}
    for name, runs in sorted(groups.items()):
        summaries = [run['summary'] for run in runs]
        unresolved = [run for run in runs
                      if (run['summary']['t_confirm_B'] is None or run['summary']['t_confirm_B'] > landmark)
                      and (run['summary']['recovered_time'] is None or run['summary']['recovered_time'] > landmark)]
        representatives = {}
        for outcome in ['confirmed', 'recovered', 'unresolved']:
            eligible = [run for run in runs if ('confirmed' if run['summary']['event_observed'] else
                        'recovered' if run['summary']['recovered_time'] is not None else 'unresolved') == outcome]
            selected = min(eligible, key=lambda run: run['seed']) if eligible else None
            representatives[outcome] = None if selected is None else dict(seed=selected['seed'], path=selected['path'])
        results[name] = dict(
            n=len(runs), damage=summaries[0]['damage'],
            outcomes=[outcome_summary(summaries, time) for time in times],
            spatial=[spatial_summary(runs, time) for time in spatial_times],
            progression=progression_summary(runs, times, keys), representatives=representatives,
            unresolved_followup=dict(
                landmark=landmark, n=len(unresolved), original_n=len(runs),
                pending_confirmation=sum(run['spatial'][str(landmark)]['pending_confirmation'] for run in unresolved),
                extinct_resource_pending=sum(run['spatial'][str(landmark)]['defector_fraction'] == 0 for run in unresolved),
                final=outcome_summary([run['summary'] for run in unresolved], times[-1]) if unresolved else None),
        )
    files = sorted([*PROJECT_ROOT.glob('code/*.py'), *PROJECT_ROOT.glob('experiments/*.py')])
    identity = dict(campaign=config_hash(campaign),
                    manifests={str(directory.relative_to(root)): file_hash(directory / 'COMPLETE') for _, _, directory in jobs},
                    analysis_sources={str(path.relative_to(PROJECT_ROOT)): file_hash(path) for path in files})
    return dict(identity, groups=results, resource_contrasts=resource_contrasts(results), analysis=analysis,
                input_versions=sorted({(s['run_commit'], s['source_hash']) for _, s, _ in jobs}),
                interpretation='finite-window first-depletion and conditional growth descriptions; not causal, metastable or infinite-time conclusions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT / 'data' / 'nucleation')
    args = parser.parse_args()
    report = growth_report(args.config, args.root)
    target = args.root / 'analysis' / config_hash({key: report[key] for key in ['campaign', 'manifests', 'analysis_sources']})
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'growth.json', report)
    print(target / 'growth.json')
