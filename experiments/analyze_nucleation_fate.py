"""N06 fate analysis: per-mother probabilities, arm contrasts, state descriptors and verification.

Revision 2 (external review 0e2f052): mothers additionally carry the J_pre
diagnostic (one resource substep ahead of the strategy phase, matching the
observation phase of the drift calibration); every continuation run's initial
field is verified against its arm-transformed mother; arm contrasts and group
risk differences carry mother-level bootstrap and normal-approximation
intervals; descriptor entanglement is reported through the Spearman matrix and
partial rank correlations; exact-N_D strata are tabulated.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_nucleation_interface import _auc, _spearman, _stats
from nucleation_interface import (interface_observables, resource_equilibrium,
                                  resource_substep)
from research_clusters import analyze_clusters
from research_common import PROJECT_ROOT, config_hash, file_hash, verify_complete, write_json
from run_nucleation_fate import expand_fate_campaign

DESCRIPTOR_KEYS = ['defectors', 'interface_edges', 'n_clusters', 'largest_share',
                   'eta_mean', 'resource_mean', 'boundary_c_mean', 'boundary_c_mean_eq',
                   'J', 'J_pre', 'J_eq', 'delta_J']


def mother_descriptors(mother_path: Path, model: dict) -> dict:
    size = model['size']
    with np.load(mother_path, allow_pickle=False) as stored:
        strategy = stored['strategy'].copy()
        resource = stored['resource'].copy()
    interface = interface_observables(strategy, resource, size, model['temptation'],
                                      model['coupling'], model['strength'],
                                      model['selection_noise'])
    r_pre = resource_substep(strategy, resource, size, model['growth'],
                             model['depletion'], model['diffusion'])
    predictor = interface_observables(strategy, r_pre, size, model['temptation'],
                                      model['coupling'], model['strength'],
                                      model['selection_noise'])
    r_eq, _ = resource_equilibrium(strategy, resource, size, model['growth'],
                                   model['depletion'], model['diffusion'])
    adapted = interface_observables(strategy, r_eq, size, model['temptation'],
                                    model['coupling'], model['strength'],
                                    model['selection_noise'])
    clusters = analyze_clusters((strategy == 0).reshape(size, size))
    areas = clusters['areas']
    largest_share = float(areas.max() / areas.sum()) if areas.size else 0.0
    eta_mean = float(np.abs(resource - r_eq).mean())
    return dict(defectors=interface['defectors'],
                interface_edges=interface['interface_edges'],
                boundary_c_mean=interface['boundary_c_mean'],
                boundary_c_mean_eq=adapted['boundary_c_mean'],
                J=interface['J'], J_pre=predictor['J'], J_eq=adapted['J'],
                delta_J=interface['J'] - adapted['J'],
                eta_mean=eta_mean, resource_mean=float(resource.mean()),
                largest_share=largest_share,
                n_clusters=int(areas.size))


def verify_child_initial(config: dict, directory: Path) -> dict:
    """Check the run's initial field equals the mother after the declared arm."""
    fate = config['fate']
    model = config['model']
    size = model['size']
    with np.load(PROJECT_ROOT / fate['mother_path'], allow_pickle=False) as stored:
        strategy = stored['strategy']
        resource = stored['resource']
        mother_meta = json.loads(str(stored['metadata_json']))
    parent_model = dict(mother_meta['parameters'])
    parent_model['size'] = mother_meta['L']
    if parent_model != model:
        raise ValueError(f'mother model identity mismatch: {directory}')
    with np.load(directory / 'states' / 'initial.npz', allow_pickle=False) as stored:
        child_strategy = stored['strategy']
        child_resource = stored['resource']
    if not np.array_equal(child_strategy, strategy):
        raise ValueError(f'child strategy differs from mother: {directory}')
    arm = fate['arm']
    if arm == 'original':
        if not np.array_equal(child_resource, resource):
            raise ValueError(f'original-arm child resource differs from mother: {directory}')
        return dict(arm=arm, max_abs_resource_difference=0.0)
    if arm == 'adapted':
        r_eq, _ = resource_equilibrium(strategy, resource, size, model['growth'],
                                       model['depletion'], model['diffusion'])
        difference = float(np.max(np.abs(child_resource - r_eq)))
    else:
        rolled = np.roll(resource.reshape(size, size),
                         (int(fate['shift'][0]), int(fate['shift'][1])), (0, 1)).reshape(-1)
        difference = float(np.max(np.abs(child_resource - rolled)))
    if difference > 1e-12:
        raise ValueError(f'{arm}-arm child resource differs from transformation: {directory}')
    return dict(arm=arm, max_abs_resource_difference=difference)


def wilson(successes: int, n: int) -> list:
    if n == 0:
        return [0., 1.]
    z = 1.959963984540054
    share = successes / n
    denominator = 1 + z * z / n
    centre = (share + z * z / (2 * n)) / denominator
    half = z * np.sqrt(share * (1 - share) / n + z * z / (4 * n * n)) / denominator
    return [max(0., centre - half), min(1., centre + half)]


def _mean_interval(values, seed=20261006):
    """Mother-level bootstrap percentile interval plus a normal approximation."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return None
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    normal_half = 1.959963984540054 * sd / np.sqrt(values.size)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(10000, values.size), replace=True).mean(axis=1)
    return dict(n=int(values.size), mean=mean, sd=sd,
                bootstrap95=[float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
                normal_approx95=[mean - normal_half, mean + normal_half],
                note='mother-level resampling; post-hoc approximate intervals, no multiplicity control')


def _spearman_matrix(rows, keys):
    size = len(keys)
    matrix = np.eye(size)
    for a in range(size):
        for b in range(a + 1, size):
            value = _spearman([row[keys[a]] for row in rows], [row[keys[b]] for row in rows])
            matrix[a, b] = matrix[b, a] = 0.0 if value is None else value
    return matrix


def _partial_from_matrix(matrix, keys, target, given, condition):
    indices = [keys.index(target), keys.index(given)]
    indices += [keys.index(c) for c in condition]
    sub = matrix[np.ix_(indices, indices)]
    try:
        inverse = np.linalg.inv(sub)
    except np.linalg.LinAlgError:
        return None
    return float(-inverse[0, 1] / np.sqrt(abs(inverse[0, 0] * inverse[1, 1])))


def fate_report(config_path: Path, root: Path) -> dict:
    campaign = json.loads(config_path.read_text())
    cells = expand_fate_campaign(campaign, root)
    jobs = []
    verification = []
    for config, seed in cells:
        directory = root / config['experiment_id'] / config_hash(config) / f'seed-{seed:010d}'
        summary = verify_complete(directory)
        if summary['technical_status'] != 'complete':
            raise ValueError(f'technical failure: {directory}')
        verification.append(verify_child_initial(config, directory))
        jobs.append((config, summary, directory))
    if len({summary['seed_id'] for _, summary, _ in jobs}) != len(jobs):
        raise ValueError('independent clone seeds required')
    model = campaign['base']['model']
    descriptors = {}
    mothers = defaultdict(list)
    for config, summary, directory in jobs:
        fate = config['fate']
        if fate['mother_path'] not in descriptors:
            descriptors[fate['mother_path']] = mother_descriptors(
                PROJECT_ROOT / fate['mother_path'], model)
        mothers[fate['mother_path']].append((fate['arm'], summary))
    groups = {}
    for mother_path, runs in mothers.items():
        descriptor = descriptors[mother_path]
        per_arm = {}
        shape = None
        for arm, summary in runs:
            per_arm.setdefault(arm, []).append(summary)
            shape = summary['damage']['shape'] + '_' + str(summary['damage']['count'])
        entry = dict(descriptors=descriptor, arms={})
        for arm, copies in sorted(per_arm.items()):
            hits = sum(copy['hit_B'] for copy in copies)
            recovers = sum(copy['hit_A'] for copy in copies)
            entry['arms'][arm] = dict(n=len(copies), q_B=hits / len(copies),
                                      q_A=recovers / len(copies),
                                      q_U=1 - (hits + recovers) / len(copies),
                                      wilson_q_B=wilson(hits, len(copies)),
                                      median_confirm=_stats([copy['t_confirm_B'] for copy in copies
                                                             if copy['t_confirm_B'] is not None]))
        groups.setdefault(shape, []).append(
            dict(mother_path=mother_path, parent_run_id=runs[0][1]['fate']['parent_run_id'],
                 source_time=runs[0][1]['fate']['parent_source_time'],
                 parent_defectors=runs[0][1]['fate']['parent_defectors'], **entry))

    # arm contrasts with mother-level paired intervals
    contrasts = {}
    for shape, mothers_list in groups.items():
        rows = []
        for mother in mothers_list:
            original = mother['arms']['original']
            for arm in ('adapted', 'shifted'):
                if arm in mother['arms']:
                    rows.append(dict(parent_run_id=mother['parent_run_id'], arm=arm,
                                     delta_q_B=mother['arms'][arm]['q_B'] - original['q_B']))
        contrast = {}
        for arm in ('adapted', 'shifted'):
            values = [row['delta_q_B'] for row in rows if row['arm'] == arm]
            if values:
                contrast[arm] = dict(paired=_mean_interval(values),
                                     mothers_lower=sum(value < 0 for value in values))
        contrasts[shape] = dict(rows=rows, summary=contrast)

    # group risk difference on the original arm and every arm (mother-level)
    group_difference = {}
    for arm in ('original', 'adapted', 'shifted'):
        random_q = [m['arms'][arm]['q_B'] for m in groups.get('random_16', []) if arm in m['arms']]
        compact_q = [m['arms'][arm]['q_B'] for m in groups.get('compact_16', []) if arm in m['arms']]
        if random_q and compact_q:
            difference = np.asarray(random_q) - np.asarray(compact_q)
            group_difference[arm] = dict(
                mean_difference=float(difference.mean()),
                interval=_mean_interval(difference.tolist()),
                note='random minus compact mother means; post-hoc approximate, mother as unit')

    # descriptor entanglement and partial rank correlations against q_B
    entanglement = {}
    for shape, mothers_list in groups.items():
        rows = [dict(mother['descriptors'],
                     q_B=mother['arms']['original']['q_B']) for mother in mothers_list]
        keys = [key for key in [*DESCRIPTOR_KEYS, 'q_B']
                if len({row[key] for row in rows}) > 1]
        matrix = _spearman_matrix(rows, keys)
        partials = {}
        if 'q_B' in keys:
            if 'eta_mean' in keys and 'interface_edges' in keys:
                partials['qB_given_eta_partialling_edges'] = _partial_from_matrix(
                    matrix, keys, 'q_B', 'eta_mean', ['interface_edges'])
                partials['qB_given_edges_partialling_eta'] = _partial_from_matrix(
                    matrix, keys, 'q_B', 'interface_edges', ['eta_mean'])
            if 'eta_mean' in keys and {'interface_edges', 'resource_mean'} <= set(keys):
                partials['qB_given_eta_partialling_edges_resource'] = _partial_from_matrix(
                    matrix, keys, 'q_B', 'eta_mean', ['interface_edges', 'resource_mean'])
            if 'J_pre' in keys and {'interface_edges', 'eta_mean'} <= set(keys):
                partials['qB_given_Jpre_partialling_edges_eta'] = _partial_from_matrix(
                    matrix, keys, 'q_B', 'J_pre', ['interface_edges', 'eta_mean'])
        entanglement[shape] = dict(
            keys=keys,
            spearman_matrix={keys[a]: {keys[b]: float(matrix[a, b]) for b in range(len(keys))}
                             for a in range(len(keys))},
            partials=partials,
            note='Spearman/partial Spearman on 15 mothers including q_B; entangled '
                 'descriptors, not causal separation')

    # exact-N_D stratification on the original arm
    strata = {}
    for shape, mothers_list in groups.items():
        by_count = defaultdict(list)
        for mother in mothers_list:
            by_count[int(mother['descriptors']['defectors'])].append(
                mother['arms']['original']['q_B'])
        strata[shape] = {str(count): dict(n=len(values), mean_q_B=float(np.mean(values)))
                         for count, values in sorted(by_count.items())}

    # state-variable rank prediction of q_B (original arm)
    state_prediction = {}
    for shape, mothers_list in groups.items():
        original_q = [mother['arms']['original']['q_B'] for mother in mothers_list]
        for key in keys_for_prediction():
            values = [mother['descriptors'][key] for mother in mothers_list]
            if np.std(values) < 1e-12 or np.std(original_q) < 1e-12:
                continue
            state_prediction[f'{shape}/{key}'] = dict(
                spearman=_spearman(values, original_q), n=len(values))
    files = sorted([*PROJECT_ROOT.glob('code/*.py'), *PROJECT_ROOT.glob('experiments/*.py')])
    identity = dict(campaign=config_hash(campaign),
                    manifests={str(directory.relative_to(root)): file_hash(directory / 'COMPLETE')
                               for _, _, directory in jobs},
                    analysis_sources={str(path.relative_to(PROJECT_ROOT)): file_hash(path)
                                      for path in files})
    summary_cells = {}
    for shape, mothers_list in groups.items():
        for arm in ('original', 'adapted', 'shifted'):
            rows = [mother['arms'][arm] for mother in mothers_list if arm in mother['arms']]
            if rows:
                summary_cells[f'{shape}/{arm}'] = dict(
                    n_mothers=len(rows), n_runs=sum(row['n'] for row in rows),
                    mean_q_B=float(np.mean([row['q_B'] for row in rows])),
                    mean_q_A=float(np.mean([row['q_A'] for row in rows])),
                    mean_q_U=float(np.mean([row['q_U'] for row in rows])))
    verification_summary = dict(
        runs_verified=len(verification),
        arms={arm: dict(n=sum(1 for row in verification if row['arm'] == arm),
                        max_abs_resource_difference=max(
                            (row['max_abs_resource_difference'] for row in verification
                             if row['arm'] == arm), default=0.0))
              for arm in ('original', 'adapted', 'shifted')},
        note='every continuation initial field verified against its arm-transformed mother state')
    return dict(identity, summary=summary_cells, groups=groups, contrasts=contrasts,
                group_difference=group_difference, entanglement=entanglement,
                exact_defector_strata=strata, state_prediction=state_prediction,
                verification=verification_summary,
                input_versions=sorted({(summary['run_commit'], summary['source_hash'])
                                       for _, summary, _ in jobs}),
                interpretation=('per-mother finite-horizon fate probabilities, arm contrasts with '
                                'mother-level intervals, descriptor entanglement and verified child '
                                'initial states; unresolved remains unresolved; no infinite-time '
                                'committor, no causal decomposition, no equivalence claims'))


def keys_for_prediction():
    return ['defectors', 'interface_edges', 'n_clusters', 'largest_share',
            'J', 'J_pre', 'J_eq', 'delta_J', 'eta_mean', 'resource_mean', 'boundary_c_mean']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT / 'data' / 'nucleation')
    args = parser.parse_args()
    report = fate_report(args.config, args.root)
    target = args.root / 'analysis' / config_hash(
        {key: report[key] for key in ['campaign', 'manifests', 'analysis_sources']})
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'fate.json', report)
    print(target / 'fate.json')
