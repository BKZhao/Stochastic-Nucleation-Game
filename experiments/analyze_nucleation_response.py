"""Reconstruct paired population and resource responses from completed trajectories."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from analyze_nucleation_moments import defector_series
from nucleation_paper_inputs import ANALYSIS, REPORTS, load_report
from research_common import PROJECT_ROOT, config_hash, file_hash, verify_complete, write_json
from run_nucleation_fate import expand_fate_campaign

CAMPAIGNS = {
    'translation': 'configs/nucleation/n06_direction_dose.json',
    'allocation': 'configs/nucleation/n06_mechanism_dose.json',
}


def paired_curve(values: np.ndarray, draws: np.ndarray) -> dict:
    """Resample complete configurations; columns are observation times."""
    means = values[draws].mean(axis=1)
    return {'mean': values.mean(axis=0).tolist(),
            'interval95': np.quantile(means, [.025, .975], axis=0).T.tolist(),
            'drift_mean': np.diff(values.mean(axis=0)).tolist(),
            'drift_interval95': np.quantile(np.diff(means, axis=1),
                                           [.025, .975], axis=0).T.tolist()}


def campaign_response(name: str, relative: str) -> dict:
    config_path = PROJECT_ROOT/relative
    campaign = json.loads(config_path.read_text())
    horizon = campaign['base']['protocol']['t_max_sweeps']
    count = campaign['base']['model']['size']**2
    raw = defaultdict(list)
    resources = defaultdict(list)
    covariances = defaultdict(list)
    checked = set()
    covariance_error = 0.
    initial_mean_error = 0.
    manifests = []
    for config, seed in expand_fate_campaign(campaign, PROJECT_ROOT/'data/nucleation'):
        directory = PROJECT_ROOT/'data/nucleation/N06'/config_hash(config)/f'seed-{seed:010d}'
        summary = verify_complete(directory)
        if summary['executed_sweeps'] != horizon:
            raise ValueError(f'short response stopped early: {directory}')
        fate = config['fate']
        arm = fate['arm'] if fate['arm'] != 'shifted' else f"shifted{tuple(fate['shift'])}"
        values = defector_series(directory, count, horizon).astype(float)
        key = (fate['mother_path'], arm)
        raw[key].append(values-values[0])
        with np.load(directory/'trajectory.npz', allow_pickle=False) as stored:
            columns = stored['columns'].tolist()
            data = stored['values']
            rho = data[:, columns.index('cooperation')]
            resource = data[:, columns.index('resource')]
            if not np.array_equal(data[:, columns.index('time')], np.arange(horizon+1)):
                raise ValueError('resource reconstruction requires every sweep')
        g, depletion = config['model']['growth'], config['model']['depletion']
        if g == depletion:
            raise ValueError('covariance cannot be reconstructed when growth equals depletion')
        covariance = ((np.diff(resource)-g*rho[:-1]+depletion*resource[:-1])
                      /(depletion-g)-rho[:-1]*resource[:-1])
        resources[key].append(resource)
        covariances[key].append(covariance)
        if key not in checked:
            with np.load(directory/'states/initial.npz', allow_pickle=False) as initial:
                strategy, field = initial['strategy'], initial['resource']
                direct = np.mean(strategy*field)-strategy.mean()*field.mean()
                covariance_error = max(covariance_error, abs(covariance[0]-direct))
                initial_mean_error = max(initial_mean_error, abs(resource[0]-field.mean()),
                                         abs(rho[0]-strategy.mean()))
            checked.add(key)
        manifests.append({'path': str((directory/'COMPLETE').relative_to(PROJECT_ROOT)),
                          'sha256': file_hash(directory/'COMPLETE')})
    paths = sorted({path for path, _ in raw})
    arms = sorted({arm for _, arm in raw})
    if len(paths) != 48 or any(len(raw[path, arm]) != 16 for path in paths for arm in arms):
        raise ValueError('expected 48 configurations and 16 continuations per arm')
    means = {arm: np.stack([np.mean(raw[path, arm], axis=0) for path in paths]) for arm in arms}
    delta = {arm: value-means['original'] for arm, value in means.items() if arm != 'original'}
    draws = np.random.default_rng(20261008).integers(0, len(paths), size=(10000, len(paths)))
    curves = {arm: paired_curve(value, draws) for arm, value in delta.items()}
    resource_means = {arm: np.stack([np.mean(resources[path, arm], axis=0) for path in paths])
                      for arm in arms}
    covariance_means = {arm: np.stack([np.mean(covariances[path, arm], axis=0) for path in paths])
                        for arm in arms}
    resource_delta = {arm: value-resource_means['original']
                      for arm, value in resource_means.items() if arm != 'original'}
    covariance_delta = {arm: value-covariance_means['original']
                        for arm, value in covariance_means.items() if arm != 'original'}
    resource_curves = {arm: paired_curve(value, draws) for arm, value in resource_delta.items()}
    covariance_curves = {arm: paired_curve(value, draws) for arm, value in covariance_delta.items()}
    # Means of complete physical paths are used only for descriptive phase-plane trajectories.
    initial_rho = np.empty(len(paths))
    for index, path in enumerate(paths):
        with np.load(PROJECT_ROOT/path, allow_pickle=False) as stored:
            initial_rho[index] = stored['strategy'].mean()
    phase = {arm: {'cooperation': (initial_rho[:, None]-means[arm]/count).mean(axis=0).tolist(),
                   'resource': resource_means[arm].mean(axis=0).tolist()} for arm in arms}
    if covariance_error > 1e-10 or initial_mean_error > 1e-12:
        raise ValueError('balance reconstruction disagrees with the saved initial fields')
    if name == 'translation':
        for label, target in [('small', np.sqrt(5)), ('medium', np.sqrt(314))]:
            selected = [arm for arm in delta if abs(np.linalg.norm(
                [int(x) for x in arm[8:-1].split(',')])-target) < 1e-8]
            if len(selected) != 4:
                raise ValueError('four rotations are required before configuration averaging')
            curves[label] = paired_curve(np.mean([delta[arm] for arm in selected], axis=0), draws)
            resource_curves[label] = paired_curve(
                np.mean([resource_delta[arm] for arm in selected], axis=0), draws)
            covariance_curves[label] = paired_curve(
                np.mean([covariance_delta[arm] for arm in selected], axis=0), draws)
            phase[label] = {observable: np.mean([phase[arm][observable] for arm in selected],
                                                axis=0).tolist()
                            for observable in ['cooperation', 'resource']}
    frozen = load_report('direction')['dose']['rows'] if name == 'translation' else load_report('mechanism')['dose_rows']
    reference = {(row['seed'], row['arm']): row for row in frozen}
    max_error = 0.
    for index, path in enumerate(paths):
        seed = path.split('seed-')[1].split('/')[0]
        for arm in arms:
            row = reference[seed, arm]
            max_error = max(max_error, abs(means[arm][index, 1]-row['a1']))
    if max_error > 1e-12:
        raise ValueError(f'first-sweep reconstruction differs from frozen analysis: {max_error}')
    mean_abs_final = float(np.mean([np.abs(values[:, -1]).mean() for values in delta.values()]))
    expected = load_report('r3')['impulse']['per_source']['direction' if name == 'translation' else 'mechanism']['mean_abs_dD20']
    if not np.isclose(mean_abs_final, expected, atol=1e-12, rtol=0):
        raise ValueError('20-sweep response differs from frozen R3 readout')
    return {'config': relative, 'config_sha256': file_hash(config_path), 'time': list(range(horizon+1)),
            'n_configurations': len(paths), 'continuations_per_arm': 16, 'n_runs': len(manifests),
            'curves': curves, 'resource_curves': resource_curves,
            'covariance_curves': covariance_curves, 'phase': phase,
            'covariance_time': list(range(horizon)),
            'balance_checks': {'unique_initial_states': len(checked),
                               'initial_covariance_max_error': float(covariance_error),
                               'initial_mean_max_error': float(initial_mean_error)},
            'manifests': manifests, 'first_sweep_max_error': max_error,
            'mean_abs_delta_n20': mean_abs_final,
            'scope': 'Paired ensemble increments; pointwise configuration bootstrap. '
                     'Covariance reconstructed per trajectory from the exact balance at t=0,...,19; '
                     'initial values checked against all distinct saved initial fields. '
                     'Phase-plane paths are ensemble means, not a deterministic vector field. '
                     'No mediation inference.'}


def main() -> None:
    results = {name: campaign_response(name, relative) for name, relative in CAMPAIGNS.items()}
    sources = [Path(__file__), PROJECT_ROOT/'experiments/analyze_nucleation_moments.py',
               PROJECT_ROOT/'experiments/run_nucleation_fate.py',
               PROJECT_ROOT/'code/b_eco_unified.py', REPORTS['direction'],
               REPORTS['mechanism'], REPORTS['r3']]
    identity = {'sources': {str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in sources},
                'inputs': {name: row['manifests'] for name, row in results.items()}}
    path = ANALYSIS/config_hash(identity)/'response.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {'identity': identity, 'groups': results, 'bootstrap_draws': 10000})
    write_json(PROJECT_ROOT/'paper/response_inputs.json',
               {'analysis': str(path.relative_to(PROJECT_ROOT)), 'sha256': file_hash(path)})
    print(path)
    print({name: {k: row[k] for k in ['n_runs', 'first_sweep_max_error', 'mean_abs_delta_n20']}
           for name, row in results.items()})


if __name__ == '__main__':
    main()
