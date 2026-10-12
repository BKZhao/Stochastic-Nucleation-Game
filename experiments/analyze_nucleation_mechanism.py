"""Mechanism round analysis: local allocation, dose chain, coupling-0 control.

Loads the frozen mechanism preregistration (exact per-arm J_pre predictions,
linear sensitivities, invariants; written before any continuation run), the
fate campaign, the dose companion and the coupling-0 identity check.

Primary pre-registered test: the mother-level paired contrast
q_B(interface_harm) - q_B(interface_protect) is positive with a mother-level
cluster bootstrap 95% interval excluding zero. Secondary: the shuffle
magnitude bound, the defector-shuffle equivalence margin, the dose chain
dJ_pre -> a_1 -> delta_q_B, and the linear-response calibration block. The
coupling-0 negative control (strategy-trajectory identity across arms at
zero resource-payoff coupling) is read from its report file.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_nucleation_direction import _cluster_indices, _spearman_bootstrap
from analyze_nucleation_confirmatory import (_arm_runs_digest, hits_of,
                                              load_campaign_outcomes, q_B)
from analyze_nucleation_fate import _mean_interval
from analyze_nucleation_moments import defector_series
from nucleation_rearrange import MECHANISM_ARMS
from research_common import (PROJECT_ROOT, config_hash, file_hash,
                              research_root, verify_complete, write_json)
from run_nucleation_fate import expand_fate_campaign

PREREG = 'data/nucleation/N06/campaigns/preregistered_mechanism_predictions.json'
FATE_CONFIG = 'configs/nucleation/n06_mechanism_fate.json'
DOSE_CONFIG = 'configs/nucleation/n06_mechanism_dose.json'
COUPLING0_REPORT = 'data/nucleation/analysis/mechanism_coupling0.json'
MECHANISM_ARMS_PLUS = ('original', *MECHANISM_ARMS)
DRAWS = 10000


def _seed_of(mother_path):
    return mother_path.split('seed-')[1].split('/')[0]


def _cluster_mean(values, groups, draws=DRAWS, seed=20261007):
    """Mother-level cluster bootstrap of the mean of paired values."""
    mothers = sorted(set(groups))
    index_of = {mother: position for position, mother in enumerate(mothers)}
    values = np.asarray(values, dtype=float)
    groups = np.asarray([index_of[group] for group in groups])
    rng = np.random.default_rng(seed)
    take = rng.integers(0, len(mothers), size=(draws, len(mothers)))
    samples = np.empty(draws)
    for draw in range(draws):
        indices = _cluster_indices(groups, take[draw])
        samples[draw] = values[indices].mean()
    return [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))]


def mechanism_rows(outcomes):
    rows = []
    for mother_path, arms in outcomes.items():
        base = q_B(arms['original'])
        row = dict(mother_path=mother_path, seed=_seed_of(mother_path),
                   q_B={}, hits={}, n={}, deltas={})
        for arm, copies in arms.items():
            row['q_B'][arm] = q_B(copies)
            row['hits'][arm] = hits_of(copies)
            row['n'][arm] = len(copies)
            if arm != 'original':
                row['deltas'][arm] = q_B(copies) - base
        rows.append(row)
    rows.sort(key=lambda row: row['seed'])
    return rows


def paired_contrast(rows, arm_high, arm_low):
    values = [row['q_B'][arm_high] - row['q_B'][arm_low] for row in rows]
    groups = [row['seed'] for row in rows]
    interval = _cluster_mean(values, groups)
    return dict(arm_high=arm_high, arm_low=arm_low, n_mothers=len(values),
                mean=float(np.mean(values)), interval=interval,
                mothers_lower=sum(value < 0 for value in values))


def arm_deltas(rows):
    table = {}
    for arm in MECHANISM_ARMS:
        values = [row['deltas'][arm] for row in rows]
        groups = [row['seed'] for row in rows]
        table[arm] = dict(mean=float(np.mean(values)),
                          interval=_cluster_mean(values, groups),
                          mothers_lower=sum(value < 0 for value in values),
                          n_mothers=len(values))
    return table


def dose_rows(root: Path, prereg):
    campaign_config = json.loads((PROJECT_ROOT / DOSE_CONFIG).read_text())
    model = campaign_config['base']['model']
    horizon = campaign_config['base']['protocol']['t_max_sweeps']
    node_count = model['size'] ** 2
    changes = defaultdict(list)
    run_identity = defaultdict(list)
    for config, seed in expand_fate_campaign(campaign_config, root):
        directory = root / 'N06' / config_hash(config) / f'seed-{seed:010d}'
        verify_complete(directory)
        fate = config['fate']
        series = defector_series(directory, node_count, horizon)
        changes[(fate['mother_path'], fate['arm'])].append(float(series[1] - series[0]))
        run_identity[(fate['mother_path'], fate['arm'])].append(
            (str(directory.relative_to(root)), file_hash(directory / 'COMPLETE')))
    by_seed = {entry['seed']: entry for entry in prereg['mothers']}
    rows = []
    for (mother_path, arm), values in sorted(changes.items()):
        seed = _seed_of(mother_path)
        entry = by_seed.get(seed)
        if entry is None:
            continue
        row = dict(seed=seed, arm=arm, n=len(values),
                   a1=float(np.mean(values)),
                   a1_clone_se=float(np.std(values, ddof=1) / np.sqrt(len(values)))
                   if len(values) > 1 else None,
                   J_pre=entry['arms'][arm]['J_pre'] if arm != 'original'
                   else entry['original']['J_pre'],
                   dJ_pre_exact=0.0 if arm == 'original'
                   else entry['arms'][arm]['dJ_pre_exact'],
                   dJ_pre_linear=0.0 if arm == 'original'
                   else entry['arms'][arm]['dJ_pre_linear'],
                   dJ_pre_harm_pool=0.0 if arm == 'original'
                   else entry['arms'][arm]['dJ_pre_linear'],
                   runs=_arm_runs_digest(run_identity[(mother_path, arm)]))
        rows.append(row)
    original_by_seed = {row['seed']: row for row in rows if row['arm'] == 'original'}
    for row in rows:
        base = original_by_seed.get(row['seed'])
        if base is not None:
            row['da1'] = row['a1'] - base['a1']
    return rows


def linear_calibration(pairs_rows):
    """OLS of dJ_pre_exact on dJ_pre_linear over mother-arm rows."""
    x = np.asarray([row['dJ_pre_linear'] for row in pairs_rows], dtype=float)
    y = np.asarray([row['dJ_pre_exact'] for row in pairs_rows], dtype=float)
    mothers = sorted({row['seed'] for row in pairs_rows})
    groups = np.asarray([mothers.index(row['seed']) for row in pairs_rows])
    slope, intercept = np.polyfit(x, y, 1)
    rng = np.random.default_rng(20261007)
    slopes = []
    for _ in range(4000):
        take = rng.integers(0, len(mothers), size=len(mothers))
        indices = _cluster_indices(groups, take)
        slopes.append(float(np.polyfit(x[indices], y[indices], 1)[0]))
    return dict(n_rows=len(pairs_rows), n_mothers=len(mothers),
                slope=float(slope),
                slope_bootstrap95=[float(np.quantile(slopes, .025)),
                                   float(np.quantile(slopes, .975))],
                intercept=float(intercept),
                spearman=_spearman_bootstrap(pairs_rows, 'dJ_pre_linear',
                                             'dJ_pre_exact', draws=4000),
                note=('calibration of the exact first-sweep response on the linear '
                      'sensitivity prediction across mechanism arms; the slope is '
                      'expected BELOW one because the full rearrangement leaves the '
                      'small-perturbation regime (tanh saturation dampens the '
                      'response); reported, not a pass/fail gate'))


def mechanism_round_report(root: Path) -> dict:
    prereg = json.loads((PROJECT_ROOT / PREREG).read_text())
    campaign_config = json.loads((PROJECT_ROOT / FATE_CONFIG).read_text())
    if prereg['campaign_config_sha256'] != file_hash(PROJECT_ROOT / FATE_CONFIG):
        raise ValueError('preregistration was computed for a different campaign config')
    outcomes, _, identities = load_campaign_outcomes(
        PROJECT_ROOT / FATE_CONFIG, root)
    if set(next(iter(outcomes.values())).keys()) != set(MECHANISM_ARMS_PLUS):
        raise ValueError('fate campaign arm set does not match the mechanism arms')
    rows = mechanism_rows(outcomes)
    dose = dose_rows(root, prereg)
    by_arm = defaultdict(list)
    for row in dose:
        by_arm[row['arm']].append(row)

    paired_with_delta = []
    for row in dose:
        if row['arm'] != 'original' and 'da1' in row:
            delta_lookup = {r['seed']: r['deltas'].get(row['arm'])
                            for r in rows}
            row['delta_qB'] = delta_lookup.get(row['seed'])
            if row['delta_qB'] is not None:
                paired_with_delta.append(row)

    primary = paired_contrast(rows, 'interface_harm', 'interface_protect')
    contrast_abs = abs(primary['mean'])
    shuffle_arm = arm_deltas(rows)
    defector = shuffle_arm['defector_shuffle']
    chain = dict(
        dJpre_to_a1=_spearman_bootstrap(
            [row for row in dose if row['arm'] != 'original'],
            'dJ_pre_exact', 'a1', draws=4000),
        dJpre_to_da1=_spearman_bootstrap(
            [row for row in dose if row['arm'] != 'original' and 'da1' in row],
            'dJ_pre_exact', 'da1', draws=4000),
        da1_to_delta=_spearman_bootstrap(paired_with_delta, 'da1', 'delta_qB',
                                         draws=4000),
        a1_calibration_per_arm={
            arm: dict(slope=float(np.polyfit(
                [row['J_pre'] for row in by_arm[arm]],
                [row['a1'] for row in by_arm[arm]], 1)[0]))
            for arm in MECHANISM_ARMS_PLUS})
    secondary = dict(
        shuffle_bound=dict(
            mean=shuffle_arm['interface_shuffle']['mean'],
            bound=0.5 * contrast_abs,
            satisfied=abs(shuffle_arm['interface_shuffle']['mean'])
            < 0.5 * contrast_abs if contrast_abs > 0 else None),
        defector_margin=dict(
            interval=defector['interval'], margin=0.25 * contrast_abs,
            within=abs(defector['interval'][0]) <= 0.25 * contrast_abs
            and abs(defector['interval'][1]) <= 0.25 * contrast_abs
            if contrast_abs > 0 else None),
        linear_calibration=linear_calibration(
            [row for row in dose if row['arm'] != 'original']))
    coupling0 = json.loads((PROJECT_ROOT / COUPLING0_REPORT).read_text())
    identity = dict(
        model=campaign_config['base']['model'],
        preregistration=dict(path=PREREG, sha256=file_hash(PROJECT_ROOT / PREREG)),
        fate_config=dict(path=FATE_CONFIG, sha256=file_hash(PROJECT_ROOT / FATE_CONFIG)),
        dose_config=dict(path=DOSE_CONFIG, sha256=file_hash(PROJECT_ROOT / DOSE_CONFIG)),
        coupling0_report=dict(path=COUPLING0_REPORT,
                              sha256=file_hash(PROJECT_ROOT / COUPLING0_REPORT)),
        analysis_sources={
            str(path.relative_to(PROJECT_ROOT)): file_hash(path)
            for path in sorted([*PROJECT_ROOT.glob('code/*.py'),
                                *PROJECT_ROOT.glob('experiments/*.py')])})
    return dict(
        identity=identity, primary=primary, arm_deltas=shuffle_arm,
        dose_rows=dose, chain=chain, secondary=secondary,
        coupling0=dict(n_mothers=coupling0['n_mothers'],
                       all_strategy_trajectories_identical=coupling0[
                           'all_strategy_trajectories_identical'],
                       trivial_identity_mothers=coupling0['trivial_identity_mothers'],
                       note=coupling0['interpretation']),
        mothers=rows,
        interpretation=('controlled local-allocation mechanism round: primary '
                        'paired contrast, secondary bounds and margins, dose '
                        'chain and linear calibration; finite-horizon fate '
                        'statements only'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT / 'data' / 'nucleation')
    args = parser.parse_args()
    report = mechanism_round_report(args.root)
    target = research_root(args.root) / 'analysis' / config_hash(report['identity'])
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'mechanism.json', report)
    print(target / 'mechanism.json')
