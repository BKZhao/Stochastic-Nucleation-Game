"""Direction-controlled round analysis: pit-gap mechanism, direction contrasts, precision.

Loads the frozen preregistration (state-level B(d), J_pre(d), pit geometry; written
before any continuation run of the round) and the four round campaigns: the
direction-controlled fate campaign, its short-horizon dose companion, and the two
clone-augmentation campaigns for the 50 existing compact mothers.

Primary pre-registered test: Spearman(dB(d), delta_q_B(d)) < 0 across mother-
direction pairs with a mother-level bootstrap 95% interval, where
dB = boundary_c_mean(shifted) - boundary_c_mean(original) is the resource read by
interface cooperators after the displacement. Secondary: direction means at fixed
magnitude with paired bootstrap, the magnitude curve, the state-level pit-width
check, and the dose mechanism chain B(d) -> J_pre(d) -> a_1(d) -> delta_q_B(d).
The precision section pools 16 + 32 clones per mother, re-runs the
risk-heterogeneity diagnostic and screens descriptors against the sharpened
per-mother q_B (discovery level, no confirmatory claim).
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from analyze_nucleation_confirmatory import (MODEL_SPECS, _arm_runs_digest,
                                             heterogeneity_diagnostic, hits_of,
                                             load_campaign_outcomes, predict, q_B)
from analyze_nucleation_fate import _mean_interval, mother_descriptors
from analyze_nucleation_interface import _auc, _spearman
from analyze_nucleation_moments import defector_series
from nucleation_interface import (interface_observables, pit_observables,
                                  resource_substep)
from preregister_direction_round import torus_magnitude
from research_common import PROJECT_ROOT, config_hash, file_hash, verify_complete, write_json
from run_nucleation_fate import expand_fate_campaign

PREREG = 'data/nucleation/N06/campaigns/preregistered_direction_predictions.json'
DIRECTION_CONFIG = 'configs/nucleation/n06_direction_controls.json'
DOSE_CONFIG = 'configs/nucleation/n06_direction_dose.json'
PRECISION_HELDOUT_CONFIG = 'configs/nucleation/n06_precision_compact_heldout.json'
PRECISION_TRAINING_CONFIG = 'configs/nucleation/n06_precision_compact_training.json'
HELDOUT_DESCRIPTORS = 'data/nucleation/analysis/84e608ec76114c63/confirmation.json'
TRAINING_FATE = 'data/nucleation/analysis/50a1d6f6a0f2b399/fate.json'


def _seed_of(mother_path):
    return mother_path.split('seed-')[1].split('/')[0]


def _arm_magnitudes(campaign_config):
    size = campaign_config['base']['model']['size']
    magnitudes = {'original': 0.0}
    for entry in campaign_config['arms']:
        if isinstance(entry, dict) and entry['arm'] == 'shifted':
            magnitudes[f"shifted{tuple(entry['shift'])}"] = torus_magnitude(
                entry['shift'], size)
    return magnitudes


def mother_rows(outcomes, magnitudes, run_identity=None):
    """One row per mother: q_B/hits/n per arm, paired deltas and run digests."""
    rows = []
    for mother_path, arms in outcomes.items():
        base = q_B(arms['original'])
        row = dict(mother_path=mother_path, seed=_seed_of(mother_path),
                   q_B={}, hits={}, n={}, deltas={}, magnitudes={},
                   runs=(run_identity or {}).get(mother_path, {}))
        for arm, copies in arms.items():
            row['q_B'][arm] = q_B(copies)
            row['hits'][arm] = hits_of(copies)
            row['n'][arm] = len(copies)
            row['magnitudes'][arm] = magnitudes.get(arm)
            if arm != 'original':
                row['deltas'][arm] = q_B(copies) - base
        rows.append(row)
    rows.sort(key=lambda row: row['seed'])
    return rows


def direction_pairs(rows, prereg):
    """Join per-mother per-direction outcomes onto the frozen state-level predictions."""
    by_seed = {entry['seed']: entry for entry in prereg['mothers']}
    pairs = []
    for row in rows:
        entry = by_seed.get(row['seed'])
        if entry is None:
            continue
        for arm, delta in row['deltas'].items():
            key = arm[len('shifted'):]
            pairs.append(dict(
                seed=row['seed'], mother_path=row['mother_path'], arm=arm,
                magnitude=row['magnitudes'][arm], direction=key,
                delta_qB=delta, base_qB=row['q_B']['original'],
                q_B_shifted=row['q_B'][arm], hits=row['hits'][arm], n=row['n'][arm],
                dB=entry['boundary_c_shift'][key] - entry['boundary_c_original'],
                J_pre=entry['J_pre_shift'][key],
                B_shifted=entry['boundary_c_shift'][key]))
    return pairs


def _cluster_indices(groups, take):
    """Concatenate mother blocks in the drawn order, preserving multiplicities.

    A cluster bootstrap must replicate the whole block of a drawn unit each time
    it is drawn; selecting the union of drawn mothers would instead produce a
    random subset and understate the resampling variance.
    """
    count = int(groups.max()) + 1
    blocks = [np.flatnonzero(groups == index) for index in range(count)]
    return np.concatenate([blocks[int(index)] for index in take])


def _spearman_bootstrap(pairs, x_key, y_key, draws=10000, seed=20261006):
    """Mother-level cluster bootstrap of Spearman(x, y) over mother-direction pairs."""
    mothers = sorted({pair['seed'] for pair in pairs})
    x = np.asarray([pair[x_key] for pair in pairs], dtype=float)
    y = np.asarray([pair[y_key] for pair in pairs], dtype=float)
    groups = np.asarray([mothers.index(pair['seed']) for pair in pairs])
    rng = np.random.default_rng(seed)
    observed = _spearman(x.tolist(), y.tolist())
    samples = []
    for _ in range(draws):
        take = rng.integers(0, len(mothers), size=len(mothers))
        indices = _cluster_indices(groups, take)
        samples.append(_spearman(x[indices].tolist(), y[indices].tolist()))
    interval = [float(np.quantile(samples, .025)), float(np.quantile(samples, .975))] \
        if samples else None
    return dict(spearman=observed, n_pairs=int(len(pairs)), n_mothers=len(mothers),
                bootstrap95=interval, draws=len(samples),
                note=('mother-level cluster bootstrap: a drawn mother contributes its '
                      'full set of pairs each time it is drawn'))


def _magnitude_mother_means(pairs, magnitude):
    """Per-mother mean of the direction deltas at one magnitude.

    All direction deltas of a mother share the same original-arm baseline, so the
    magnitude-level effect must be aggregated per mother before any resampling.
    """
    by_mother = defaultdict(list)
    for pair in pairs:
        if abs(pair['magnitude'] - magnitude) < 1e-9:
            by_mother[pair['seed']].append(pair['delta_qB'])
    return {seed: float(np.mean(values)) for seed, values in by_mother.items()}


def _paired_magnitude_contrast(pairs, high, low):
    """Mother-paired difference of magnitude-level averaged responses."""
    first = _magnitude_mother_means(pairs, high)
    second = _magnitude_mother_means(pairs, low)
    common = sorted(set(first) & set(second))
    values = [first[seed] - second[seed] for seed in common]
    return dict(n_mothers=len(common), mean=float(np.mean(values)),
                interval=_mean_interval(values),
                note='per-mother magnitude averages, mother-level resampling')


def _direction_permutation_p(matrix, draws=10000, seed=20261006):
    """Calibration of the direction-mean spread under within-mother exchangeability.

    Direction labels are permuted within each mother; the p-value is the fraction
    of permutation spreads at or above the observed spread. The non-negative
    max-min statistic has no valid percentile test by itself.
    """
    rng = np.random.default_rng(seed)
    observed = float(np.nanmean(matrix, axis=1).max() - np.nanmean(matrix, axis=1).min())
    counts = 0
    for _ in range(draws):
        permuted = matrix.copy()
        for mother in range(matrix.shape[1]):
            permuted[:, mother] = matrix[rng.permutation(matrix.shape[0]), mother]
        means = np.nanmean(permuted, axis=1)
        if float(means.max() - means.min()) >= observed:
            counts += 1
    return (counts + 1) / (draws + 1)


def direction_report(pairs, rows, prereg, root: Path) -> dict:
    magnitudes = sorted({pair['magnitude'] for pair in pairs})
    arms = {}
    for arm in sorted({pair['arm'] for pair in pairs}):
        selected = [pair['delta_qB'] for pair in pairs if pair['arm'] == arm]
        arms[arm] = dict(
            magnitude=next(pair['magnitude'] for pair in pairs if pair['arm'] == arm),
            n_mothers=len(selected), mean_delta=float(np.mean(selected)),
            interval=_mean_interval(selected),
            mothers_lower=sum(value < 0 for value in selected),
            mean_dB=float(np.mean([pair['dB'] for pair in pairs
                                   if pair['arm'] == arm])))
    primary = _spearman_bootstrap(pairs, 'dB', 'delta_qB')
    within_magnitude = {}
    for magnitude in magnitudes:
        selected = [pair for pair in pairs if pair['magnitude'] == magnitude]
        if len(selected) >= 6:
            within_magnitude[f'{magnitude}'] = _spearman_bootstrap(
                selected, 'dB', 'delta_qB', draws=4000)
    directions = {}
    for magnitude in magnitudes:
        selected = [pair for pair in pairs if pair['magnitude'] == magnitude]
        mother_list = sorted({pair['seed'] for pair in selected})
        index_of = {seed: position for position, seed in enumerate(mother_list)}
        columns = {}
        for pair in selected:
            column = columns.setdefault(pair['direction'],
                                        np.full(len(mother_list), np.nan))
            column[index_of[pair['seed']]] = pair['delta_qB']
        means = {direction: float(np.nanmean(values))
                 for direction, values in columns.items()}
        matrix = np.stack([values for values in columns.values()])
        rng = np.random.default_rng(20261006)
        spread_draws = []
        for _ in range(10000):
            take = rng.choice(matrix.shape[1], size=matrix.shape[1], replace=True)
            means_drawn = np.nanmean(matrix[:, take], axis=1)
            spread_draws.append(float(means_drawn.max() - means_drawn.min()))
        observed_spread = max(means.values()) - min(means.values())
        directions[f'{magnitude}'] = dict(
            n=len(selected), direction_means=means,
            observed_spread=observed_spread,
            spread_bootstrap95=[float(np.quantile(spread_draws, .025)),
                                float(np.quantile(spread_draws, .975))],
            spread_permutation_p=_direction_permutation_p(matrix),
            note=('descriptive spread of direction means with estimation uncertainty '
                  '(mother-level paired resampling); the max-min statistic is '
                  'non-negative so the percentile interval is not a test of equal '
                  'direction means - the permutation p-value (direction labels '
                  'exchanged within each mother, post-hoc) calibrates it'))
    magnitude_curve = {}
    for magnitude in magnitudes:
        mother_means = _magnitude_mother_means(pairs, magnitude)
        values = list(mother_means.values())
        pairs_at_magnitude = sum(1 for pair in pairs
                                 if abs(pair['magnitude'] - magnitude) < 1e-9)
        magnitude_curve[f'{magnitude}'] = dict(
            n_mothers=len(values),
            n_directions_per_mother=(pairs_at_magnitude // len(values)) if values else 0,
            mean_delta=float(np.mean(values)),
            interval=_mean_interval(values),
            note='per-mother mean over directions first, then mother-level '
                 'resampling (direction deltas share the original-arm baseline)')
    small, medium, large = magnitudes[0], magnitudes[1], magnitudes[2]
    magnitude_contrasts = dict(
        medium_minus_small=_paired_magnitude_contrast(pairs, medium, small),
        large_minus_medium=_paired_magnitude_contrast(pairs, large, medium))
    magnitude_contrasts['large_minus_medium']['note'] += (
        '; the large magnitude carries a single direction, so this contrast mixes '
        'magnitude with direction-set change')
    original_rows = rows
    heterogeneity = heterogeneity_diagnostic(
        [dict(q_B=row['q_B']['original'], n=row['n']['original']) for row in original_rows])
    widths = [entry['pit']['half_recovery_distance'] for entry in prereg['mothers']]
    smallest = min(pair['magnitude'] for pair in pairs)
    state_check = dict(
        half_recovery_distances={str(value): widths.count(value) for value in widths},
        n_mothers_with_width_at_least_smallest_magnitude=sum(
            1 for width in widths if width is not None and width >= smallest),
        note=('the half-recovery distance measures how thick the OUTER recovery band '
              'of the resource field is around the defector region (all defector '
              'cells fall into distance bin 0); it is not the pit diameter and does '
              'not measure the overlap between the shifted pit and the strategy '
              'patch. It shows no discriminative power in this sample, and no '
              'mother has a width at or above the smallest tested magnitude, so '
              'the |d| < width side of the registered condition has no testable '
              'samples. This is not evidence that no spatial scale exists, nor '
              'that B is a sufficient replacement geometry'))
    return dict(n_mothers=len(rows), mothers=rows, arms=arms, primary=primary,
                within_magnitude=within_magnitude, direction_contrasts=directions,
                magnitude_curve=magnitude_curve, magnitude_contrasts=magnitude_contrasts,
                original_heterogeneity=heterogeneity, state_check=state_check)


def dose_chain(root: Path, prereg, direction_pairs_list) -> dict:
    campaign_config = json.loads((PROJECT_ROOT / DOSE_CONFIG).read_text())
    campaign = json.loads((PROJECT_ROOT / DIRECTION_CONFIG).read_text())
    model = campaign['base']['model']
    horizon = campaign_config['base']['protocol']['t_max_sweeps']
    node_count = model['size'] ** 2
    changes = defaultdict(list)
    run_identity = defaultdict(list)
    for config, seed in expand_fate_campaign(campaign_config, root):
        directory = root / 'N06' / config_hash(config) / f'seed-{seed:010d}'
        verify_complete(directory)
        fate = config['fate']
        arm = fate['arm'] if fate['arm'] != 'shifted' else f"shifted{tuple(fate['shift'])}"
        series = defector_series(directory, node_count, horizon)
        changes[(fate['mother_path'], arm)].append(float(series[1] - series[0]))
        run_identity[(fate['mother_path'], arm)].append(
            (str(directory.relative_to(root)), file_hash(directory / 'COMPLETE')))
    by_seed = {entry['seed']: entry for entry in prereg['mothers']}
    delta_lookup = {(pair['seed'], pair['arm']): pair['delta_qB']
                    for pair in direction_pairs_list}
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
                   runs=_arm_runs_digest(run_identity[(mother_path, arm)]))
        if arm == 'original':
            row.update(J_pre=entry['J_pre_original'], dB=0.0, magnitude=0.0,
                       delta_qB=None)
        else:
            key = arm[len('shifted'):]
            row.update(J_pre=entry['J_pre_shift'][key],
                       dB=entry['boundary_c_shift'][key] - entry['boundary_c_original'],
                       delta_qB=delta_lookup.get((seed, arm)))
        rows.append(row)
    original_by_seed = {row['seed']: row for row in rows if row['arm'] == 'original'}
    for row in rows:
        if row['arm'] == 'original':
            continue
        shift = tuple(int(value) for value in
                      row['arm'][len('shifted') + 1:-1].split(','))
        row['magnitude'] = torus_magnitude(list(shift), 48)
        base = original_by_seed.get(row['seed'])
        if base is not None:
            row['dJ_pre'] = row['J_pre'] - base['J_pre']
            row['da1'] = row['a1'] - base['a1']
    shifted = [row for row in rows if row['arm'] != 'original']
    original = [row for row in rows if row['arm'] == 'original']
    paired_rows = [row for row in shifted if 'da1' in row]
    paired_with_delta = [row for row in paired_rows if row['delta_qB'] is not None]
    chain = dict(
        n_shifted_rows=len(shifted),
        Jpre_to_a1=_spearman_bootstrap(shifted, 'J_pre', 'a1', draws=4000),
        dB_to_a1=_spearman_bootstrap(shifted, 'dB', 'a1', draws=4000),
        a1_to_delta=_spearman_bootstrap(
            [row for row in shifted if row['delta_qB'] is not None],
            'a1', 'delta_qB', draws=4000),
        original_Jpre_to_a1=_spearman_bootstrap(original, 'J_pre', 'a1', draws=4000),
        dJpre_to_da1=_spearman_bootstrap(paired_rows, 'dJ_pre', 'da1', draws=4000),
        dJpre_to_delta=_spearman_bootstrap(paired_with_delta, 'dJ_pre', 'delta_qB',
                                           draws=4000),
        da1_to_delta=_spearman_bootstrap(paired_with_delta, 'da1', 'delta_qB',
                                         draws=4000),
        calibration=_a1_calibration(shifted),
        note=('identification chain on first-sweep drift; a1 is the mean defector '
              'change of sweep 1 over 16 clones per mother and direction. Absolute '
              'links use a1(d) versus the paired long-horizon difference; the '
              'paired links use the differences from the same mother original arm '
              '(dJ_pre, da1). High rank correlation is not a quantitative '
              'calibration - see the calibration block for slope, intercept and '
              'residuals'))
    return dict(rows=rows, chain=chain)


def _a1_calibration(rows):
    """OLS calibration of a1 on J_pre with cluster-bootstrap residual mean.

    The OLS-with-intercept residual mean is zero by construction, so the
    unbiasedness evidence lives in the *uncalibrated* error
    epsilon = a1 - J_pre (cluster-bootstrap interval) plus slope/intercept
    intervals, not in the fitted residual mean.
    """
    x = np.asarray([row['J_pre'] for row in rows], dtype=float)
    y = np.asarray([row['a1'] for row in rows], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    epsilon = y - x
    mothers = sorted({row['seed'] for row in rows})
    groups = np.asarray([mothers.index(row['seed']) for row in rows])
    rng = np.random.default_rng(20261006)

    def _cluster_draws(statistic):
        values = []
        for _ in range(4000):
            take = rng.integers(0, len(mothers), size=len(mothers))
            indices = _cluster_indices(groups, take)
            values.append(statistic(indices))
        return values

    epsilon_means = _cluster_draws(lambda indices: float(epsilon[indices].mean()))
    slope_draws, intercept_draws, residual_means = [], [], []
    for _ in range(4000):
        take = rng.integers(0, len(mothers), size=len(mothers))
        indices = _cluster_indices(groups, take)
        drawn_slope, drawn_intercept = np.polyfit(x[indices], y[indices], 1)
        slope_draws.append(float(drawn_slope))
        intercept_draws.append(float(drawn_intercept))
        residual_means.append(float(residual[indices].mean()))
    return dict(n=len(rows), slope=float(slope), intercept=float(intercept),
                slope_bootstrap95=[float(np.quantile(slope_draws, .025)),
                                   float(np.quantile(slope_draws, .975))],
                intercept_bootstrap95=[float(np.quantile(intercept_draws, .025)),
                                       float(np.quantile(intercept_draws, .975))],
                epsilon_mean=float(epsilon.mean()),
                epsilon_bootstrap95=[float(np.quantile(epsilon_means, .025)),
                                     float(np.quantile(epsilon_means, .975))],
                residual_mean=float(residual.mean()),
                residual_mean_bootstrap95=[float(np.quantile(residual_means, .025)),
                                           float(np.quantile(residual_means, .975))],
                residual_sd=float(residual.std(ddof=1)),
                mean_clone_se=float(np.mean([row['a1_clone_se'] for row in rows])),
                note=('OLS of a1 on J_pre over shifted mother-direction rows. '
                      'epsilon = a1 - J_pre is the uncalibrated prediction error '
                      '(mother-level cluster bootstrap interval); the fitted '
                      'residual mean is zero by construction with an intercept and '
                      'is kept only for completeness, never as unbiasedness '
                      'evidence. residual_sd is compared against the mean '
                      'clone-level standard error of a1'))


def precision_report(root: Path, model: dict) -> dict:
    heldout_outcomes, _, heldout_runs = load_campaign_outcomes(
        PROJECT_ROOT / 'configs/nucleation/n06_heldout_compact.json', root)
    precision_outcomes, _, precision_runs = load_campaign_outcomes(
        PROJECT_ROOT / PRECISION_HELDOUT_CONFIG, root)
    descriptor_source = json.loads((PROJECT_ROOT / HELDOUT_DESCRIPTORS).read_text())
    descriptors = {row['mother_path']: row['descriptors']
                   for row in descriptor_source['heldout']['heldout_mothers']}
    heldout = _combine_clones([heldout_outcomes, precision_outcomes])
    training_rows = _training_compact_rows()
    training_new_outcomes, _, training_runs = load_campaign_outcomes(
        PROJECT_ROOT / PRECISION_TRAINING_CONFIG, root)
    training = _combine_clones([training_rows, training_new_outcomes])

    def enrich(rows):
        for row in rows:
            if 'descriptors' not in row:
                row['descriptors'] = descriptors.get(row['mother_path'], {})
            pit = pit_observables(
                *mother_arrays(row['mother_path'], model), model['size'])
            row['pit'] = dict(pit_floor=pit['pit_floor'], background=pit['background'],
                              half_recovery_distance=pit['half_recovery_distance'])
            row['mother_hash'] = file_hash(PROJECT_ROOT / row['mother_path'])
            row['runs'] = dict(
                prior_16clones=(heldout_runs.get(row['mother_path'], {}) or {}).get(
                    'original', 'see analysis 50a1d6f6a0f2b399'),
                precision_32clones=(precision_runs.get(row['mother_path'], {})
                                    or training_runs.get(row['mother_path'], {})
                                    or {}).get('original'))
        return rows

    heldout = enrich(heldout)
    training = enrich(training)
    keys = ['defectors', 'interface_edges', 'n_clusters', 'resource_mean', 'eta_mean',
            'boundary_c_mean', 'J', 'J_pre']
    screening = {}
    for name, rows in (('heldout_35', heldout), ('training_15', training),
                       ('pooled_50', heldout + training)):
        q = [row['q_B'] for row in rows]

        def _ranked(values):
            return len({float(value) for value in values}) >= 2

        screening[name] = dict(
            n_mothers=len(rows),
            heterogeneity=heterogeneity_diagnostic(
                [dict(q_B=row['q_B'], n=row['n']) for row in rows]),
            spearman={key: _spearman([row['descriptors'][key] for row in rows], q)
                      for key in keys
                      if all(key in row['descriptors'] for row in rows)
                      and _ranked([row['descriptors'][key] for row in rows])},
            spearman_pit={key: _spearman([row['pit'][key] for row in rows], q)
                          for key in ('pit_floor', 'half_recovery_distance')
                          if _ranked([row['pit'][key] for row in rows])})
    return dict(heldout_mothers=heldout, training_mothers=training,
                descriptor_screening=screening,
                frozen_model_regrade=_frozen_model_evaluation(heldout,
                                                              descriptor_source),
                descriptor_source=dict(path=HELDOUT_DESCRIPTORS,
                                       sha256=file_hash(PROJECT_ROOT / HELDOUT_DESCRIPTORS)),
                note=('clone augmentation: heldout 35 mothers have 16+32=48 clones, '
                      'training 15 mothers have 32+32=64 clones (the original fate '
                      'campaign ran 32 clones for the training mothers); '
                      'heterogeneity diagnostics use each mother actual n. Descriptor '
                      'associations are discovery-level on previously used mothers, '
                      'no confirmatory claim'))


def _frozen_model_evaluation(heldout_rows, descriptor_source):
    """Re-grade the frozen confirmatory-round predictions on sharpened q_B labels.

    The M0/M1/M2 weights, standardisation and SOURCE group means are read from the
    revision-2 confirmation report without refitting, and evaluated against the
    48-clone q_B of the same 35 held-out compact mothers. This separates the
    effect of sharper labels from any model refitting.
    """
    models = descriptor_source['heldout']['models']
    rows = [row for row in heldout_rows if row.get('descriptors')]
    truth = np.asarray([row['q_B'] for row in rows], dtype=float)
    predictions = {'SOURCE': np.full(len(rows),
                                     models['SOURCE']['group_means']['compact_16'])}
    for name, keys in MODEL_SPECS.items():
        frozen = models[name]
        design = np.asarray([[row['descriptors'][key] for key in keys] for row in rows])
        mean = np.asarray([frozen['standardisation'][key][0] for key in keys])
        scale = np.asarray([frozen['standardisation'][key][1] for key in keys])
        weights = np.asarray([frozen['weights']['intercept']]
                             + [frozen['weights'][key] for key in keys])
        predictions[name] = predict(weights, design, mean, scale)
    labels = (truth > 0.5).astype(int)

    def _metric(probability):
        distinct = np.unique(probability).size >= 2
        return dict(
            spearman=_spearman(probability.tolist(), truth.tolist()) if distinct else None,
            auc=_auc(labels.tolist(), probability.tolist())
            if 0 < labels.sum() < labels.size else None,
            mse_qB=float(np.mean((probability - truth) ** 2)))

    metrics = {name: _metric(probability) for name, probability in predictions.items()}
    rng = np.random.default_rng(20261006)
    indices = np.arange(len(rows))
    deltas = {}
    for pair, (a, b) in (('M1_minus_SOURCE', ('M1', 'SOURCE')),
                         ('M1_minus_M0', ('M1', 'M0'))):
        samples = defaultdict(list)
        for _ in range(4000):
            take = rng.choice(indices, size=indices.size, replace=True)
            sub_truth = truth[take]
            sub_labels = (sub_truth > 0.5).astype(int)
            if 0 < sub_labels.sum() < sub_labels.size:
                samples['auc'].append(
                    _auc(sub_labels.tolist(), predictions[a][take].tolist())
                    - _auc(sub_labels.tolist(), predictions[b][take].tolist()))
            samples['mse_qB'].append(
                float(np.mean((predictions[a][take] - sub_truth) ** 2)
                      - np.mean((predictions[b][take] - sub_truth) ** 2)))
        deltas[pair] = {metric: dict(
            mean=float(np.mean(values)) if values else None,
            bootstrap95=[float(np.quantile(values, .025)),
                         float(np.quantile(values, .975))] if values else None)
            for metric, values in samples.items()}
    return dict(n_mothers=len(rows), metrics=metrics, paired_deltas=deltas,
                previous_labels_metrics=descriptor_source['heldout']['within_group'][
                    'compact_16'],
                note=('frozen predictions from analysis 84e608ec76114c63 re-graded '
                      'against 48-clone q_B on the same 35 held-out compact mothers; '
                      'no refitting, so any metric change reflects label precision'))


def mother_arrays(mother_path, model):
    with np.load(PROJECT_ROOT / mother_path, allow_pickle=False) as stored:
        return stored['strategy'].copy(), stored['resource'].copy()


def _training_compact_rows():
    report = json.loads((PROJECT_ROOT / TRAINING_FATE).read_text())
    rows = []
    for shape, mothers in report['groups'].items():
        if not shape.startswith('compact'):
            continue
        for mother in mothers:
            if 'original' not in mother['arms']:
                continue
            rows.append(dict(
                mother_path=mother['mother_path'], seed=_seed_of(mother['mother_path']),
                q_B=mother['arms']['original']['q_B'], n=mother['arms']['original']['n'],
                hits=round(mother['arms']['original']['q_B']
                           * mother['arms']['original']['n']),
                descriptors=mother['descriptors']))
    return rows


def _combine_clones(sources):
    """Pool clone counts per mother across campaigns; returns one row per mother.

    Accepts campaign outcome maps (mother -> arm -> run summaries) or plain row
    lists carrying hits/n (and optional descriptors) per mother.
    """
    combined = {}
    for source in sources:
        if isinstance(source, dict):
            items = [(mother_path, hits_of(arms['original']), len(arms['original']), None)
                     for mother_path, arms in source.items() if 'original' in arms]
        else:
            items = [(row['mother_path'], row['hits'], row['n'], row.get('descriptors'))
                     for row in source]
        for mother_path, source_hits, source_n, descriptors in items:
            row = combined.setdefault(mother_path, dict(
                mother_path=mother_path, seed=_seed_of(mother_path), hits=0, n=0))
            row['hits'] += source_hits
            row['n'] += source_n
            if descriptors:
                row['descriptors'] = descriptors
    for row in combined.values():
        row['q_B'] = row['hits'] / row['n']
    return sorted(combined.values(), key=lambda row: row['seed'])


def _post_substep_boundary(prereg, model):
    """Boundary-cooperator mean resource AFTER the resource substep, per mother/shift.

    The preregistered B(d) uses the raw shifted field; the strategy phase of the
    next sweep actually evaluates payoffs on the substep of that field - the same
    field J_pre is built from. This diagnostic keeps the two observation time
    points apart instead of folding them into one average.
    """
    size = model['size']
    values = {}
    for entry in prereg['mothers']:
        with np.load(PROJECT_ROOT / entry['mother_path'], allow_pickle=False) as stored:
            strategy = stored['strategy'].copy()
            resource = stored['resource'].copy()
        fields = {'original': resource}
        for shift in prereg['shifts']:
            key = f'({shift[0]}, {shift[1]})'
            fields[key] = np.roll(resource.reshape(size, size),
                                  (int(shift[0]), int(shift[1])), (0, 1)).reshape(-1)
        for key, field in fields.items():
            r_pre = resource_substep(strategy, field, size, model['growth'],
                                     model['depletion'], model['diffusion'])
            observables = interface_observables(
                strategy, r_pre, size, model['temptation'], model['coupling'],
                model['strength'], model['selection_noise'])
            values[(entry['seed'], key)] = observables['boundary_c_mean']
    return values


def direction_round_report(root: Path) -> dict:
    prereg = json.loads((PROJECT_ROOT / PREREG).read_text())
    campaign_config = json.loads((PROJECT_ROOT / DIRECTION_CONFIG).read_text())
    if (file_hash(PROJECT_ROOT / DIRECTION_CONFIG)
            != prereg['campaign_config_sha256']):
        raise ValueError('preregistration was computed for a different campaign config')
    magnitudes = _arm_magnitudes(campaign_config)
    outcomes, _, identities = load_campaign_outcomes(PROJECT_ROOT / DIRECTION_CONFIG, root)
    rows = mother_rows(outcomes, magnitudes, identities)
    pairs = direction_pairs(rows, prereg)
    model = campaign_config['base']['model']
    post = _post_substep_boundary(prereg, model)
    post_original = {seed: value for (seed, key), value in post.items()
                     if key == 'original'}
    for pair in pairs:
        shifted_value = post.get((pair['seed'], pair['direction']))
        base = post_original.get(pair['seed'])
        pair['B_post'] = shifted_value
        pair['dB_post'] = (shifted_value - base if shifted_value is not None
                           and base is not None else None)
    direction = direction_report(pairs, rows, prereg, root)
    direction['post_substep_diagnostic'] = dict(
        dB_post_to_delta=_spearman_bootstrap(
            [pair for pair in pairs if pair['dB_post'] is not None],
            'dB_post', 'delta_qB', draws=4000),
        dB_post_vs_dB=_spearman_bootstrap(pairs, 'dB_post', 'dB', draws=4000),
        note=('post-hoc diagnostic: B_post is the boundary-cooperator mean resource '
              'of the shifted field after the resource substep (the field the next '
              'strategy phase evaluates), whereas the preregistered B uses the raw '
              'shifted field; both time points are reported separately'))
    dose = dose_chain(root, prereg, pairs)
    precision = precision_report(root, model)
    identity = dict(
        model=model,
        preregistration=dict(path=PREREG, sha256=file_hash(PROJECT_ROOT / PREREG)),
        direction_config=dict(path=DIRECTION_CONFIG,
                              sha256=file_hash(PROJECT_ROOT / DIRECTION_CONFIG)),
        dose_config=dict(path=DOSE_CONFIG, sha256=file_hash(PROJECT_ROOT / DOSE_CONFIG)),
        precision_configs=dict(
            heldout=dict(path=PRECISION_HELDOUT_CONFIG,
                         sha256=file_hash(PROJECT_ROOT / PRECISION_HELDOUT_CONFIG)),
            training=dict(path=PRECISION_TRAINING_CONFIG,
                          sha256=file_hash(PROJECT_ROOT / PRECISION_TRAINING_CONFIG))),
        analysis_sources={
            str(path.relative_to(PROJECT_ROOT)): file_hash(path)
            for path in sorted([*PROJECT_ROOT.glob('code/*.py'),
                                *PROJECT_ROOT.glob('experiments/*.py')])})
    return dict(identity=identity, direction=direction, dose=dose,
                precision=precision,
                pairs=pairs,
                interpretation=('direction-controlled confirmatory round: primary '
                                'pre-registered pit-gap test, secondary direction and '
                                'magnitude structure, identification chain and clone-'
                                'augmented heterogeneity diagnostics; no causal or '
                                'infinite-time claims'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT / 'data' / 'nucleation')
    args = parser.parse_args()
    report = direction_round_report(args.root)
    target = args.root / 'analysis' / config_hash(report['identity'])
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'direction.json', report)
    print(target / 'direction.json')
