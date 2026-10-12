"""Confirmatory analysis: multi-shift direction, uniform decomposition, held-out M0/M1/M2.

Model specification is frozen before the held-out outcomes are inspected:
M0 = [defectors]; M1 = [defectors, interface_edges, n_clusters, resource_mean];
M2 = M1 + [eta_mean]. Ridge logistic (lambda=1, standardized by training moments,
unpenalised intercept, Newton iterations), fit only on the training mothers of the
original fate campaign, evaluated on held-out mothers with Spearman, AUC (q_B>0.5)
and MSE against per-mother q_B estimates (16-32 clones).

Revision 2 (external review of 8c42997767cf432f): the Newton update previously
omitted the ridge term from the gradient while keeping it in the Hessian, so the
iteration did not optimise the declared penalised objective; its fixed point
solved the unpenalised problem and non-convergence left residual dependence on
the iteration count. The gradient now matches the declared objective
sum_i CE(y_i, p_i) + (penalty/2) * beta' R beta (tests cover the objective,
gradient, convergence and agreement with an independent optimiser). The held-out
comparison additionally reports a source-group-only baseline (training pooled
group mean, no descriptors), within-group metrics and paired differences against
that baseline, per-mother descriptors/predictions/counts, standardisation
parameters, post-hoc risk-heterogeneity diagnostics that separate mother-level
variance from branch-level binomial noise, and input/run identities. The 'brier'
field is renamed 'mse_qB': it is the MSE against per-mother clone-level risk
estimates, not a branch-level Brier score (the branch-noise term is shared by
all models, so paired differences remain meaningful). The dose-response campaign
additionally exports a per-mother predicted-vs-observed first-sweep calibration
against the preregistered J_pre predictions, with residuals epsilon = a1 - J_pre
and clone-level standard errors as the primary bias measure.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_nucleation_fate import _mean_interval, mother_descriptors
from analyze_nucleation_interface import _auc, _spearman
from analyze_nucleation_moments import defector_series
from research_common import PROJECT_ROOT, config_hash, file_hash, verify_complete, write_json
from run_nucleation_fate import expand_fate_campaign

TRAINING_FATE = 'data/nucleation/analysis/50a1d6f6a0f2b399/fate.json'
PREREGISTERED_PREDICTIONS = 'data/nucleation/N06/campaigns/preregistered_multishift_predictions.json'
DOSE_CONFIG = 'configs/nucleation/n06_multishift_dose.json'
MODEL_SPECS = {
    'M0': ['defectors'],
    'M1': ['defectors', 'interface_edges', 'n_clusters', 'resource_mean'],
    'M2': ['defectors', 'interface_edges', 'n_clusters', 'resource_mean', 'eta_mean'],
}
BOOTSTRAP_PAIRS = (('M2_minus_M1', ('M2', 'M1')),
                   ('M1_minus_M0', ('M1', 'M0')),
                   ('M1_minus_SOURCE', ('M1', 'SOURCE')),
                   ('SOURCE_minus_M0', ('SOURCE', 'M0')))


def _biased(design):
    return np.hstack([np.ones((np.asarray(design, dtype=float).shape[0], 1)),
                      np.asarray(design, dtype=float)])


def penalised_objective(weights, design, outcome, penalty=1.0):
    """Declared objective: sum cross-entropy + (penalty/2)|beta|^2, intercept unpenalised."""
    biased = _biased(design)
    outcome = np.asarray(outcome, dtype=float)
    linear = biased @ weights
    cross_entropy = -(outcome * linear - np.logaddexp(0.0, linear)).sum()
    ridge = 0.5 * penalty * float(np.asarray(weights)[1:] @ np.asarray(weights)[1:])
    return float(cross_entropy + ridge)


def penalised_gradient(weights, design, outcome, penalty=1.0):
    """Gradient of penalised_objective; zero at the declared optimum."""
    biased = _biased(design)
    outcome = np.asarray(outcome, dtype=float)
    weights = np.asarray(weights, dtype=float)
    probability = 1.0 / (1.0 + np.exp(-(biased @ weights)))
    regulariser = penalty * np.diag(np.r_[0.0, np.ones(biased.shape[1] - 1)])
    return biased.T @ (probability - outcome) + regulariser @ weights


def ridge_logistic(design, outcome, penalty=1.0, iterations=60):
    """Newton iterations on the declared penalised objective (intercept unpenalised).

    Returns (weights, solver_info); solver_info records iterations run, the final
    Newton step size and the max absolute objective gradient at the solution.
    """
    rows, columns = design.shape
    biased = np.hstack([np.ones((rows, 1)), design])
    weights = np.zeros(columns + 1)
    regulariser = penalty * np.diag(np.r_[0.0, np.ones(columns)])
    info = dict(iterations=0, final_max_step=None, max_abs_objective_gradient=None)
    for _ in range(iterations):
        probability = 1.0 / (1.0 + np.exp(-(biased @ weights)))
        gradient = biased.T @ (probability - outcome) + regulariser @ weights
        curvature = biased.T @ ((probability * (1 - probability))[:, None] * biased) + regulariser
        step = np.linalg.solve(curvature, gradient)
        weights -= step
        info['iterations'] += 1
        info['final_max_step'] = float(np.max(np.abs(step)))
        if info['final_max_step'] < 1e-9:
            break
    info['max_abs_objective_gradient'] = float(np.max(np.abs(
        penalised_gradient(weights, design, outcome, penalty))))
    return weights, info


def predict(weights, design, mean, scale):
    standard = (np.asarray(design, dtype=float) - mean) / scale
    biased = np.hstack([np.ones((standard.shape[0], 1)), standard])
    return 1.0 / (1.0 + np.exp(-(biased @ weights)))


def _arm_runs_digest(entries):
    digest = hashlib.sha256('\n'.join(f'{path}:{complete_hash}'
                                      for path, complete_hash in sorted(entries)).encode())
    return dict(n_runs=len(entries), complete_digest=digest.hexdigest()[:16])


def load_campaign_outcomes(config_path: Path, root: Path):
    campaign = json.loads(config_path.read_text())
    cells = expand_fate_campaign(campaign, root)
    outcomes = defaultdict(lambda: defaultdict(list))
    damage_name = {}
    run_identity = defaultdict(dict)
    for config, seed in cells:
        directory = root / 'N06' / config_hash(config) / f'seed-{seed:010d}'
        summary = verify_complete(directory)
        fate = config['fate']
        arm = fate['arm'] if fate['arm'] != 'shifted' else f"shifted{tuple(fate['shift'])}"
        outcomes[fate['mother_path']][arm].append(summary)
        damage_name[fate['mother_path']] = (summary['damage']['shape'] + '_'
                                            + str(summary['damage']['count']))
        run_identity[fate['mother_path']].setdefault(arm, []).append(
            (str(directory.relative_to(root)), file_hash(directory / 'COMPLETE')))
    identities = {mother: {arm: _arm_runs_digest(entries)
                           for arm, entries in sorted(arms.items())}
                  for mother, arms in run_identity.items()}
    return outcomes, damage_name, identities


def q_B(copies):
    return sum(copy['hit_B'] for copy in copies) / len(copies)


def hits_of(copies):
    return int(sum(bool(copy['hit_B']) for copy in copies))


def training_mothers():
    report = json.loads((PROJECT_ROOT / TRAINING_FATE).read_text())
    rows = []
    for shape, mothers in report['groups'].items():
        for mother in mothers:
            if 'original' in mother['arms']:
                rows.append(dict(group=shape, mother_path=mother['mother_path'],
                                 parent_run_id=mother['parent_run_id'],
                                 q_B=mother['arms']['original']['q_B'],
                                 n=mother['arms']['original']['n'],
                                 hits=round(mother['arms']['original']['q_B']
                                            * mother['arms']['original']['n']),
                                 descriptors=mother['descriptors']))
    return rows


def heldout_mothers(outcomes, damage_name, model):
    rows = []
    for mother_path, arms in outcomes.items():
        if 'original' not in arms:
            continue
        copies = arms['original']
        descriptors = mother_descriptors(PROJECT_ROOT / mother_path, model)
        rows.append(dict(group=damage_name[mother_path], mother_path=mother_path,
                         parent_run_id=copies[0]['fate']['parent_run_id'],
                         q_B=q_B(copies), n=len(copies), hits=hits_of(copies),
                         descriptors=descriptors))
    return rows


def _metrics(probability, truth):
    probability = np.asarray(probability, dtype=float)
    truth = np.asarray(truth, dtype=float)
    labels = (truth > 0.5).astype(int)
    return dict(
        spearman=_spearman(probability.tolist(), truth.tolist())
        if np.unique(probability).size >= 2 else None,
        auc=_auc(labels.tolist(), probability.tolist())
        if 0 < labels.sum() < labels.size else None,
        mse_qB=float(np.mean((probability - truth) ** 2)))


def source_baseline(training, heldout):
    """Per-source-group training clone-pooled mean; uses no descriptors or held-out data."""
    totals = defaultdict(lambda: [0.0, 0])
    for row in training:
        bucket = totals[row['group']]
        bucket[0] += row['q_B'] * row['n']
        bucket[1] += row['n']
    means = {group: value / count for group, (value, count) in totals.items()}
    overall = sum(value for value, _ in totals.values()) / sum(
        count for _, count in totals.values())
    prediction = np.asarray([means.get(row['group'], overall) for row in heldout])
    return prediction, dict(group_means=means, overall_mean=overall)


def _paired_deltas(predictions, heldout, pairs, seed=20261006, draws=4000):
    truth = np.asarray([row['q_B'] for row in heldout], dtype=float)
    labels = (truth > 0.5).astype(int)
    indices = np.arange(len(heldout))
    groups = sorted({row['group'] for row in heldout})
    scopes = [('pooled', indices)] + [
        (f'within_{group}', indices[np.asarray([row['group'] == group for row in heldout])])
        for group in groups]
    rng = np.random.default_rng(seed)
    deltas = {pair: {scope: defaultdict(list) for scope, _ in scopes} for pair, _ in pairs}
    for _ in range(draws):
        take = rng.choice(indices, size=indices.size, replace=True)
        for pair, (a, b) in pairs:
            for scope, membership in scopes:
                if scope == 'pooled':
                    chosen = take
                else:
                    chosen = take[np.isin(take, membership)]
                if chosen.size < 3:
                    continue
                sub_scope_truth = truth[chosen]
                sub_scope_labels = labels[chosen]
                usable = 0 < sub_scope_labels.sum() < sub_scope_labels.size
                pa, pb = predictions[a][chosen], predictions[b][chosen]
                if usable:
                    deltas[pair][scope]['auc'].append(
                        _auc(sub_scope_labels.tolist(), pa.tolist())
                        - _auc(sub_scope_labels.tolist(), pb.tolist()))
                if np.unique(pa).size >= 2 and np.unique(pb).size >= 2:
                    deltas[pair][scope]['spearman'].append(
                        _spearman(pa.tolist(), sub_scope_truth.tolist())
                        - _spearman(pb.tolist(), sub_scope_truth.tolist()))
                deltas[pair][scope]['mse_qB'].append(
                    float(np.mean((pa - sub_scope_truth) ** 2)
                          - np.mean((pb - sub_scope_truth) ** 2)))
    intervals = {}
    for pair, scopes in deltas.items():
        intervals[pair] = {scope: {metric: dict(
            mean=float(np.mean(values)) if values else None,
            bootstrap95=[float(np.quantile(values, .025)), float(np.quantile(values, .975))]
            if values else None) for metric, values in metrics.items() if values}
            for scope, metrics in scopes.items()}
    return intervals


def heterogeneity_diagnostic(rows, seed=20261006, draws=20000):
    """Post-hoc: mother-level q_B variance minus mean branch-level binomial noise.

    p-value from a parametric bootstrap of the common-risk binomial model
    (all mothers share the pooled clone-level risk; branch independence assumed).
    """
    q = np.asarray([row['q_B'] for row in rows], dtype=float)
    n = np.asarray([row['n'] for row in rows], dtype=int)
    if q.size < 3 or np.any(n < 2):
        return None
    variance = float(q.var(ddof=1))
    noise = float(np.mean(q * (1 - q) / (n - 1)))
    statistic = variance - noise
    pooled = float((q * n).sum() / n.sum())
    rng = np.random.default_rng(seed)
    simulated = rng.binomial(n[None, :], pooled, size=(draws, n.size)) / n[None, :]
    sim_variance = simulated.var(axis=1, ddof=1)
    sim_noise = np.mean(simulated * (1 - simulated) / (n[None, :] - 1), axis=1)
    exceedances = int(np.count_nonzero(sim_variance - sim_noise >= statistic))
    return dict(n_mothers=int(q.size), mean_qB=float(q.mean()), pooled_qB=pooled,
                sample_variance=variance, branch_noise=noise,
                excess_variance=statistic,
                excess_sd_pp=float(np.sqrt(max(statistic, 0.0)) * 100),
                homogeneity_bootstrap_p=(exceedances + 1) / (draws + 1),
                draws=draws, seed=seed,
                note=('post-hoc diagnostic: sample variance of per-mother q_B minus '
                      'mean q(1-q)/(n-1) branch noise; p from parametric bootstrap of '
                      'a common-risk binomial model, branch independence assumed'))


def fit_and_evaluate(training, heldout):
    truth = np.asarray([row['q_B'] for row in heldout], dtype=float)
    predictions = {}
    source, source_means = source_baseline(training, heldout)
    predictions['SOURCE'] = source
    results = {'SOURCE': dict(
        **_metrics(source, truth), definition='training pooled clone-level group '
        'mean per source group; no descriptors, no held-out fitting', **source_means)}
    for name, keys in MODEL_SPECS.items():
        design = np.asarray([[row['descriptors'][key] for key in keys] for row in training])
        outcome = np.asarray([row['q_B'] for row in training], dtype=float)
        mean, scale = design.mean(axis=0), design.std(axis=0)
        scale[scale < 1e-12] = 1.0
        weights, info = ridge_logistic((design - mean) / scale, outcome)
        test_design = np.asarray([[row['descriptors'][key] for key in keys] for row in heldout])
        probability = predict(weights, test_design, mean, scale)
        predictions[name] = probability
        results[name] = dict(
            **_metrics(probability, truth),
            weights={key: float(value) for key, value in zip(['intercept', *keys], weights)},
            standardisation={key: [float(m), float(s)] for key, m, s in zip(keys, mean, scale)},
            solver=info)
    intervals = _paired_deltas(predictions, heldout, BOOTSTRAP_PAIRS)
    groups = sorted({row['group'] for row in heldout})
    within = {group: {name: _metrics(predictions[name][
        np.asarray([row['group'] == group for row in heldout])],
        truth[np.asarray([row['group'] == group for row in heldout])])
        for name in predictions} for group in groups}
    heterogeneity = {group: heterogeneity_diagnostic(
        [row for row in heldout if row['group'] == group]) for group in groups}
    return results, intervals, predictions, within, heterogeneity


def multishift_contrasts(outcomes, training, run_identity):
    by_path = {row['mother_path']: row for row in training}
    groups = defaultdict(list)
    for mother_path, arms in outcomes.items():
        if 'original' in arms or mother_path not in by_path:
            continue
        base = by_path[mother_path]
        for arm, copies in arms.items():
            groups[(base['group'], arm)].append(
                dict(mother_path=mother_path, parent_run_id=base['parent_run_id'],
                     delta=q_B(copies) - base['q_B'], n=len(copies),
                     hits=hits_of(copies), q_B=q_B(copies),
                     base_q_B=base['q_B'], base_n=base['n'],
                     runs=run_identity.get(mother_path, {}).get(arm)))
    table = {}
    for (group, arm), rows in sorted(groups.items()):
        values = [row['delta'] for row in rows]
        table[f'{group}/{arm}'] = dict(
            n_mothers=len(rows), mean_delta=float(np.mean(values)),
            interval=_mean_interval(values),
            mothers_lower=sum(value < 0 for value in values),
            mothers=[dict(mother_path=row['mother_path'], **{
                key: value for key, value in row.items() if key != 'mother_path'})
                     for row in rows])
    return table


def _calibration_rows(by_seed, changes):
    """Join (mother_path, arm) first-sweep changes onto preregistered J_pre values."""
    rows = []
    for (mother_path, arm), values in sorted(changes.items()):
        seed_id = mother_path.split('seed-')[1].split('/')[0]
        entry = by_seed.get(seed_id)
        if entry is None:
            continue
        predicted = entry['J_pre_shift'][arm[len('shifted'):]]
        observed = float(np.mean(values))
        rows.append(dict(
            group=entry['group'], seed=seed_id, arm=arm, n=len(values),
            j_pre_predicted=float(predicted), a1_observed=observed,
            a1_clone_se=float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1 else None,
            residual=observed - float(predicted),
            ratio=observed / predicted if abs(predicted) >= 1.0 else None))
    return rows


def dose_calibration(root: Path) -> dict:
    """Per-mother predicted-vs-observed first-sweep drift for the dose campaign.

    Prediction (preregistered before execution): a1 = J_pre of the arm-shifted
    state. Observed: mean first-sweep defector change over clones, with the
    clone-level standard error. The residual epsilon = a1 - J_pre is the primary
    quantity; the ratio a1/J_pre is only reported where |J_pre| >= 1 so that
    near-zero predictions do not produce unstable ratios.
    """
    config_path = PROJECT_ROOT / DOSE_CONFIG
    prereg_path = PROJECT_ROOT / PREREGISTERED_PREDICTIONS
    if not config_path.exists() or not prereg_path.exists():
        return None
    prereg = json.loads(prereg_path.read_text())
    by_seed = {entry['seed']: entry for entry in prereg['mothers']}
    campaign = json.loads(config_path.read_text())
    horizon = campaign['base']['protocol']['t_max_sweeps']
    node_count = campaign['base']['model']['size'] ** 2
    changes = defaultdict(list)
    for config, seed in expand_fate_campaign(campaign, root):
        directory = root / 'N06' / config_hash(config) / f'seed-{seed:010d}'
        verify_complete(directory)
        fate = config['fate']
        arm = fate['arm'] if fate['arm'] != 'shifted' else f"shifted{tuple(fate['shift'])}"
        series = defector_series(directory, node_count, horizon)
        changes[(fate['mother_path'], arm)].append(float(series[1] - series[0]))
    rows = _calibration_rows(by_seed, changes)
    summary = {}
    for (group, arm) in sorted({(row['group'], row['arm']) for row in rows}):
        selected = [row for row in rows if row['group'] == group and row['arm'] == arm]
        residuals = [row['residual'] for row in selected]
        ratios = [row['ratio'] for row in selected if row['ratio'] is not None]
        summary[f'{group}/{arm}'] = dict(
            n_mothers=len(selected), mean_residual=float(np.mean(residuals)),
            residual_interval=_mean_interval(residuals),
            median_ratio=float(np.median(ratios)) if ratios else None,
            n_ratio_stable=len(ratios))
    return dict(prediction=prereg['prediction'], registered=prereg['registered'],
                rows=rows, summary=summary,
                ratio_threshold='|J_pre| >= 1; ratios near zero-predicted mothers are unstable',
                note=('per-mother calibration of the preregistered drift prediction; '
                      'epsilon = a1 - J_pre with clone-level standard errors replaces '
                      'the ratio as the primary bias measure'))


def confirmatory_report(root: Path) -> dict:
    training = training_mothers()
    for row in training:
        row['mother_hash'] = file_hash(PROJECT_ROOT / row['mother_path'])
    with open(PROJECT_ROOT / 'configs/nucleation/n06_fate_continuation.json') as stream:
        model = json.load(stream)['base']['model']
    report = {}
    heldout_rows = []
    heldout_configs = {}
    heldout_run_identity = {}
    for name in ('n06_heldout_random.json', 'n06_heldout_compact.json'):
        path = PROJECT_ROOT / 'configs/nucleation' / name
        if not path.exists():
            continue
        outcomes, damage_name, identities = load_campaign_outcomes(path, root)
        heldout_rows.extend(heldout_mothers(outcomes, damage_name, model))
        heldout_configs[f'configs/nucleation/{name}'] = file_hash(path)
        heldout_run_identity.update(identities)
    for row in heldout_rows:
        row['mother_hash'] = file_hash(PROJECT_ROOT / row['mother_path'])
        row['runs'] = heldout_run_identity.get(row['mother_path'], {}).get('original')
    if heldout_rows:
        results, intervals, predictions, within, heterogeneity = fit_and_evaluate(
            training, heldout_rows)
        export = []
        for index, row in enumerate(heldout_rows):
            export.append(dict(
                mother_path=row['mother_path'], group=row['group'],
                parent_run_id=row['parent_run_id'], q_B=row['q_B'], hits=row['hits'],
                n=row['n'], mother_hash=row['mother_hash'], runs=row['runs'],
                descriptors=row['descriptors'],
                predictions={name: float(probability[index])
                             for name, probability in predictions.items()}))
        training_export = [dict(
            mother_path=row['mother_path'], group=row['group'], q_B=row['q_B'],
            hits=row['hits'], n=row['n'], mother_hash=row['mother_hash'],
            descriptors=row['descriptors']) for row in training]
        report['heldout'] = dict(
            n_training=len(training), n_heldout=len(heldout_rows),
            groups={group: sum(1 for row in heldout_rows if row['group'] == group)
                    for group in {row['group'] for row in heldout_rows}},
            models=results, deltas=intervals, within_group=within,
            heterogeneity=heterogeneity, model_specs=MODEL_SPECS,
            training_mothers=training_export, heldout_mothers=export,
            note=('ridge logistic lambda=1 standardized, unpenalised intercept, fit on '
                  'training mothers only; AUC ranks the thresholded labels q_B>0.5; '
                  'mse_qB is the MSE against per-mother clone-level q_B estimates, so '
                  'it carries branch-level binomial noise shared by all models and '
                  'paired differences remain meaningful; SOURCE baseline uses training '
                  'pooled group means only'))
    multishift_path = PROJECT_ROOT / 'configs/nucleation/n06_multishift_controls.json'
    outcomes, _, identities = load_campaign_outcomes(multishift_path, root)
    report['multishift'] = multishift_contrasts(outcomes, training, identities)
    dose_path = PROJECT_ROOT / DOSE_CONFIG
    prereg_path = PROJECT_ROOT / PREREGISTERED_PREDICTIONS
    if dose_path.exists() and prereg_path.exists():
        report['dose_calibration'] = dose_calibration(root)
    files = sorted([*PROJECT_ROOT.glob('code/*.py'), *PROJECT_ROOT.glob('experiments/*.py')])
    identity = dict(
        analysis_sources={str(path.relative_to(PROJECT_ROOT)): file_hash(path) for path in files},
        training_fate=dict(path=TRAINING_FATE, sha256=file_hash(PROJECT_ROOT / TRAINING_FATE)),
        heldout_configs=heldout_configs,
        multishift_config=dict(path='configs/nucleation/n06_multishift_controls.json',
                               sha256=file_hash(multishift_path)),
        dose_config=dict(path=DOSE_CONFIG, sha256=file_hash(dose_path)),
        preregistered_predictions=dict(path=PREREGISTERED_PREDICTIONS,
                                       sha256=file_hash(prereg_path)))
    return dict(identity=identity, **report,
                interpretation=('confirmatory screening: pre-registered displacement directions, '
                                'total-matched structure decomposition and held-out nested model '
                                'comparison; no causal or infinite-time claims'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT_ROOT / 'data' / 'nucleation')
    args = parser.parse_args()
    report = confirmatory_report(args.root)
    target = args.root / 'analysis' / config_hash(report['identity'])
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'confirmation.json', report)
    print(target / 'confirmation.json')
