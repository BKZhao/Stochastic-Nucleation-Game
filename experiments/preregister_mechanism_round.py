"""Freeze the mechanism-round state-level predictions before any continuation runs.

Reads the mechanism campaign configuration (mother selection rules, arms, model),
selects the same outcome-blind identity-bound mothers as the direction round, and
computes for each mother and each rearrangement arm: the constructed resource
field's invariants (total, value multiset, boundary multiset), the exact
J_pre of the arranged state, the linear-sensitivity prediction
dJ_lin = -sum w_u (r~_arm - r~_original)_u with the weights of the original
post-substep field, and the pre/post-substep boundary means. None of these
quantities touch a continuation outcome; the file is the pre-registered
prediction record for the mechanism round.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from nucleation_interface import (interface_observables, interface_sensitivity,
                                  resource_substep)
from nucleation_rearrange import (MECHANISM_ARMS, boundary_cooperator_mask,
                                  rearranged_resource)
from research_common import PROJECT_ROOT, file_hash, write_json
from run_nucleation_fate import select_mothers

CAMPAIGN_CONFIG = 'configs/nucleation/n06_mechanism_fate.json'
OUTPUT = 'data/nucleation/N06/campaigns/preregistered_mechanism_predictions.json'


def _interface(strategy, resource, model):
    return interface_observables(strategy, resource, model['size'],
                                 model['temptation'], model['coupling'],
                                 model['strength'], model['selection_noise'])


def mechanism_predictions(strategy, resource, model, mother_digest):
    """Per-arm arranged-field quantities for one mother (no continuation data)."""
    size = model['size']
    mask = boundary_cooperator_mask(strategy, size)
    post_original = resource_substep(strategy, resource, size, model['growth'],
                                     model['depletion'], model['diffusion'])
    sensitivity = interface_sensitivity(strategy, post_original, size,
                                        model['temptation'], model['coupling'],
                                        model['strength'], model['selection_noise'])
    weights = sensitivity['weights']
    j_pre_original = _interface(strategy, post_original, model)['J']
    b_original = float(np.asarray(resource, dtype=float).reshape(size, size)[mask].mean())
    b_post_original = float(post_original.reshape(size, size)[mask].mean())
    arms = {}
    for arm in MECHANISM_ARMS:
        arranged = rearranged_resource(strategy, resource, model, arm, mother_digest)
        post_arm = resource_substep(strategy, arranged, size, model['growth'],
                                    model['depletion'], model['diffusion'])
        delta_post = post_arm - post_original
        arms[arm] = dict(
            total_resource=float(arranged.sum()),
            multiset_preserved=bool(np.array_equal(np.sort(np.asarray(resource, dtype=float)),
                                                   np.sort(arranged))),
            boundary_multiset_preserved=bool(np.array_equal(
                np.sort(np.asarray(resource, dtype=float).reshape(size, size)[mask]),
                np.sort(arranged.reshape(size, size)[mask]))),
            J_pre=_interface(strategy, post_arm, model)['J'],
            dJ_pre_exact=float(_interface(strategy, post_arm, model)['J']
                               - j_pre_original),
            dJ_pre_linear=float(-(weights * delta_post).sum()),
            boundary_c_mean=float(arranged.reshape(size, size)[mask].mean()),
            boundary_c_mean_post=float(post_arm.reshape(size, size)[mask].mean()),
            dB=float(arranged.reshape(size, size)[mask].mean() - b_original),
            dB_post=float(post_arm.reshape(size, size)[mask].mean() - b_post_original))
    weights_masked = weights.reshape(size, size)[mask]
    top = np.sort(weights_masked)[-5:] if len(weights_masked) >= 5 else weights_masked
    return dict(
        original=dict(J_pre=float(j_pre_original),
                      boundary_c_mean=b_original,
                      boundary_c_mean_post=b_post_original,
                      n_boundary=int(mask.sum()),
                      total_resource=float(np.asarray(resource, dtype=float).sum())),
        sensitivity=dict(total_weight=float(weights.sum()),
                         weight_max=float(weights_masked.max()) if len(weights_masked) else 0.0,
                         top5_weight_share=float(top.sum() / weights.sum())
                         if weights.sum() > 0 else 0.0),
        arms=arms)


def main() -> None:
    campaign = json.loads((PROJECT_ROOT / CAMPAIGN_CONFIG).read_text())
    model = campaign['base']['model']
    root = PROJECT_ROOT / 'data' / 'nucleation'
    mothers = select_mothers(root, campaign['mothers']['campaign'],
                             campaign['mothers']['rules'], bind_identity=True)
    if len(mothers) != 48 or mothers[0]['name'] != 'compact_16':
        raise ValueError('mechanism round expects the 48 direction-round mothers')
    rows = []
    for mother in mothers:
        with np.load(PROJECT_ROOT / mother['relative_path'], allow_pickle=False) as stored:
            strategy = stored['strategy'].copy()
            resource = stored['resource'].copy()
        predictions = mechanism_predictions(strategy, resource, model,
                                            mother['mother_file_sha256'])
        for arm, block in predictions['arms'].items():
            if not block['multiset_preserved'] or not block['boundary_multiset_preserved']:
                raise ValueError(f'multiset invariant violated: {mother["seed"]} {arm}')
            if abs(block['total_resource'] - predictions['original']['total_resource']) > 1e-12:
                raise ValueError(f'total resource invariant violated: {mother["seed"]} {arm}')
        rows.append(dict(seed=mother['seed'], group=mother['name'],
                         mother_path=mother['relative_path'],
                         mother_file_sha256=mother['mother_file_sha256'],
                         **predictions))
    mean_harm = float(np.mean([abs(row['arms']['interface_harm']['dJ_pre_linear'])
                               for row in rows]))
    mean_leakage = float(np.mean([abs(row['arms']['defector_shuffle']['dJ_pre_linear'])
                                   for row in rows]))
    if mean_harm < 3.0 * mean_leakage:
        raise ValueError(f'interface arms must dominate the diffusion leakage: '
                         f'harm {mean_harm:.3f} vs defector {mean_leakage:.3f}')
    report = dict(
        registered=datetime.now(timezone.utc).isoformat(),
        campaign_config=CAMPAIGN_CONFIG,
        campaign_config_sha256=file_hash(PROJECT_ROOT / CAMPAIGN_CONFIG),
        arms=list(MECHANISM_ARMS), n_mothers=len(rows), mothers=rows,
        prediction=(
            'PRIMARY (pre-registered): at fixed strategy, total resource, value '
            'distribution and pre-substep boundary-cooperator mean, the mother-level '
            'paired contrast q_B(interface_harm) - q_B(interface_protect) is POSITIVE '
            '(extreme anti-sensitivity allocation depletes more than pro-sensitivity '
            'allocation); confirmation requires the mother-level cluster bootstrap 95% '
            'interval on the mean paired contrast to exclude zero. SECONDARY: (a) the '
            'absolute mean paired response of interface_shuffle stays below half the '
            'primary contrast (random allocation is weaker than the w_u-paired extremes); '
            '(b) the defector_shuffle mean paired response 95% interval lies within '
            '+/- 0.25 times the primary contrast (resource change on the defector '
            'side reaches boundary cooperators only through one diffusion substep '
            'with prefactor diffusion/4, so its linear prediction is an order of '
            'magnitude weaker than the interface arms - a graded, equivalence-style '
            'control, not an exact null); (c) the dose companion shows '
            'Spearman(dJ_pre, a_1 difference) > 0 with cluster bootstrap interval '
            'excluding zero, and the a_1 difference rank-links to delta_q_B; (d) the '
            'linear sensitivity prediction is REPORTED as a calibration: OLS of '
            'dJ_exact on dJ_linear across mother-arm rows with a cluster bootstrap '
            'slope interval, expected BELOW one because the full rearrangement '
            'leaves the small-perturbation regime (tanh saturation dampens the '
            'response); signs and orderings, not magnitudes, are the linear '
            'theory claim. Multiplicity: one primary test; secondaries carry '
            'intervals and margins but no independent confirmatory claim.'),
        outcome_blindness=('all quantities are computed from mother states and the '
                           'frozen construction rules before any continuation run of '
                           'this round exists; the pooled interface linear prediction '
                           'is asserted to dominate the defector-side diffusion '
                           'leakage by at least a factor of three at '
                           'preregistration time'))
    write_json(PROJECT_ROOT / OUTPUT, report)
    print(PROJECT_ROOT / OUTPUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
