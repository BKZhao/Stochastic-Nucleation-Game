"""Freeze the direction-round state-level predictions before any continuation runs.

Reads the direction campaign configuration (mother selection rules, shift vectors,
model), selects the same outcome-blind mothers, and computes for each mother:
the descriptor set, the resource-pit geometry, the interface resource B(d) of
every shifted state, and the first-sweep drift predictor J_pre of every shifted
state. None of these quantities touch a continuation outcome; the file is the
pre-registered prediction record for the direction-controlled round.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from analyze_nucleation_fate import mother_descriptors
from nucleation_interface import (interface_observables, pit_observables,
                                  resource_substep)
from research_common import PROJECT_ROOT, file_hash, write_json
from run_nucleation_fate import select_mothers

CAMPAIGN_CONFIG = 'configs/nucleation/n06_direction_controls.json'
OUTPUT = 'data/nucleation/N06/campaigns/preregistered_direction_predictions.json'


def shifted_state_quantities(strategy, resource, model, shift):
    """Interface observables of the shifted field and of its resource substep."""
    size = model['size']
    rolled = np.roll(resource.reshape(size, size), (int(shift[0]), int(shift[1])),
                     (0, 1)).reshape(-1)
    interface = interface_observables(strategy, rolled, size, model['temptation'],
                                      model['coupling'], model['strength'],
                                      model['selection_noise'])
    r_pre = resource_substep(strategy, rolled, size, model['growth'],
                             model['depletion'], model['diffusion'])
    predictor = interface_observables(strategy, r_pre, size, model['temptation'],
                                      model['coupling'], model['strength'],
                                      model['selection_noise'])
    return interface['boundary_c_mean'], interface['J'], predictor['J']


def torus_magnitude(shift, size):
    """Euclidean norm of the torus-reduced displacement (per-axis wrap)."""
    wrapped = [min(abs(v) % size, size - abs(v) % size) for v in shift]
    return float(np.hypot(*wrapped))


def main() -> None:
    campaign = json.loads((PROJECT_ROOT / CAMPAIGN_CONFIG).read_text())
    model = campaign['base']['model']
    shifts = [entry['shift'] for entry in campaign['arms']
              if isinstance(entry, dict) and entry['arm'] == 'shifted']
    root = PROJECT_ROOT / 'data' / 'nucleation'
    mothers = select_mothers(root, campaign['mothers']['campaign'],
                             campaign['mothers']['rules'], bind_identity=True)
    rows = []
    for mother in mothers:
        path = PROJECT_ROOT / mother['relative_path']
        with np.load(path, allow_pickle=False) as stored:
            strategy = stored['strategy'].copy()
            resource = stored['resource'].copy()
        descriptors = mother_descriptors(path, model)
        pit = pit_observables(strategy, resource, model['size'])
        boundary_shift, j_shift, j_pre_shift = {}, {}, {}
        for shift in shifts:
            key = f'({shift[0]}, {shift[1]})'
            boundary, _, predictor = shifted_state_quantities(
                strategy, resource, model, shift)
            boundary_shift[key] = float(boundary)
            j_pre_shift[key] = float(predictor)
        rows.append(dict(
            seed=mother['seed'], group=mother['name'], mother_path=mother['relative_path'],
            mother_file_sha256=mother['mother_file_sha256'],
            descriptors=descriptors,
            pit=dict(pit_floor=pit['pit_floor'], background=pit['background'],
                     half_recovery_distance=pit['half_recovery_distance'],
                     pit_profile=pit['pit_profile']),
            boundary_c_original=descriptors['boundary_c_mean'],
            J_pre_original=descriptors['J_pre'],
            boundary_c_shift=boundary_shift, J_pre_shift=j_pre_shift))
    magnitudes = sorted({round(torus_magnitude(shift, model['size']), 4) for shift in shifts})
    report = dict(
        registered=datetime.now(timezone.utc).isoformat(),
        campaign_config=CAMPAIGN_CONFIG,
        campaign_config_sha256=file_hash(PROJECT_ROOT / CAMPAIGN_CONFIG),
        shifts=[list(shift) for shift in shifts], magnitudes=magnitudes,
        n_mothers=len(rows), mothers=rows,
        prediction=(
            'PRIMARY (pre-registered): across mother-direction pairs the interface-'
            'resource change dB(d) = boundary_c_mean(shifted) - boundary_c_mean(original) '
            'predicts the paired long-horizon response with Spearman(dB, delta_q_B) < 0 '
            '(higher resource read by boundary cooperators after the shift, lower '
            'depletion risk); mother-level bootstrap 95% interval on the correlation '
            'must exclude zero for confirmation. SECONDARY: (a) at fixed magnitude 17.72 '
            'the four direction means differ (omnibus max-min, bootstrap); (b) the '
            'small-magnitude (2.24) responses are weaker and more direction-variable '
            'than medium ones; (c) per-mother state-level check: B(d) stays near B(0) '
            'when |d| is below the pit half-recovery distance and approaches the '
            'background beyond it. The characteristic-length claim is only confirmed '
            'if (c) holds AND the primary test passes. Multiplicity: one primary test; '
            'secondaries carry intervals but no confirmatory claim on their own.'),
        outcome_blindness=('all quantities are computed from mother states before any '
                           'continuation run of this round exists'))
    write_json(PROJECT_ROOT / OUTPUT, report)
    print(PROJECT_ROOT / OUTPUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
