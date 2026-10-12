"""Negative control for the mechanism round: strategy-trajectory identity at coupling 0.

For every identity-bound mechanism-round mother and every arm, the mother state
is loaded, the arm's resource rearrangement is applied with the REAL model
(the same construction as the fate round), and the continuation is run with
coupling = 0 (resources never enter payoffs, R_eff == 1) under the SAME random
seed across arms. The resource substep is deterministic and consumes no random
numbers, and the strategy update reads payoffs only through R_eff, so the
strategy trajectories must be bitwise identical across arms whenever the
observed differences are carried by the resource -> payoff channel. The joint
depletion endpoint is NOT compared: it contains a resource threshold, which
legitimately differs across arms. The check also records that the resource
fields actually differ across arms, so identity is not trivial.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from b_eco_unified import seed_numba, step_state
from run_nucleation_fate import apply_arm, load_mother
from research_common import PROJECT_ROOT, file_hash, write_json

CAMPAIGN_CONFIG = 'configs/nucleation/n06_mechanism_fate.json'
MECHANISM_CHECK_ARMS = ('original', 'interface_protect', 'interface_harm',
                        'interface_shuffle', 'defector_shuffle')
CHECK_SWEEPS = 500
CHECKPOINT_EVERY = 100
RUN_SEED_OFFSET = 7717


def _strategy_digest(state):
    return hashlib.sha256(np.ascontiguousarray(state.strategy).tobytes()).hexdigest()


def _resource_digest(state):
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(state.resource, dtype=np.float64)).tobytes()).hexdigest()


def mother_identity_check(mother, model, check_sweeps, checkpoint_every):
    """Run all arms of one mother at coupling 0 with one shared seed."""
    size = model['size']
    state, _ = load_mother(PROJECT_ROOT / mother['relative_path'], size)
    seed = (int(mother['mother_file_sha256'][:8], 16) + RUN_SEED_OFFSET) % (2 ** 32)
    run_model = dict(model)
    run_model['coupling'] = 0.0
    run_model.pop('size')
    trajectories = {}
    resource_fingerprints = {}
    for arm in MECHANISM_CHECK_ARMS:
        state, _ = load_mother(PROJECT_ROOT / mother['relative_path'], size)
        apply_arm(state, arm, [0, 0], model, mother['mother_file_sha256'])
        seed_numba(seed)
        checkpoints = {0: _strategy_digest(state)}
        for sweep in range(1, check_sweeps + 1):
            step_state(state, **run_model)
            if sweep % checkpoint_every == 0:
                checkpoints[sweep] = _strategy_digest(state)
        trajectories[arm] = checkpoints
        resource_fingerprints[arm] = _resource_digest(state)
    identical = all(trajectories[arm] == trajectories['original']
                    for arm in MECHANISM_CHECK_ARMS)
    resources_differ = len(set(resource_fingerprints.values())) > 1
    return dict(seed=mother['seed'], mother_path=mother['relative_path'],
                check_sweeps=check_sweeps, shared_seed=seed,
                strategy_trajectories_identical=identical,
                resource_fields_differ=resources_differ,
                strategy_digest_final=trajectories['original'][check_sweeps],
                resource_digests={arm: value[:16] for arm, value
                                  in resource_fingerprints.items()},
                note=('strategy digests at sweep 0 and every checkpoint_every '
                      'sweeps; resource digests taken after arm application'))


def main() -> None:
    campaign = json.loads((PROJECT_ROOT / CAMPAIGN_CONFIG).read_text())
    model = dict(campaign['base']['model'])
    root = PROJECT_ROOT / 'data' / 'nucleation'
    from run_nucleation_fate import select_mothers
    mothers = select_mothers(root, campaign['mothers']['campaign'],
                             campaign['mothers']['rules'], bind_identity=True)
    rows = [mother_identity_check(mother, model, CHECK_SWEEPS, CHECKPOINT_EVERY)
            for mother in mothers]
    failures = [row['seed'] for row in rows if not row['strategy_trajectories_identical']]
    trivial = [row['seed'] for row in rows if not row['resource_fields_differ']]
    report = dict(
        campaign_config=CAMPAIGN_CONFIG,
        campaign_config_sha256=file_hash(PROJECT_ROOT / CAMPAIGN_CONFIG),
        n_mothers=len(rows), check_sweeps=CHECK_SWEEPS,
        all_strategy_trajectories_identical=not failures,
        failures=failures, trivial_identity_mothers=trivial, mothers=rows,
        interpretation=('negative control: with the resource -> payoff coupling set to '
                        'zero (a model parameter, the kernel update rule is untouched) '
                        'the strategy trajectories of all arms are bitwise identical '
                        'under a shared seed, so any fate-round differences must flow '
                        'through the resource -> payoff channel rather than through '
                        'task identity or data handling'))
    output = PROJECT_ROOT / 'data' / 'nucleation' / 'analysis' / 'mechanism_coupling0.json'
    write_json(output, report)
    print(output)
    print(json.dumps(dict(all_identical=not failures, trivial=trivial,
                          n=len(rows))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sweeps', type=int, default=CHECK_SWEEPS)
    args = parser.parse_args()
    CHECK_SWEEPS = args.check_sweeps
    main()
