"""Post-confirmation continuation: does a confirmed collapse ever reverse?

Takes one confirmed-collapse final state per (mother, core arm) from the
direction-round fate campaign (12 smallest-seed mothers, three core arms),
loads the microscopic state at confirmation, and continues it for 3000 sweeps
WITHOUT the stopping rule, under four fresh clone seeds. Reports per
(mother, arm) the maximum cooperation and mean resource after continuation
and whether the cooperation fraction ever re-exceeds the 0.2 collapse
threshold. This decides whether the paper says 'persistent macroscopic
depletion' or the stronger 'irreversible collapse'.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from b_eco_unified import SimulationState, seed_numba, step_state
from research_common import (PROJECT_ROOT, config_hash, file_hash,
                              research_root, verify_complete, write_json)
from run_nucleation_fate import expand_fate_campaign

DIRECTION_CONFIG = 'configs/nucleation/n06_direction_controls.json'
CONTINUATION_SWEEPS = 3000
CLONE_SEEDS = 4
MOTHERS = 12
ARMS = ('original', 'shifted(2, 1)', 'shifted(5, -17)')


def _arm_key(fate):
    return fate['arm'] if fate['arm'] != 'shifted' else f"shifted{tuple(fate['shift'])}"


def main() -> None:
    root = research_root(PROJECT_ROOT / 'data' / 'nucleation')
    campaign = json.loads((PROJECT_ROOT / DIRECTION_CONFIG).read_text())
    cells = expand_fate_campaign(campaign, root)
    by_arm = {}
    for config, seed in cells:
        fate = config['fate']
        by_arm.setdefault(_arm_key(fate), []).append(
            (fate['mother_path'], seed, config))
    mothers_seen = []
    chosen = []
    for mother_path, seed, config in sorted(by_arm['original']):
        run_dir = root / 'N06' / config_hash(config) / f'seed-{seed:010d}'
        summary = json.loads((run_dir / 'summary.json').read_text())
        if summary['t_confirm_B'] is not None and mother_path not in mothers_seen:
            mothers_seen.append(mother_path)
        if len(mothers_seen) >= MOTHERS:
            break
    mother_set = set(mothers_seen[:MOTHERS])
    for arm in ARMS:
        taken = set()
        for mother_path, seed, config in sorted(by_arm.get(arm, [])):
            if mother_path not in mother_set or mother_path in taken:
                continue
            run_dir = root / 'N06' / config_hash(config) / f'seed-{seed:010d}'
            summary = json.loads((run_dir / 'summary.json').read_text())
            if summary['t_confirm_B'] is not None:
                chosen.append((arm, mother_path, run_dir, summary['t_confirm_B']))
                taken.add(mother_path)
    rows = []
    model = dict(campaign['base']['model'])
    size = model.pop('size')
    for arm, mother_path, run_dir, confirm_time in chosen:
        with np.load(run_dir / 'states' / 'final.npz', allow_pickle=False) as stored:
            state = SimulationState(size=size, neighbors=stored['neighbors'].copy(),
                                    degree=stored['degree'].copy(),
                                    strategy=stored['strategy'].copy(),
                                    resource=stored['resource'].copy(),
                                    permutation=stored['permutation'].copy())
            meta = json.loads(str(stored['metadata_json']))
        outcomes = []
        for clone in range(CLONE_SEEDS):
            with np.load(run_dir / 'states' / 'final.npz', allow_pickle=False) as stored:
                clone_state = SimulationState(size=size, neighbors=stored['neighbors'].copy(),
                                              degree=stored['degree'].copy(),
                                              strategy=stored['strategy'].copy(),
                                              resource=stored['resource'].copy(),
                                              permutation=stored['permutation'].copy())
            run_digest = hashlib.sha256(str(run_dir.relative_to(root)).encode()).digest()
            seed_words = np.frombuffer(run_digest, dtype='<u4').tolist()
            clone_seed = int(np.random.SeedSequence(
                [2026100717, *seed_words, clone]).generate_state(1)[0])
            seed_numba(clone_seed)
            max_rho = float(clone_state.strategy.mean())
            final_res = None
            for sweep in range(1, CONTINUATION_SWEEPS + 1):
                rho, res = step_state(clone_state, **model)
                max_rho = max(max_rho, float(rho))
            outcomes.append(dict(seed=clone_seed, max_cooperation=max_rho,
                                 final_resource=float(res)))
        rows.append(dict(
            arm=arm, mother_path=mother_path, run=str(run_dir.relative_to(root)),
            confirm_time=confirm_time, mother_defectors=meta['fate']['parent_defectors'],
            clones=outcomes,
            max_cooperation=max(o['max_cooperation'] for o in outcomes),
            mean_final_resource=float(np.mean([o['final_resource'] for o in outcomes]))))
    reversals = [row for row in rows if row['max_cooperation'] > 0.2]
    report = dict(
        direction_config=DIRECTION_CONFIG,
        direction_config_sha256=file_hash(PROJECT_ROOT / DIRECTION_CONFIG),
        continuation_sweeps=CONTINUATION_SWEEPS, clone_seeds=CLONE_SEEDS,
        n_states=len(rows), n_reversals=len(reversals), rows=rows,
        interpretation=('post-confirmation continuation without the stopping rule: '
                        'a reversal is any continuation whose cooperation fraction '
                        're-exceeds the 0.2 collapse threshold within the continuation '
                        'window; zero reversals supports "persistent macroscopic '
                        'depletion", any reversal demotes the claim to a '
                        'finite-horizon statement'))
    identity = dict(direction_config_sha256=report['direction_config_sha256'],
                    analysis_sha256=file_hash(Path(__file__)),
                    mothers=MOTHERS, continuation_sweeps=CONTINUATION_SWEEPS,
                    clone_seeds=CLONE_SEEDS)
    report['identity'] = identity
    report['n_mothers'] = len({row['mother_path'] for row in rows})
    output = root / 'analysis' / config_hash(identity) / 'post_confirmation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite an existing report: {output}')
    write_json(output, report)
    print(output)
    print(json.dumps(dict(n=len(rows), reversals=len(reversals))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
