"""N06 same-scale fate continuation from population-crossing mother states.

Three pre-registered resource arms per mother: the original field, the fixed-strategy
equilibrium field, and a fixed periodic translation of the original field. Each
replica is one seeded uninterrupted job from the full microscopic mother state;
mothers are selected by frozen seed-based rules before any continuation runs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import json
from multiprocessing import get_context
from pathlib import Path
import time
import traceback

import numpy as np

from b_eco_unified import SimulationState, seed_numba, step_state
from nucleation_campaign import expand_campaign
from nucleation_interface import resource_equilibrium
from nucleation_observation import observe_clusters, CLUSTER_COLUMNS, LINK_COLUMNS
from nucleation_rearrange import MECHANISM_ARMS, rearranged_resource
from nucleation_runner import TRAJECTORY_COLUMNS
from research_events import CollapseDetector
from research_state import state_arrays, state_hash
from research_common import (PROJECT_ROOT, config_hash, file_hash, provenance,
                             research_root, start_run, finish_run, verify_complete,
                             write_json, write_npz)

FATE_ARMS = ('original', 'adapted', 'shifted', 'uniform_live', 'uniform_eq',
             *MECHANISM_ARMS)


def validate_fate_config(config: dict) -> None:
    model = config['model']
    required = {'size', 'temptation', 'growth', 'depletion', 'coupling', 'strength',
                'selection_noise', 'mutation', 'diffusion'}
    if set(model) != required or model['size'] < 3 or int(model['size']) != model['size']:
        raise ValueError('model must contain the nine original API parameters; integer L >= 3')
    if not all(np.isfinite(value) for value in model.values()):
        raise ValueError('model parameters must be finite')
    for key in ['growth', 'depletion', 'mutation', 'diffusion']:
        if not 0 <= model[key] <= 1:
            raise ValueError(f'{key} must be in [0,1]')
    protocol = config['protocol']
    for key in ['t_max_sweeps', 'sample_every', 'snapshot_every']:
        if type(protocol[key]) is not int or protocol[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if type(protocol['stop_after_collapse']) is not bool:
        raise ValueError('stop_after_collapse must be boolean')
    if set(config['collapse']) != {'rho_max', 'resource_max', 'confirm_sweeps'}:
        raise ValueError('collapse configuration must not contain internal event state')
    CollapseDetector(**config['collapse'])
    fate = config.get('fate')
    if (type(fate) is not dict
            or not {'mother_path', 'arm', 'shift'}.issubset(fate)
            or fate['arm'] not in FATE_ARMS
            or type(fate['shift']) is not list
            or len(fate['shift']) != 2
            or any(type(v) is not int for v in fate['shift'])
            or config['experiment_id'] != 'N06'):
        raise ValueError('fate requires mother_path, a known arm and an integer shift pair')
    identity_keys = {'mother_file_sha256', 'mother_run_id', 'mother_time', 'mother_defectors'}
    if not set(fate).issubset({'mother_path', 'arm', 'shift'} | identity_keys):
        raise ValueError('fate accepts only the mother path, arm, shift and identity keys')
    if fate['arm'] in MECHANISM_ARMS and 'mother_file_sha256' not in fate:
        raise ValueError('mechanism arms require identity binding: the frozen '
                         'permutation seed derives from the mother file digest')
    if 'mother_file_sha256' in fate and (type(fate['mother_file_sha256']) is not str
                                         or len(fate['mother_file_sha256']) != 64
                                         or any(c not in '0123456789abcdef'
                                                for c in fate['mother_file_sha256'])):
        raise ValueError('mother_file_sha256 must be a 64-character lowercase hex digest')
    if not 0 <= config['clusters']['resource_cutoff'] <= 1:
        raise ValueError('invalid cluster resource cutoff')


def load_mother(mother_path: Path, size: int) -> tuple[SimulationState, dict]:
    """Rebuild the mother SimulationState from a lossless crossing snapshot."""
    with np.load(mother_path, allow_pickle=False) as stored:
        if int(stored['size']) != size:
            raise ValueError(f'mother lattice size mismatch: {mother_path}')
        state = SimulationState(size=size,
                                neighbors=stored['neighbors'].copy(),
                                degree=stored['degree'].copy(),
                                strategy=stored['strategy'].copy(),
                                resource=stored['resource'].copy(),
                                permutation=stored['permutation'].copy())
        metadata = json.loads(str(stored['metadata_json']))
    counter = metadata['event_counter']
    if counter['streak'] != 0 or counter['confirm_time'] is not None:
        raise ValueError(f'mother must carry a clean event clock: {mother_path}')
    return state, metadata


def apply_arm(state: SimulationState, arm: str, shift: list, model: dict,
              mother_digest: str = None) -> None:
    if arm == 'original':
        return
    if arm == 'adapted':
        state.resource[:], _ = resource_equilibrium(state.strategy, state.resource,
                                                    state.size, model['growth'],
                                                    model['depletion'], model['diffusion'])
        return
    if arm == 'uniform_live':
        state.resource[:] = float(state.resource.mean())
        return
    if arm == 'uniform_eq':
        r_eq, _ = resource_equilibrium(state.strategy, state.resource, state.size,
                                       model['growth'], model['depletion'], model['diffusion'])
        state.resource[:] = float(r_eq.mean())
        return
    if arm in MECHANISM_ARMS:
        full_model = dict(model, size=state.size)
        state.resource[:] = rearranged_resource(state.strategy, state.resource,
                                                 full_model, arm, mother_digest)
        return
    rolled = np.roll(state.resource.reshape(state.size, state.size),
                     (int(shift[0]), int(shift[1])), (0, 1)).reshape(-1)
    state.resource[:] = rolled


def _check_mother_identity(config: dict, mother_path: Path) -> None:
    """If the task binds a mother digest, refuse to run against a different file."""
    fate = config['fate']
    if 'mother_file_sha256' not in fate:
        return
    if file_hash(mother_path) != fate['mother_file_sha256']:
        raise ValueError(f'mother file no longer matches the frozen digest: {mother_path}')


def run_fate_replica(job: tuple[dict, int, str, dict]) -> dict:
    config, seed, root, provenance_info = job
    validate_fate_config(config)
    final = Path(root) / config['experiment_id'] / config_hash(config) / f'seed-{seed:010d}'
    if final.exists():
        summary = verify_complete(final)
        if summary['source_hash'] != provenance_info['source_hash']:
            raise ValueError(f'completed task has a different source version: {final}')
        _check_mother_identity(config, PROJECT_ROOT / config['fate']['mother_path'])
        return summary
    staging, final = start_run(Path(root), config, seed)
    try:
        summary = _simulate_fate(config, seed, staging, provenance_info)
        finish_run(staging, final)
        return summary
    except Exception as error:
        write_json(staging / 'FAILED.json', dict(technical_status='failed', error=repr(error),
                                                 traceback=traceback.format_exc(), seed_id=seed))
        raise


def _simulate_fate(config: dict, seed: int, directory: Path, provenance_info: dict) -> dict:
    model = dict(config['model'])
    size = model.pop('size')
    protocol = config['protocol']
    horizon = protocol['t_max_sweeps']
    fate = config['fate']
    mother_path = PROJECT_ROOT / fate['mother_path']
    _check_mother_identity(config, mother_path)
    state, mother_metadata = load_mother(mother_path, size)
    parent_model = dict(mother_metadata['parameters'])
    parent_model['size'] = mother_metadata['L']
    if parent_model != dict(config['model']):
        raise ValueError(f'mother model identity mismatch: {mother_path}')
    apply_arm(state, fate['arm'], fate['shift'], model,
              mother_digest=fate.get('mother_file_sha256'))
    seed_numba(seed)
    detector = CollapseDetector(**config['collapse'])
    initial_hash = state_hash(state)
    lineage = dict(parent_run_id=mother_metadata['run_id'],
                   parent_config_hash=mother_metadata['config_hash'],
                   parent_seed=mother_metadata['seed_id'],
                   parent_source_time=mother_metadata['local_time'],
                   parent_defectors=int(np.count_nonzero(state.strategy == 0)),
                   arm=fate['arm'], clone_seed=seed, mother_path=fate['mother_path'])
    metadata = dict(provenance_info, run_id=f'seed-{seed:010d}', experiment_id='N06',
                    config_hash=config_hash(config), seed_id=seed, L=size, parameters=model,
                    initial_state_hash=initial_hash, protocol=protocol,
                    preparation_outcome='mother_crossing_snapshot',
                    time_origin='mother_crossing', source_time=mother_metadata['local_time'],
                    event_definition_version='joint_depletion_confirmation_v1',
                    fate=lineage, damage=mother_metadata['damage'],
                    mutation_attempts=None, retained_new_defects=None, successful_nuclei=None,
                    microscopic_mutation_counters='not_instrumented',
                    lineage_resolution='mother_state_continuation')
    write_json(directory / 'provenance.json', metadata)

    def snapshot(name: str, sweep: int) -> None:
        write_npz(directory / 'states' / name, state_arrays(state),
                  dict(metadata, local_time=sweep, global_time=mother_metadata['local_time'] + sweep,
                       event_counter=asdict(detector)))
    snapshot('initial.npz', 0)
    rows, clusters, links = [], [], []
    previous = {}
    started = time.perf_counter()
    recovered_time = None
    rho, res = float(state.strategy.mean()), float(state.resource.mean())
    for sweep in range(horizon + 1):
        if sweep:
            rho, res = step_state(state, **model)
            detector.update(sweep, rho, res)
            if model['mutation'] == 0 and recovered_time is None and np.all(state.strategy == 1) and res >= .6:
                recovered_time = sweep
        stopping = detector.confirm_time is not None and protocol['stop_after_collapse']
        if sweep % protocol['sample_every'] == 0 or sweep == horizon or stopping:
            metrics, new_clusters, new_links, previous = observe_clusters(
                state, sweep, config['clusters']['resource_cutoff'], previous)
            rows.append([sweep, rho, res, *[metrics[key] for key in TRAJECTORY_COLUMNS[3:-1]],
                         detector.streak])
            clusters.extend(new_clusters)
            links.extend(new_links)
        if sweep and sweep % protocol['snapshot_every'] == 0:
            snapshot(f't-{sweep:08d}.npz', sweep)
        if stopping:
            break
    elapsed = time.perf_counter() - started
    snapshot('final.npz', sweep)
    confirmed = detector.confirm_time is not None
    summary = dict(metadata, t_entry_B=detector.entry_time, t_confirm_B=detector.confirm_time,
                   censor_time=None if confirmed else horizon, event_observed=confirmed,
                   observed_time=detector.confirm_time if confirmed else horizon,
                   technical_status='complete', executed_sweeps=sweep,
                   recovered_time=recovered_time,
                   recovery_definition='all_cooperators_and_mean_resource_at_least_0.6_mu_zero_only',
                   elapsed_seconds=elapsed, site_visits=size * size * sweep,
                   final_cooperation=rho, final_resource=res,
                   hit_B=confirmed, hit_A=recovered_time is not None,
                   unresolved=not confirmed and recovered_time is None,
                   horizon=horizon, arm=fate['arm'], clone_id=seed,
                   max_strategy_cluster=max(row[6] for row in rows),
                   max_depleted_cluster=max(row[3] for row in rows))
    write_npz(directory / 'trajectory.npz',
              {'values': np.asarray(rows), 'columns': np.array(TRAJECTORY_COLUMNS)}, metadata)
    write_npz(directory / 'clusters.npz',
              {'values': np.asarray(clusters, dtype=float).reshape(-1, len(CLUSTER_COLUMNS)),
               'columns': np.array(CLUSTER_COLUMNS)}, metadata)
    write_npz(directory / 'lineage.npz',
              {'values': np.asarray(links, dtype=np.int64).reshape(-1, len(LINK_COLUMNS)),
               'columns': np.array(LINK_COLUMNS)}, metadata)
    write_npz(directory / 'events.npz',
              dict(t_entry_B=np.array(-1 if detector.entry_time is None else detector.entry_time),
                   t_confirm_B=np.array(-1 if not confirmed else detector.confirm_time),
                   event_observed=np.array(confirmed)),
              dict(metadata, event_counter=asdict(detector)))
    write_json(directory / 'summary.json', summary)
    return summary


def select_mothers(root: Path, campaign_id: str, rules: dict, bind_identity: bool = False,
                   snapshot: str = 'growth-0.05.npz',
                   defector_window: list = None) -> list[dict]:
    """Frozen, outcome-blind mother selection from a growth campaign's task manifest.

    Candidates are expanded from the campaign's own (config, seed) list, each parent
    run is verified complete, and the snapshot's event clock must be clean; with
    ``bind_identity`` the selected rows carry the mother file digest and identity
    for embedding into continuation task keys. ``snapshot`` is either a
    defector-fraction crossing state (``growth-0.05.npz``) or a periodic state
    (``t-00010000.npz``); periodic states additionally accept a ``defector_window``
    ``[lo, hi]`` count filter so that fixed-time, count-matched samples can be
    selected without ever touching a continuation outcome.
    """
    import re
    if not re.fullmatch(r'(growth-0\.\d+|t-\d{8})\.npz', snapshot):
        raise ValueError('snapshot must be a defector-fraction crossing state name '
                         'or a periodic t-XXXXXXXX state name')
    if defector_window is not None:
        if (type(defector_window) is not list or len(defector_window) != 2
                or any(type(v) is not int for v in defector_window)
                or not 0 <= defector_window[0] <= defector_window[1]):
            raise ValueError('defector_window must be two non-decreasing integers [lo, hi]')
    directory = root / 'N02' / 'campaigns' / campaign_id
    if not directory.is_dir():
        raise ValueError(f'unknown mother campaign: {campaign_id}')
    mothers_campaign = json.loads((directory / 'campaign.json').read_text())
    if mothers_campaign['base']['experiment_id'] != 'N02':
        raise ValueError(f'mother campaign must be an N02 seeds campaign: {campaign_id}')
    candidates = {}
    for config, seed in expand_campaign(mothers_campaign):
        run_dir = root / 'N02' / config_hash(config) / f'seed-{seed:010d}'
        mother = run_dir / 'states' / snapshot
        if not mother.exists():
            continue
        verify_complete(run_dir)
        damage = config['damage']
        name = f"{damage['shape']}_{damage['count']}"
        with np.load(mother, allow_pickle=False) as stored:
            metadata = json.loads(str(stored['metadata_json']))
            counter = metadata['event_counter']
            if counter['streak'] != 0 or counter['confirm_time'] is not None:
                continue
            defectors = int(np.count_nonzero(stored['strategy'] == 0))
            if defector_window is not None and not defector_window[0] <= defectors <= defector_window[1]:
                continue
            local_time = metadata['local_time']
            run_id = metadata['run_id']
        row = dict(name=name, seed=f'{seed:010d}',
                   relative_path=str(mother.relative_to(PROJECT_ROOT))
                   if PROJECT_ROOT in mother.parents else str(mother),
                   config_hash=config_hash(config))
        if bind_identity:
            row.update(mother_file_sha256=file_hash(mother),
                       mother_run_id=run_id, mother_time=local_time,
                       mother_defectors=defectors)
        candidates.setdefault(name, []).append(row)
    selected = []
    for name, rule in rules.items():
        pool = sorted(candidates.get(name, []), key=lambda row: row['seed'])
        count = rule['count']
        if rule['rule'] == 'all':
            chosen = pool
        elif rule['rule'] == 'smallest_seed':
            chosen = pool[:count]
        elif rule['rule'] == 'largest_seed':
            chosen = pool[-count:] if count else []
        else:
            raise ValueError(f'unknown mother selection rule: {rule["rule"]}')
        if len(chosen) != count:
            raise ValueError(f'insufficient mothers for {name}: need {count}, found {len(chosen)}')
        selected.extend(chosen)
    return selected


def expand_fate_campaign(campaign: dict, root: Path) -> list[tuple[dict, int]]:
    bind = bool(campaign.get('bind_mother_identity'))
    mothers = campaign['mothers']
    snapshot = mothers.get('snapshot', 'growth-0.05.npz')
    selected = select_mothers(root, mothers['campaign'], mothers['rules'],
                              bind_identity=bind, snapshot=snapshot,
                              defector_window=mothers.get('defector_window'))
    cells = []
    for mother in selected:
        for entry in campaign['arms']:
            if type(entry) is dict:
                arm = entry['arm']
                shift = list(entry.get('shift', campaign['shift']))
            else:
                arm, shift = entry, list(campaign['shift'])
            config = deepcopy(campaign['base'])
            fate = dict(mother_path=mother['relative_path'], arm=arm, shift=shift)
            if bind:
                fate.update(mother_file_sha256=mother['mother_file_sha256'],
                            mother_run_id=mother['mother_run_id'],
                            mother_time=mother['mother_time'],
                            mother_defectors=mother['mother_defectors'])
            config['fate'] = fate
            validate_fate_config(config)
            replicas = (campaign['replicas_original'] if arm == 'original'
                        else campaign['replicas_intervention'])
            digest = int(config_hash(config), 16)
            for replica in range(replicas):
                sequence = np.random.SeedSequence([campaign['master_seed'], digest, replica])
                seed = int(sequence.generate_state(1)[0])
                cells.append((config, seed))
    if len({(config_hash(config), seed) for config, seed in cells}) != len(cells):
        raise ValueError('duplicate jobs in fate campaign')
    return cells


def summarize_fate(results: list[dict]) -> dict:
    cells = {}
    for result in results:
        if result['technical_status'] != 'complete':
            raise ValueError('technical failures cannot enter fate estimates')
        key = (result['damage']['shape'] + '_' + str(result['damage']['count']), result['arm'])
        cells.setdefault(key, []).append(result)
    groups = {}
    for (name, arm), runs in sorted(cells.items()):
        mothers = {}
        for run in runs:
            mothers.setdefault(run['fate']['parent_run_id'], []).append(run)
        mother_rows = []
        for parent, copies in sorted(mothers.items()):
            hits = sum(copy['hit_B'] for copy in copies)
            recovers = sum(copy['hit_A'] for copy in copies)
            mother_rows.append(dict(parent_run_id=parent,
                                    source_time=copies[0]['fate']['parent_source_time'],
                                    parent_defectors=copies[0]['fate']['parent_defectors'],
                                    n=len(copies), q_B=hits / len(copies),
                                    q_A=recovers / len(copies),
                                    q_U=1 - (hits + recovers) / len(copies)))
        groups[f'{name}/{arm}'] = dict(
            n_runs=len(runs), n_mothers=len(mothers), mothers=mother_rows,
            mean_q_B=float(np.mean([row['q_B'] for row in mother_rows])),
            mean_q_A=float(np.mean([row['q_A'] for row in mother_rows])),
            mean_q_U=float(np.mean([row['q_U'] for row in mother_rows])))
    return dict(completed_runs=len(results), groups=groups,
                interpretation=('same-scale finite-horizon fate probabilities per mother and arm; '
                                'committor-style screening, not infinite-time or causal claims'),
                total_site_visits=sum(run['site_visits'] for run in results))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--output-root', type=Path, default=PROJECT_ROOT / 'data' / 'nucleation')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    campaign = json.loads(args.config.read_text())
    if args.workers < 1:
        raise ValueError('invalid worker count')
    root = research_root(args.output_root)
    cells = expand_fate_campaign(campaign, root)
    upper = sum(config['model']['size'] ** 2 * config['protocol']['t_max_sweeps']
                for config, _ in cells)
    print(json.dumps(dict(jobs=len(cells), maximum_site_visits=upper, workers=args.workers)), flush=True)
    if args.dry_run:
        return
    source = provenance(root)
    directory = root / 'N06' / 'campaigns' / config_hash(campaign)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'campaign.json', campaign)
    write_json(directory / 'provenance.json', source)
    jobs = [(config, seed, str(root), source) for config, seed in cells]
    results = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as executor:
        futures = {executor.submit(run_fate_replica, job): index for index, job in enumerate(jobs)}
        for future in as_completed(futures):
            results.append(future.result())
            print(json.dumps(dict(completed=len(results), total=len(jobs),
                                  elapsed_seconds=round(time.perf_counter() - started, 2),
                                  events=sum(row['hit_B'] for row in results))), flush=True)
    report = summarize_fate(results)
    report['elapsed_seconds'] = time.perf_counter() - started
    report['source_hash'] = source['source_hash']
    write_json(directory / 'summary.json', report)
    print(f'summary: {directory / "summary.json"}', flush=True)


if __name__ == '__main__':
    main()
