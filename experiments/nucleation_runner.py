"""One complete replica per job; persistence is checked every physical sweep."""
from dataclasses import asdict
from pathlib import Path
import time
import traceback

import numpy as np
from b_eco_unified import create_simulation_state, step_state
from research_events import CollapseDetector
from research_state import state_arrays,state_hash
from nucleation_seeds import seed_mask
from nucleation_progress import GrowthTracker
from nucleation_interface import (equilibrium_observables,interface_observables,
                                  resource_equilibrium,resource_substep)
from nucleation_observation import observe_clusters,CLUSTER_COLUMNS,LINK_COLUMNS
from research_common import (config_hash,start_run,finish_run,write_json,write_npz,verify_complete)

TRAJECTORY_COLUMNS=['time','cooperation','resource','n_max_0','n_clusters_0',
                    'depleted_area','n_max_1','n_clusters_1','collapse_streak']
INTERFACE_COLUMNS=['time','defectors','interface_edges','J','J_pre_next',
                   'boundary_c_mean','boundary_c_min']
INTERFACE_EQUILIBRIUM_COLUMNS=['time','defectors','interface_edges','J_eq',
                               'boundary_c_mean_eq','boundary_c_min_eq',
                               'resource_mean_eq','equilibrium_sweeps']


def validate_config(config: dict) -> None:
    model=config['model']
    required={'size','temptation','growth','depletion','coupling','strength','selection_noise','mutation','diffusion'}
    if set(model)!=required or model['size']<3 or int(model['size'])!=model['size']:
        raise ValueError('model must contain the nine original API parameters; integer L >= 3')
    if not all(np.isfinite(value) for value in model.values()):
        raise ValueError('model parameters must be finite')
    if model['selection_noise']<=0 or model['coupling']<=0 or model['strength']<=0:
        raise ValueError('startup study requires K > 0 and positive coupling and strength')
    for key in ['growth','depletion','mutation','diffusion']:
        if not 0<=model[key]<=1:
            raise ValueError(f'{key} must be in [0,1]')
    for key in ['t_max_sweeps','sample_every','snapshot_every']:
        value=config['protocol'][key]
        if type(value) is not int or value<=0:
            raise ValueError(f'{key} must be a positive integer')
    if type(config['protocol']['stop_after_collapse']) is not bool:
        raise ValueError('stop_after_collapse must be boolean')
    stop_recovery=config['protocol'].get('stop_after_recovery',False)
    if type(stop_recovery) is not bool:
        raise ValueError('stop_after_recovery must be boolean')
    if stop_recovery and model['mutation'] != 0:
        raise ValueError('recovery stopping requires the all-C absorbing state at mutation=0')
    if set(config['collapse']) != {'rho_max','resource_max','confirm_sweeps'}:
        raise ValueError('collapse configuration must not contain internal event state')
    CollapseDetector(**config['collapse'])
    if stop_recovery and config['collapse']['rho_max'] >= 1:
        raise ValueError('recovery stopping requires a depletion cooperation threshold below one')
    if not 0<=config['clusters']['resource_cutoff']<=1:
        raise ValueError('invalid cluster resource cutoff')
    if config['experiment_id']=='N01' and (config.get('damage') is not None or model['mutation']<=0):
        raise ValueError('N01 requires positive mutation and no planted seed')
    if config['experiment_id'] in ('N02','N03') and config.get('damage') is None:
        raise ValueError('N02/N03 requires an explicit damage layout')
    if config.get('damage') is not None:
        value=config['damage'].get('resource_value',0.)
        if type(value) not in (int,float) or not np.isfinite(value) or not 0<=value<=1:
            raise ValueError('damage resource_value must be a finite number in [0,1]')
        field=config['damage'].get('resource_field','uniform')
        if field not in ('uniform','equilibrium'):
            raise ValueError("damage resource_field must be 'uniform' or 'equilibrium'")
    if 'growth' in config:
        growth=config['growth']
        if (set(growth) != {'defector_fractions','snapshot_crossings'}
                or type(growth['snapshot_crossings']) is not bool):
            raise ValueError('growth requires defector_fractions and boolean snapshot_crossings')
        GrowthTracker(model['size']**2,tuple(growth['defector_fractions']))
    if 'interface' in config:
        section=config['interface']
        times=section.get('equilibrium_times') if type(section) is dict else None
        if (type(section) is not dict or set(section) != {'equilibrium_times'}
                or type(times) is not list
                or any(type(value) is not int or value<0
                       or value>config['protocol']['t_max_sweeps'] for value in times)
                or any(a>=b for a,b in zip(times,times[1:]))):
            raise ValueError('interface requires strictly increasing equilibrium_times within the horizon')


def run_replica(job: tuple[dict,int,str,dict]) -> dict:
    config,seed,root,provenance=job
    validate_config(config)
    final=Path(root)/config['experiment_id']/config_hash(config)/f'seed-{seed:010d}'
    if final.exists():
        summary=verify_complete(final)
        if summary['source_hash']!=provenance['source_hash']:
            raise ValueError(f'completed task has a different source version: {final}')
        return summary
    staging,final=start_run(Path(root),config,seed)
    try:
        summary=_simulate(config,seed,staging,provenance)
        finish_run(staging,final)
        return summary
    except Exception as error:
        # Preserve technical failures separately and propagate the original exception.
        write_json(staging/'FAILED.json',dict(technical_status='failed',error=repr(error),
                                            traceback=traceback.format_exc(),seed_id=seed))
        raise


def _simulate(config: dict, seed: int, directory: Path, provenance: dict) -> dict:
    model=dict(config['model'])
    size=model.pop('size')
    protocol=config['protocol']
    horizon=protocol['t_max_sweeps']
    state=create_simulation_state(size,seed,'all_c',1.)
    detector=CollapseDetector(**config['collapse'])
    growth=config.get('growth')
    progress=GrowthTracker(size**2,tuple(growth['defector_fractions'])) if growth else None
    damage=config.get('damage')
    center=None
    if damage:
        geometry_rng=np.random.default_rng(np.random.SeedSequence([seed,811]))
        center=geometry_rng.integers(0,size,2).tolist()
        mask=seed_mask(size,damage['count'],damage['shape'],tuple(center),geometry_rng)
        state.strategy[mask.ravel()]=0
        state.resource[mask.ravel()]=damage.get('resource_value',0.)
        if damage.get('resource_field','uniform')=='equilibrium':
            # Replace the whole field with the unique fixed point of the resource
            # substep under the frozen planted strategy pattern; deterministic,
            # no random numbers consumed. Keeps the strategy protocol identical.
            state.resource[:],_=resource_equilibrium(state.strategy,state.resource,size,
                                                     model['growth'],model['depletion'],
                                                     model['diffusion'])
    initial_hash=state_hash(state)
    metadata=dict(provenance,run_id=f'seed-{seed:010d}',experiment_id=config['experiment_id'],
                  config_hash=config_hash(config),seed_id=seed,L=size,parameters=model,
                  initial_state_hash=initial_hash,protocol=protocol,
                  preparation_outcome='not_applicable_no_preparation',
                  time_origin='post_seed' if damage else 'initial_state',
                  event_definition_version='joint_depletion_confirmation_v1',
                  source_time=0,geometry_center=center,damage=damage,
                  mutation_attempts=None,retained_new_defects=None,successful_nuclei=None,
                  microscopic_mutation_counters='not_instrumented',
                  lineage_resolution='sampled_overlap_only; not successful nucleus identification')
    write_json(directory/'provenance.json',metadata)
    interface=config.get('interface')
    equilibrium_times=set(interface['equilibrium_times']) if interface else set()
    interface_rows,interface_equilibrium_rows=[],[]
    def snapshot(name: str, sweep: int) -> None:
        write_npz(directory/'states'/name,state_arrays(state),
                  dict(metadata,local_time=sweep,global_time=sweep,event_counter=asdict(detector),
                       growth=None if progress is None else progress.record()))
    snapshot('initial.npz',0)
    rows,clusters,links=[],[],[]
    previous={}
    started=time.perf_counter()
    recovered_time=None
    rho,res=float(state.strategy.mean()),float(state.resource.mean())
    for sweep in range(horizon+1):
        if sweep:
            rho,res=step_state(state,**model)
            detector.update(sweep,rho,res)
            if model['mutation']==0 and recovered_time is None and np.all(state.strategy==1) and res>=.6:
                recovered_time=sweep
        if progress is not None:
            for threshold in progress.update(sweep,int(round((1-rho)*size*size))):
                if growth['snapshot_crossings']:
                    snapshot(f'growth-{threshold}.npz',sweep)
        if interface is not None:
            current=interface_observables(state.strategy,state.resource,size,
                                           model['temptation'],model['coupling'],
                                           model['strength'],model['selection_noise'])
            r_pre=resource_substep(state.strategy,state.resource,size,model['growth'],
                                   model['depletion'],model['diffusion'])
            predictor=interface_observables(state.strategy,r_pre,size,
                                            model['temptation'],model['coupling'],
                                            model['strength'],model['selection_noise'])
            interface_rows.append([sweep,current['defectors'],current['interface_edges'],
                                   current['J'],predictor['J'],
                                   current['boundary_c_mean'],current['boundary_c_min']])
            if sweep in equilibrium_times:
                equilibrium=equilibrium_observables(state.strategy,state.resource,size,
                                                    model['temptation'],model['coupling'],
                                                    model['strength'],model['selection_noise'],
                                                    growth=model['growth'],
                                                    depletion=model['depletion'],
                                                    diffusion=model['diffusion'])
                interface_equilibrium_rows.append([sweep,equilibrium['defectors'],
                                                   equilibrium['interface_edges'],equilibrium['J'],
                                                   equilibrium['boundary_c_mean'],
                                                   equilibrium['boundary_c_min'],
                                                   equilibrium['resource_mean_eq'],
                                                   equilibrium['equilibrium_sweeps']])
        stopping_B=detector.confirm_time is not None and protocol['stop_after_collapse']
        stopping_A=recovered_time is not None and protocol.get('stop_after_recovery',False)
        stopping=stopping_B or stopping_A
        if sweep%protocol['sample_every']==0 or sweep==horizon or stopping:
            metrics,new_clusters,new_links,previous=observe_clusters(
                state,sweep,config['clusters']['resource_cutoff'],previous)
            rows.append([sweep,rho,res,*[metrics[key] for key in TRAJECTORY_COLUMNS[3:-1]],detector.streak])
            clusters.extend(new_clusters)
            links.extend(new_links)
        if sweep and sweep%protocol['snapshot_every']==0:
            snapshot(f't-{sweep:08d}.npz',sweep)
        if stopping:
            break
    elapsed=time.perf_counter()-started
    snapshot('final.npz',sweep)
    confirmed=detector.confirm_time is not None
    summary=dict(metadata,t_entry_B=detector.entry_time,t_confirm_B=detector.confirm_time,
                 censor_time=None if confirmed else horizon,event_observed=confirmed,
                 observed_time=detector.confirm_time if confirmed else horizon,
                 technical_status='complete',executed_sweeps=sweep,
                 termination_reason=('confirmed_B' if stopping_B else 'absorbing_A'
                                     if stopping_A else 'horizon'),
                 final_state_time=sweep,
                 recovered_time=recovered_time,
                 recovery_definition='all_cooperators_and_mean_resource_at_least_0.6_mu_zero_only',
                 elapsed_seconds=elapsed,site_visits=size*size*sweep,
                 final_cooperation=rho,final_resource=res,
                 growth=None if progress is None else progress.record(),
                 max_strategy_cluster=max(row[6] for row in rows),
                 max_depleted_cluster=max(row[3] for row in rows))
    write_npz(directory/'trajectory.npz',{'values':np.asarray(rows),'columns':np.array(TRAJECTORY_COLUMNS)},metadata)
    write_npz(directory/'clusters.npz',{'values':np.asarray(clusters,dtype=float).reshape(-1,len(CLUSTER_COLUMNS)),
                                     'columns':np.array(CLUSTER_COLUMNS)},metadata)
    write_npz(directory/'lineage.npz',{'values':np.asarray(links,dtype=np.int64).reshape(-1,len(LINK_COLUMNS)),
                                    'columns':np.array(LINK_COLUMNS)},metadata)
    write_npz(directory/'events.npz',dict(t_entry_B=np.array(-1 if detector.entry_time is None else detector.entry_time),
                                        t_confirm_B=np.array(-1 if not confirmed else detector.confirm_time),
                                        event_observed=np.array(confirmed)),
               dict(metadata,event_counter=asdict(detector)))
    if interface is not None:
        write_npz(directory/'interface.npz',
                  {'values':np.asarray(interface_rows,dtype=float),
                   'columns':np.array(INTERFACE_COLUMNS),
                   'equilibrium_values':np.asarray(interface_equilibrium_rows,dtype=float),
                   'equilibrium_columns':np.array(INTERFACE_EQUILIBRIUM_COLUMNS)},
                  dict(metadata,interface_definition_version='frozen_first_visit_tanh_v1',
                       interface_note=('J uses the end-of-sweep resource field; J_pre_next applies one '
                                       'resource substep to that field and is the exact first-visit '
                                       'predictor of the next sweep strategy phase; equilibrium rows '
                                       'iterate the resource substep with frozen strategies. NaN boundary '
                                       'statistics mean no cooperator has a defector neighbour.')))
    write_json(directory/'summary.json',summary)
    return summary
