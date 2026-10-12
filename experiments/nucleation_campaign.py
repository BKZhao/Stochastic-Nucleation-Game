"""JSON campaign expansion, deterministic seeds, process jobs and complete-only summaries."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from copy import deepcopy
import itertools
import json
from multiprocessing import get_context
from pathlib import Path
import time

import numpy as np
from nucleation_runner import run_replica,validate_config
from nucleation_survival import survival_summary,wilson_interval
from research_common import PROJECT_ROOT,config_hash,provenance,research_root,write_json


def expand_campaign(campaign: dict) -> list[tuple[dict,int]]:
    axes=campaign.get('grid',{})
    if 'damage' in axes and any(key.startswith('damage.') for key in axes):
        raise ValueError('whole damage layouts cannot be combined with damage subfield axes')
    cells=[]
    for values in itertools.product(*axes.values()):
        config=deepcopy(campaign['base'])
        for key,value in zip(axes,values):
            if key == 'damage':
                config['damage']=deepcopy(value)
            else:
                owner,field=key.split('.')
                config.setdefault(owner,{})[field]=value
        validate_config(config)
        digest=int(config_hash(config),16)
        for replica in range(campaign['replicas']):
            sequence=np.random.SeedSequence([campaign['master_seed'],digest,replica])
            seed=int(sequence.generate_state(1)[0])
            cells.append((config,seed))
    if len({(config_hash(config),seed) for config,seed in cells})!=len(cells):
        raise ValueError('duplicate jobs in campaign')
    return cells


def summarize(results: list[dict], expected: int) -> dict:
    groups={}
    for result in results:
        if result['technical_status']!='complete':
            raise ValueError('technical failures cannot enter survival estimates')
        groups.setdefault(result['config_hash'],[]).append(result)
    cells=[]
    for key,runs in sorted(groups.items()):
        horizon=runs[0]['protocol']['t_max_sweeps']
        stat=survival_summary([run['observed_time'] for run in runs],
                              [run['event_observed'] for run in runs],horizon)
        events=stat['events']
        cells.append(dict(config_hash=key,parameters=runs[0]['parameters'],L=runs[0]['L'],
                          damage=runs[0]['damage'],**stat,collapse_risk=events/len(runs),
                          wilson95=wilson_interval(events,len(runs)),
                          zero_event_upper95=1-.05**(1/len(runs)) if events==0 else None,
                          recovered=sum(run['recovered_time'] is not None for run in runs),
                          not_collapsed_not_recovered=sum(not run['event_observed'] and run['recovered_time'] is None for run in runs),
                          max_strategy_cluster=max(run['max_strategy_cluster'] for run in runs),
                          max_depleted_cluster=max(run['max_depleted_cluster'] for run in runs)))
    return dict(expected_runs=expected,completed_runs=len(results),cells=cells,
                interpretation=('pre-specified geometry confirmation only; no metastability or single-nucleus claim'
                                if results and all(row['experiment_id']=='N03' for row in results)
                                else 'parameter screening only; no metastability or single-nucleus claim'),
                total_site_visits=sum(run['site_visits'] for run in results))


def main(experiment_id: str) -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output-root',type=Path,default=PROJECT_ROOT/'data'/'nucleation')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    campaign=json.loads(args.config.read_text())
    if campaign['base']['experiment_id']!=experiment_id or args.workers<1:
        raise ValueError('experiment mismatch or invalid worker count')
    cells=expand_campaign(campaign)
    upper=sum(config['model']['size']**2*config['protocol']['t_max_sweeps'] for config,_ in cells)
    print(json.dumps(dict(jobs=len(cells),maximum_site_visits=upper,workers=args.workers)),flush=True)
    if args.dry_run:
        return
    root=research_root(args.output_root)
    source=provenance(root)
    campaign_id=config_hash(campaign)
    directory=root/experiment_id/'campaigns'/campaign_id
    directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'campaign.json',campaign)
    write_json(directory/'provenance.json',source)
    jobs=[(config,seed,str(root),source) for config,seed in cells]
    results=[]
    event_count=0
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=get_context('spawn')) as executor:
        futures={executor.submit(run_replica,job):index for index,job in enumerate(jobs)}
        for future in as_completed(futures):
            result=future.result()
            results.append(result)
            event_count+=int(result['event_observed'])
            print(json.dumps(dict(completed=len(results),total=len(jobs),
                                  elapsed_seconds=round(time.perf_counter()-started,2),
                                  events=event_count)),flush=True)
    report=summarize(results,len(jobs))
    report['elapsed_seconds']=time.perf_counter()-started
    report['source_hash']=source['source_hash']
    write_json(directory/'summary.json',report)
    print(f'summary: {directory / "summary.json"}',flush=True)
