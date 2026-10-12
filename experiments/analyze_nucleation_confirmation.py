"""Analyze only the pre-specified independent random-versus-compact confirmation."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from analyze_nucleation_followup import load_campaign,read_trace
from nucleation_campaign import summarize
from nucleation_survival import fisher_two_sided,risk_difference_interval
from research_common import PROJECT_ROOT,config_hash,file_hash,write_json


FROZEN_CAMPAIGN_HASH='ece1d6b170fb3c1c'

def confirmation_report(config_path: Path, root: Path) -> dict:
    campaign=json.loads(config_path.read_text())
    if config_hash(campaign)!=FROZEN_CAMPAIGN_HASH:
        raise ValueError('campaign does not match the pre-registered confirmation protocol')
    specification=campaign['analysis']
    if specification != dict(primary_contrast='random_minus_compact',test='fisher_two_sided_probability_ordering',
                             alpha=.05,confidence=.95,interim_testing=False,difference_interval='newcombe_wilson_no_continuity_correction'):
        raise ValueError('analysis must match the frozen single-comparison protocol')
    jobs=load_campaign(config_path,root)
    shared=[]
    costs=[]
    for config,summary,directory in jobs:
        if config['experiment_id']!='N03':
            raise ValueError('confirmation must use independent N03 jobs')
        read_trace(config,summary,directory)
        comparable=deepcopy(config)
        comparable['damage'].pop('shape')
        shared.append(config_hash(comparable))
        with np.load(directory/'states'/'initial.npz',allow_pickle=False) as initial:
            strategy=initial['strategy']; resource=initial['resource']
            damaged=strategy==0
            budget=config['damage']['count']
            if (int(damaged.sum())!=budget or not np.isin(strategy,[0,1]).all()
                    or not np.array_equal(resource,1.-damaged.astype(float))):
                raise ValueError(f'initial strategy/resource damage budget mismatch: {directory}')
            costs.append(budget)
    if len(set(shared))!=1:
        raise ValueError('comparison groups differ beyond geometry')
    cells=summarize([summary for _,summary,_ in jobs],len(jobs))['cells']
    groups={cell['damage']['shape']:cell for cell in cells}
    if len(cells)!=2 or set(groups)!={'random','compact'} or any(cell['n']!=campaign['replicas'] for cell in cells):
        raise ValueError('both pre-specified groups must have the full planned sample size')
    random,compact=groups['random'],groups['compact']
    args=(random['events'],random['n'],compact['events'],compact['n'])
    difference=random['collapse_risk']-compact['collapse_risk']
    pvalue=fisher_two_sided(*args)
    sources={str(path.relative_to(PROJECT_ROOT)):file_hash(path) for path in sorted([*PROJECT_ROOT.glob('code/*.py'),*PROJECT_ROOT.glob('experiments/*.py')])}
    return dict(campaign_hash=config_hash(campaign),specification=specification,
                groups=groups,risk_difference=difference,newcombe95=risk_difference_interval(*args),
                fisher_two_sided_p=pvalue,significant=pvalue<.05,
                random_higher_supported=pvalue<.05 and difference>0,
                cost_checks=dict(runs=len(costs),nodes_each=costs[0],resource_loss_each=costs[0]),
                input_versions=sorted({(s['run_commit'],s['source_hash']) for _,s,_ in jobs}),
                manifests={str(path.relative_to(root)):file_hash(path/'COMPLETE') for _,_,path in jobs},
                analysis_sources=sources,
                scope='Only the two frozen geometry generators at L48 b1.16 D.40 mu0 m16 T10000; no spontaneous nucleation claim')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=PROJECT_ROOT/'data'/'nucleation')
    args=parser.parse_args()
    report=confirmation_report(args.config,args.root)
    identity={key:report[key] for key in ['campaign_hash','analysis_sources','manifests']}
    output=args.root/'analysis'/config_hash(identity)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'confirmation.json',report)
    print(output)
