"""Checksum selected campaign jobs and report background drift and seed risk."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from nucleation_campaign import expand_campaign,summarize
from nucleation_diagnostics import RunTrace,landmark_summary,paired_window_drift
from research_common import PROJECT_ROOT,config_hash,file_hash,verify_complete,write_json

LANDMARKS=[100,500,1000,2000,5000,10000]
WINDOW_PAIRS=[((100,500),(1500,1900)),((5000,5400),(9500,9900))]


def load_campaign(path: Path, root: Path) -> list[tuple[dict,dict,Path]]:
    campaign=json.loads(path.read_text())
    jobs=[]
    for config,seed in expand_campaign(campaign):
        directory=root/config['experiment_id']/config_hash(config)/f'seed-{seed:010d}'
        summary=verify_complete(directory)
        saved=json.loads((directory/'config.json').read_text())
        if saved!=config or summary['config_hash']!=config_hash(config) or summary['seed_id']!=seed:
            raise ValueError(f'campaign identity mismatch: {directory}')
        if summary['technical_status']!='complete':
            raise ValueError(f'technical failure: {directory}')
        jobs.append((config,summary,directory))
    if len({row[1]['source_hash'] for row in jobs})!=1:
        raise ValueError('a campaign must use one frozen source version')
    return jobs


def read_trace(config: dict, summary: dict, directory: Path) -> RunTrace:
    horizon=config['protocol']['t_max_sweeps']
    event=summary['event_observed']
    confirm=summary['t_confirm_B']
    if event:
        entry=summary['t_entry_B']
        if confirm is None or entry is None or not 1<=entry<=confirm<=horizon:
            raise ValueError('invalid observed event clock')
        if confirm-entry!=config['collapse']['confirm_sweeps']-1:
            raise ValueError('entry and confirmation do not match persistence')
        expected_end=confirm if config['protocol']['stop_after_collapse'] else horizon
        if summary['observed_time']!=confirm or summary['censor_time'] is not None:
            raise ValueError('observed event must not be administrative censoring')
    else:
        expected_end=horizon
        if confirm is not None or summary['t_entry_B'] is not None or summary['observed_time']!=horizon or summary['censor_time']!=horizon:
            raise ValueError('invalid administrative censoring clock')
    if summary['executed_sweeps']!=expected_end:
        raise ValueError('incomplete execution must not enter biological survival analysis')
    with np.load(directory/'trajectory.npz',allow_pickle=False) as stored:
        columns=stored['columns'].tolist()
        values=stored['values']
    if 'collapse_streak' not in columns:
        raise ValueError('background diagnostics require online collapse_streak (schema 2)')
    time=values[:,columns.index('time')]
    end=summary['executed_sweeps']
    expected=np.unique(np.append(np.arange(0,end+1,config['protocol']['sample_every']),end))
    if not np.array_equal(time,expected) or not np.isfinite(values).all():
        raise ValueError(f'invalid or missing trajectory samples: {directory}')
    fields={name:values[:,columns.index(name)] for name in ['cooperation','resource','n_max_1','collapse_streak']}
    return RunTrace(summary['run_id'],summary['t_confirm_B'],config['protocol']['t_max_sweeps'],time,
                    fields['cooperation'],fields['resource'],fields['n_max_1']/config['model']['size']**2,
                    fields['collapse_streak'])


def background_diagnostics(jobs: list[tuple[dict,dict,Path]]) -> list[dict]:
    groups=defaultdict(list)
    for job in jobs:
        groups[job[1]['config_hash']].append(job)
    results=[]
    for key,group in sorted(groups.items()):
        traces=[read_trace(*job) for job in group]
        cell=summarize([job[1] for job in group],len(group))['cells'][0]
        results.append(dict(config_hash=key,cell=cell,
                            landmarks=[landmark_summary(traces,time,5000) for time in LANDMARKS],
                            drift=[paired_window_drift(traces,early,late) for early,late in WINDOW_PAIRS]))
    return sorted(results,key=lambda result:result['cell']['parameters']['temptation'])


def tables(report: dict) -> str:
    lines=['# 第二轮完整诊断表','',
           '用途：独立探索批次的完整统计；输入：配置指定且COMPLETE校验通过的任务；输出：风险、条件寿命与漂移表。Owner：研究执行者。',
           '', '所有计数以独立轨迹为单位；条件寿命是确认耗竭寿命，不是成核等待。活动筛选不是已校准的恢复核心A。', '',
           '## N01 无条件结果','',
           '| b | 耗竭/n | Wilson 95% | 中位确认时间 | RMST(20000) |',
           '|---|---|---|---|---|']
    for group in report['background']:
        cell=group['cell']; low,high=cell['wilson95']
        median='未达到' if cell['median'] is None else f'{cell["median"]:.0f}'
        lines.append(f'| {cell["parameters"]["temptation"]} | {cell["events"]}/{cell["n"]} | [{low:.3f}, {high:.3f}] | {median} | {cell["rmst"]:.2f} |')
    lines+=['','## N01 各预定时点的队列','',
            '| b | t | 已确认 | 在险 | 当前持续段 | 活动筛选通过 | rho中位 | r中位 | 最大D簇比例中位 | 条件5000步事件/n | 条件RMST |',
            '|---|---|---|---|---|---|---|---|---|---|---|']
    for group in report['background']:
        b=group['cell']['parameters']['temptation']
        for row in group['landmarks']:
            medians=['—' if row[key] is None else f'{row[key]["median"]:.4f}' for key in ['cooperation','resource','max_defector_fraction']]
            conditional=row['conditional']
            risk='—' if conditional is None else f'{conditional["events"]}/{conditional["n"]}'
            rmst='—' if conditional is None else f'{conditional["rmst"]:.2f}'
            lines.append(f'| {b} | {row["landmark"]} | {row["confirmed_before_or_at"]} | {row["at_risk"]} | {row["pending_confirmation"]} | {row["screen_active"]} | '+ ' | '.join(medians)+f' | {risk} | {rmst} |')
    lines+=['','## N01 同一幸存者队列的窗口差','',
            '后窗减前窗；bootstrap以整条轨迹为单位，2000次。队列条件为存活至后窗末端，存在事后选择，不能据此证明平稳。','',
            '| b | 前窗→后窗 | 队列n | rho差 [95%] | r差 [95%] | 最大D簇比例差 [95%] |',
            '|---|---|---|---|---|---|']
    for group in report['background']:
        for row in group['drift']:
            values=[]
            for key in ['cooperation','resource','defector_fraction']:
                if key not in row:
                    values.append('—'); continue
                stat=row[key]; low,high=stat['bootstrap95']
                values.append(f'{stat["difference"]:.4f} [{low:.4f}, {high:.4f}]')
            lines.append(f'| {group["cell"]["parameters"]["temptation"]} | {row["early"]}→{row["late"]} | {row["n"]} | '+ ' | '.join(values)+' |')
    lines+=['','## N02 逐格结果','',
            '| D_r | m | 形状 | 耗竭/n | Wilson 95% | 恢复 | 未耗竭未恢复 | RMST(10000) |',
            '|---|---|---|---|---|---|---|---|']
    for cell in sorted(report['seeds']['cells'],key=lambda c:(c['parameters']['diffusion'],c['damage']['count'],c['damage']['shape'])):
        low,high=cell['wilson95']; damage=cell['damage']
        lines.append(f'| {cell["parameters"]["diffusion"]} | {damage["count"]} | {damage["shape"]} | {cell["events"]}/{cell["n"]} | [{low:.3f}, {high:.3f}] | {cell["recovered"]} | {cell["not_collapsed_not_recovered"]} | {cell["rmst"]:.2f} |')
    return '\n'.join(lines).rstrip()+'\n'


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--background',type=Path,required=True)
    parser.add_argument('--seeds',type=Path,required=True)
    parser.add_argument('--root',type=Path,default=PROJECT_ROOT/'data'/'nucleation')
    args=parser.parse_args()
    background=load_campaign(args.background,args.root)
    seeds=load_campaign(args.seeds,args.root)
    manifests={str(directory.relative_to(args.root)):file_hash(directory/'COMPLETE') for _,_,directory in background+seeds}
    analysis_files=sorted([*PROJECT_ROOT.glob('code/*.py'),*PROJECT_ROOT.glob('experiments/*.py')])
    sources={str(path.relative_to(PROJECT_ROOT)):file_hash(path) for path in analysis_files}
    identity=dict(manifests=manifests,analysis_sources=sources)
    report=dict(identity,background=background_diagnostics(background),
                seeds=summarize([summary for _,summary,_ in seeds],len(seeds)),
                input_versions={name:sorted({(s['run_commit'],s['source_hash']) for _,s,_ in jobs}) for name,jobs in [('background',background),('seeds',seeds)]},
                interpretation='exploratory; conditional survival and drift, not nucleation waiting or stationarity proof')
    target=args.root/'analysis'/config_hash(identity)
    target.mkdir(parents=True,exist_ok=True)
    write_json(target/'diagnostics.json',report)
    (target/'tables.md').write_text(tables(report))
    print(target)


if __name__=='__main__':
    main()
