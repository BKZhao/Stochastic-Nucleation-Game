"""Plot fixed-window outcomes and explicitly conditional spatial progression examples."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def label(name: str) -> str:
    if name == 'compact_1':
        return 'Single site'
    return name.replace('compact_16', 'Compact 16').replace('random_16', 'Random 16').replace('_r', ', r=')


def plot_summary(report: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.5), layout='constrained')
    groups = report['groups']
    names = list(groups)
    for index, (name, group) in enumerate(groups.items()):
        color = '#009E73' if name == 'compact_1' else '#0072B2' if name.startswith('compact') else '#D55E00'
        style = '--' if name.endswith('_r1') else '-'
        outcomes = group['outcomes']
        risk = np.array([row['collapse_risk'] for row in outcomes])
        intervals = np.array([row['wilson95'] for row in outcomes])
        axes[0].errorbar([row['horizon']/1000 for row in outcomes], risk*100,
                        yerr=np.array([risk-intervals[:, 0], intervals[:, 1]-risk])*100,
                        color=color, ls=style, marker='o', capsize=3, label=label(name))
        bottom = 0
        for key, fill, title in [('events', '#D55E00', 'Confirmed'), ('recovered', '#009E73', 'Recovered'),
                                 ('unresolved', '#BBBBBB', 'Unresolved')]:
            value = outcomes[-1][key]
            axes[1].bar(index, value, bottom=bottom, color=fill, label=title if index == 0 else None)
            if value:
                axes[1].text(index, bottom+value/2, str(value), ha='center', va='center', fontsize=9)
            bottom += value
        crossing = group['progression']['crossings']
        for step, (key, row) in enumerate(crossing.items()):
            stat = row['spatial']['largest_share']
            if stat is not None:
                x = step + (index-(len(groups)-1)/2)*.13
                axes[2].errorbar(x, stat['median'],
                                yerr=[[stat['median']-stat['q25']], [stat['q75']-stat['median']]],
                                fmt='o', color=color, mfc='white' if style == '--' else color, capsize=3)
    axes[0].set(xlabel='Observation window (1,000 sweeps)', ylabel='First confirmed depletion (%)',
                ylim=(0, 100), title='a  Risk; marginal Wilson 95%')
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set(xticks=range(len(names)), xticklabels=[label(name).replace(', ', '\n') for name in names],
                ylabel='Number of trajectories', title='b  Final-window outcomes',
                ylim=(0, max(group['n'] for group in groups.values())*1.25))
    axes[1].tick_params(axis='x', labelsize=8)
    axes[1].legend(frameon=False, fontsize=8, loc='upper center', ncol=3, columnspacing=.8, handlelength=1.)
    axes[2].set(xticks=range(4), xticklabels=['5%', '20%', '50%', '80%'], ylim=(0, 1.05),
                xlabel='First reached total-D fraction', ylabel='Largest D cluster / all D nodes',
                title='c  Conditional median and IQR')
    fig.suptitle('L=48, b=1.16, diffusion=0.40, mutation=0; independent intervention cohorts', fontsize=11)
    fig.supxlabel('Panel c conditions on reaching each threshold; cohort sizes differ. Thresholds are not nucleus labels.', fontsize=9)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'growth.{suffix}', dpi=200)
    plt.close(fig)


def plot_representatives(report: dict, output: Path) -> None:
    target = output / 'representatives'
    target.mkdir(parents=True, exist_ok=True)
    strategy_colors = ListedColormap(['#D55E00', '#0072B2'])
    files = [('initial.npz', 'Initial'), ('growth-0.05.npz', 'First 5% D'),
             ('growth-0.2.npz', 'First 20% D'), ('growth-0.5.npz', 'First 50% D'), ('final.npz', 'Final observed')]
    for name, group in report['groups'].items():
        for outcome, selected in group['representatives'].items():
            if selected is None:
                continue
            fig, axes = plt.subplots(2, len(files), figsize=(10.5, 4.3), layout='constrained')
            resource_image = None
            for index, (filename, title) in enumerate(files):
                path = Path(selected['path']) / 'states' / filename
                if not path.exists():
                    for row in axes[:, index]:
                        row.set_axis_off()
                    axes[0, index].set_title(f'{title}\nNot reached', fontsize=9)
                    continue
                with np.load(path, allow_pickle=False) as state:
                    meta = json.loads(str(state['metadata_json']))
                    size = meta['L']
                    strategy = state['strategy'].reshape(size, size)
                    resource = state['resource'].reshape(size, size)
                axes[0, index].imshow(strategy, vmin=0, vmax=1, cmap=strategy_colors, interpolation='nearest')
                resource_image = axes[1, index].imshow(resource, vmin=0, vmax=1, cmap='viridis', interpolation='nearest')
                axes[0, index].set_title(f'{title}\nt={meta["local_time"]:,}', fontsize=9)
                for row in axes[:, index]:
                    row.set_xticks([])
                    row.set_yticks([])
            axes[0, 0].set_ylabel('Strategy\nD orange; C blue', fontsize=9)
            axes[1, 0].set_ylabel('Resource', fontsize=9)
            fig.colorbar(resource_image, ax=list(axes[1]), shrink=.7, label='Resource')
            fig.suptitle(f'{label(name)}; final outcome: {outcome}; smallest eligible seed={selected["seed"]}', fontsize=10)
            fig.supxlabel('Outcome-selected example; unequal time gaps; not a typical or causal nucleus path.', fontsize=9)
            fig.savefig(target / f'{name}-{outcome}.png', dpi=180)
            plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    plot_summary(report, args.output)
    plot_representatives(report, args.output)
