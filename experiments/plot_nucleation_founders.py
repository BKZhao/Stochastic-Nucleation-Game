"""Show founder risk benchmarks and the explicitly post-hoc early burden comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_founders(report: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout='constrained')
    colors = {'single': '#009E73', 'compact': '#0072B2', 'random': '#D55E00'}
    groups = {'single': report['single'], **report['comparison']}
    for index, name in [(0, 'single'), (1, 'compact'), (3, 'random')]:
        cell = groups[name]
        risk, (low, high) = cell['collapse_risk'], cell['wilson95']
        axes[0].errorbar(index, 100 * risk, yerr=np.array([[risk-low], [high-risk]]) * 100,
                        fmt='o', color=colors[name], capsize=5, ms=7)
        axes[0].text(index, high * 100 + 2, f"{cell['events']}/{cell['n']}", ha='center', fontsize=9)
    benchmark = report['benchmark']
    risk, (low, high) = benchmark['risk'], benchmark['transformed_wilson95']
    axes[0].errorbar(2, risk * 100, yerr=np.array([[risk-low], [high-risk]]) * 100,
                    fmt='D', mfc='white', color='0.4', capsize=5, ms=7)
    axes[0].text(2, high * 100 + 2, 'Derived', ha='center', fontsize=9, color='0.4')
    axes[0].set(xticks=range(4), xticklabels=['Single\nsite', 'Compact\n16', 'OR\nbenchmark', 'Random\n16'],
                ylabel='Confirmed depletion by 10,000 sweeps (%)', ylim=(0, 75),
                title='a  Risk and marginal 95% intervals')
    for name, label in [('random', 'Random 16'), ('compact', 'Compact 16'), ('single', 'Single site')]:
        rows = report['early_posthoc']['groups'][name]
        axes[1].plot([r['time'] for r in rows], [r['mean_defectors'] for r in rows],
                     color=colors[name], marker='o', ms=3, label=label)
    rows = report['early_posthoc']['additive_mean_benchmark']
    axes[1].plot([r['time'] for r in rows], [r['mean_defectors'] for r in rows],
                 color='0.4', ls='--', label='16 x single-site mean')
    axes[1].set(xlabel='Time after damage (sweeps)', ylabel='Mean total defector count',
                title='b  Early burden: post-hoc description', ylim=(0, None))
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle('L=48, b=1.16, diffusion=0.40, mutation=0; D/r0 interventions', fontsize=11)
    fig.supxlabel('OR: 1-(1-P1)^16. No formal model test. Early means include all 256 runs per group.', fontsize=9)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'pdf']:
        fig.savefig(output / f'founders.{suffix}', dpi=220)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, default=Path('figures/nucleation/founders'))
    args = parser.parse_args()
    plot_founders(json.loads(args.report.read_text()), args.output)
