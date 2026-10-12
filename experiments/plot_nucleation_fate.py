"""N06 fate continuation: per-mother probabilities, arm contrasts and state-variable prediction."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'compact_16': '#0072B2', 'random_16': '#D55E00'}


def jitter(axis, x, values, color):
    spread = 0.16
    rng = np.random.default_rng(0)
    for value in values:
        axis.plot(x + rng.uniform(-spread, spread), value, 'o', color=color,
                  alpha=.75, ms=6, mec='white', mew=.6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, default=Path('figures/nucleation/fate'))
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    groups = report['groups']
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), layout='constrained')

    # a  per-mother q_B on the original field
    for index, (name, color) in enumerate([('compact_16', COLORS['compact_16']),
                                           ('random_16', COLORS['random_16'])]):
        mothers = groups[name]
        values = [mother['arms']['original']['q_B'] for mother in mothers]
        jitter(axes[0, 0], index, values, color)
        axes[0, 0].plot([index - .3, index + .3], [np.mean(values)] * 2, color='k', lw=1.4)
        axes[0, 0].text(index, 1.02, f'mean {np.mean(values):.2f}', ha='center', fontsize=9)
    axes[0, 0].axhline(.5, color='#888', lw=.8, ls=':')
    axes[0, 0].set(xticks=[0, 1], xticklabels=['Compact-16\nsources', 'Random-16\nsources'],
                   ylabel='$q_B$ (40,000-sweep fate)', ylim=(0, 1.12),
                   title='a  Depletion probability per mother, original field')

    # b  arm contrasts with mother-level intervals
    labels, positions = [], []
    for index, (name, arm) in enumerate([('compact_16', 'adapted'), ('compact_16', 'shifted'),
                                         ('random_16', 'adapted'), ('random_16', 'shifted')]):
        summary = report['contrasts'][name]['summary'][arm]
        paired = summary['paired']
        rows = [row for row in report['contrasts'][name]['rows'] if row['arm'] == arm]
        values = [row['delta_q_B'] for row in rows]
        color = COLORS[name]
        jitter(axes[0, 1], index, values, color)
        mean = paired['mean']
        low, high = paired['bootstrap95']
        axes[0, 1].plot([index - .3, index + .3], [mean] * 2, color='k', lw=1.4)
        axes[0, 1].plot([index, index], [low, high], color='k', lw=1.2, alpha=.8)
        axes[0, 1].plot([index - .08, index + .08], [low] * 2, color='k', lw=1.2, alpha=.8)
        axes[0, 1].plot([index - .08, index + .08], [high] * 2, color='k', lw=1.2, alpha=.8)
        crossing = ' CI crosses 0' if low <= 0 <= high else ' CI excludes 0'
        axes[0, 1].text(index, high + .035, f'{mean:+.3f}\n{low:+.3f},{high:+.3f}',
                        ha='center', fontsize=7.5)
        labels.append(f"{'compact' if name.startswith('compact') else 'random'}/{arm}{crossing}")
        positions.append(index)
    axes[0, 1].axhline(0., color='k', lw=.8, alpha=.6)
    axes[0, 1].set(xticks=positions, xticklabels=labels, ylabel='$\\Delta q_B$ versus original arm',
                   ylim=(-.45, .45),
                   title='b  Resource-arm contrasts (mother bootstrap 95%)')
    axes[0, 1].tick_params(axis='x', labelrotation=18)
    for tick in axes[0, 1].get_xticklabels():
        tick.set_horizontalalignment('right')

    # c  q_B versus resource non-adaptation eta
    for name in ('compact_16', 'random_16'):
        mothers = groups[name]
        eta = [mother['descriptors']['eta_mean'] for mother in mothers]
        values = [mother['arms']['original']['q_B'] for mother in mothers]
        axes[1, 0].plot(eta, values, 'o', color=COLORS[name], ms=7, alpha=.85,
                        mec='white', mew=.6, label=f"{'compact' if name.startswith('compact') else 'random'} sources")
        correlation = report['state_prediction'][f'{name}/eta_mean']['spearman']
        axes[1, 0].text(.03, .06 + .12 * (name == 'compact_16'),
                        f"Spearman {correlation:+.2f} (n=15)", transform=axes[1, 0].transAxes,
                        color=COLORS[name], fontsize=9)
    axes[1, 0].axhline(.5, color='#888', lw=.8, ls=':')
    axes[1, 0].set(xlabel='Resource non-adaptation $\\eta=\\langle|r-r_{eq}|\\rangle$',
                   ylabel='$q_B$', xlim=(0, None), ylim=(0, 1.02),
                   title='c  Fate tracks the resource-adaptation lag')

    # d  outcome composition per group and arm
    arms = ['original', 'adapted', 'shifted']
    positions, labels = [], []
    index = 0
    for name in ('compact_16', 'random_16'):
        for arm in arms:
            rows = [mother['arms'][arm] for mother in groups[name] if arm in mother['arms']]
            if not rows:
                continue
            q_b = np.mean([row['q_B'] for row in rows])
            q_a = np.mean([row['q_A'] for row in rows])
            q_u = np.mean([row['q_U'] for row in rows])
            axes[1, 1].bar(index, q_b, color=COLORS[name])
            axes[1, 1].bar(index, q_a, bottom=q_b, color='#009E73')
            axes[1, 1].bar(index, q_u, bottom=q_b + q_a, color='#BBBBBB')
            positions.append(index)
            labels.append(f"{'compact' if name.startswith('compact') else 'random'}/{arm}")
            index += 1
    axes[1, 1].set(xticks=positions, xticklabels=labels, ylabel='Probability',
                   ylim=(0, 1), title='d  Fate composition by source and resource arm')
    axes[1, 1].tick_params(axis='x', labelrotation=18)
    for tick in axes[1, 1].get_xticklabels():
        tick.set_horizontalalignment('right')
    axes[1, 0].legend(frameon=False, fontsize=8, loc='upper left')

    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output / f'fate.{suffix}', dpi=200)
    print(args.output / 'fate.png')


if __name__ == '__main__':
    main()
