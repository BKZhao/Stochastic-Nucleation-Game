"""Early-window interface diagnostics: establishment trajectories, marginality, drift calibration."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

COLORS = {'compact_1': '#009E73', 'compact_16': '#0072B2', 'random_16': '#D55E00',
          'compact_16_eq': '#56B4E9', 'random_16_eq': '#E69F00'}
LABELS = {'compact_1': 'Single (fresh)', 'compact_16': 'Compact 16 (fresh)',
          'random_16': 'Random 16 (fresh)', 'compact_16_eq': 'Compact 16 (equilibrium field)',
          'random_16_eq': 'Random 16 (equilibrium field)'}


def series(group, column, times):
    rows = group['trajectory']
    median, low, high = [], [], []
    for t in times:
        stat = rows[str(t)][column]
        median.append(stat['median'])
        low.append(stat['q25'])
        high.append(stat['q75'])
    return np.asarray(median), np.asarray(low), np.asarray(high)


def band(axis, x, median, low, high, color, label):
    axis.plot(x, median, color=color, lw=1.6, label=label)
    axis.fill_between(x, low, high, color=color, alpha=.18, lw=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('early', type=Path, help='analysis interface.json for the fresh-field campaign')
    parser.add_argument('--intervention', type=Path, default=None,
                        help='analysis interface.json for the equilibrium-field campaign')
    parser.add_argument('--output', type=Path, default=Path('figures/nucleation/early_interface'))
    args = parser.parse_args()
    early = json.loads(args.early.read_text())
    intervention = json.loads(args.intervention.read_text()) if args.intervention else None

    merged = dict(early['groups'])
    if intervention:
        for name, group in intervention['groups'].items():
            merged[name + '_eq'] = group
    times = [int(t) for t in merged[list(merged)[0]]['trajectory']]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6), layout='constrained')

    # a  Defector population
    for name, group in sorted(merged.items()):
        median, low, high = series(group, 'defectors', times)
        band(axes[0, 0], times, median, low, high, COLORS.get(name, '#888'), LABELS.get(name, name))
    axes[0, 0].set(xlabel='Sweep', ylabel='Defectors $N_D$ (median, IQR)',
                   title='a  Damage population', xscale='log')

    # b  Interface diagnostic
    for name, group in sorted(merged.items()):
        median, low, high = series(group, 'J', times)
        band(axes[0, 1], times, median, low, high, COLORS.get(name, '#888'), LABELS.get(name, name))
    axes[0, 1].axhline(0., color='k', lw=.8, alpha=.6)
    axes[0, 1].set(xlabel='Sweep', ylabel='$J$ (median, IQR)', ylim=(-4.2, 17),
                   title='b  Interface diagnostic self-stabilises at marginality', xscale='log')

    # c  Boundary cooperator resource
    for name, group in sorted(merged.items()):
        median, low, high = series(group, 'boundary_c_mean', times)
        band(axes[0, 2], times, median, low, high, COLORS.get(name, '#888'), LABELS.get(name, name))
    axes[0, 2].set(xlabel='Sweep', ylabel='Boundary cooperator resource',
                   title='c  Resource adaptation at the interface', xscale='log', ylim=(0.5, 1.02))

    # d  Drift calibration (pooled, binned)
    pooled = early['pooled_calibration']['binned']
    predictor = np.array([cell['J_pre_mean'] for cell in pooled])
    response = np.array([cell['delta_ND_mean'] for cell in pooled])
    count = np.array([cell['count'] for cell in pooled], dtype=float)
    size = 12 + 60 * (count / count.max())
    axes[1, 0].scatter(predictor, response, s=size, color='#0072B2', alpha=.85, edgecolor='none')
    grid = np.linspace(predictor.min(), predictor.max(), 32)
    slope = early['pooled_calibration']['ols']['slope']
    axes[1, 0].plot(grid, grid, color='#BBBBBB', lw=1, ls='--', label='frozen-state $\\Delta N_D=J$')
    axes[1, 0].plot(grid, slope * grid, color='#D55E00', lw=1.6,
                    label=f'fitted $\\Delta N_D={slope:.2f}\\,J$')
    axes[1, 0].axhline(0., color='k', lw=.8, alpha=.6)
    axes[1, 0].axvline(0., color='k', lw=.8, alpha=.6)
    axes[1, 0].set(xlabel='$J$ predictor (one resource substep ahead)',
                   ylabel='Realised $\\Delta N_D$ per sweep (bin mean)',
                   title='d  Per-sweep drift calibration, pooled')
    axes[1, 0].legend(frameon=False, fontsize=8)

    # e  Initial interface diagnostic vs first-sweep response
    xpos, labels = [], []
    for index, (name, group) in enumerate(sorted(merged.items())):
        initial = group['initial']
        color = COLORS.get(name, '#888')
        axes[1, 1].errorbar(index, initial['J_pre_t0']['median'],
                            yerr=[[initial['J_pre_t0']['median'] - initial['J_pre_t0']['q25']],
                                  [initial['J_pre_t0']['q75'] - initial['J_pre_t0']['median']]],
                            fmt='o', color=color, capsize=3, ms=7)
        axes[1, 2].errorbar(index, initial['delta_ND_sweep1']['mean'],
                            yerr=[[initial['delta_ND_sweep1']['mean'] - initial['delta_ND_sweep1']['q25']],
                                  [initial['delta_ND_sweep1']['q75'] - initial['delta_ND_sweep1']['mean']]],
                            fmt='o', color=color, capsize=3, ms=7)
        xpos.append(index)
        labels.append(LABELS.get(name, name))
    axes[1, 1].axhline(0., color='k', lw=.8, alpha=.6)
    axes[1, 1].set(xticks=xpos, xticklabels=[], ylabel='$J$ before first sweep',
                   title='e  Initial interface diagnostic')
    axes[1, 2].axhline(0., color='k', lw=.8, alpha=.6)
    axes[1, 2].set(xticks=xpos, xticklabels=labels, ylabel='Realised $\\Delta N_D$, sweep 1',
                   title='f  First-sweep response')
    for axis in (axes[1, 2],):
        axis.tick_params(axis='x', labelrotation=32, labelsize=8)
        for tick in axis.get_xticklabels():
            tick.set_horizontalalignment('right')

    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(args.output / f'interface.{suffix}', dpi=200)
    print(args.output / 'interface.png')


if __name__ == '__main__':
    main()
