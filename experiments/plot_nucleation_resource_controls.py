"""Adapted resource field and fixed-strategy relaxation."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

import pubstyle as ps
from nucleation_interface import resource_equilibrium
from nucleation_paper_inputs import OUTPUT, record_figure
from research_common import PROJECT_ROOT, file_hash


def main() -> None:
    pointer = json.loads((PROJECT_ROOT/'paper/structure_inputs.json').read_text())
    path = PROJECT_ROOT/pointer['analysis']
    if file_hash(path) != pointer['sha256']:
        raise ValueError('structure analysis changed')
    report = json.loads(path.read_text())['resource']
    example = report['example']
    ps.apply_pre()
    fig, axes = plt.subplots(1, 2, figsize=(ps.WIDTH_DOUBLE, 2.85), constrained_layout=True)
    state_path = PROJECT_ROOT/example['path']
    with np.load(state_path, allow_pickle=False) as state:
        strategy, resource = state['strategy'], state['resource']
    equilibrium, steps = resource_equilibrium(strategy, resource, 48, .1, .5, .4)
    ax = axes[0]
    im = ax.imshow(equilibrium.reshape(48, 48), origin='lower', interpolation='nearest',
                   cmap=ps.SEQ, vmin=0, vmax=1, rasterized=True)
    contour = ax.contour(strategy.reshape(48, 48), levels=[.5], colors=[ps.INK], linewidths=.6)
    contour.set_path_effects([pe.Stroke(linewidth=1.2, foreground='white'), pe.Normal()])
    ax.plot([3, 15], [3, 3], color='white', lw=1.4)
    ax.set(xticks=[], yticks=[], title='fixed-strategy equilibrium')
    colorbar = fig.colorbar(im, ax=ax, fraction=.042, pad=.02, ticks=[0, .5, 1])
    colorbar.set_label(r'$r_{{\rm eq},i}$')
    ps.heatmap_frame(ax)
    ax = axes[1]
    time = np.arange(61)
    for arm, color, style, label in [('small', ps.AMBER, '-', r'$\mathbf{d}=(2,1)$'),
                                    ('medium', ps.BLUE, '--', r'$\mathbf{d}=(5,-17)$')]:
        values = np.array([row['arms'][arm]['relaxation'] for row in report['rows']])
        low, median, high = np.quantile(values, [.25, .5, .75], axis=0)
        ax.semilogy(time, median, color=color, ls=style, label=label)
    ax.semilogy(time, .9**time, color=ps.INK, ls=':', label=r'$0.9^t$ bound')
    ax.set(xlabel=r'$t$ (resource steps)',
           ylabel=r'$\|\Delta r_t\|_\infty/\|\Delta r_0\|_\infty$', xlim=(0, 60))
    ax.legend(fontsize=8, loc='lower left')
    ps.clean_axes(ax)
    for ax, letter in zip(axes, 'ab'):
        ps.panel_label(ax, letter, x=0, y=1.04)
    ps.save(fig, 'figS06_resource_controls', OUTPUT)
    record_figure('figS06_resource_controls', __file__,
                  [path, state_path, PROJECT_ROOT/'code/nucleation_interface.py'],
                  'Equilibrium field for the fixed strategy used in the alignment figure, solved to tolerance 1e-13. '
                  'Reference parameters g=0.1, delta=0.5, D=0.4; 12-site bar. '
                  'Fixed-strategy difference decays deterministically: median and '
                  'variation computed across 48 configurations; only the median is drawn. '
                  'The bound is not a fitted coupled-system relaxation time. '
                  'Mean-balance residuals are reported numerically, not as a redundant identity scatter.')


if __name__ == '__main__':
    main()
