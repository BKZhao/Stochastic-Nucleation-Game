"""Physical dependence on horizon, system size, and the collapse threshold."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pubstyle as ps
from nucleation_paper_inputs import OUTPUT, REPORTS, load_report, record_figure


def response(ax, x, rows, color, label=None, style='-'):
    means = np.array([row['mean'] for row in rows])
    bounds = np.array([row['interval'] for row in rows])
    ax.plot(x, means, color=color, label=label, lw=1.1, ls=style, marker='o', ms=4)
    if len(x) > 1:
        ax.fill_between(x, bounds[:, 0], bounds[:, 1], color=color, alpha=.13, lw=0)
    else:
        ax.errorbar(x, means, yerr=[means-bounds[:, 0], bounds[:, 1]-means],
                    color=color, ls='none', capsize=3)


def main() -> None:
    ps.apply_pre()
    r3, robust = load_report('r3'), load_report('robustness')
    fig, axes = plt.subplots(1, 3, figsize=(ps.WIDTH_DOUBLE, 2.65), constrained_layout=True)
    xs = [20, 40, 80]
    response(axes[0], xs, [r3['window']['windows'][f'H{x*1000}']['medium_minus_small'] for x in xs],
             ps.BLUE)
    axes[0].set(xlabel=r'$H$ ($10^3$ sweeps)', xticks=xs, title='observation time', ylim=(-.19, .07))
    cells = {row['name']: row for row in robust['cells']}
    response(axes[1], [64, 96], [cells[f'L{size}_nd110']['medium_minus_small'] for size in [64, 96]],
             ps.BLUE, r'$n_D\simeq110$')
    response(axes[1], [64], [cells['L64_5pct']['medium_minus_small']], ps.AMBER, '5% fraction')
    axes[1].set(xlabel=r'$L$', xticks=[64, 96], xlim=(59, 101), title='system size', ylim=(-.19, .07))
    axes[1].legend(fontsize=8, loc='lower right')
    for persistence, color, style in [(100, ps.BLUE, '-'), (500, ps.RED, '--')]:
        response(axes[2], [.15, .25],
                 [r3['endpoint'][f'endpoint_t{threshold}_c{persistence}']['medium_minus_small']
                  for threshold in [15, 25]], color, f'{persistence} sweeps', style)
    axes[2].set(xlabel=r'$\rho_*=r_*$', xticks=[.15, .25], xlim=(.13, .27),
                title='collapse threshold', ylim=(-.39, .13))
    axes[2].legend(fontsize=8, loc='lower right')
    axes[0].set_ylabel(r'$q_B^{\rm medium}-q_B^{\rm small}$')
    for ax, letter in zip(axes, 'abc'):
        ax.axhline(0, color=ps.MUTED, lw=.8, ls=':')
        ps.clean_axes(ax)
        ps.panel_label(ax, letter, x=0, y=1.04)
    ps.save(fig, 'figS02_robustness', OUTPUT)
    record_figure('figS02_robustness', __file__, [REPORTS['r3'], REPORTS['robustness']],
                  'Numeric physical coordinates throughout. Pointwise paired 95% intervals; '
                  'segments between measured values guide the eye, with no fitted size or threshold law. '
                  'Only horizon values share a continuation sample. Saturated and ineligible cases '
                  'are reported in prose, never displayed as precise zero effects.')


if __name__ == '__main__':
    main()
