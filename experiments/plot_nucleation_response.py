"""Time-dependent population response and coupled resource dynamics."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import pubstyle as ps
from nucleation_paper_inputs import OUTPUT, record_figure
from research_common import PROJECT_ROOT, file_hash

PANELS = [
    ('translation', 'resource translation',
     [('small', 'short shift', ps.AMBER, '-'),
      ('medium', 'long shift', ps.BLUE, '--')]),
    ('allocation', 'interface redistribution',
     [('interface_protect', 'low bias', ps.BLUE, '-'),
      ('interface_harm', 'high bias', ps.RED, '--')]),
]


def draw_curve(ax, time, mean, color, style, label):
    ax.plot(time, mean, color=color, ls=style, label=label, lw=1.4)


def main() -> None:
    pointer = json.loads((PROJECT_ROOT/'paper/response_inputs.json').read_text())
    path = PROJECT_ROOT/pointer['analysis']
    if file_hash(path) != pointer['sha256']:
        raise ValueError('response analysis changed')
    report = json.loads(path.read_text())
    ps.apply_pre()
    fig, axes = plt.subplots(2, 2, figsize=(5.9, 3.45), constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=.08, hspace=.14)
    for column, (source, title, entries) in enumerate(PANELS):
        group = report['groups'][source]
        for arm, label, color, style in entries:
            curve = group['curves'][arm]
            draw_curve(axes[0, column], group['time'][:-1], curve['drift_mean'],
                      color, style, label)
            draw_curve(axes[1, column], group['time'], curve['mean'],
                      color, style, label)
        axes[0, column].set(title=title, ylabel=r'$\Delta a(t)$ (MCS$^{-1}$)')
        axes[0, column].legend(loc='best', fontsize=9)
        axes[1, column].set(ylabel=r'$\Delta\langle n_D(t)\rangle$')
    for ax, letter in zip(axes.flat, 'abcd'):
        ax.set(xlabel=r'$t$ (MCS)', xlim=(0, 20), xticks=[0, 5, 10, 15, 20])
        ax.axhline(0, color=ps.MUTED, ls=':', lw=.8, zorder=0)
        ps.clean_axes(ax)
        ps.panel_label(ax, letter, x=0, y=1.04)
    for ax in axes[0]:
        ax.set_xlabel('')
        ax.tick_params(labelbottom=False)
    ps.save(fig, 'fig07_cumulative_response', OUTPUT)
    record_figure('fig07_cumulative_response', __file__, [path],
                  'Paired instantaneous and cumulative population differences against original. '
                  '48 configurations per sample, 16 continuations per arm; four translation directions '
                  'averaged within magnitude first. Curves show ensemble means without interval shading. '
                  'Uncertainty remains in the analysis; no smoothing or fitted time scales.')

    fig, axes = plt.subplots(2, 2, figsize=(5.9, 4.30), constrained_layout=True)
    for column, (source, title, entries) in enumerate(PANELS):
        group = report['groups'][source]
        ax = axes[0, column]
        for arm, label, color, style in [('original', 'original', ps.INK, ':'), *entries]:
            phase = group['phase'][arm]
            x, y = np.asarray(phase['cooperation']), np.asarray(phase['resource'])
            ax.plot(x, y, color=color, ls=style, lw=1.4, label=label)
            for index in [2, 10, 17]:
                ax.annotate('', xy=(x[index+1], y[index+1]), xytext=(x[index], y[index]),
                            arrowprops=dict(arrowstyle='->', color=color, lw=.9,
                                            shrinkA=0, shrinkB=0, mutation_scale=9))
            ax.plot(x[-1], y[-1], marker='o', ms=3, color=color)
        origin = group['phase']['original']
        ax.plot(origin['cooperation'][0], origin['resource'][0], marker='o', mfc='white',
                mec=ps.INK, ms=5, zorder=5)
        ax.set(xlabel=r'$\langle\rho(t)\rangle$', ylabel=r'$\langle\bar r(t)\rangle$', title=title)
        ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.01), ncol=3,
                  fontsize=8.5, handlelength=1.0, columnspacing=.65, handletextpad=.35)
        ax.set_title(title, pad=24)
        ax.ticklabel_format(useOffset=False, axis='both')
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax = axes[1, column]
        for arm, label, color, style in entries:
            curve = group['covariance_curves'][arm]
            draw_curve(ax, group['covariance_time'], 1e3*np.asarray(curve['mean']),
                      color, style, label)
        ax.axhline(0, color=ps.MUTED, ls=':', lw=.8)
        ax.set(xlabel=r'$t$ (MCS)', ylabel=r'$10^3\,\Delta\langle C_{sr}(t)\rangle$',
               xlim=(0, 20), xticks=[0, 5, 10, 15, 20])
    for ax, letter in zip(axes.flat, 'abcd'):
        ps.clean_axes(ax)
        ps.panel_label(ax, letter, x=0, y=1.35 if letter in 'ab' else 1.04)
    ps.save(fig, 'fig08_coupled_response', OUTPUT)
    record_figure('fig08_coupled_response', __file__, [path],
                  'Top: ensemble-mean paths over MCS 0..20, initial open circle, final filled circles, '
                  'arrows in increasing time. These are not a deterministic phase portrait or a probability '
                  'flow. Bottom: strategy-resource covariance differences over MCS 0..19, reconstructed '
                  'trajectory by trajectory from the exact resource balance and checked at saved initial '
                  'fields. Curves show ensemble means; paired uncertainty remains in the underlying analysis.')


if __name__ == '__main__':
    main()
