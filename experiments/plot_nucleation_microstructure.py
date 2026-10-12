"""Physical interface figures from checked motifs and resource-layer observables."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patheffects as pe
import numpy as np

import pubstyle as ps
from analyze_nucleation_microstructure import load_state, motif_bias
from nucleation_interface import resource_substep
from nucleation_paper_inputs import OUTPUT, record_figure
from research_common import PROJECT_ROOT, file_hash


COLORS = {'original': ps.INK, 'small': ps.AMBER, 'medium': ps.BLUE}
LABELS = {'original': 'original', 'small': 'short shift', 'medium': 'long shift'}
STYLES = {'original': '-', 'small': '--', 'medium': '-.'}


def frame_field(ax: plt.Axes, strategy: np.ndarray) -> None:
    contour = ax.contour(strategy, levels=[.5], colors=[ps.INK], linewidths=.6)
    contour.set_path_effects([pe.Stroke(linewidth=1.15, foreground='white'), pe.Normal()])
    ax.plot([3, 15], [3, 3], color=ps.INK, lw=2)
    ax.text(9, 4.5, '12', ha='center', va='bottom', fontsize=8)
    ax.set(xticks=[], yticks=[])
    ps.heatmap_frame(ax)


def initial_figure(report: dict, analysis_path: Path) -> None:
    groups = report['initial']['groups']
    fig, axes = plt.subplots(2, 3, figsize=(ps.WIDTH_DOUBLE, 4.65), constrained_layout=True)
    matrices = []
    for col, (key, label) in enumerate((('compact_16', 'compact'), ('random_16', 'dispersed'))):
        row = groups[key]
        s, r = load_state(PROJECT_ROOT/row['example'])
        ax = axes[0, col]
        ax.imshow(s, origin='lower', cmap=ListedColormap([ps.RED, '#edf3f9']), vmin=0, vmax=1,
                  interpolation='nearest', rasterized=True)
        frame_field(ax, s)
        ax.set_title(label, fontsize=10)
        ax.text(.97, .04, r'$m=16$', transform=ax.transAxes, ha='right', fontsize=9)
        histogram = np.asarray(row['mean_motifs'])
        matrices.append(histogram)
        ax = axes[1, col]
        fraction = histogram/histogram.sum()
        im = ax.imshow(fraction, origin='lower', cmap=ps.SEQ, vmin=0, vmax=1,
                       extent=(.5, 4.5, -.5, 3.5), interpolation='nearest')
        for a, c in np.argwhere(fraction > 0):
            ax.text(c+1, a, f'{100*fraction[a,c]:.1f}', ha='center', va='center',
                    color='white' if fraction[a,c] > .55 else ps.INK, fontsize=8)
        ax.set(xlabel=r'$k_C(D)$', ylabel=r'$k_C(C)$', xticks=[1, 2, 3, 4], yticks=[0, 1, 2, 3])
        ax.text(.03, .06, r'$\langle P\rangle=$'+f'{row["mean_edges"]:.1f}', transform=ax.transAxes,
                fontsize=8, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .85, 'pad': 1.5})
        ps.heatmap_frame(ax)
    fig.colorbar(im, ax=axes[1, :2], shrink=.88, pad=.015, fraction=.028,
                 ticks=[0, .5, 1], label=r'$\langle H_{ac}\rangle/\langle P\rangle$')
    for ax, varying in ((axes[0, 2], 'diffusion'), (axes[1, 2], 'payoff')):
        domain = np.linspace(.2, .6, 161) if varying == 'diffusion' else np.linspace(1.12, 1.2, 161)
        for histogram, color, label in zip(matrices, [ps.BLUE, ps.RED], ['compact', 'dispersed']):
            values = [motif_bias(histogram, 1.16 if varying == 'diffusion' else x,
                                 x if varying == 'diffusion' else .4)/16 for x in domain]
            ax.plot(domain, values, color=color, label=label)
            ax.plot(.4 if varying == 'diffusion' else 1.16, motif_bias(histogram, 1.16, .4)/16,
                    'o', color=color, ms=4)
        b = 1.16 if varying == 'diffusion' else domain
        d = domain if varying == 'diffusion' else .4
        ax.plot(domain, np.tanh((4*b-3*(1+.5*(1-d/2)))/.2), color=ps.INK2, ls=':', lw=1.1,
                label='isolated defector')
        ax.axhline(0, color=ps.MUTED, lw=.6)
        ax.set(xlabel=r'$D$' if varying == 'diffusion' else r'$b$',
               ylabel=r'$\langle J_{\rm pre}(0)\rangle/m$', ylim=(-.4, 1.08), xlim=(domain[0], domain[-1]))
        ax.set_xticks([.2, .4, .6] if varying == 'diffusion' else [1.12, 1.16, 1.2])
        ps.clean_axes(ax)
    axes[0, 2].legend(loc='center right', fontsize=7.5)
    for ax, letter in zip(axes.flat, 'abcdef'):
        ps.panel_label(ax, letter, x=0, y=1.04)
    ps.save(fig, 'fig09_initial_interfaces', OUTPUT)
    record_figure('fig09_initial_interfaces', __file__,
                  [analysis_path, PROJECT_ROOT/'experiments/analyze_nucleation_microstructure.py',
                   PROJECT_ROOT/'code/nucleation_interface.py']+[PROJECT_ROOT/g['example'] for g in groups.values()],
                  'All 64 preparations per geometry. Illustrations: first lexical archived path, independent of outcomes. '
                  'Initial strategy maps: red defectors, pale-blue cooperators. '
                  'Motif frequencies are pooled edges, labels in percent. Curves evaluate the exact frozen initial-state formula, '
                  'not new collapse probabilities or stochastic simulations; dots mark reference parameters. Scale bars: 12 sites.')


def layer_figure(report: dict, analysis_path: Path) -> None:
    layers = report['layers']
    fig = plt.figure(figsize=(ps.WIDTH_DOUBLE, 6.4))
    side = .36
    height = side*ps.WIDTH_DOUBLE/6.4
    axes = np.array([fig.add_axes([x, y, side, height])
                     for y in (.565, .08) for x in (.08, .59)]).reshape(2, 2)
    s, r = load_state(PROJECT_ROOT/layers['example'])
    original = resource_substep(s.ravel(), r.ravel(), len(s), .1, .5, .4).reshape(s.shape)
    shifted = np.roll(r, (5, -17), (0, 1))
    displaced = resource_substep(s.ravel(), shifted.ravel(), len(s), .1, .5, .4).reshape(s.shape)
    ax = axes[0, 0]
    maximum = max(abs(displaced-original).max(), 1e-12)
    im = ax.imshow(displaced-original, origin='lower', cmap=ps.DIV, vmin=-maximum, vmax=maximum,
                   interpolation='nearest', rasterized=True)
    frame_field(ax, s)
    fig.colorbar(im, cax=fig.add_axes([.45, .565, .014, height]), label=r'$\Delta r_i^{+}$')
    ax.set_title('resource redistribution', fontsize=9)
    bins = np.asarray(layers['bins'])
    distance = bins-.5*np.sign(bins)
    for ax, moment, title in ((axes[0, 1], 'pre', 'before resource update'),
                              (axes[1, 0], 'post', 'after resource update')):
        for arm in COLORS:
            ax.plot(distance, layers['arms'][arm][moment], color=COLORS[arm], ls=STYLES[arm],
                    marker='o', ms=2.5, label=LABELS[arm])
        ax.axvline(0, color=ps.MUTED, lw=.7, ls=':')
        ax.text(.06, .07, 'D', transform=ax.transAxes, color=ps.RED)
        ax.text(.92, .07, 'C', transform=ax.transAxes, color=ps.BLUE)
        ax.set(xlabel=r'$\ell$ (lattice sites)', ylabel=r'$\langle r\rangle_{\ell}$' if moment == 'pre' else r'$\langle r^{+}\rangle_{\ell}$',
               xlim=(-4, 12), ylim=(0, 1.03), title=title)
        ax.set_title(title, fontsize=9)
        ps.clean_axes(ax)
    axes[0, 1].legend(loc='lower right', bbox_to_anchor=(1, .12), fontsize=8)
    ax = axes[1, 1]
    for arm in COLORS:
        gaps = np.asarray(layers['arms'][arm]['gaps'])
        ax.step(gaps, np.arange(1, len(gaps)+1)/len(gaps), where='post', color=COLORS[arm],
                ls=STYLES[arm], label=LABELS[arm])
    ax.axvline(0, color=ps.MUTED, lw=.7, ls=':')
    ax.set(xlabel=r'$\Pi_D-\Pi_C$', ylabel='interface-edge fraction', ylim=(0, 1.03))
    ps.clean_axes(ax)
    for ax, letter in zip(axes.flat, 'abcd'):
        ax.set_box_aspect(1)
        ps.panel_label(ax, letter, x=0, y=1.04)
    ps.save(fig, 'fig10_resource_layers', OUTPUT)
    record_figure('fig10_resource_layers', __file__,
                  [analysis_path, PROJECT_ROOT/'experiments/analyze_nucleation_microstructure.py',
                   PROJECT_ROOT/'code/nucleation_interface.py', PROJECT_ROOT/layers['example']],
                  'All 48 saved first-5%-crossing configurations used in the direction ensemble. '
                  'Short/long labels denote vectors (2,1) and (5,-17), with no angular pooling. Four equal square axes. '
                  'Periodic nearest-opposite-strategy graph distance minus half a site, negative in D, positive in C. '
                  'Profiles pool sites per shell; full bin counts are in the analysis. Display range -4 to 12 sites. '
                  'CDF pools every undirected C-D edge after one resource update. '
                  'Map is medium-minus-original for first lexical path; no outcome selection. Scale bar: 12 sites.')


def main() -> None:
    pointer = json.loads((PROJECT_ROOT/'paper/microstructure_inputs.json').read_text())
    path = PROJECT_ROOT/pointer['analysis']
    if file_hash(path) != pointer['sha256']:
        raise ValueError('microstructure analysis changed')
    report = json.loads(path.read_text())
    ps.apply_pre()
    initial_figure(report, path)
    layer_figure(report, path)


if __name__ == '__main__':
    main()
