"""Spatial resource alignment and its translation invariants."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

import pubstyle as ps
from nucleation_paper_inputs import OUTPUT, record_figure
from research_common import PROJECT_ROOT, file_hash


def main() -> None:
    ps.apply_pre()
    pointer = json.loads((PROJECT_ROOT/'paper/structure_inputs.json').read_text())
    structure_path = PROJECT_ROOT/pointer['analysis']
    if file_hash(structure_path) != pointer['sha256']:
        raise ValueError('structure analysis changed')
    example = json.loads(structure_path.read_text())['resource']['example']
    state_path = PROJECT_ROOT/example['path']
    with np.load(state_path, allow_pickle=False) as state:
        strategy = state['strategy'].reshape(48, 48)
        resource = state['resource'].reshape(48, 48)
    fig = plt.figure(figsize=(ps.WIDTH_DOUBLE, 6.3))
    side = .365
    height = side*ps.WIDTH_DOUBLE/6.3
    axes = [fig.add_axes([x, y, side, height]) for y in (.555, .075) for x in (.075, .545)]
    for ax, shift, title, letter in zip(axes[:2], [(0, 0), (5, -17)],
                                         ['original field', 'translated field'], 'ab'):
        im = ax.imshow(np.roll(resource, shift, (0, 1)), origin='lower',
                       interpolation='nearest', cmap=ps.SEQ, vmin=0, vmax=1, rasterized=True)
        contour = ax.contour(strategy, levels=[.5], colors=[ps.INK], linewidths=.7)
        contour.set_path_effects([pe.Stroke(linewidth=1.35, foreground='white'), pe.Normal()])
        ax.set(xticks=[], yticks=[], title=title)
        ax.plot([3, 15], [3, 3], color='white', lw=1.6)
        ps.heatmap_frame(ax)
        ps.panel_label(ax, letter, x=0, y=1.035)
    fig.colorbar(im, cax=fig.add_axes([.929, .555, .014, height]), ticks=[0, .5, 1]).set_label(r'$r_i$')
    limit = max(np.max(np.abs(example[key])) for key in ['cross_covariance', 'autocorrelation'])
    for ax, field, title, letter in zip(axes[2:], ['cross_covariance', 'autocorrelation'],
             ['strategy–resource alignment', 'resource autocorrelation'], 'cd'):
        im = ax.pcolormesh(example['displacements'], example['displacements'], example[field],
                          cmap=ps.DIV, vmin=-limit, vmax=limit, shading='nearest', rasterized=True)
        if field == 'cross_covariance':
            for (d1, d2), color, marker in [((0, 0), 'white', 'o'), ((5, -17), ps.INK, 's')]:
                ax.plot(d2, d1, marker=marker, color=color, ms=5, mec=ps.INK, mew=.8, ls='none')
        ax.set(xlabel=r'$d_x$ (sites)', ylabel=r'$d_y$ (sites)', aspect='equal',
               xticks=[-20, 0, 20], yticks=[-20, 0, 20], title=title)
        ps.heatmap_frame(ax)
        ps.panel_label(ax, letter, x=0, y=1.045)
    fig.colorbar(im, cax=fig.add_axes([.929, .075, .014, height])).set_label('covariance')
    ps.save(fig, 'fig02_interventions', OUTPUT)
    record_figure('fig02_interventions', __file__, [structure_path, state_path],
                  'Four equal square panels. One fixed strategy configuration; original and (5,-17)-translated '
                  'resources share the same resource color scale and 12-site bars. Full periodic cross-covariance '
                  'and resource autocorrelation share a signed scale. Circle/square identify the displayed fields. '
                  'Structural maps are not maps of collapse probability. No displacement-risk inset.')


if __name__ == '__main__':
    main()
