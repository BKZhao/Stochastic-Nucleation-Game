"""Matched-population resource fields and conditional cluster distributions."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap
import numpy as np

import pubstyle as ps
from nucleation_paper_inputs import OUTPUT, record_figure
from research_common import PROJECT_ROOT, file_hash

STYLES = [('compact_16', 'compact', ps.BLUE, '-'),
          ('random_16', 'dispersed', ps.RED, '--')]


def load_field(run: Path, name: str):
    path = run/'states'/name
    with np.load(path, allow_pickle=False) as state:
        meta = json.loads(str(state['metadata_json']))
        size = meta['L']
        center = meta['geometry_center']
        shift = (size//2-center[0], size//2-center[1])
        s = np.roll(state['strategy'].reshape(size, size), shift, (0, 1))
        r = np.roll(state['resource'].reshape(size, size), shift, (0, 1))
    return path, meta, s, r


def ecdf(ax, values, color, style, label):
    values = np.sort(values)
    ax.step(np.r_[values[0], values], np.r_[0, np.arange(1, len(values)+1)/len(values)],
            color=color, ls=style, where='post', lw=1.4, label=label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.analysis.read_text())
    pointer = json.loads((PROJECT_ROOT/'paper/structure_inputs.json').read_text())
    structure_path = PROJECT_ROOT/pointer['analysis']
    if file_hash(structure_path) != pointer['sha256']:
        raise ValueError('structure input changed')
    structure = json.loads(structure_path.read_text())['growth']['groups']
    ps.apply_pre()
    fig, axes = plt.subplots(2, 4, figsize=(ps.WIDTH_DOUBLE, 3.95),
                             constrained_layout=True)
    fields, paths = [], [args.analysis.resolve()]
    names = ['initial.npz', 'growth-0.05.npz', 'growth-0.2.npz', 'growth-0.8.npz']
    stages = ['initial', '5% defectors', '20% defectors', '80% defectors']
    for row, (key, label, color, style) in enumerate(STYLES):
        run = PROJECT_ROOT/report['groups'][key]['snapshot_run']
        for col, name in enumerate(names):
            path, meta, strategy, resource = load_field(run, name)
            paths.append(path)
            ax = axes[row, col]
            im = ax.imshow(resource, origin='lower', interpolation='nearest', cmap=ps.SEQ,
                           vmin=0, vmax=1, rasterized=True)
            contour = ax.contour(strategy, levels=[.5], colors=[ps.INK], linewidths=.55)
            contour.set_path_effects([pe.Stroke(linewidth=1.15, foreground='white'), pe.Normal()])
            ax.set(xticks=[], yticks=[])
            ax.set_title(stages[col]+chr(10)+f"$t={meta['local_time']}$", fontsize=8.5)
            if col == 0:
                ax.set_ylabel(label, fontsize=10)
            ax.plot([3, 15], [3, 3], color='white', lw=1.6,
                    path_effects=[pe.Stroke(linewidth=2.6, foreground=ps.INK), pe.Normal()])
            ps.heatmap_frame(ax)
            ax.text(.04, .96, '('+'abcdefgh'[row*4+col]+')', transform=ax.transAxes,
                    va='top', color='white', fontsize=10, fontweight='bold',
                    path_effects=[pe.Stroke(linewidth=1.3, foreground=ps.INK), pe.Normal()])
            fields.append(ax)
    cb = fig.colorbar(im, ax=fields, fraction=.022, pad=.015, ticks=[0, .5, 1])
    cb.set_label(r'$r_i$')
    ps.save(fig, 'fig02_growth_fields', OUTPUT)
    record_figure('fig02_growth_fields', __file__, paths,
                  'Resource maps and unchanged-coordinate strategy contours, selected confirmed-B examples. '
                  'Columns: initial and first 5%, 20%, 80% crossings; times are actual MCS, not common times. '
                  'All stages use one fixed translation per example and the same resource scale. '
                  '12-site bars. Outcome-conditioned illustrations; ensemble statistics remain separate.')

    fig, axes = plt.subplots(4, 5, figsize=(ps.WIDTH_DOUBLE, 6.4), constrained_layout=True)
    paths = [args.analysis.resolve()]
    for group, (key, label, color, style) in enumerate(STYLES):
        run = PROJECT_ROOT/report['groups'][key]['snapshot_run']
        for col, name in enumerate(['initial.npz', 'growth-0.05.npz', 'growth-0.2.npz',
                                    'growth-0.5.npz', 'growth-0.8.npz']):
            path, meta, s, r = load_field(run, name); paths.append(path)
            for offset, field in enumerate([s, r]):
                ax = axes[2*group+offset, col]
                im = ax.imshow(field, origin='lower', interpolation='nearest', vmin=0, vmax=1,
                               cmap=ListedColormap([ps.RED, '#f0f3f6']) if offset==0 else ps.SEQ,
                               rasterized=True)
                if offset:
                    ax.contour(s, levels=[.5], colors=[ps.INK], linewidths=.3)
                ax.set(xticks=[], yticks=[])
                if offset==0:
                    ax.set_title(f"t = {meta['local_time']}", fontsize=9)
                if col==0:
                    ax.set_ylabel(label+'\n'+('strategy' if offset==0 else 'resource'))
                ax.plot([3, 15], [3, 3], color='white', lw=1.2,
                        path_effects=[pe.Stroke(linewidth=2.2, foreground=ps.INK), pe.Normal()])
                ps.heatmap_frame(ax)
        ps.panel_label(axes[2*group, 0], 'ab'[group], x=-.15, y=1.04)
    cb=fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.022, pad=.012, ticks=[0,.5,1])
    cb.set_label(r'$r_i$')
    ps.save(fig, 'figS03_full_growth_fields', OUTPUT)
    record_figure('figS03_full_growth_fields', __file__, paths,
                  'Same outcome-conditioned examples as main growth figure. Columns: initial, '
                  'first 5%, 20%, 50%, 80%; strategy and resource shown separately. Not typical trajectories.')

    fig, axes=plt.subplots(2, 2, figsize=(ps.WIDTH_DOUBLE, 4.9), constrained_layout=True)
    for row, quantity in enumerate(['clusters', 'time']):
        for col, fraction in enumerate([.05, .2]):
            ax=axes[row,col]
            for key,label,color,style in STYLES:
                entries=structure[key][str(fraction)]['rows']
                values=[r[quantity]/(1000 if quantity=='time' else 1) for r in entries]
                ecdf(ax,values,color,style,f'{label} ({len(entries)})')
            ax.set(xlabel=r'cluster count $n_{\rm cl}$' if row==0 else r'first-crossing time ($10^3$ MCS)',
                   ylabel='cumulative fraction', ylim=(-.03,1.03), title=f'first {100*fraction:g}% crossing')
            if row==0:
                ax.legend(loc='lower right',fontsize=8)
            ps.clean_axes(ax);ps.panel_label(ax,'abcd'[2*row+col],x=0,y=1.04)
    ps.save(fig,'figS04_growth_distributions',OUTPUT)
    record_figure('figS04_growth_distributions',__file__,[structure_path],
                  'Unsmoothed ECDFs of cluster counts and first-passage times, conditional on reaching '
                  'each threshold by T=40000. Reacher counts are shown; not paired across geometries.')


if __name__ == '__main__':
    main()
