"""Cell-resolved finite-time phase maps; contour guides disclose interpolation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

import pubstyle as ps
from nucleation_paper_inputs import OUTPUT, record_figure
from research_common import PROJECT_ROOT, file_hash, write_json


def matrix(rows, b_values, d_values, field):
    lookup = {(r['b'], r['diffusion']): r for r in rows}
    return np.asarray([[field(lookup[b, d]) for b in b_values] for d in d_values])


def map_panel(axis, b, diffusion, values, title, *, cmap=ps.SEQ_A, vmin=0, vmax=1):
    plotted = axis.pcolormesh(b, diffusion, values, shading='nearest',
                             rasterized=True, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.plot([1.16], [.4], '+', color='white', ms=6, mew=1.2,
              path_effects=[pe.Stroke(linewidth=2.2, foreground=ps.INK), pe.Normal()])
    axis.set(xlabel=r'payoff parameter $b$', ylabel=r'diffusion $D$',
             xticks=[1.12, 1.16, 1.20], yticks=[.2, .4, .6])
    axis.set_box_aspect(.95)
    axis.set_title(title, fontsize=9, pad=5)
    ps.heatmap_frame(axis)
    return plotted

def depletion_section(axis, report, diffusion=.4):
    """Compare geometries at one measured diffusion, at the recorded parameter values."""
    for shape, label, color, style in (
            ('compact', 'compact', ps.BLUE, '-'),
            ('random', 'dispersed', ps.RED, '--')):
        cells = sorted((r for r in report['cells'] if r['shape'] == shape
                        and abs(r['diffusion']-diffusion) < 1e-10),
                       key=lambda r: r['b'])
        if len(cells) != len(report['axes']['model.temptation']):
            raise ValueError(f'incomplete depletion section: {shape}, D={diffusion}')
        b = [r['b'] for r in cells]
        axis.plot(b, [r['probability']['B'] for r in cells],
                  color=color, ls=style, lw=1.4, label=label)
    axis.axvline(1.16, color=ps.MUTED, lw=.7, zorder=0)
    axis.set(xlim=(min(b), max(b)), ylim=(-.04, 1.04),
             xlabel=r'payoff parameter $b$', ylabel=r'$P_B(T)$',
             xticks=[1.12, 1.16, 1.20], yticks=[0, .5, 1])
    axis.set_title(r'$D=0.4$', fontsize=9, pad=5)
    axis.set_box_aspect(.95)
    axis.legend(loc='upper left', fontsize=9)
    ps.clean_axes(axis)


def phase_figures(report, analysis):
    ps.apply_pre()
    plt.rcParams.update({'hatch.linewidth': .35})
    b = report['axes']['model.temptation']
    diffusion = report['axes']['model.diffusion']
    shapes = {shape: [r for r in report['cells'] if r['shape'] == shape]
              for shape in ('compact', 'random')}
    fig, axes = plt.subplots(2, 2, figsize=(5.9, 5.45),
                             constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=.04, h_pad=.05, wspace=.055, hspace=.055)
    for axis, shape, letter in zip(axes[0], shapes, 'ab'):
        values = matrix(shapes[shape], b, diffusion, lambda row: row['probability']['B'])
        im = map_panel(axis, b, diffusion, values,
                       'compact' if shape == 'compact' else 'dispersed')
        if values.min() < .5 < values.max():
            axis.contour(b, diffusion, values, levels=[.5], colors=['white'], linewidths=.9)
        ps.add_colorbar(fig, im, axis, r'$P_B(T)$', ticks=[0, .5, 1], fraction=.04, pad=.025)
        ps.panel_label(axis, letter, x=0, y=1.06)
    differences = {}
    for outcome in ('A', 'B', 'U'):
        differences[outcome] = (
            matrix(shapes['random'], b, diffusion, lambda row: row['probability'][outcome])
            - matrix(shapes['compact'], b, diffusion, lambda row: row['probability'][outcome]))
    if not np.allclose(sum(differences.values()), 0, atol=1e-14, rtol=0):
        raise ValueError('outcome contrasts must sum to zero cellwise')
    im = map_panel(axes[1, 0], b, diffusion, differences['B'], 'geometry contrast',
                   cmap=ps.DIV, vmin=-1, vmax=1)
    ps.add_colorbar(fig, im, axes[1, 0], r'$\Delta P_B(T)$',
                    ticks=[-1, 0, 1], fraction=.04, pad=.025)
    ps.panel_label(axes[1, 0], 'c', x=0, y=1.06)
    depletion_section(axes[1, 1], report)
    ps.panel_label(axes[1, 1], 'd', x=0, y=1.06)
    ps.save(fig, 'fig01_phase', OUTPUT)
    selection = (f"Complete {len(b)} by {len(diffusion)} measured grid. "
                 "Four panels: two collapse-probability maps, dispersed-minus-compact contrast, "
                 "and D=0.4 section. 64 paths per cell, T=10000, L=48, m=16. "
                 "Colors are cell estimates; white half-probability contours are interpolated guides. "
                 "No interval artists, smoothing, extrapolation or significance masks. "
                 "All three outcome contrasts were checked to sum to zero.")
    record_figure('fig01_phase', __file__, [analysis], selection)
    write_json(OUTPUT/'fig01_phase.statistics.json', {
        'source_sha256': file_hash(analysis),
        'sampled_domain': {'b': [min(b), max(b)], 'D': [min(diffusion), max(diffusion)]},
        'probability_limits': [0, 1], 'contrast_limits': [-1, 1],
        'difference_definition': 'P_B(dispersed) - P_B(compact)',
        'contrasts': report['contrasts'],
        'reference': next(r for r in report['contrasts']
                          if abs(r['b']-1.16) < 1e-10 and abs(r['diffusion']-.4) < 1e-10),
    })

    fig, axes = plt.subplots(2, 2, figsize=(5.9, 5.25),
                             constrained_layout=True)
    for row_index, outcome in enumerate(('A', 'U')):
        for col_index, shape in enumerate(shapes):
            axis = axes[row_index, col_index]
            values = matrix(shapes[shape], b, diffusion,
                            lambda r: r['probability'][outcome])
            im = map_panel(axis, b, diffusion, values,
                           ('compact' if shape == 'compact' else 'dispersed')+': '+('recovery' if outcome == 'A' else 'unresolved'))
            ps.panel_label(axis, 'abcd'[2*row_index+col_index], x=0, y=1.07)
        ps.add_colorbar(fig, im, axes[row_index, :].tolist(), '$P_'+outcome+'(T)$',
                        ticks=[0, .5, 1], fraction=.03, pad=.025)
    ps.save(fig, 'figS01_recovery_unresolved', OUTPUT)
    record_figure('figS01_recovery_unresolved', __file__, [analysis],
                  'Complete sampled grid; 64 trajectories per cell, T=10000, L=48, m=16. '
                  'Cellwise A/U probabilities on [0,1], no interpolation or extrapolation. '
                  'Cross: b=1.16, D=0.4. P_A+P_B+P_U=1 cellwise.')
    write_json(PROJECT_ROOT/'paper/phase_inputs.json',
               {'analysis': str(analysis.relative_to(PROJECT_ROOT)),
                'sha256': file_hash(analysis), 'b_points': len(b),
                'diffusion_points': len(diffusion),
                'runs': sum(r['n'] for r in report['cells'])})



def parameter_sections(report, analysis):
    """Nine physically paired sections: diffusion regimes by competing outcomes."""
    diffusion_values = [.2, .4, .6]
    outcomes = ['A', 'B', 'U']
    fig, axes = plt.subplots(3, 3, figsize=(6.3, 5.9),
                             sharex=True, sharey=True, constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=.065, w_pad=.045, hspace=.10, wspace=.035)
    for row, outcome in enumerate(outcomes):
        for column, diffusion in enumerate(diffusion_values):
            axis = axes[row, column]
            for shape, label, color, style in [
                    ('compact', 'compact', ps.BLUE, '-'),
                    ('random', 'dispersed', ps.RED, '--')]:
                cells = sorted((item for item in report['cells'] if item['shape']==shape
                                and abs(item['diffusion']-diffusion)<1e-10), key=lambda item:item['b'])
                if len(cells) != len(report['axes']['model.temptation']):
                    raise ValueError('incomplete measured section')
                payoff = [item['b'] for item in cells]
                axis.plot(payoff, [item['probability'][outcome] for item in cells],
                          color=color, ls=style, lw=1.25, label=label)
            axis.set(xlim=(1.12, 1.20), ylim=(-.04, 1.04),
                     xticks=[1.12, 1.16, 1.20], yticks=[0, .5, 1])
            if column == 0:
                axis.set_ylabel('$P_'+outcome+'(T)$')
            if row == 0:
                axis.set_title(f'$D={diffusion:.1f}$')
            if row == 2:
                axis.set_xlabel(r'$b$')
            axis.axhline(0, color=ps.MUTED, lw=.45, ls=':', zorder=0)
            axis.axhline(1, color=ps.MUTED, lw=.45, ls=':', zorder=0)
            ps.clean_axes(axis)
            ps.panel_label(axis, 'abcdefghi'[3*row+column], x=0, y=1.04)
    axes[0, 0].legend(loc='lower left', fontsize=9)
    ps.save(fig, 'fig05_phase_sections', OUTPUT)
    record_figure('fig05_phase_sections', __file__, [analysis],
                  'Nine panels: recovery, collapse and unresolved outcomes as rows; D=0.2,0.4,0.6 as columns. '
                  '64 trajectories at every sampled payoff value, T=10000. Only two geometry curves per panel. '
                  'All 41 payoff values retained as sampled probabilities, no smoothing. '
                  'Axis margins expose P=0 and P=1 curves; all probabilities remain in [0,1].')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    analysis = args.analysis.resolve()
    phase_figures(json.loads(analysis.read_text()), analysis)
    parameter_sections(json.loads(analysis.read_text()), analysis)
    print(OUTPUT)


if __name__ == '__main__':
    main()
