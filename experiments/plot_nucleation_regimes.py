"""Recovery, collapse and unresolved regimes with all-trajectory collapse times."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.patches import Patch
import numpy as np

import pubstyle as ps
from nucleation_paper_inputs import OUTPUT, record_figure
from plot_nucleation_phase import matrix
from research_common import PROJECT_ROOT, file_hash, write_json

REGIMES = {
    'A': ('recovery', ps.BLUE),
    'B': ('collapse', ps.RED),
    'U': ('unresolved dynamics', ps.AMBER),
    None: ('no resolved majority', '#d6d9de'),
}


def format_panel(axis, title: str, letter: str) -> None:
    axis.set(xlabel=r'payoff parameter $b$', ylabel=r'diffusion $D$',
             xticks=[1.12, 1.16, 1.20], yticks=[.2, .4, .6])
    axis.set_box_aspect(.88)
    axis.set_title(title, fontsize=9, pad=5)
    ps.heatmap_frame(axis)
    ps.panel_label(axis, letter, x=0, y=1.055)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, required=True)
    args = parser.parse_args()
    analysis = args.analysis.resolve()
    report = json.loads(analysis.read_text())
    phase_path = PROJECT_ROOT/report['identity']['phase_analysis']
    if file_hash(phase_path) != report['identity']['phase_sha256']:
        raise ValueError('regime and probability maps refer to different phase data')
    ps.apply_pre()
    b = report['axes']['model.temptation']
    diffusion = report['axes']['model.diffusion']
    fig = plt.figure(figsize=(5.9, 5.3), layout='constrained')
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, .035))
    axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(2)]
                       for row in range(2)])
    codes = {key: i for i, key in enumerate(REGIMES)}
    colors = ListedColormap([color for _, color in REGIMES.values()])
    norm = BoundaryNorm(np.arange(-.5, 4.5), colors.N)
    time_colors = ps.SEQ_A.copy()
    time_colors.set_bad('#eeeeee')
    time_norm = LogNorm(vmin=report['protocol']['collapse']['confirm_sweeps'],
                        vmax=report['horizon'])
    statistics = {}
    for col, (shape, label) in enumerate((('compact', 'compact'),
                                          ('random', 'dispersed'))):
        cells = [c for c in report['cells'] if c['shape'] == shape]
        values = matrix(cells, b, diffusion, lambda c: codes[c['resolved_majority']])
        axes[0, col].pcolormesh(b, diffusion, values, shading='nearest',
                               cmap=colors, norm=norm, rasterized=True)
        format_panel(axes[0, col], label, 'ab'[col])
        half_times = matrix(cells, b, diffusion,
                            lambda c: np.nan if c['t_half_B'] is None else c['t_half_B'])
        im = axes[1, col].pcolormesh(b, diffusion, np.ma.masked_invalid(half_times),
                                    shading='nearest', cmap=time_colors,
                                    norm=time_norm, rasterized=True)
        format_panel(axes[1, col], 'collapse half-time', 'cd'[col])
        statistics[shape] = {
            'regime_cell_counts': {str(key): sum(c['resolved_majority'] == key for c in cells)
                                   for key in REGIMES},
            'half_time_measured_cells': int(np.isfinite(half_times).sum()),
            'half_time_not_reached_cells': int(np.isnan(half_times).sum()),
            'measured_half_time_range': [float(np.nanmin(half_times)),
                                         float(np.nanmax(half_times))],
        }
    fig.legend([Patch(facecolor=color) for _, color in REGIMES.values()],
               [label for label, _ in REGIMES.values()],
               loc='outside upper center', ncol=2, fontsize=9,
               frameon=False, columnspacing=1.1, handlelength=1.3)
    color_axis = fig.add_subplot(grid[1, 2])
    colorbar = fig.colorbar(im, cax=color_axis, ticks=[250, 1000, 3000, 10000])
    colorbar.set_ticklabels(['250', r'$10^3$', r'$3\times10^3$', r'$10^4$'])
    colorbar.set_label(r'$t^{B}_{1/2}$ (MCS)', fontsize=9)
    colorbar.outline.set_edgecolor(ps.MUTED)
    colorbar.outline.set_linewidth(.6)
    color_axis.tick_params(labelsize=9, width=.7, length=2.4)
    color_axis.set_box_aspect(26)
    fig.legend([Patch(facecolor='#eeeeee', edgecolor=ps.FAINT)],
               [r'(c,d): 50% collapse not reached by $T=10\,000$'],
               loc='outside lower center', frameon=False, fontsize=9)
    ps.save(fig, 'fig00_dynamical_regimes', OUTPUT)
    record_figure('fig00_dynamical_regimes', __file__, [
        analysis, phase_path, PROJECT_ROOT/'experiments/analyze_nucleation_regimes.py',
        PROJECT_ROOT/'code/nucleation_survival.py',
        PROJECT_ROOT/'experiments/plot_nucleation_phase.py',
    ], 'Same 41x41 grid and all 215168 trajectories as the probability maps. '
       'Top: A/B/U where the pointwise Wilson 95% lower limit is above 1/2; '
       'gray means no resolved majority. Bottom: empirical first time 32 of all 64 '
       'trajectories have confirmed B, not the median of the collapsed subset. '
       'Gray means the half-time was not reached by T=10000, not zero or infinity. '
       'No interpolation, equilibrium-phase inference or simultaneous confidence claim.')
    write_json(OUTPUT/'fig00_dynamical_regimes.statistics.json', {
        'source_sha256': file_hash(analysis), 'horizon': report['horizon'],
        'groups': statistics,
    })
    print(json.dumps(statistics))


if __name__ == '__main__':
    main()
