"""Conditional drift and fluctuation curves from verified short continuations."""
import json
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import pubstyle as ps
from analyze_nucleation_moments import DELTAS, defector_series
from nucleation_paper_inputs import OUTPUT, REPORTS, load_report, record_figure
from research_common import PROJECT_ROOT, file_hash, verify_complete, write_json

ARMS = {'original': ('original', ps.INK, '-', 'o'),
        'adapted': ('adapted', ps.BLUE, '--', 's'),
        'shifted': ('translated', ps.RED, '-.', '^')}
BOOTSTRAP_SEED = 20261008
BOOTSTRAP_SAMPLES = 10000


def verify_moments(report: dict) -> dict:
    """Reconstruct all reported moments from checksum-verified trajectories."""
    observed = defaultdict(list)
    root = PROJECT_ROOT/'data/nucleation'
    for relative, digest in report['manifests'].items():
        directory = root/relative
        if file_hash(directory/'COMPLETE') != digest:
            raise ValueError(f'changed continuation manifest: {directory}')
        summary = verify_complete(directory)
        key = (summary['fate']['mother_path'], summary['arm'])
        observed[key].append(defector_series(directory, summary['L']**2, 20))
    checked = 0
    for mothers in report['groups'].values():
        for mother in mothers:
            for arm in ARMS:
                series = np.asarray(observed.pop((mother['mother_path'], arm)))
                if len(series) != mother[arm]['n'] or len(series) != 64:
                    raise ValueError('continuation count mismatch')
                for delta in DELTAS:
                    change = series[:, delta]-series[:, 0]
                    actual = [change.mean()/delta, change.var(ddof=1)/delta]
                    expected = [mother[arm]['moments'][str(delta)][name]
                                for name in ('a_delta', 'b_delta')]
                    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
                    checked += 2
    if observed:
        raise ValueError('unassigned continuation groups')
    return {'verified_runs': len(report['manifests']), 'verified_moments': checked}


def main() -> None:
    report = load_report('moments')
    verification = verify_moments(report)
    ps.apply_pre()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    fig, axes = plt.subplots(2, 2, figsize=(ps.WIDTH_DOUBLE, 5.0),
                             sharex=True, sharey='col', constrained_layout=True)
    plotted = {}
    for row, source in enumerate(('compact_16', 'random_16')):
        mothers = report['groups'][source]
        indices = rng.integers(0, len(mothers), (BOOTSTRAP_SAMPLES, len(mothers)))
        plotted[source] = {}
        for arm, (label, color, style, marker) in ARMS.items():
            values = np.asarray([[[m[arm]['moments'][str(d)][name]
                                   for name in ('a_delta', 'b_delta')]
                                  for d in DELTAS] for m in mothers])
            mean = values.mean(axis=0)
            ci = np.percentile(values[indices].mean(axis=1), [2.5, 97.5], axis=0)
            plotted[source][arm] = {'mean': mean.tolist(), 'interval95': ci.tolist()}
            for col in range(2):
                axis = axes[row, col]
                axis.plot(DELTAS, mean[:, col], color=color, ls=style, lw=1.4,
                          marker=marker, ms=3.5, markerfacecolor='white', mew=.75,
                          label=label)
        for col, ylabel in enumerate((
                r'mean drift $\langle a_\Delta\rangle$'+'\n(defectors / MCS)',
                r'variance rate $\langle b_\Delta\rangle$'
                +'\n'+r'(defectors$^2$ / MCS)')):
            axis = axes[row, col]
            axis.set_xscale('log')
            axis.set(xlim=(.9, 22), ylabel=ylabel)
            axis.set_xticks(DELTAS, [str(d) for d in DELTAS])
            axis.set_title(('compact' if row == 0 else 'dispersed')+' perturbation',
                           fontsize=9, pad=4)
            if row == 1:
                axis.set_xlabel(r'window $\Delta$ (MCS)')
            if col == 0:
                axis.axhline(0, color=ps.MUTED, lw=.65, zorder=0)
            ps.clean_axes(axis)
            ps.panel_label(axis, 'abcd'[2*row+col], x=0, y=1.055)
    axes[0, 1].set_ylim(bottom=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside upper center', ncol=3,
               frameon=False, fontsize=9, handlelength=2.8)
    ps.save(fig, 'fig06_conditional_moments', OUTPUT)
    campaign = PROJECT_ROOT/'data/nucleation/N06/campaigns'/report['campaign']/'campaign.json'
    record_figure('fig06_conditional_moments', __file__, [REPORTS['moments'], campaign,
                   PROJECT_ROOT/'experiments/analyze_nucleation_moments.py'],
                  'First-5%-crossing configurations: 15 compact and 15 dispersed; '
                  '64 fresh continuations per configuration and treatment; 5760 total. '
                  'Original, fixed-strategy-equilibrium adapted, and periodic resource '
                  'translation (17,12); reference physical parameters. '
                  'Average conditional mean increment / Delta and unbiased within-configuration '
                  'increment variance / Delta at Delta=1,2,5,10,20 MCS. '
                  'Count-based rates in defectors/MCS and defectors^2/MCS, not probabilities; '
                  'signed drift is retained and variance-rate axes start at zero. '
                  'Lines join only these windows, with no fitted scaling law. '
                  'Pointwise 95% percentile bootstrap intervals are retained in the statistics file, not drawn; 10000 whole-configuration '
                  'resamples; all arms and windows of a configuration remain paired; seed 20261008. '
                  'These are finite-MCS empirical moments, not frozen first-visit identities.')
    write_json(OUTPUT/'fig06_conditional_moments.statistics.json',
               {'deltas': DELTAS, 'bootstrap_seed': BOOTSTRAP_SEED,
                'bootstrap_samples': BOOTSTRAP_SAMPLES, **verification,
                'source_sha256': file_hash(REPORTS['moments']), 'groups': plotted})
    print(json.dumps(verification))


if __name__ == '__main__':
    main()
