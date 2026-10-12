"""Interface-resource displacement-response map plus the mechanism channels.

Panel layout: (a) ensemble-mean dB(d) over the torus of displacements with the
nine measured displacement vectors overlaid, (b) the most anisotropic example
mother with that mother's own measured arm responses, (c) the primary
pre-registered test as a scatter - interface-resource change dB versus the
paired fate response for all 432 mother-direction pairs, coloured by magnitude,
(d) the transient channel - measured first-sweep drift a_1 versus the same
response. B(d) is computed for every displacement at once through the circular
cross-correlation of the boundary-cooperator mask with the resource field and
validated against the direct interface_observables computation at the
preregistered shifts. The background is a resource-interface cross-correlation
map, not a system phase diagram: it is labelled a displacement-response map.
Binned channel means carry mother-level cluster bootstrap 95% intervals.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from nucleation_interface import interface_observables
from preregister_direction_round import torus_magnitude
from research_common import PROJECT_ROOT

DEFAULT_ANALYSIS = 'data/nucleation/analysis/f0d02c561cd72ebe/direction.json'
DEFAULT_PREREG = 'data/nucleation/N06/campaigns/preregistered_direction_predictions.json'


def boundary_mask(strategy, size):
    """Cooperators adjacent to at least one defector (the interface band)."""
    grid = np.asarray(strategy).reshape(size, size)
    cooperators = (np.roll(grid, 1, 0) + np.roll(grid, -1, 0)
                   + np.roll(grid, 1, 1) + np.roll(grid, -1, 1))
    return ((grid == 1) & (cooperators < 4)).astype(float)


def b_map(mask, resource, size):
    """B(d) for every torus displacement via circular cross-correlation.

    B(d) = mean resource read by boundary cooperators after np.roll(resource, d),
    i.e. sum(mask[x] * resource[x - d]) / sum(mask) for all d at once.
    """
    field = np.asarray(resource, dtype=float).reshape(size, size)
    correlation = np.fft.ifft2(np.conj(np.fft.fft2(mask)) * np.fft.fft2(field)).real
    index = (-np.arange(size)) % size
    return correlation[np.ix_(index, index)] / mask.sum()


def _reduced_grid(size):
    return np.ix_(np.arange(-size // 2, size // 2) % size,
                  np.arange(-size // 2, size // 2) % size)


def _display_shift(dx, dy, size):
    """Wrap a torus displacement into the [-size/2, size/2) display range."""
    return ((int(dx) + size // 2) % size - size // 2,
            (int(dy) + size // 2) % size - size // 2)


def load_mothers(prereg_path: Path):
    prereg = json.loads(prereg_path.read_text())
    rows = []
    for entry in prereg['mothers']:
        with np.load(PROJECT_ROOT / entry['mother_path'], allow_pickle=False) as stored:
            strategy = stored['strategy'].copy()
            resource = stored['resource'].copy()
        rows.append(dict(entry=entry, strategy=strategy, resource=resource))
    return prereg, rows


def _arm_magnitude(arm, size):
    if arm == 'original':
        return 0.0
    shift = tuple(int(value) for value in arm[len('shifted') + 1:-1].split(','))
    return torus_magnitude(list(shift), size)


def _binned(x, y, mothers, bins=8, minimum=3, draws=2000, seed=20261006):
    """Equal-count bin means of y sorted by x with mother-cluster bootstrap 95%.

    The bin assignment and the pooled bin mean stay descriptive; the interval
    resamples mothers with replacement within the bin, so a drawn mother
    contributes its full set of in-bin pairs each time it is drawn. Plain
    s/sqrt(n) bars would treat mother-direction pairs as independent and
    understate the mother-level uncertainty.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mothers = np.asarray(mothers)
    order = np.argsort(x)
    xs, ys, ms = x[order], y[order], mothers[order]
    edges = np.linspace(0, len(xs), bins + 1).astype(int)
    rng = np.random.default_rng(seed)
    centres, means, intervals = [], [], []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop - start < minimum:
            continue
        values = ys[start:stop]
        unique = sorted(set(ms[start:stop].tolist()))
        if len(unique) >= 2:
            block_indices = {mother: np.flatnonzero(ms[start:stop] == mother)
                             for mother in unique}
            samples = []
            for _ in range(draws):
                take = rng.integers(0, len(unique), size=len(unique))
                indices = np.concatenate(
                    [start + block_indices[unique[int(index)]] for index in take])
                samples.append(float(ys[indices].mean()))
            interval = [float(np.quantile(samples, .025)),
                        float(np.quantile(samples, .975))]
        else:
            interval = None
        centres.append(float(xs[start:stop].mean()))
        means.append(float(values.mean()))
        intervals.append(interval)
    return centres, means, intervals


def _magnitude_groups(magnitudes):
    """Marker groups from the magnitudes actually present, rounded to 2 decimals."""
    grouped = {}
    for value in magnitudes:
        grouped.setdefault(round(float(value), 2), []).append(value)
    palette = (('#D55E00', '^'), ('#0072B2', 'o'), ('#009E73', 's'),
               ('#CC79A7', 'D'), ('#E69F00', 'v'))
    return [(value, palette[position % len(palette)][0],
             palette[position % len(palette)][1])
            for position, value in enumerate(sorted(grouped))]


def _channel_scatter(axis, x_values, y_values, mothers, magnitudes, x_label, title,
                     correlation, x_zero=False):
    for magnitude, colour, marker in _magnitude_groups(magnitudes):
        selected = [index for index, value in enumerate(magnitudes)
                    if abs(value - magnitude) < 0.3]
        if not selected:
            continue
        axis.plot([x_values[index] for index in selected],
                  [y_values[index] for index in selected], marker, color=colour,
                  ms=4, alpha=.55, mec='none', label=f'|d|={magnitude:.2f}')
    centres, means, intervals = _binned(x_values, y_values, mothers)
    drawn = [value for value in intervals if value is not None]
    if drawn:
        low = [mean - interval[0] if interval else np.nan
               for mean, interval in zip(means, intervals)]
        high = [interval[1] - mean if interval else np.nan
                for mean, interval in zip(means, intervals)]
        axis.errorbar(centres, means, yerr=[low, high], fmt='k-', lw=1.6,
                      capsize=3, ms=5, label='binned mean (mother bootstrap 95%)')
    else:
        axis.plot(centres, means, 'k-', lw=1.6, label='binned mean (descriptive)')
    axis.axhline(0, color='k', lw=0.8)
    if x_zero:
        axis.axvline(0, color='k', lw=0.8)
    interval = correlation['bootstrap95']
    axis.set(xlabel=x_label, ylabel='paired $\\Delta q_B$',
             title=f"{title}\nSpearman = {correlation['spearman']:.3f} "
                   f"[{interval[0]:.3f}, {interval[1]:.3f}]")
    axis.legend(fontsize=8, loc='best')


def _arm_markers(axis, arm_deltas, size, annotate_label):
    """Overlay measured arm responses on a displacement map panel.

    ``arm_deltas`` maps arm names (e.g. 'shifted(2, 1)') to mean responses.
    Displacements are wrapped into the display range, so the torus-equivalent
    (23, 42) is drawn at (23, -6).
    """
    for arm, mean_delta in sorted(arm_deltas.items()):
        if arm == 'original' or mean_delta is None:
            continue
        dx, dy = (int(value) for value in arm[len('shifted') + 1:-1].split(','))
        dx, dy = _display_shift(dx, dy, size)
        marker = 'o' if mean_delta < 0 else '^'
        colour = '#0072B2' if mean_delta < 0 else '#D55E00'
        axis.plot([dx], [dy], marker, color=colour, mec='k', ms=9, mew=0.8)
        axis.annotate(f"{mean_delta:+.2f}", (dx, dy), xytext=(4, 4),
                      textcoords='offset points', fontsize=7, color=colour)
    axis.plot([0], [0], 'k+', ms=10, mew=1.5)


def _mother_arm_deltas(pairs, seed):
    """Mean paired response per arm for one mother, from the analysis pairs."""
    grouped = {}
    for pair in pairs:
        if pair['seed'] == seed:
            grouped.setdefault(pair['arm'], []).append(pair['delta_qB'])
    return {arm: float(np.mean(values)) for arm, values in grouped.items()}


def response_figure(analysis: dict, prereg: dict, mothers: list, output: Path) -> dict:
    model = analysis['identity']['model']
    size = model['size']
    maps = []
    for mother in mothers:
        mask = boundary_mask(mother['strategy'], size)
        maps.append(b_map(mask, mother['resource'], size))
    shifts = [tuple(shift) for shift in prereg['shifts']]
    direct = {}
    for mother in mothers:
        entry = mother['entry']
        for shift in shifts:
            rolled = np.roll(mother['resource'].reshape(size, size), shift,
                             (0, 1)).reshape(-1)
            observables = interface_observables(
                mother['strategy'], rolled, size, model['temptation'],
                model['coupling'], model['strength'], model['selection_noise'])
            key = (entry['seed'], shift)
            direct[key] = observables['boundary_c_mean']
    deviations = []
    for mother, b in zip(mothers, maps):
        for shift in shifts:
            key = (mother['entry']['seed'], shift)
            deviations.append(abs(b[shift[0] % size, shift[1] % size] - direct[key]))
    max_deviation = float(max(deviations))

    ensemble = np.mean([b - b[0, 0] for b in maps], axis=0)
    medium = ['(5, -17)', '(17, 5)', '(-5, 17)', '(-17, -5)']
    anisotropy = []
    for mother, b in zip(mothers, maps):
        values = [mother['entry']['boundary_c_shift'][key] for key in medium]
        anisotropy.append(max(values) - min(values))
    example_index = int(np.argmax(anisotropy))
    example = maps[example_index] - maps[example_index][0, 0]
    example_seed = mothers[example_index]['entry']['seed']

    display = ensemble[_reduced_grid(size)]
    example_display = example[_reduced_grid(size)]
    extent = [-size // 2, size // 2 - 1, -size // 2, size // 2 - 1]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), layout='constrained')
    ensemble_arms = {arm: row['mean_delta']
                     for arm, row in analysis['direction']['arms'].items()}
    mother_arms = _mother_arm_deltas(analysis['pairs'], example_seed)
    for axis, field, title, arm_deltas in (
            (axes[0, 0], display, 'Ensemble mean dB(d), 48 mothers', ensemble_arms),
            (axes[0, 1], example_display,
             f'Most anisotropic mother (seed {example_seed})',
             mother_arms)):
        image = axis.imshow(field.T, origin='lower', extent=extent, cmap='viridis',
                            vmin=-0.06, vmax=0.45, interpolation='nearest')
        axis.contour(np.arange(-size // 2, size // 2),
                     np.arange(-size // 2, size // 2), field.T,
                     levels=[0.0], colors='w', linewidths=0.8)
        _arm_markers(axis, arm_deltas, size,
                     'ensemble' if axis is axes[0, 0] else 'this mother')
        axis.set(xlabel='displacement dx', ylabel='displacement dy', title=title)
        plt.colorbar(image, ax=axis, label='B(d) - B(0)', shrink=0.85)

    pairs = analysis['pairs']
    dB_values = [pair['dB'] for pair in pairs]
    delta_values = [pair['delta_qB'] for pair in pairs]
    pair_magnitudes = [pair['magnitude'] for pair in pairs]
    pair_seeds = [pair['seed'] for pair in pairs]
    _channel_scatter(axes[1, 0], dB_values, delta_values, pair_seeds,
                     pair_magnitudes,
                     'interface-resource change $B(d)-B(0)$',
                     'Primary mechanism: resource band vs fate',
                     analysis['direction']['primary'], x_zero=True)

    dose_rows = [row for row in analysis['dose']['rows']
                 if row['delta_qB'] is not None and row['arm'] != 'original']
    _channel_scatter(axes[1, 1], [row['a1'] for row in dose_rows],
                     [row['delta_qB'] for row in dose_rows],
                     [row['seed'] for row in dose_rows],
                     [_arm_magnitude(row['arm'], size) for row in dose_rows],
                     'measured first-sweep drift $a_1$',
                     'Transient channel: drift vs fate',
                     analysis['dose']['chain']['a1_to_delta'])
    axes[1, 1].annotate('small-|d| pairs sit at $B(d)-B(0)\\approx 0$',
                        xy=(0.02, 0.95), xycoords='axes fraction', va='top',
                        fontsize=8, color='dimgray')

    primary = analysis['direction']['primary']
    fig.suptitle(f"Interface-resource displacement-response map; primary "
                 f"pre-registered test "
                 f"Spearman(dB, $\\Delta q_B$) = {primary['spearman']:.3f} "
                 f"[{primary['bootstrap95'][0]:.3f}, {primary['bootstrap95'][1]:.3f}]",
                 fontsize=11)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'displacement_response.{suffix}', dpi=220)
    plt.close(fig)
    return dict(max_fft_vs_direct_deviation=max_deviation,
                example_seed=example_seed,
                example_anisotropy=float(max(anisotropy)),
                example_arm_deltas={arm: round(value, 4)
                                    for arm, value in mother_arms.items()},
                ensemble_dB_at_origin=float(ensemble[0, 0]),
                ensemble_dB_at_tested_shifts={
                    str(shift): float(np.mean([b[shift[0] % size, shift[1] % size]
                                               - b[0, 0] for b in maps]))
                    for shift in shifts})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', type=Path, default=PROJECT_ROOT / DEFAULT_ANALYSIS)
    parser.add_argument('--prereg', type=Path, default=PROJECT_ROOT / DEFAULT_PREREG)
    parser.add_argument('--output', type=Path, default=Path('figures/nucleation/direction'))
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    if 'model' not in analysis.get('identity', {}):
        raise SystemExit('analysis report predates the identity model block; '
                         'rerun analyze_nucleation_direction.py first')
    prereg, mothers = load_mothers(args.prereg)
    diagnostics = response_figure(analysis, prereg, mothers, args.output)
    print(json.dumps(diagnostics))
    print(args.output / 'displacement_response.png')
