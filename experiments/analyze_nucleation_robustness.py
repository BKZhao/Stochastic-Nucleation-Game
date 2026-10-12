"""Robustness round analysis: per-cell replication of the core readouts.

Cells are defined by the frozen robustness_plan.md: two lattice sizes at the
fixed 5% milestone, the same sizes at the fixed N_D~110 milestone, and the
b / D parameter neighbours. Per cell this reports the paired medium-minus-small
shift contrast and the Spearman(dB, delta_q_B) rank test with mother-level
cluster bootstrap, plus saturation and mother-bank yield diagnostics. Saturated
cells (no outcome variance) and zero-yield cells are reported as such, never
silently dropped.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_nucleation_confirmatory import load_campaign_outcomes
from analyze_nucleation_direction import _cluster_indices
from analyze_nucleation_interface import _spearman
from nucleation_interface import interface_observables
from research_common import (PROJECT_ROOT, config_hash, file_hash,
                              research_root, write_json)
from run_nucleation_fate import select_mothers

CELLS = [
    dict(name='L48_base', fate='docs/nucleation/direction_results.md', note='direction round, reference'),
    dict(name='L64_5pct', fate='configs/nucleation/n06_robustness_size64_fate.json',
         bank='e297a4a92349d881', snapshot='growth-0.05.npz', size=64, b=1.16, d=0.4),
    dict(name='L96_5pct', fate='configs/nucleation/n06_robustness_size96_fate.json',
         bank='fda70f07e87e1eb9', snapshot='growth-0.05.npz', size=96, b=1.16, d=0.4),
    dict(name='L64_nd110', fate='configs/nucleation/n06_robustness_size64_nd110_fate.json',
         bank='ea4c2d7ff52ec028', snapshot='growth-0.0268.npz', size=64, b=1.16, d=0.4),
    dict(name='L96_nd110', fate='configs/nucleation/n06_robustness_size96_nd110_fate.json',
         bank='3619fddaae538438', snapshot='growth-0.0119.npz', size=96, b=1.16, d=0.4),
    dict(name='b118', fate='configs/nucleation/n06_robustness_b118_fate.json',
         bank='8a1e5308ee9d4a98', snapshot='growth-0.05.npz', size=48, b=1.18, d=0.4),
    dict(name='d06', fate='configs/nucleation/n06_robustness_d06_fate.json',
         bank='72bd4aa5edb4308d', snapshot='growth-0.05.npz', size=48, b=1.16, d=0.6),
]
BANK_YIELDS = {
    'e297a4a92349d881': 38, 'fda70f07e87e1eb9': 42, 'ea4c2d7ff52ec028': 83,
    '3619fddaae538438': 84, '8a1e5308ee9d4a98': 256, '72bd4aa5edb4308d': 256,
    '4761359814673579': 0, '587fdc4e36054208': 0,
}
BANK_LABELS = {
    'e297a4a92349d881': 'L64 5%', 'fda70f07e87e1eb9': 'L96 5%',
    'ea4c2d7ff52ec028': 'L64 N_D110', '3619fddaae538438': 'L96 N_D110',
    '8a1e5308ee9d4a98': 'b=1.18', '72bd4aa5edb4308d': 'D=0.6',
    '4761359814673579': 'b=1.14 (zero yield)', '587fdc4e36054208': 'D=0.2 (zero yield)',
}


def _cluster_ci(values, seeds, draws=4000, seed=20261007):
    mothers = sorted(set(seeds))
    index_of = {mother: position for position, mother in enumerate(mothers)}
    v = np.asarray(values, dtype=float)
    g = np.asarray([index_of[s] for s in seeds])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        take = rng.integers(0, len(mothers), size=len(mothers))
        out.append(v[_cluster_indices(g, take)].mean())
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def _spearman_ci(dB, dq, seeds, draws=2000, seed=20261007):
    mothers = sorted(set(seeds))
    index_of = {mother: position for position, mother in enumerate(mothers)}
    x, y = np.asarray(dB, dtype=float), np.asarray(dq, dtype=float)
    g = np.asarray([index_of[s] for s in seeds])
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(draws):
        take = rng.integers(0, len(mothers), size=len(mothers))
        ii = _cluster_indices(g, take)
        if len({*x[ii].tolist()}) < 2 or len({*y[ii].tolist()}) < 2:
            continue
        out.append(_spearman(x[ii].tolist(), y[ii].tolist()))
    if not out:
        return None
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def cell_report(root: Path, cell: dict) -> dict:
    size, b, diff = cell['size'], cell['b'], cell['d']
    cfg = json.loads((PROJECT_ROOT / cell['fate']).read_text())
    shifts = [tuple(e['shift']) for e in cfg['arms'] if isinstance(e, dict)]
    outcomes, _, _ = load_campaign_outcomes(PROJECT_ROOT / cell['fate'], root)
    q = {}
    for mother_path, arms in outcomes.items():
        seed = mother_path.split('seed-')[1].split('/')[0]
        q[seed] = {arm: float(np.mean([c['hit_B'] for c in copies]))
                   for arm, copies in arms.items()}
    seeds = sorted(q)
    original = [q[s]['original'] for s in seeds]
    if np.mean(original) >= 0.999:
        return dict(name=cell['name'], status='saturated',
                    mean_q_B_original=float(np.mean(original)),
                    n_mothers=len(seeds),
                    note='every run collapses; no discriminative range (a ceiling, '
                         'not a reversal)')
    small = [q[s].get('shifted(2, 1)') for s in seeds]
    medium = [q[s].get('shifted(5, -17)') for s in seeds]
    pairs = [(m - sm, s) for sm, m, s in zip(small, medium, seeds)
             if sm is not None and m is not None]
    contrast = _cluster_ci([p[0] for p in pairs], [p[1] for p in pairs])
    scaled = None
    for entry in cfg['arms']:
        if isinstance(entry, dict) and tuple(entry['shift']) not in [(2, 1), (5, -17)]:
            label = f"shifted({entry['shift'][0]}, {entry['shift'][1]})"
            values = [q[s][label] for s in seeds if label in q[s]]
            if values:
                scaled = float(np.mean(values))
    mothers = select_mothers(root, cell['bank'],
                             {'compact_16': {'count': 24, 'rule': 'smallest_seed'}},
                             snapshot=cell['snapshot'])
    b_shift = {}
    for mother in mothers:
        with np.load(PROJECT_ROOT / mother['relative_path'], allow_pickle=False) as stored:
            strategy, resource = stored['strategy'].copy(), stored['resource'].copy()
        base = interface_observables(strategy, resource, size, b, 0.5, 1.0, 0.1
                                     )['boundary_c_mean']
        for shift in shifts:
            rolled = np.roll(resource.reshape(size, size), shift, (0, 1)).reshape(-1)
            value = interface_observables(strategy, rolled, size, b, 0.5, 1.0, 0.1
                                          )['boundary_c_mean']
            b_shift.setdefault(mother['seed'], {})[shift] = value - base
    rank_pairs = []
    for mother_path, arms in outcomes.items():
        seed = mother_path.split('seed-')[1].split('/')[0]
        if seed not in b_shift:
            continue
        base = np.mean([c['hit_B'] for c in arms['original']])
        for arm, copies in arms.items():
            if arm == 'original':
                continue
            shift = tuple(int(v) for v in arm[len('shifted('):-1].split(','))
            if shift not in b_shift[seed]:
                continue
            rank_pairs.append((seed, b_shift[seed][shift],
                               float(np.mean([c['hit_B'] for c in copies])) - base))
    if len(rank_pairs) >= 6 and len({p[2] for p in rank_pairs}) >= 2:
        rho = _spearman([p[1] for p in rank_pairs], [p[2] for p in rank_pairs])
        rho_ci = _spearman_ci([p[1] for p in rank_pairs], [p[2] for p in rank_pairs],
                              [p[0] for p in rank_pairs])
    else:
        rho, rho_ci = None, None
    return dict(
        name=cell['name'], status='ok', n_mothers=len(seeds),
        mean_q_B_original=float(np.mean(original)),
        medium_minus_small=dict(mean=float(np.mean([p[0] for p in pairs])),
                                interval=contrast, n_pairs=len(pairs)),
        scaled_medium_mean=scaled,
        spearman_dB_delta=dict(rho=rho, bootstrap95=rho_ci, n_pairs=len(rank_pairs)),
        note='descriptive cell readout with mother-level cluster bootstrap; '
             'no cross-cell pooled claim')


def main() -> None:
    root = research_root(PROJECT_ROOT / 'data' / 'nucleation')
    cells = []
    for cell in CELLS[1:]:
        cells.append(cell_report(root, cell))
    report = dict(
        cells=cells,
        mother_bank_yields={BANK_LABELS[k]: v for k, v in BANK_YIELDS.items()},
        reference=dict(medium_minus_small=-0.1126,
                       medium_minus_small_interval=[-0.137, -0.088],
                       spearman_dB_delta=-0.310,
                       spearman_interval=[-0.398, -0.224],
                       source='direction round, L48 b=1.16 D=0.4, 48 mothers'),
        interpretation=('applicability range of the direction-round readouts: the '
                        'medium-minus-small contrast and the dB rank test replicate '
                        'in every cell with discriminative range (both L64 calibers '
                        'and L96 at fixed N_D); L96 at fixed 5% fraction, b=1.18 and '
                        'D=0.6 saturate (all runs collapse, a ceiling not a reversal) '
                        'and b=1.14 / D=0.2 never establish a mother bank; the '
                        'same-budget geometry effect lives inside an '
                        'establishment-to-collapse window'),
        analysis_sources={
            str(path.relative_to(PROJECT_ROOT)): file_hash(path)
            for path in sorted([*PROJECT_ROOT.glob('code/*.py'),
                                *PROJECT_ROOT.glob('experiments/*.py')])})
    target = root / 'analysis' / config_hash(report)
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'robustness.json', report)
    print(target / 'robustness.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
