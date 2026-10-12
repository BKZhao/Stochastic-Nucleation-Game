"""Interface motifs and resource layers from archived configurations, without simulation."""
import json
from pathlib import Path

import numpy as np

from analyze_nucleation_physics import cohort_rows
from nucleation_interface import interface_observables, resource_substep
from nucleation_paper_inputs import ANALYSIS, REPORTS, load_report
from research_common import PROJECT_ROOT, config_hash, file_hash, write_json


def neighbors(field: np.ndarray) -> list[np.ndarray]:
    return [np.roll(field, shift, axis) for axis in (0, 1) for shift in (-1, 1)]


def load_state(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as state:
        size = json.loads(str(state['metadata_json']))['L']
        return state['strategy'].reshape(size, size), state['resource'].reshape(size, size)


def edge_data(strategy: np.ndarray, resource: np.ndarray, payoff: float = 1.16) -> dict:
    counts = sum(neighbors(strategy))
    histogram = np.zeros((4, 4), dtype=int)
    gaps = []
    for adjacent, adjacent_counts in zip(neighbors(strategy), neighbors(counts)):
        mask = (strategy == 1) & (adjacent == 0)
        a, c = counts[mask].astype(int), adjacent_counts[mask].astype(int)
        np.add.at(histogram, (a, c-1), 1)
        gaps.extend((payoff*c-(.5+resource[mask])*a).tolist())
    return {'motifs': histogram, 'gaps': np.asarray(gaps)}


def motif_bias(histogram: np.ndarray, payoff: float, diffusion: float) -> float:
    a = np.arange(4)[:, None]
    c = np.arange(1, 5)[None, :]
    gap = payoff*c-a*(1+.5*(1-diffusion*(4-a)/2))
    return float(np.sum(histogram*np.tanh(gap/.2))/4)


def distance_to(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        raise ValueError('distance requires a nonempty target set')
    distance = np.where(mask, 0, -1)
    frontier = mask.copy()
    level = 0
    while np.any(distance < 0):
        frontier = np.logical_or.reduce(neighbors(frontier)) & (distance < 0)
        if not frontier.any():
            raise ValueError('periodic lattice distance failed to reach all sites')
        level += 1
        distance[frontier] = level
    return distance


def signed_distance(strategy: np.ndarray) -> np.ndarray:
    return np.where(strategy == 1, distance_to(strategy == 0), -distance_to(strategy == 1))


def initial_geometry() -> dict:
    groups, manifests = cohort_rows(load_report('early_interface'))
    result = {}
    maximum_error = 0.
    for key in ('compact_16', 'random_16'):
        matrices, sources = [], []
        for directory, _ in sorted(groups[key], key=lambda item: str(item[0])):
            path = directory/'states/initial.npz'
            s, r = load_state(path)
            if np.count_nonzero(s == 0) != 16 or not np.array_equal(s, r):
                raise ValueError(f'initial preparation mismatch: {path}')
            histogram = edge_data(s, r)['motifs']
            matrices.append(histogram)
            sources.append({'path': str(path.relative_to(PROJECT_ROOT)), 'sha256': file_hash(path)})
            for b, diffusion in ((1.12, .2), (1.16, .4), (1.20, .6)):
                updated = resource_substep(s.ravel(), r.ravel(), len(s), .1, .5, diffusion)
                measured = interface_observables(s, updated, len(s), b, .5, 1., .1)
                if int(histogram.sum()) != measured['interface_edges']:
                    raise ValueError('edge motif count differs from interface owner')
                maximum_error = max(maximum_error, abs(motif_bias(histogram, b, diffusion)-measured['J']))
        mean = np.mean(matrices, axis=0)
        result[key] = {'sources': sources, 'n': len(sources), 'mean_motifs': mean.tolist(),
                       'mean_edges': float(mean.sum()), 'mean_bias': motif_bias(mean, 1.16, .4),
                       'example': sources[0]['path']}
    if maximum_error > 1e-12:
        raise ValueError(f'motif identity mismatch: {maximum_error}')
    return {'groups': result, 'manifests': manifests, 'max_bias_error': maximum_error,
            'selection': 'All 64 archived preparations per geometry; illustrations use first path in lexical order, independent of fate.'}


def resource_layers() -> dict:
    pointer = json.loads((PROJECT_ROOT/'paper/structure_inputs.json').read_text())
    path = PROJECT_ROOT/pointer['analysis']
    if file_hash(path) != pointer['sha256']:
        raise ValueError('structure input hash mismatch')
    sources = json.loads(path.read_text())['resource']['rows']
    bins = np.r_[np.arange(-48, 0), np.arange(1, 49)]
    counts = np.zeros(len(bins), dtype=int)
    arms = {key: {'pre_sum': np.zeros(len(bins)), 'post_sum': np.zeros(len(bins)), 'gaps': []}
            for key in ('original', 'small', 'medium')}
    max_bias_error = 0.
    max_distance_error = 0
    for index, entry in enumerate(sources):
        state_path = PROJECT_ROOT/entry['path']
        if file_hash(state_path) != entry['sha256']:
            raise ValueError(f'configuration hash mismatch: {state_path}')
        s, r = load_state(state_path)
        distance = signed_distance(s)
        if index == 0:
            sites = np.argwhere(np.ones_like(s))
            for value in (0, 1):
                target = np.argwhere(s != value)
                delta = np.abs(sites[:, None, :]-target[None, :, :])
                direct = np.minimum(delta, len(s)-delta).sum(axis=2).min(axis=1).reshape(s.shape)
                mask = s == value
                max_distance_error = max(max_distance_error, int(np.max(np.abs(np.abs(distance[mask])-direct[mask]))))
        counts += np.array([np.count_nonzero(distance == d) for d in bins])
        for key, shift in (('original', (0, 0)), ('small', (2, 1)), ('medium', (5, -17))):
            field = np.roll(r, shift, (0, 1))
            updated = resource_substep(s.ravel(), field.ravel(), len(s), .1, .5, .4).reshape(s.shape)
            for moment, values in (('pre_sum', field), ('post_sum', updated)):
                arms[key][moment] += np.array([values[distance == d].sum() for d in bins])
            gaps = edge_data(s, updated)['gaps']
            measured = interface_observables(s, updated, len(s), 1.16, .5, 1., .1)
            max_bias_error = max(max_bias_error, abs(np.tanh(gaps/.2).sum()/4-measured['J']))
            arms[key]['gaps'].extend(gaps.tolist())
    if counts.sum() != len(sources)*48**2 or max_distance_error or max_bias_error > 1e-11:
        raise ValueError('resource-layer validation failed')
    support = counts > 0
    result = {}
    for key, arm in arms.items():
        result[key] = {name.replace('_sum', ''): (arm[name][support]/counts[support]).tolist()
                       for name in ('pre_sum', 'post_sum')}
        result[key]['gaps'] = np.sort(arm['gaps']).tolist()
    return {'bins': bins[support].tolist(), 'site_counts': counts[support].tolist(), 'arms': result,
            'sources': [{'path': row['path'], 'sha256': row['sha256']} for row in sources],
            'example': sources[0]['path'], 'n': len(sources),
            'checks': {'distance_error': max_distance_error, 'bias_error': max_bias_error},
            'weighting': 'Pooled sites for each signed periodic graph-distance shell; pooled undirected C-D edges for payoff distributions. Single displayed translation vector per magnitude; no continuation outcomes used.'}


def main() -> None:
    initial, layers = initial_geometry(), resource_layers()
    inputs = [Path(__file__), PROJECT_ROOT/'experiments/analyze_nucleation_physics.py',
              PROJECT_ROOT/'code/nucleation_interface.py', REPORTS['early_interface'],
              PROJECT_ROOT/'paper/structure_inputs.json']
    identity = {'sources': {str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in inputs},
                'initial_states': [r for group in initial['groups'].values() for r in group['sources']],
                'grown_states': layers['sources']}
    path = ANALYSIS/config_hash(identity)/'microstructure.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {'identity': identity, 'initial': initial, 'layers': layers})
    write_json(PROJECT_ROOT/'paper/microstructure_inputs.json',
               {'analysis': str(path.relative_to(PROJECT_ROOT)), 'sha256': file_hash(path)})
    print(path)
    print('initial', {key: {k: row[k] for k in ('n', 'mean_edges', 'mean_bias')} for key, row in initial['groups'].items()})
    print('validation', initial['max_bias_error'], layers['checks'])
    print('boundary resources', {key: dict(zip(layers['bins'], row['pre'])) for key, row in layers['arms'].items()})


if __name__ == '__main__':
    main()
