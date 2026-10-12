"""Conditional cluster distributions and deterministic resource controls."""
import json
from pathlib import Path

import numpy as np

from analyze_nucleation_physics import cohort_rows
from nucleation_growth_analysis import state_metrics
from nucleation_interface import resource_substep
from nucleation_paper_inputs import ANALYSIS, REPORTS, load_report
from research_common import PROJECT_ROOT, config_hash, file_hash, write_json


def growth_distributions() -> dict:
    groups, manifests = cohort_rows(load_report('growth'))
    result = {}
    for key in ['compact_16', 'random_16']:
        result[key] = {}
        for fraction in [.05, .2, .5, .8]:
            rows = []
            for directory, summary in groups[key]:
                time = summary['growth']['first_crossings'][str(fraction)]
                if time is None:
                    continue
                path = directory/'states'/f'growth-{fraction}.npz'
                with np.load(path, allow_pickle=False) as state:
                    metadata = json.loads(str(state['metadata_json']))
                    if time != metadata['local_time']:
                        raise ValueError(f'first-crossing clock mismatch: {path}')
                    metrics = state_metrics(state['strategy'], state['resource'], metadata['L'], .3)
                rows.append({'path': str(path.relative_to(PROJECT_ROOT)), 'sha256': file_hash(path),
                             'time': time, **metrics})
            result[key][str(fraction)] = {'total': len(groups[key]), 'reached': len(rows), 'rows': rows}
    return {'groups': result, 'manifests': manifests}


def covariance(strategy: np.ndarray, resource: np.ndarray) -> float:
    return float(np.mean((strategy-strategy.mean())*(resource-resource.mean())))


def autocorrelation(resource: np.ndarray) -> np.ndarray:
    centered = resource-resource.mean()
    return np.fft.ifft2(np.abs(np.fft.fft2(centered))**2).real/resource.size


def resource_controls() -> dict:
    rows = sorted(load_report('direction')['direction']['mothers'], key=lambda row: row['mother_path'])
    results = []
    selected = None
    max_errors = {'mean': 0., 'autocorrelation': 0., 'balance': 0., 'contraction': 0.}
    for row in rows:
        path = PROJECT_ROOT/row['mother_path']
        with np.load(path, allow_pickle=False) as state:
            metadata = json.loads(str(state['metadata_json']))
            size = metadata['L']
            s = state['strategy'].reshape(size, size)
            r = state['resource'].reshape(size, size)
        base = resource_substep(s.ravel(), r.ravel(), size, .1, .5, .4).reshape(size, size)
        entry = {'path': row['mother_path'], 'sha256': file_hash(path), 'arms': {}}
        for label, shift in [('small', (2, 1)), ('medium', (5, -17))]:
            translated = np.roll(r, shift, (0, 1))
            if not np.array_equal(np.sort(r.ravel()), np.sort(translated.ravel())):
                raise ValueError('translation changed the resource multiset')
            nxt = resource_substep(s.ravel(), translated.ravel(), size, .1, .5, .4).reshape(size, size)
            dc = covariance(s, translated)-covariance(s, r)
            dr = float(nxt.mean()-base.mean())
            original, changed = r.ravel().copy(), translated.ravel().copy()
            initial = np.max(np.abs(original-changed))
            contraction = [1.]
            for _ in range(60):
                original = resource_substep(s.ravel(), original, size, .1, .5, .4)
                changed = resource_substep(s.ravel(), changed, size, .1, .5, .4)
                contraction.append(float(np.max(np.abs(original-changed))/initial))
            errors = {'mean': abs(float(translated.mean()-r.mean())),
                      'autocorrelation': float(np.max(np.abs(autocorrelation(r)-autocorrelation(translated)))),
                      'balance': abs(dr-.4*dc),
                      'contraction': float(max(0., np.max(np.asarray(contraction)-.9**np.arange(61))))}
            for name, value in errors.items():
                max_errors[name] = max(max_errors[name], value)
            entry['arms'][label] = {'delta_covariance': dc, 'delta_next_mean': dr,
                                    'relaxation': contraction}
        results.append(entry)
        if selected is None:
            displacements = list(range(-size//2, size//2))
            cross = np.array([[covariance(s, np.roll(r, (dy, dx), (0, 1)))
                               for dx in displacements] for dy in displacements])
            selected = {'path': row['mother_path'], 'displacements': displacements,
                        'cross_covariance': cross.tolist(),
                        'autocorrelation': np.fft.fftshift(autocorrelation(r)).tolist()}
    if max(max_errors.values()) > 1e-12:
        raise ValueError(f'resource identity check failed: {max_errors}')
    return {'rows': results, 'example': selected, 'max_errors': max_errors,
            'scope': 'Fixed-strategy deterministic relaxation, not the relaxation time of coupled dynamics.'}


def main() -> None:
    growth = growth_distributions()
    resource = resource_controls()
    sources = [Path(__file__), PROJECT_ROOT/'experiments/analyze_nucleation_physics.py',
               PROJECT_ROOT/'experiments/nucleation_growth_analysis.py',
               PROJECT_ROOT/'code/research_clusters.py', PROJECT_ROOT/'code/nucleation_interface.py',
               REPORTS['growth'], REPORTS['direction']]
    identity = {'sources': {str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in sources},
                'growth_manifests': growth['manifests'],
                'configurations': [{'path': r['path'], 'sha256': r['sha256']} for r in resource['rows']]}
    path = ANALYSIS/config_hash(identity)/'structure.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {'identity': identity, 'growth': growth, 'resource': resource})
    write_json(PROJECT_ROOT/'paper/structure_inputs.json',
               {'analysis': str(path.relative_to(PROJECT_ROOT)), 'sha256': file_hash(path)})
    print(path)
    print('resource identity errors', resource['max_errors'])


if __name__ == '__main__':
    main()
