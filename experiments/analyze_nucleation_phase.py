"""Complete-grid finite-time fate estimates; does not infer steady-state phases."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from nucleation_campaign import expand_campaign
from nucleation_survival import risk_difference_interval, wilson_interval
from research_common import PROJECT_ROOT, config_hash, file_hash, verify_complete, write_json


def phase_report(config_path: Path, root: Path) -> dict:
    campaign = json.loads(config_path.read_text())
    groups = defaultdict(list)
    run_hashes = []
    source_hashes = set()
    for config, seed in expand_campaign(campaign):
        directory = root / config['experiment_id'] / config_hash(config) / f'seed-{seed:010d}'
        summary = verify_complete(directory)
        if summary['protocol'] != config['protocol'] or summary['damage'] != config['damage']:
            raise ValueError(f'run identity mismatch: {directory}')
        if summary['event_observed'] and summary['recovered_time'] is not None:
            raise ValueError(f'A and B cannot coexist with zero mutation: {directory}')
        with np.load(directory / 'states/initial.npz', allow_pickle=False) as initial:
            mask = initial['strategy'] == 0
            if int(mask.sum()) != 16 or not np.array_equal(initial['resource'], 1.0 - mask):
                raise ValueError(f'initial damage budget mismatch: {directory}')
        key = (config['model']['temptation'], config['model']['diffusion'],
               config['damage']['shape'])
        groups[key].append(summary)
        source_hashes.add(summary['source_hash'])
        run_hashes.append((str(directory.relative_to(root)), file_hash(directory / 'COMPLETE')))
    if len(source_hashes) != 1:
        raise ValueError('phase campaign mixes simulation source versions')
    cells = []
    for (b, diffusion, shape), runs in sorted(groups.items()):
        n = len(runs)
        if n != campaign['replicas']:
            raise ValueError(f'incomplete cell: {(b, diffusion, shape)}')
        counts = {'B': sum(r['event_observed'] for r in runs),
                  'A': sum(r['recovered_time'] is not None for r in runs)}
        counts['U'] = n - counts['A'] - counts['B']
        cells.append(dict(b=b, diffusion=diffusion, shape=shape, n=n, counts=counts,
                          probability={key: value/n for key, value in counts.items()},
                          interval95={key: wilson_interval(value, n) for key, value in counts.items()}))
    expected_cells = np.prod([len(v) for v in campaign['grid'].values()])
    if len(cells) != expected_cells:
        raise ValueError('the Cartesian grid is incomplete')
    lookup = {(r['b'], r['diffusion'], r['shape']): r for r in cells}
    contrasts = []
    for b in campaign['grid']['model.temptation']:
        for diffusion in campaign['grid']['model.diffusion']:
            random = lookup[b, diffusion, 'random']
            compact = lookup[b, diffusion, 'compact']
            contrasts.append(dict(
                b=b, diffusion=diffusion,
                difference=random['probability']['B']-compact['probability']['B'],
                interval95=risk_difference_interval(random['counts']['B'], random['n'],
                                                    compact['counts']['B'], compact['n'])))
    identity = dict(config=str(config_path.relative_to(PROJECT_ROOT)),
                    config_sha256=file_hash(config_path), simulation_sources=sorted(source_hashes),
                    analyzer_sha256=file_hash(Path(__file__)),
                    runs_digest=config_hash(sorted(run_hashes)))
    return dict(identity=identity, protocol=campaign['base'], replicas=campaign['replicas'],
                axes=campaign['grid'], n_runs=len(run_hashes), cells=cells, contrasts=contrasts,
                interpretation='Finite-time outcome probabilities after planted damage; '
                'unresolved outcomes remain U. No interpolation, steady-state phase boundary, '
                'critical nucleus or simultaneous-significance claim.',
                runs=[dict(path=path, complete_sha256=digest) for path, digest in sorted(run_hashes)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path,
                        default=PROJECT_ROOT/'configs/nucleation/n02_paper_phase_map.json')
    args = parser.parse_args()
    report = phase_report(args.config, PROJECT_ROOT/'data/nucleation')
    target = PROJECT_ROOT/'data/nucleation/analysis'/config_hash(report['identity'])/'phase.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, report)
    print(target)


if __name__ == '__main__':
    main()
