"""Finite-time outcome regimes and collapse half-times from frozen run records."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from nucleation_survival import wilson_interval
from research_common import PROJECT_ROOT, config_hash, file_hash, write_json


def resolved_majority(counts: dict[str, int]) -> str | None:
    """Require a pointwise Wilson lower limit above one half."""
    if set(counts) != {'A', 'B', 'U'} or any(
            type(v) is not int or v < 0 for v in counts.values()):
        raise ValueError('nonnegative integer A/B/U counts required')
    total = sum(counts.values())
    if total == 0:
        raise ValueError('at least one trajectory required')
    return next((outcome for outcome, count in counts.items()
                 if wilson_interval(count, total)[0] > .5), None)


def collapse_half_time(times: list[int], total: int, horizon: int) -> int | None:
    """First time half of all trajectories have confirmed B, including ties."""
    if type(total) is not int or total < 1 or type(horizon) is not int or horizon < 1:
        raise ValueError('positive integer ensemble size and horizon required')
    if len(times) > total or any(type(t) is not int or not 0 <= t <= horizon for t in times):
        raise ValueError('invalid confirmed-event times or count')
    rank = (total + 1) // 2
    return sorted(times)[rank - 1] if len(times) >= rank else None


def verified_json(path: Path, expected: str) -> dict:
    """Read one checksummed record without rereading its contents."""
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f'checksum mismatch: {path}')
    return json.loads(content)


def regimes_report(phase_path: Path, root: Path) -> dict:
    phase = json.loads(phase_path.read_text())
    horizon = phase['protocol']['protocol']['t_max_sweeps']
    baseline = phase['protocol']
    expected = {(c['b'], c['diffusion'], c['shape']): c for c in phase['cells']}
    counts = {key: dict(A=0, B=0, U=0) for key in expected}
    times = defaultdict(list)
    records_digest = hashlib.sha256()
    seen = set()
    for index, record in enumerate(phase['runs'], 1):
        if record['path'] in seen:
            raise ValueError(f"duplicate trajectory: {record['path']}")
        seen.add(record['path'])
        directory = root / record['path']
        manifest = verified_json(directory/'COMPLETE', record['complete_sha256'])
        summary = verified_json(directory/'summary.json', manifest['summary.json'])
        parameters = summary['parameters']
        key = (parameters['temptation'], parameters['diffusion'], summary['damage']['shape'])
        if key not in expected:
            raise ValueError(f'unexpected parameter cell: {directory}')
        model = {**baseline['model'], 'temptation': key[0], 'diffusion': key[1]}
        size = model.pop('size')
        damage = {**baseline['damage'], 'shape': key[2]}
        if (parameters != model or summary['L'] != size or summary['damage'] != damage
                or summary['protocol'] != baseline['protocol']
                or summary['source_hash'] not in phase['identity']['simulation_sources']
                or summary['technical_status'] != 'complete'):
            raise ValueError(f'run identity mismatch: {directory}')
        confirmed, recovered = summary['t_confirm_B'], summary['recovered_time']
        if summary['event_observed']:
            if (type(confirmed) is not int or not 0 <= confirmed <= horizon
                    or recovered is not None):
                raise ValueError(f'invalid collapse event: {directory}')
            if confirmed - summary['t_entry_B'] + 1 != baseline['collapse']['confirm_sweeps']:
                raise ValueError(f'confirmation-clock mismatch: {directory}')
            outcome = 'B'
            times[key].append(confirmed)
        else:
            if confirmed is not None:
                raise ValueError(f'inconsistent event flag: {directory}')
            if recovered is not None and (type(recovered) is not int
                                           or not 0 <= recovered <= horizon):
                raise ValueError(f'invalid recovery time: {directory}')
            outcome = 'A' if recovered is not None else 'U'
        counts[key][outcome] += 1
        records_digest.update((record['path']+'\0'+manifest['summary.json']+'\n').encode())
        if index % 25000 == 0:
            print(f'verified {index}/{phase["n_runs"]} event records', flush=True)
    if len(seen) != phase['n_runs']:
        raise ValueError('trajectory count differs from frozen phase analysis')
    cells = []
    for key, cell in expected.items():
        if counts[key] != cell['counts'] or sum(counts[key].values()) != cell['n']:
            raise ValueError(f'outcome counts differ from frozen analysis: {key}')
        cells.append({
            'b': key[0], 'diffusion': key[1], 'shape': key[2], 'n': cell['n'],
            'counts': counts[key], 'resolved_majority': resolved_majority(counts[key]),
            't_half_B': collapse_half_time(times[key], cell['n'], horizon),
            'confirmed_times_B': sorted(times[key]),
        })
    identity = {
        'phase_analysis': str(phase_path.relative_to(PROJECT_ROOT)),
        'phase_sha256': file_hash(phase_path), 'analyzer_sha256': file_hash(Path(__file__)),
        'interval_source_sha256': file_hash(PROJECT_ROOT/'code/nucleation_survival.py'),
        'event_records_digest': records_digest.hexdigest(),
    }
    return {
        'identity': identity, 'horizon': horizon, 'axes': phase['axes'],
        'protocol': baseline, 'n_runs': len(seen), 'cells': cells,
        'definitions': {
            'regime': 'A/B/U only when the pointwise Wilson 95% lower bound exceeds 1/2; otherwise no resolved majority. No simultaneous coverage claim.',
            't_half_B': 'inf{t: number of confirmed B events by t / all trajectories >= 1/2}; null means not reached by T, not zero or infinite time.',
            'scope': 'Finite-time event outcomes, not stationary phases. Recovery trajectories remain in the denominator; no conditional median over B-only trajectories.',
            'verification': 'All COMPLETE and summary.json checksums verified against the frozen phase analysis; counts rechecked cellwise. Simulation trajectories are not rerun.',
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase-analysis', type=Path, required=True)
    args = parser.parse_args()
    report = regimes_report(args.phase_analysis.resolve(), PROJECT_ROOT/'data/nucleation')
    target = PROJECT_ROOT/'data/nucleation/analysis'/config_hash(report['identity'])/'regimes.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    write_json(target, report)
    write_json(PROJECT_ROOT/'paper/regimes_inputs.json', {
        'analysis': str(target.relative_to(PROJECT_ROOT)), 'sha256': file_hash(target),
        'phase_analysis': report['identity']['phase_analysis'],
        'phase_sha256': report['identity']['phase_sha256'],
    })
    print(target)


if __name__ == '__main__':
    main()
