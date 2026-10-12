"""Refresh a complete phase campaign, its figures, and the verified paper PDF."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from research_common import PROJECT_ROOT, config_hash, file_hash, write_json

PAPER = PROJECT_ROOT/'paper'
STATUS = PAPER/'refresh_status.json'


def status(stage: str, **details) -> None:
    write_json(STATUS, {'stage': stage,
                       'updated_utc': datetime.now(timezone.utc).isoformat(), **details})
    print(json.dumps({'stage': stage, **details}), flush=True)


def run_python(script: str, *args: str) -> str:
    result = subprocess.run([sys.executable, str(PROJECT_ROOT/'experiments'/script), *args],
                            cwd=PROJECT_ROOT, text=True, capture_output=True, check=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, end='')
    return result.stdout.strip()


def complete_campaign(config_path: Path, wait_pid: int | None) -> None:
    config = json.loads(config_path.read_text())
    campaign = PROJECT_ROOT/'data/nucleation/N02/campaigns'/config_hash(config)
    expected = config['replicas']
    for values in config['grid'].values():
        expected *= len(values)
    deadline = time.monotonic()+24*3600
    while not (campaign/'summary.json').is_file():
        if wait_pid is None:
            raise FileNotFoundError(f'campaign not complete: {campaign}')
        if time.monotonic() > deadline:
            raise TimeoutError('campaign did not finish within 24 hours')
        command = Path(f'/proc/{wait_pid}/cmdline')
        if not command.exists() or b'run_nucleation_seeds.py' not in command.read_bytes():
            raise RuntimeError('simulation process ended without a complete campaign summary')
        status('waiting_for_complete_campaign', campaign=str(campaign.relative_to(PROJECT_ROOT)),
               expected_runs=expected, simulation_pid=wait_pid)
        time.sleep(30)
    report = json.loads((campaign/'summary.json').read_text())
    if report['expected_runs'] != expected or report['completed_runs'] != expected:
        raise ValueError('campaign summary is incomplete')
    if json.loads((campaign/'campaign.json').read_text()) != config:
        raise ValueError('campaign configuration changed')


def compile_pdf(reuse_figures: bool = False) -> dict:
    status('building_manuscript')
    build_args = ('--reuse-figures',) if reuse_figures else ()
    print(run_python('build_nucleation_manuscript.py', *build_args), flush=True)
    log_dir = PROJECT_ROOT/'tmp/paper-start'
    log_dir.mkdir(parents=True, exist_ok=True)
    # Each entry point uses the other document's aux file through xr-hyper.
    # Two alternating builds resolve both directions even from a clean directory.
    for cycle in range(2):
        for document in ('main', 'supplement'):
            log_path = log_dir/f'{document}-build-{cycle + 1}.log'
            with log_path.open('w') as log:
                subprocess.run([str(PROJECT_ROOT/'.tools/bin/tectonic'), '--keep-logs',
                                '--keep-intermediates', f'{document}.tex'], cwd=PAPER,
                               stdout=log, stderr=subprocess.STDOUT, check=True)
    forbidden = ['There were undefined references', 'There were undefined citations',
                 'Overfull \\hbox', 'Missing character:', 'LaTeX Error',
                 'multiply defined', 'Rerun to get cross-references right', 'Invalid page number']
    documents = {}
    for document in ('main', 'supplement'):
        log = (PAPER/f'{document}.log').read_text(errors='replace')
        failures = [token for token in forbidden if token in log]
        if failures:
            raise ValueError(f'{document} PDF log checks failed: '+', '.join(failures))
        documents[document] = {
            'path': f'paper/{document}.pdf',
            'pdf_sha256': file_hash(PAPER/f'{document}.pdf'),
            'log_sha256': file_hash(PAPER/f'{document}.log')}
    record = {'built_utc': datetime.now(timezone.utc).isoformat(),
              'pdf_sha256': documents['main']['pdf_sha256'],
              'documents': documents,
              'build_inputs_sha256': file_hash(PAPER/'build_inputs.json'),
              'phase': json.loads((PAPER/'phase_inputs.json').read_text()),
              'log_checks': 'Both PDFs: no undefined or duplicate references, missing characters or overfull boxes.',
              'visual_review': 'Both newly refreshed PDFs still require visual inspection.'}
    write_json(PAPER/'pdf_build.json', record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-figures', action='store_true',
                        help='Typeset frozen figure assets and record unavailable raw inputs.')
    parser.add_argument('--phase-config', type=Path)
    parser.add_argument('--wait-pid', type=int)
    args = parser.parse_args()
    if args.wait_pid is not None and args.phase_config is None:
        parser.error('--wait-pid requires --phase-config')
    try:
        if args.phase_config is not None:
            config_path = args.phase_config.resolve()
            complete_campaign(config_path, args.wait_pid)
            status('verifying_complete_grid')
            analysis = run_python('analyze_nucleation_phase.py', '--config', str(config_path))
            if not Path(analysis).is_file():
                raise FileNotFoundError('phase analyzer did not return one analysis file')
            status('plotting_complete_grid', analysis=analysis)
            print(run_python('plot_nucleation_phase.py', '--analysis', analysis), flush=True)
        record = compile_pdf(args.reuse_figures)
        status('complete', **record)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        details = {'error': str(error)}
        if isinstance(error, subprocess.CalledProcessError):
            details.update(stdout=error.stdout, stderr=error.stderr)
        status('failed', **details)
        raise


if __name__ == '__main__':
    main()
