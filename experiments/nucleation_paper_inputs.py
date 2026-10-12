"""Frozen evidence inputs and provenance for the finite-time collapse paper."""
import json
from pathlib import Path

from research_common import PROJECT_ROOT, file_hash, write_json

ANALYSIS = PROJECT_ROOT / 'data/nucleation/analysis'
OUTPUT = PROJECT_ROOT / 'figures/nucleation/paper'
REPORTS = {
    'moments': ANALYSIS / '412167a8b811e426/moments.json',
    'growth': ANALYSIS / '9cf40e627ab5987f/growth.json',
    'early_interface': ANALYSIS / '9d20420d06db4296/interface.json',
    'resource_control': ANALYSIS / 'c20406591965431c/growth.json',
    'confirmation': ANALYSIS / '84e608ec76114c63/confirmation.json',
    'direction': ANALYSIS / 'f0d02c561cd72ebe/direction.json',
    'mechanism': ANALYSIS / '102a4bb967083a9b/mechanism.json',
    'robustness': ANALYSIS / '658e0f7d77cd7dae/robustness.json',
    'r3': ANALYSIS / '2e9372f61ffee64b/r3.json',
    'post_confirmation': ANALYSIS / '88ad386a4538e7fb/post_confirmation.json',
    'geometry': ANALYSIS / '71b13393bb500003/confirmation.json',
}


def load_report(name: str) -> dict:
    return json.loads(REPORTS[name].read_text())


def record_figure(stem: str, script: str, inputs: list[Path], selection: str) -> None:
    """Bind each exported figure to exact input files and plotting sources."""
    sources = [Path(script), Path(__file__), PROJECT_ROOT/'experiments/pubstyle.py']
    assets = [OUTPUT/f'{stem}.{suffix}' for suffix in ('pdf', 'png')]
    def describe(path):
        return {'path': str(path.relative_to(PROJECT_ROOT)), 'sha256': file_hash(path)}
    write_json(OUTPUT/f'{stem}.json', {
        'inputs': [describe(p) for p in inputs],
        'plotting_sources': [describe(p) for p in sources],
        'assets': [describe(p) for p in assets],
        'selection': selection,
    })
