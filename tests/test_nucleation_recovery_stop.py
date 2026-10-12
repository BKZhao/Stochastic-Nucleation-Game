"""Absorption stopping preserves finite-horizon fate in the real simulation."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from nucleation_runner import run_replica, validate_config


class RecoveryStopping(unittest.TestCase):
    def config(self):
        config = json.loads(Path('configs/nucleation/n02_single_founder.json').read_text())['base']
        config['model'].update(size=8, temptation=1.0)
        config['protocol'].update(t_max_sweeps=150, sample_every=50, snapshot_every=150)
        return config

    def test_same_seed_fate_and_clock_agree_with_full_horizon(self):
        early_count = 0
        with tempfile.TemporaryDirectory() as tmp:
            root = str(Path(tmp)/'nucleation')
            for seed in [3, 7, 19]:
                full = self.config()
                early = deepcopy(full)
                early['protocol']['stop_after_recovery'] = True
                a = run_replica((full, seed, root, {'source_hash': 'test'}))
                b = run_replica((early, seed, root, {'source_hash': 'test'}))
                for key in ['event_observed', 'recovered_time', 't_confirm_B',
                            'observed_time', 'censor_time']:
                    self.assertEqual(a[key], b[key], (seed, key))
                if b['termination_reason'] == 'absorbing_A':
                    early_count += 1
                    self.assertEqual(b['executed_sweeps'], b['recovered_time'])
                    self.assertLess(b['final_state_time'], 150)
                    self.assertEqual(a['final_cooperation'], 1.)
                    self.assertEqual(b['final_cooperation'], 1.)
                    self.assertGreaterEqual(b['final_resource'], .6)
                    self.assertEqual(b['observed_time'], 150)
        self.assertGreater(early_count, 0)

    def test_rejects_nonabsorbing_recovery(self):
        config = self.config()
        config['protocol']['stop_after_recovery'] = True
        config['model']['mutation'] = .001
        with self.assertRaisesRegex(ValueError, 'mutation=0'):
            validate_config(config)
        config['model']['mutation'] = 0
        config['collapse']['rho_max'] = 1
        with self.assertRaisesRegex(ValueError, 'below one'):
            validate_config(config)
