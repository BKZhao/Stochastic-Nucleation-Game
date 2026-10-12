"""Protect exact growth clocks, historical outcome partitions and observation neutrality."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nucleation_campaign import expand_campaign
from nucleation_runner import run_replica
from research_common import config_hash


class Progress(unittest.TestCase):
    def test_transient_crossing_is_recorded_even_after_retreat(self):
        from nucleation_progress import GrowthTracker
        tracker = GrowthTracker(100, (.05, .2, .5, .8))
        for time, count in enumerate([1, 6, 1, 20, 0]):
            tracker.update(time, count)
        self.assertEqual(tracker.first_crossings, {'0.05': 1, '0.2': 3, '0.5': None, '0.8': None})
        self.assertEqual(tracker.extinction_time, 4)
        with self.assertRaises(ValueError):
            tracker.update(6, 0)

    def test_integer_population_threshold_and_initial_crossing(self):
        from nucleation_progress import GrowthTracker
        tracker = GrowthTracker(9, (.2, .8))
        self.assertEqual(tracker.update(0, 2), ('0.2',))
        self.assertEqual(tracker.update(1, 7), ())
        self.assertEqual(tracker.update(2, 8), ('0.8',))

    def test_invalid_thresholds_rejected(self):
        from nucleation_progress import GrowthTracker
        for fractions in [(), (.2, .2), (.8, .2), (0.,), (1.1,), (True,), (float('nan'),)]:
            with self.subTest(fractions=fractions), self.assertRaises(ValueError):
                GrowthTracker(100, fractions)

    def test_historical_outcome_counts_do_not_use_future_outcomes(self):
        from nucleation_progress import outcome_summary
        def row(event=None, recovery=None):
            return dict(technical_status='complete', protocol={'t_max_sweeps': 40},
                        parameters={'mutation': 0.}, t_confirm_B=event, recovered_time=recovery)
        rows = [row(event=10), row(event=11), row(recovery=10), row(recovery=11), row()]
        at10 = outcome_summary(rows, 10)
        self.assertEqual((at10['events'], at10['recovered'], at10['unresolved']), (1, 1, 3))
        at40 = outcome_summary(rows, 40)
        self.assertEqual((at40['events'], at40['recovered'], at40['unresolved']), (2, 2, 1))

    def test_growth_observation_does_not_change_simulated_fields(self):
        base = json.loads(Path('configs/nucleation/n02_single_founder.json').read_text())['base']
        base['model']['size'] = 16
        base['damage']['count'] = 16
        base['protocol'].update(t_max_sweeps=20, sample_every=1, snapshot_every=20)
        extra = deepcopy(base)
        extra['growth'] = {'defector_fractions': [.05, .2, .5, .8], 'snapshot_crossings': True}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'nucleation'
            plain = run_replica((base, 91, str(root), {'source_hash': 'test'}))
            observed = run_replica((extra, 91, str(root), {'source_hash': 'test'}))
            folders = [root / 'N02' / config_hash(c) / 'seed-0000000091' for c in [base, extra]]
            arrays = []
            for folder in folders:
                with np.load(folder / 'states' / 'final.npz') as state:
                    arrays.append((state['strategy'], state['resource']))
            for left, right in zip(*arrays):
                np.testing.assert_array_equal(left, right)
            self.assertEqual(plain['t_confirm_B'], observed['t_confirm_B'])
            self.assertEqual(observed['growth']['first_crossings']['0.05'], 0)
            self.assertTrue((folders[1] / 'states' / 'growth-0.05.npz').exists())
            from nucleation_growth_analysis import read_growth_run
            record = read_growth_run(extra, observed, folders[1], [0, 1, 20])
            self.assertEqual(record['crossing_states']['0.05']['time'], 0)
            self.assertEqual(record['spatial']['0']['defector_fraction'], 16 / 256)

    def test_empty_defector_state_has_no_largest_share(self):
        from nucleation_growth_analysis import state_metrics
        result = state_metrics(np.ones(16, dtype=np.int8), np.ones(16), 4, .3)
        self.assertIsNone(result['largest_share'])
        self.assertEqual(result['clusters'], 0)
        self.assertFalse(result['winding'])


class LayoutCampaign(unittest.TestCase):
    def test_three_explicit_layouts_and_independent_configs(self):
        campaign = json.loads(Path('configs/nucleation/n02_single_founder.json').read_text())
        campaign['replicas'] = 2
        layouts = [{'count': 1, 'shape': 'compact'}, {'count': 16, 'shape': 'compact'},
                   {'count': 16, 'shape': 'random'}]
        campaign['grid'] = {'damage': deepcopy(layouts)}
        jobs = expand_campaign(campaign)
        self.assertEqual(len(jobs), 6)
        self.assertEqual({(c['damage']['count'], c['damage']['shape']) for c, _ in jobs},
                         {(1, 'compact'), (16, 'compact'), (16, 'random')})
        jobs[0][0]['damage']['count'] = 99
        self.assertEqual(campaign['grid']['damage'], layouts)

    def test_ambiguous_layout_axes_rejected(self):
        campaign = json.loads(Path('configs/nucleation/n02_single_founder.json').read_text())
        campaign['grid'] = {'damage': [{'count': 1, 'shape': 'compact'}], 'damage.count': [16]}
        with self.assertRaises(ValueError):
            expand_campaign(campaign)


class ResourceIntervention(unittest.TestCase):
    def test_strategy_only_damage_preserves_initial_resources(self):
        from analyze_nucleation_founders import validate_jobs
        base = json.loads(Path('configs/nucleation/n02_single_founder.json').read_text())['base']
        base['model']['size'] = 16
        base['damage'] = {'count': 16, 'shape': 'random', 'resource_value': 1.}
        base['protocol'].update(t_max_sweeps=3, sample_every=1, snapshot_every=3)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'nucleation'
            summary = run_replica((base, 119, str(root), {'source_hash': 'test'}))
            folder = root / 'N02' / config_hash(base) / 'seed-0000000119'
            with np.load(folder / 'states' / 'initial.npz') as state:
                self.assertEqual(int((state['strategy'] == 0).sum()), 16)
                self.assertEqual(float((1-state['resource']).sum()), 0.)
            validate_jobs([(base, summary, folder)], base)

    def test_invalid_resource_value_rejected_before_simulation(self):
        from nucleation_runner import validate_config
        base = json.loads(Path('configs/nucleation/n02_single_founder.json').read_text())['base']
        for value in [-.1, 1.1, float('nan'), True, '1']:
            with self.subTest(value=value), self.assertRaises(ValueError):
                config = deepcopy(base)
                config['damage']['resource_value'] = value
                validate_config(config)
