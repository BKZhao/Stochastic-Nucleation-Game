"""N06 fate continuation: arm transforms, clone determinism, mother identity binding."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from nucleation_interface import resource_substep
from research_common import (PROJECT_ROOT, config_hash, file_hash, write_json, write_npz)
from nucleation_runner import run_replica
from nucleation_campaign import expand_campaign
from run_nucleation_fate import (apply_arm, expand_fate_campaign, load_mother,
                                 run_fate_replica, select_mothers,
                                 validate_fate_config)

MODEL = dict(size=16, temptation=1.16, growth=0.1, depletion=0.5, coupling=0.5,
             strength=1.0, selection_noise=0.1, mutation=0.0, diffusion=0.4)


def make_mother(tmp: Path) -> Path:
    """Produce a real crossing snapshot through the seed runner on a tiny lattice."""
    config = dict(experiment_id='N00', model=dict(MODEL), protocol=dict(
        t_max_sweeps=6, sample_every=2, snapshot_every=6, stop_after_collapse=True),
        collapse=dict(rho_max=.2, resource_max=.2, confirm_sweeps=250),
        clusters=dict(resource_cutoff=.3), schema_version=2, study='test',
        damage=dict(count=4, shape='random', resource_value=0.),
        growth=dict(defector_fractions=[0.01], snapshot_crossings=True))
    run_replica((config, 5, str(tmp / 'nucleation'), {'source_hash': 'test'}))
    directory = tmp / 'nucleation' / 'N00' / config_hash(config) / 'seed-0000000005'
    mother = directory / 'states' / 'growth-0.01.npz'
    if not mother.exists():  # threshold never crossed: force a snapshot as a stand-in
        import shutil
        shutil.copy(directory / 'states' / 'final.npz', mother)
    return mother


class FateContinuation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        cls.mother = make_mother(cls.root)
        cls.base = dict(experiment_id='N06', model=dict(MODEL), protocol=dict(
            t_max_sweeps=5, sample_every=2, snapshot_every=5, stop_after_collapse=True),
            collapse=dict(rho_max=.2, resource_max=.2, confirm_sweeps=250),
            clusters=dict(resource_cutoff=.3), schema_version=2, study='test')

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def fate_config(self, arm, shift=(17, 12)):
        config = deepcopy(self.base)
        config['fate'] = dict(mother_path=str(self.mother),
                              arm=arm, shift=list(shift))
        return config

    def test_load_mother_reconstructs_state_and_leaves_disk_untouched(self):
        before = self.mother.read_bytes()
        state, metadata = load_mother(self.mother, MODEL['size'])
        self.assertEqual(state.size, MODEL['size'])
        self.assertEqual(metadata['event_counter']['streak'], 0)
        self.assertEqual(metadata['experiment_id'], 'N00')
        self.assertEqual(self.mother.read_bytes(), before)

    def test_arms_transform_resource_only(self):
        original, _ = load_mother(self.mother, MODEL['size'])
        strategy_before = original.strategy.copy()
        for arm in ('original', 'adapted', 'shifted'):
            state, _ = load_mother(self.mother, MODEL['size'])
            apply_arm(state, arm, [17, 12], MODEL)
            np.testing.assert_array_equal(state.strategy, strategy_before)
            np.testing.assert_array_equal(state.neighbors, original.neighbors)
            if arm == 'original':
                np.testing.assert_array_equal(state.resource, original.resource)
            elif arm == 'adapted':
                again = resource_substep(state.strategy, state.resource, MODEL['size'],
                                         MODEL['growth'], MODEL['depletion'], MODEL['diffusion'])
                np.testing.assert_allclose(again, state.resource, atol=1e-9)
            else:
                rolled = np.roll(original.resource.reshape(MODEL['size'], MODEL['size']),
                                 (17, 12), (0, 1)).reshape(-1)
                np.testing.assert_array_equal(state.resource, rolled)
                self.assertEqual(state.resource.sum(), original.resource.sum())
                np.testing.assert_array_equal(np.sort(state.resource), np.sort(original.resource))

    def test_clone_determinism_and_distinct_seeds(self):
        config = self.fate_config('original')
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            root_a, root_b = Path(tmp_a) / 'nucleation', Path(tmp_b) / 'nucleation'
            first = run_fate_replica((deepcopy(config), 21, str(root_a), {'source_hash': 'test'}))
            # a genuine re-run in a fresh root must reproduce the same trajectory
            rerun = run_fate_replica((deepcopy(config), 21, str(root_b), {'source_hash': 'test'}))
            self.assertEqual(first['initial_state_hash'], rerun['initial_state_hash'])
            self.assertEqual(first['final_resource'], rerun['final_resource'])
            self.assertEqual(first['final_cooperation'], rerun['final_cooperation'])
            self.assertEqual(first['executed_sweeps'], rerun['executed_sweeps'])
            self.assertEqual(first['t_confirm_B'], rerun['t_confirm_B'])
            # a repeat against the same root returns the cached summary unchanged
            cached = run_fate_replica((deepcopy(config), 21, str(root_a), {'source_hash': 'test'}))
            self.assertEqual(cached, first)
            # a different clone seed diverges when horizons coincide
            other = run_fate_replica((deepcopy(config), 22, str(root_a), {'source_hash': 'test'}))
            base = root_a / 'N06' / config_hash(config)
            with np.load(base / 'seed-0000000021' / 'states' / 'final.npz') as one, \
                    np.load(base / 'seed-0000000022' / 'states' / 'final.npz') as two:
                if other['executed_sweeps'] == first['executed_sweeps']:
                    self.assertFalse(np.array_equal(one['resource'], two['resource']))
            self.assertTrue(first['hit_A'] or first['hit_B'] or first['unresolved'])
            self.assertEqual(first['arm'], 'original')
            self.assertIn('parent_run_id', first['fate'])

    def test_invalid_fate_configs_rejected(self):
        for fate in (dict(mother_path='x', arm='sideways', shift=[1, 2]),
                     dict(mother_path='x', arm='original', shift=[1.5, 2]),
                     dict(mother_path='x', arm='original'),
                     dict(arm='original', shift=[1, 2]),
                     dict(mother_path='x', arm='original', shift=[1, 2],
                          mother_file_sha256='notahexdigest')):
            config = deepcopy(self.base)
            config['fate'] = fate
            with self.subTest(fate=fate):
                with self.assertRaises(ValueError):
                    validate_fate_config(config)
        config = deepcopy(self.base)
        config['experiment_id'] = 'N02'
        config['fate'] = dict(mother_path='x', arm='original', shift=[1, 2])
        with self.assertRaises(ValueError):
            validate_fate_config(config)

    def test_fixed_time_snapshot_selection_with_defector_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'nucleation'
            mothers_campaign = dict(
                master_seed=779, replicas=3,
                base=dict(experiment_id='N02', model=dict(MODEL), protocol=dict(
                    t_max_sweeps=4, sample_every=2, snapshot_every=2, stop_after_collapse=True),
                    collapse=dict(rho_max=.2, resource_max=.2, confirm_sweeps=250),
                    clusters=dict(resource_cutoff=.3), schema_version=2, study='test',
                    damage=dict(count=4, shape='random', resource_value=0.),
                    growth=dict(defector_fractions=[0.01], snapshot_crossings=True)),
                grid={})
            campaign_id = config_hash(mothers_campaign)
            for config, seed in expand_campaign(mothers_campaign):
                run_replica((config, seed, str(root), {'source_hash': 'test'}))
            directory = root / 'N02' / 'campaigns' / campaign_id
            directory.mkdir(parents=True)
            write_json(directory / 'campaign.json', mothers_campaign)
            # periodic snapshots exist at t=2 and t=4; the wide count window
            # admits every clean state, the narrow one below the planted count
            # admits none
            selected = select_mothers(root, campaign_id,
                                      {'random_4': dict(count=3, rule='all')},
                                      bind_identity=True,
                                      snapshot='t-00000004.npz',
                                      defector_window=[0, 256])
            self.assertEqual(len(selected), 3)
            for row in selected:
                self.assertIsInstance(row['mother_defectors'], int)
            # a window below the planted defector count selects nothing, and the
            # selection fails fast rather than returning a short list
            with self.assertRaises(ValueError):
                select_mothers(root, campaign_id,
                               {'random_4': dict(count=3, rule='all')},
                               snapshot='t-00000004.npz', defector_window=[10, 20])
            mid = select_mothers(root, campaign_id,
                                 {'random_4': dict(count=1, rule='smallest_seed')},
                                 bind_identity=True,
                                 snapshot='t-00000004.npz', defector_window=[3, 5])
            for row in mid:
                self.assertIn(row['mother_defectors'], (3, 4, 5))
            # the defector_window must be a valid [lo, hi] pair
            with self.assertRaises(ValueError):
                select_mothers(root, campaign_id, {'random_4': dict(count=3, rule='all')},
                               snapshot='t-00000004.npz', defector_window=[5, 3])
            # and the snapshot name must stay a crossing or periodic state
            with self.assertRaises(ValueError):
                select_mothers(root, campaign_id, {'random_4': dict(count=3, rule='all')},
                               snapshot='final.npz')

    def test_real_micro_campaign_expansion_with_identity_binding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'nucleation'
            mothers_campaign = dict(
                master_seed=777, replicas=2,
                base=dict(experiment_id='N02', model=dict(MODEL), protocol=dict(
                    t_max_sweeps=4, sample_every=2, snapshot_every=4, stop_after_collapse=True),
                    collapse=dict(rho_max=.2, resource_max=.2, confirm_sweeps=250),
                    clusters=dict(resource_cutoff=.3), schema_version=2, study='test',
                    damage=dict(count=4, shape='random', resource_value=0.),
                    growth=dict(defector_fractions=[0.01], snapshot_crossings=True)),
                grid={})
            campaign_id = config_hash(mothers_campaign)
            for config, seed in expand_campaign(mothers_campaign):
                run_replica((config, seed, str(root), {'source_hash': 'test'}))
            directory = root / 'N02' / 'campaigns' / campaign_id
            directory.mkdir(parents=True)
            write_json(directory / 'campaign.json', mothers_campaign)
            # four planted defectors cross the 1% threshold at t=0 -> two mothers
            fate_campaign = dict(
                master_seed=888, replicas_original=2, replicas_intervention=1,
                arms=['original', 'adapted', 'shifted'], shift=[17, 12],
                bind_mother_identity=True,
                mothers=dict(campaign=campaign_id, snapshot='growth-0.01.npz',
                             rules={'random_4': dict(count=2, rule='all')}),
                base=dict(experiment_id='N06', model=dict(MODEL), protocol=dict(
                    t_max_sweeps=3, sample_every=1, snapshot_every=3, stop_after_collapse=True),
                    collapse=dict(rho_max=.2, resource_max=.2, confirm_sweeps=250),
                    clusters=dict(resource_cutoff=.3), schema_version=2, study='test'))
            cells = expand_fate_campaign(fate_campaign, root)
            self.assertEqual(len(cells), 8)  # 2 mothers x (2 + 1 + 1) replicas
            self.assertEqual(len({seed for _, seed in cells}), 8)
            counts = {}
            for config, _ in cells:
                fate = config['fate']
                counts[fate['arm']] = counts.get(fate['arm'], 0) + 1
                self.assertEqual(fate['mother_file_sha256'],
                                 file_hash(PROJECT_ROOT / fate['mother_path']))
                self.assertTrue(fate['mother_run_id'].startswith('seed-'))
                self.assertIsInstance(fate['mother_defectors'], int)
                self.assertEqual(fate['mother_defectors'], 4)
            self.assertEqual(counts, dict(original=4, adapted=2, shifted=2))
            # one end-to-end continuation honours the identity binding
            config, seed = cells[0]
            summary = run_fate_replica((deepcopy(config), seed, str(root),
                                        {'source_hash': 'test'}))
            self.assertEqual(summary['fate']['arm'], config['fate']['arm'])
            # tampering with the mother file must be rejected before any reuse
            mother_file = PROJECT_ROOT / config['fate']['mother_path']
            saved = mother_file.read_bytes()
            with np.load(mother_file, allow_pickle=False) as stored:
                fields = {key: stored[key].copy() for key in stored.files
                          if key != 'metadata_json'}
                metadata = json.loads(str(stored['metadata_json']))
            fields['resource'] = np.clip(fields['resource'] + 0.3, 0.0, 1.0)
            write_npz(mother_file, fields, metadata)
            try:
                with self.assertRaises(ValueError):
                    run_fate_replica((deepcopy(config), seed, str(root),
                                      {'source_hash': 'test'}))
            finally:
                mother_file.write_bytes(saved)


if __name__ == '__main__':
    unittest.main()
