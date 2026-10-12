"""Real backend integration with terminal events, snapshots, seeds and replay."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from b_eco_unified import create_simulation_state,seed_numba,step_state
from research_state import clone_state
from nucleation_runner import run_replica
from nucleation_campaign import expand_campaign
from research_common import config_hash,verify_complete


class Runner(unittest.TestCase):
    def config(self):
        campaign=json.loads(Path('configs/nucleation/benchmark.json').read_text())
        config=campaign['base']
        config['experiment_id']='N00'
        config['model']['size']=16
        config['model']['mutation']=0.
        config['protocol'].update(t_max_sweeps=3,sample_every=10,snapshot_every=2)
        return config

    def test_exact_horizon_event_and_lossless_snapshot(self):
        config=self.config()
        config['collapse']=dict(rho_max=1.,resource_max=1.,confirm_sweeps=3)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'nucleation'
            result=run_replica((config,7,str(root),{'source_hash':'test'}))
            self.assertEqual(result['t_entry_B'],1)
            self.assertEqual(result['t_confirm_B'],3)
            self.assertTrue(result['event_observed'])
            self.assertIsNone(result['censor_time'])
            directory=root/'N00'/config_hash(config)/'seed-0000000007'
            with np.load(directory/'states'/'final.npz') as state:
                self.assertEqual(state['resource'].dtype,np.float64)
                self.assertEqual(state['strategy'].dtype,np.int8)
                metadata=json.loads(str(state['metadata_json']))
                self.assertEqual(metadata['event_counter']['streak'],3)
            with np.load(directory/'trajectory.npz') as trajectory:
                self.assertEqual(trajectory['values'][:,0].tolist(),[0,3])
                index=trajectory['columns'].tolist().index('collapse_streak')
                self.assertEqual(trajectory['values'][:,index].tolist(),[0,3])
            from analyze_nucleation_followup import read_trace
            trace=read_trace(config,result,directory)
            self.assertEqual(trace.event_time,3)
            self.assertEqual(trace.streak.tolist(),[0,3])
            self.assertEqual(run_replica((config,7,str(root),{'source_hash':'test'})),result)
            with self.assertRaises(ValueError):
                run_replica((config,7,str(root),{'source_hash':'changed'}))

    def test_absorbing_all_c_is_censored_not_failure(self):
        config=self.config()
        with tempfile.TemporaryDirectory() as tmp:
            result=run_replica((config,9,str(Path(tmp)/'nucleation'),{'source_hash':'test'}))
            self.assertFalse(result['event_observed'])
            self.assertEqual(result['censor_time'],3)
            self.assertEqual(result['final_cooperation'],1.)
            self.assertEqual(result['final_resource'],1.)

    def test_seed_assignment_is_order_independent(self):
        campaign=json.loads(Path('configs/nucleation/n01.json').read_text())
        original={(config_hash(c),seed) for c,seed in expand_campaign(campaign)}
        campaign['grid']=dict(reversed(list(campaign['grid'].items())))
        self.assertEqual(original,{(config_hash(c),seed) for c,seed in expand_campaign(campaign)})
        self.assertEqual(len(original),384)

    def test_branch_seeds_replay_and_parent_unchanged(self):
        parent=create_simulation_state(16,51,'random',.65)
        saved=parent.strategy.copy()
        model=self.config()['model']; model.pop('size'); model['mutation']=.05
        def run(seed):
            child=clone_state(parent)
            seed_numba(seed)
            for _ in range(20):
                step_state(child,**model)
            return child
        a,b,c=run(3),run(3),run(4)
        np.testing.assert_array_equal(a.strategy,b.strategy)
        np.testing.assert_array_equal(a.resource,b.resource)
        self.assertFalse(np.array_equal(a.resource,c.resource))
        np.testing.assert_array_equal(parent.strategy,saved)

class ConfigValidation(unittest.TestCase):
    def test_rejects_silent_protocol_changes(self):
        from nucleation_runner import validate_config
        base=Runner().config()
        for owner,key,value in [('collapse','confirm_sweeps',2.5),
                                ('collapse','confirm_sweeps',True),
                                ('collapse','confirm_time',1),
                                ('model','strength',-1.),
                                ('protocol','stop_after_collapse','false')]:
            config=deepcopy(base)
            config[owner][key]=value
            with self.subTest(owner=owner,key=key,value=value):
                with self.assertRaises(ValueError):
                    validate_config(config)
