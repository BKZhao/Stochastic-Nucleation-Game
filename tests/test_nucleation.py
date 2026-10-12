"""N00 contracts for the new research layer; no old production experiments."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
from b_eco_unified import advance_state, create_simulation_state, step_state
from research_events import CollapseDetector
from research_clusters import analyze_clusters, cluster_overlaps
from research_state import clone_state
from nucleation_seeds import seed_mask
from nucleation_survival import survival_summary

MODEL = dict(temptation=1.2, growth=.1, depletion=.5, coupling=.5,
             strength=1., selection_noise=.1, mutation=.001, diffusion=.3)


class Events(unittest.TestCase):
    def test_joint_reset_and_confirmation(self):
        event = CollapseDetector(.2, .2, 3)
        values = [(.1,.9),(.9,.1),(.1,.1),(.1,.1),(.9,.1),(.1,.1),(.1,.1),(.1,.1)]
        for time, (rho, res) in enumerate(values, 1):
            event.update(time, rho, res)
        self.assertEqual((event.entry_time, event.confirm_time), (6,8))
        event.update(9,1.,1.)
        self.assertEqual(event.confirm_time,8)

    def test_clock_rejects_gaps(self):
        event = CollapseDetector(.2,.2,2)
        with self.assertRaises(ValueError):
            event.update(2,.1,.1)


class Clusters(unittest.TestCase):
    def test_seam_is_not_winding(self):
        mask = np.zeros((8,8),bool)
        mask[0,0] = mask[0,7] = True
        result = analyze_clusters(mask)
        self.assertEqual(result['areas'].tolist(), [2])
        self.assertEqual(result['perimeters'].tolist(), [6])
        self.assertFalse(result['winding'].any())
        mask[0,:] = True
        result = analyze_clusters(mask)
        self.assertTrue(result['winding'][0,1])
        self.assertFalse(result['winding'][0,0])

    def test_merge_split_overlap(self):
        old = np.zeros((8,8),bool); old[2,2] = old[2,4] = True
        new = old.copy(); new[2,3] = True
        a,b = analyze_clusters(old),analyze_clusters(new)
        self.assertEqual(len(cluster_overlaps(a['labels'],b['labels'])),2)
        self.assertEqual(len(cluster_overlaps(b['labels'],a['labels'])),2)

    def test_empty_full(self):
        self.assertEqual(len(analyze_clusters(np.zeros((8,8),bool))['areas']),0)
        full=analyze_clusters(np.ones((8,8),bool))
        self.assertEqual(full['perimeters'].tolist(),[0])
        self.assertTrue(full['winding'].all())


class Seeds(unittest.TestCase):
    def test_cost_geometry_and_translation(self):
        for shape in ['compact','stripe','four_patches','random']:
            a=seed_mask(48,64,shape,(0,0),np.random.default_rng(3))
            b=seed_mask(48,64,shape,(7,13),np.random.default_rng(3))
            self.assertEqual(int(a.sum()),64)
            np.testing.assert_array_equal(b,np.roll(a,(7,13),(0,1)))
            if shape == 'four_patches':
                self.assertEqual(len(analyze_clusters(a)['areas']),4)
            if shape == 'stripe':
                self.assertFalse(analyze_clusters(a)['winding'].any())


class Survival(unittest.TestCase):
    def test_censoring_and_rmst(self):
        out=survival_summary([2,4,4],[True,True,False],4)
        self.assertAlmostEqual(out['rmst'],10/3)
        self.assertEqual(out['median'],4)
        out=survival_summary([4,4],[False,False],4)
        self.assertIsNone(out['median'])
        self.assertEqual(out['rmst'],4)
        with self.assertRaises(ValueError):
            survival_summary([2],[False],4)


class BackendIntegration(unittest.TestCase):
    def test_readonly_hooks_and_step_advance(self):
        for seed in [5,19]:
            direct=create_simulation_state(16,seed,'random',.7)
            reference=advance_state(direct,steps=100,**MODEL)
            hooked=create_simulation_state(16,seed,'random',.7)
            event=CollapseDetector(.2,.2,3)
            trace=[]
            for t in range(1,101):
                rho,res=step_state(hooked,**MODEL)
                event.update(t,rho,res)
                analyze_clusters((hooked.strategy==0).reshape(16,16))
                analyze_clusters(((hooked.strategy==0)&(hooked.resource<.3)).reshape(16,16))
                trace.append((rho,res))
            np.testing.assert_array_equal(direct.strategy,hooked.strategy)
            np.testing.assert_array_equal(direct.resource,hooked.resource)
            np.testing.assert_array_equal(reference.cooperation,np.array(trace)[:,0])
            np.testing.assert_array_equal(reference.resource,np.array(trace)[:,1])

    def test_clone_has_no_alias(self):
        parent=create_simulation_state(16,1,'random',.7890123456789)
        child=clone_state(parent)
        for name in ['strategy','resource','neighbors','degree','permutation']:
            self.assertFalse(np.shares_memory(getattr(parent,name),getattr(child,name)))
        child.resource[0]=0
        self.assertEqual(parent.resource[0],.7890123456789)
        self.assertEqual(child.resource.dtype,np.float64)
