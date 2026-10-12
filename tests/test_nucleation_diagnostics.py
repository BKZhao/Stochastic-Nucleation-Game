"""Synthetic censoring and cohort checks for descriptive follow-up diagnostics."""
import unittest
import numpy as np
from nucleation_diagnostics import RunTrace,landmark_summary,paired_window_drift


def trace(event=None, drift=0.):
    end=20 if event is None else event
    time=np.arange(end+1)
    return RunTrace(str(event),event,20,time,np.ones(end+1)*.9+time*drift,
                    np.ones(end+1)*.8,np.ones(end+1)*.01,np.zeros(end+1,int))


class Diagnostics(unittest.TestCase):
    def test_landmark_excludes_exact_event_and_truncates_followup(self):
        result=landmark_summary([trace(5),trace(10),trace()],5,5)
        self.assertEqual(result['total'],3)
        self.assertEqual(result['confirmed_before_or_at'],1)
        self.assertEqual(result['at_risk'],2)
        self.assertEqual(result['conditional']['events'],1)
        self.assertEqual(result['conditional']['median'],5)
        self.assertEqual(result['conditional']['rmst'],5.)

    def test_pending_is_not_activity(self):
        run=trace()
        run.streak[5]=2
        out=landmark_summary([run],5,5)
        self.assertEqual(out['pending_confirmation'],1)
        self.assertEqual(out['screen_active'],0)
        self.assertEqual(out['at_risk'],1)

    def test_local_cluster_is_required_for_activity(self):
        run=trace()
        run.defector_fraction[5]=.1
        self.assertEqual(landmark_summary([run],5,5)['screen_active'],0)

    def test_no_survivors_is_not_zero_risk(self):
        out=landmark_summary([trace(2)],5,5)
        self.assertEqual(out['at_risk'],0)
        self.assertIsNone(out['conditional'])

    def test_insufficient_administrative_followup_rejected(self):
        with self.assertRaises(ValueError):
            landmark_summary([trace()],18,5)

    def test_paired_drift_and_event_at_window_end(self):
        out=paired_window_drift([trace(drift=-.01),trace(15)],(2,4),(13,15),100,42)
        self.assertEqual(out['n'],1)
        self.assertAlmostEqual(out['cooperation']['difference'],-.11)
        np.testing.assert_allclose(out['cooperation']['bootstrap95'],[-.11,-.11])

    def test_missing_sample_is_an_error(self):
        run=trace()
        run.times[5]=4
        with self.assertRaises(ValueError):
            landmark_summary([run],5,5)

class TraceValidation(unittest.TestCase):
    def test_early_technical_stop_is_not_administrative_censoring(self):
        from pathlib import Path
        import tempfile
        from analyze_nucleation_followup import read_trace
        from nucleation_runner import TRAJECTORY_COLUMNS
        config={'model':{'size':16},'protocol':{'t_max_sweeps':10,'sample_every':10,'stop_after_collapse':True},
                'collapse':{'confirm_sweeps':3}}
        summary={'run_id':'early','executed_sweeps':3,'event_observed':False,
                 't_confirm_B':None,'t_entry_B':None,'observed_time':10,'censor_time':10}
        with tempfile.TemporaryDirectory() as tmp:
            values=np.zeros((2,len(TRAJECTORY_COLUMNS))); values[:,0]=[0,3]
            np.savez(Path(tmp)/'trajectory.npz',values=values,columns=np.array(TRAJECTORY_COLUMNS))
            with self.assertRaises(ValueError):
                read_trace(config,summary,Path(tmp))
