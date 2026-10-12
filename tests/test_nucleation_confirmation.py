import unittest
from nucleation_survival import fisher_two_sided,risk_difference_interval


class ConfirmationStatistics(unittest.TestCase):
    def test_fisher_integer_probability_order(self):
        self.assertAlmostEqual(fisher_two_sided(3,4,1,4),34/70)
        self.assertAlmostEqual(fisher_two_sided(4,4,0,4),2/70)
        self.assertEqual(fisher_two_sided(0,256,0,256),1.)
        self.assertEqual(fisher_two_sided(256,256,256,256),1.)
        self.assertAlmostEqual(fisher_two_sided(8,16,0,16),fisher_two_sided(0,16,8,16))

    def test_difference_interval_includes_zero_count_uncertainty(self):
        interval=risk_difference_interval(0,256,0,256)
        self.assertLess(interval[0],0.)
        self.assertGreater(interval[1],0.)
        first=risk_difference_interval(8,16,0,16)
        reverse=risk_difference_interval(0,16,8,16)
        self.assertAlmostEqual(first[0],-reverse[1])
        self.assertAlmostEqual(first[1],-reverse[0])

    def test_invalid_binomial_counts(self):
        with self.assertRaises(ValueError):
            fisher_two_sided(4,3,1,4)

    def test_n03_uses_existing_backend_and_exact_damage(self):
        import json
        from pathlib import Path
        import tempfile
        import numpy as np
        from nucleation_runner import run_replica
        from research_common import config_hash
        config=json.loads(Path('configs/nucleation/n03_geometry_confirmation.json').read_text())['base']
        config['model']['size']=16
        config['damage']['shape']='compact'
        config['protocol'].update(t_max_sweeps=3,sample_every=1,snapshot_every=3)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'nucleation'
            result=run_replica((config,7,str(root),{'source_hash':'test'}))
            self.assertEqual(result['technical_status'],'complete')
            path=root/'N03'/config_hash(config)/'seed-0000000007'/'states'/'initial.npz'
            with np.load(path) as state:
                self.assertEqual(int((state['strategy']==0).sum()),16)
                self.assertEqual(float((1-state['resource']).sum()),16.)

    def test_modified_protocol_is_rejected_before_data_load(self):
        import json
        from pathlib import Path
        import tempfile
        from analyze_nucleation_confirmation import confirmation_report
        campaign=json.loads(Path('configs/nucleation/n03_geometry_confirmation.json').read_text())
        campaign['replicas']=32
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'changed.json'
            path.write_text(json.dumps(campaign))
            with self.assertRaisesRegex(ValueError,'pre-registered'):
                confirmation_report(path,Path(tmp)/'nucleation')
