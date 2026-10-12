import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from research_common import research_root,start_run,finish_run,write_json,write_npz,verify_complete


class ResearchIO(unittest.TestCase):
    def test_isolation_transaction_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                research_root(Path(tmp)/'data'/'main'/'nucleation')
            root=research_root(Path(tmp)/'nucleation')
            staging,final=start_run(root,{'experiment_id':'N00'},3)
            self.assertFalse(final.exists())
            for filename in ['summary.json','provenance.json']:
                write_json(staging/filename,{'valid':True})
            for filename in ['trajectory.npz','events.npz','states/initial.npz','states/final.npz',
                             'clusters.npz','lineage.npz']:
                write_npz(staging/filename,{'resource':np.array([.12345678901234567])},{})
            finish_run(staging,final)
            self.assertTrue(verify_complete(final)['valid'])
            with self.assertRaises(FileExistsError):
                start_run(root,{'experiment_id':'N00'},3)
            (final/'summary.json').write_text('{}')
            with self.assertRaises(ValueError):
                verify_complete(final)
