"""Direction round analysis: pair joining, cluster bootstrap, clone pooling."""
import unittest

import numpy as np

from analyze_nucleation_direction import (_a1_calibration, _arm_magnitudes,
                                          _cluster_indices, _combine_clones,
                                          _direction_permutation_p,
                                          _magnitude_mother_means,
                                          _paired_magnitude_contrast,
                                          _spearman_bootstrap, direction_pairs,
                                          mother_rows)
from plot_nucleation_direction import _binned, _display_shift


def _copy(hit):
    return dict(hit_B=hit)


class DirectionJoins(unittest.TestCase):
    PREREG = dict(mothers=[dict(
        seed='0000000001', boundary_c_original=0.5,
        boundary_c_shift={'(2, 1)': 0.9, '(1, -2)': 0.4},
        J_pre_shift={'(2, 1)': -10.0, '(1, -2)': -1.0})])

    def test_arm_magnitudes_use_torus_reduction(self):
        campaign = dict(base=dict(model=dict(size=48)), arms=[
            'original',
            dict(arm='shifted', shift=[2, 1]),
            dict(arm='shifted', shift=[23, 42]),
            dict(arm='shifted', shift=[5, -17])])
        magnitudes = _arm_magnitudes(campaign)
        self.assertAlmostEqual(magnitudes['shifted(2, 1)'], 2.2361, places=3)
        self.assertAlmostEqual(magnitudes['shifted(23, 42)'], 23.7697, places=3)
        self.assertAlmostEqual(magnitudes['shifted(5, -17)'], 17.7205, places=3)
        self.assertEqual(magnitudes['original'], 0.0)

    def test_mother_rows_and_pair_joining(self):
        outcomes = {'data/nucleation/N02/x/seed-0000000001/states/g.npz': {
            'original': [_copy(True)] * 8 + [_copy(False)] * 8,
            'shifted(2, 1)': [_copy(False)] * 16,
            'shifted(1, -2)': [_copy(True)] * 16}}
        magnitudes = {'original': 0.0, 'shifted(2, 1)': 2.2361, 'shifted(1, -2)': 2.2361}
        rows = mother_rows(outcomes, magnitudes)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row['seed'], '0000000001')
        self.assertAlmostEqual(row['q_B']['original'], 0.5)
        self.assertEqual(row['hits']['shifted(2, 1)'], 0)
        self.assertAlmostEqual(row['deltas']['shifted(2, 1)'], -0.5)
        pairs = direction_pairs(rows, self.PREREG)
        by_arm = {pair['arm']: pair for pair in pairs}
        self.assertAlmostEqual(by_arm['shifted(2, 1)']['dB'], 0.4)
        self.assertAlmostEqual(by_arm['shifted(1, -2)']['dB'], -0.1)
        self.assertAlmostEqual(by_arm['shifted(2, 1)']['J_pre'], -10.0)
        self.assertAlmostEqual(by_arm['shifted(2, 1)']['delta_qB'], -0.5)

    def test_spearman_bootstrap_recovers_known_negative_relation(self):
        rng = np.random.default_rng(5)
        pairs = []
        for mother in range(12):
            base = rng.uniform(0.2, 0.8)
            for direction in range(4):
                pairs.append(dict(
                    seed=f'{mother:010d}', dB=rng.uniform(0.0, 0.4),
                    delta_qB=None, y_placeholder=None))
        for pair in pairs:
            pair['delta_qB'] = 0.1 - 0.5 * pair['dB'] + 0.05 * rng.normal()
        result = _spearman_bootstrap(pairs, 'dB', 'delta_qB', draws=400)
        self.assertLess(result['spearman'], -0.7)
        self.assertIsNotNone(result['bootstrap95'])
        self.assertLess(result['bootstrap95'][1], -0.3)
        self.assertEqual(result['n_mothers'], 12)
        self.assertEqual(result['n_pairs'], 48)

    def test_cluster_indices_preserve_draw_multiplicity(self):
        # drawn sequence [A, A, C] must replicate mother A's whole block twice,
        # not select the union of drawn mothers once (external review revision 3)
        groups = np.asarray([0, 0, 1, 1, 2, 2])
        indices = _cluster_indices(groups, [0, 0, 2])
        np.testing.assert_array_equal(indices, [0, 1, 0, 1, 4, 5])
        indices = _cluster_indices(groups, [1, 2, 1])
        np.testing.assert_array_equal(indices, [2, 3, 4, 5, 2, 3])

    def test_magnitude_mother_means_average_directions_per_mother(self):
        pairs = [
            dict(seed='0000000001', magnitude=2.0, delta_qB=0.1),
            dict(seed='0000000001', magnitude=2.0, delta_qB=0.3),
            dict(seed='0000000002', magnitude=2.0, delta_qB=-0.2),
            dict(seed='0000000002', magnitude=2.0, delta_qB=0.2),
            dict(seed='0000000001', magnitude=5.0, delta_qB=0.0),
        ]
        means = _magnitude_mother_means(pairs, 2.0)
        self.assertAlmostEqual(means['0000000001'], 0.2)
        self.assertAlmostEqual(means['0000000002'], 0.0)
        self.assertNotIn('0000000001', _magnitude_mother_means(pairs, 7.0))

    def test_paired_magnitude_contrast_pairs_by_mother(self):
        pairs = []
        for mother in ('0000000001', '0000000002'):
            pairs.append(dict(seed=mother, magnitude=2.0, delta_qB=0.1))
            pairs.append(dict(seed=mother, magnitude=17.0, delta_qB=-0.3))
        contrast = _paired_magnitude_contrast(pairs, 17.0, 2.0)
        self.assertEqual(contrast['n_mothers'], 2)
        self.assertAlmostEqual(contrast['mean'], -0.4)

    def test_direction_permutation_p_detects_exchangeability(self):
        identical = np.tile(np.asarray([[0.1], [0.1], [0.1], [0.1]]), (1, 12))
        self.assertAlmostEqual(_direction_permutation_p(identical, draws=200), 1.0)
        distinct = np.tile(np.asarray([[0.4], [0.1], [0.0], [-0.2]]), (1, 12))
        self.assertLess(_direction_permutation_p(distinct, draws=200), 0.05)

    def test_combine_clones_pools_counts_and_keeps_descriptors(self):
        first_path = 'data/nucleation/N02/x/seed-0000000001/states/g.npz'
        second_path = 'data/nucleation/N02/x/seed-0000000002/states/g.npz'
        outcomes = {first_path: {'original': [_copy(True)] * 6 + [_copy(False)] * 10},
                    second_path: {'original': [_copy(True)] * 12 + [_copy(False)] * 4}}
        row_list = [dict(mother_path=first_path, seed='0000000001', hits=20, n=32,
                         descriptors=dict(defectors=116))]
        combined = _combine_clones([outcomes, row_list])
        self.assertEqual(len(combined), 2)
        first = next(row for row in combined if row['mother_path'] == first_path)
        second = next(row for row in combined if row['mother_path'] == second_path)
        self.assertEqual(first['n'], 48)
        self.assertEqual(first['hits'], 26)
        self.assertAlmostEqual(first['q_B'], 26 / 48)
        self.assertEqual(first['descriptors'], dict(defectors=116))
        self.assertEqual(second['n'], 16)

    def test_binned_interval_respects_mother_clustering(self):
        # four mothers whose pairs share a common value; the cluster bootstrap
        # must reflect mother-level spread, not the within-bin pair count
        xs = np.arange(16, dtype=float)
        ys = np.repeat([0.0, 0.2, 0.4, 0.6], 4)
        mothers = np.repeat(['a', 'b', 'c', 'd'], 4)
        centres, means, intervals = _binned(xs, ys, mothers, bins=1, draws=400)
        self.assertAlmostEqual(means[0], 0.3)
        low, high = intervals[0]
        self.assertLess(low, 0.3)
        self.assertGreater(high, 0.3)
        self.assertGreater(high - low, 0.3)
        naive = float(np.std(ys, ddof=1) / np.sqrt(len(ys)))
        self.assertGreater(high - low, 4 * naive)

    def test_binned_degenerate_single_mother_bin_has_no_interval(self):
        xs = np.arange(8, dtype=float)
        ys = np.arange(8, dtype=float) * 0.01
        mothers = np.repeat(['only'], 8)
        centres, means, intervals = _binned(xs, ys, mothers, bins=1)
        self.assertEqual(len(means), 1)
        self.assertIsNone(intervals[0])

    def test_binned_identical_mothers_collapse_to_zero_width(self):
        xs = np.arange(12, dtype=float)
        ys = np.full(12, 0.25)
        mothers = np.repeat(['a', 'b', 'c'], 4)
        centres, means, intervals = _binned(xs, ys, mothers, bins=1, draws=200)
        self.assertAlmostEqual(means[0], 0.25)
        self.assertAlmostEqual(intervals[0][1] - intervals[0][0], 0.0, places=12)

    def test_display_shift_wraps_torus_displacements(self):
        self.assertEqual(_display_shift(23, 42, 48), (23, -6))
        self.assertEqual(_display_shift(2, 1, 48), (2, 1))
        self.assertEqual(_display_shift(-24, 24, 48), (-24, -24))
        self.assertEqual(_display_shift(24, 0, 48), (-24, 0))

    def test_a1_calibration_reports_uncalibrated_epsilon_and_slope_interval(self):
        rng = np.random.default_rng(11)
        rows = []
        for mother in range(8):
            for arm in range(3):
                j_pre = float(rng.uniform(-20, 0))
                rows.append(dict(seed=f'{mother:010d}', J_pre=j_pre,
                                 a1=2.0 * j_pre + 3.0,
                                 a1_clone_se=0.05))
        block = _a1_calibration(rows)
        self.assertAlmostEqual(block['slope'], 2.0, places=6)
        self.assertAlmostEqual(block['intercept'], 3.0, places=6)
        expected_epsilon = float(np.mean([row['a1'] - row['J_pre'] for row in rows]))
        self.assertAlmostEqual(block['epsilon_mean'], expected_epsilon, places=10)
        self.assertLessEqual(block['epsilon_bootstrap95'][0], expected_epsilon)
        self.assertLessEqual(expected_epsilon, block['epsilon_bootstrap95'][1])
        self.assertLessEqual(block['slope_bootstrap95'][0], 2.0)
        self.assertLessEqual(2.0, block['slope_bootstrap95'][1])
        # the OLS-with-intercept residual mean is zero by construction and must
        # never be read as unbiasedness evidence
        self.assertAlmostEqual(block['residual_mean'], 0.0, places=10)


if __name__ == '__main__':
    unittest.main()
