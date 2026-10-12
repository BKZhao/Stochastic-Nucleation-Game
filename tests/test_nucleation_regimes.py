"""Physical outcome labels and all-trajectory collapse-time semantics."""
import unittest

from analyze_nucleation_regimes import collapse_half_time, resolved_majority


class FiniteTimeRegimes(unittest.TestCase):
    def test_unresolved_is_distinct_from_recovery(self):
        self.assertEqual(resolved_majority(dict(A=0, B=0, U=64)), 'U')
        self.assertEqual(resolved_majority(dict(A=64, B=0, U=0)), 'A')

    def test_weak_empirical_majority_is_not_resolved(self):
        self.assertIsNone(resolved_majority(dict(A=1, B=39, U=24)))
        self.assertEqual(resolved_majority(dict(A=0, B=40, U=24)), 'B')

    def test_ties_are_not_assigned_by_category_order(self):
        self.assertIsNone(resolved_majority(dict(A=32, B=32, U=0)))

    def test_invalid_counts_fail(self):
        for counts in [dict(A=0, B=0, U=0), dict(A=-1, B=64, U=1), dict(A=1, B=2)]:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                resolved_majority(counts)


class CollapseHalfTime(unittest.TestCase):
    def test_denominator_includes_noncollapsed_trajectories(self):
        self.assertIsNone(collapse_half_time(list(range(1, 32)), 64, 100))
        self.assertEqual(collapse_half_time(list(range(1, 40)), 64, 100), 32)

    def test_empirical_crossing_is_not_interpolated_median(self):
        self.assertEqual(collapse_half_time([1, 2, 90, 100], 4, 100), 2)

    def test_odd_ensemble_and_ties(self):
        self.assertEqual(collapse_half_time([9, 2, 2], 5, 10), 9)
        self.assertEqual(collapse_half_time([4, 4], 4, 10), 4)

    def test_horizon_event_counts(self):
        self.assertEqual(collapse_half_time([100]*32, 64, 100), 100)

    def test_no_events_has_no_measured_half_time(self):
        self.assertIsNone(collapse_half_time([], 64, 100))

    def test_invalid_times_fail(self):
        for times in [[101], [-1], [1.5], [1]*65]:
            with self.subTest(times=times), self.assertRaises(ValueError):
                collapse_half_time(times, 64, 100)


if __name__ == '__main__':
    unittest.main()
