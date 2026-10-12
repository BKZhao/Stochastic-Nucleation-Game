"""Analysis-layer contracts: recovery-transition calibration, binning, moment identities."""
import unittest

import numpy as np

from analyze_nucleation_interface import _binned_calibration, calibration_pairs
from lattice import build_square_lattice
from nucleation_interface import interface_observables

INTERFACE_PARAMS = dict(temptation=1.16, coupling=0.5, strength=1.0, selection_noise=0.1)


def synthetic_run(defectors_series, recovered_time, columns=7):
    values = np.zeros((len(defectors_series), columns), dtype=float)
    values[:, 1] = defectors_series
    values[:, 4] = 1.0  # arbitrary positive J predictor
    active = np.ones(len(defectors_series), dtype=bool)
    if recovered_time is not None:
        active[int(recovered_time):] = False
    return dict(seed=0, values=values, equilibrium=np.zeros((0, columns)),
                recovered_time=recovered_time, active=active, event=False, confirm=None)


class CalibrationPairs(unittest.TestCase):
    def test_final_transition_into_recovery_is_kept(self):
        # N_D = (2, 1, 0, 0): the transition 1 -> 0 enters recovery and must be kept
        run = synthetic_run([2, 1, 0, 0], recovered_time=2)
        predictor, response = calibration_pairs([run])
        self.assertEqual(response.tolist(), [-1.0, -1.0])
        self.assertEqual(predictor.tolist(), [1.0, 1.0])

    def test_post_recovery_transitions_are_dropped(self):
        run = synthetic_run([1, 0, 0, 0], recovered_time=1)
        _, response = calibration_pairs([run])
        self.assertEqual(response.tolist(), [-1.0])

    def test_unrecovered_run_keeps_all_transitions(self):
        run = synthetic_run([3, 2, 2, 1], recovered_time=None)
        _, response = calibration_pairs([run])
        self.assertEqual(response.tolist(), [-1.0, 0.0, -1.0])

    def test_time_windows(self):
        run = synthetic_run(list(range(10)), recovered_time=None)
        _, early = calibration_pairs([run], t_min=0, t_max=4)
        self.assertEqual(early.tolist(), [1.0] * 4)
        _, late = calibration_pairs([run], t_min=7)
        self.assertEqual(late.tolist(), [1.0] * 2)


class BinnedCalibration(unittest.TestCase):
    def test_counts_sum_to_observations(self):
        rng = np.random.default_rng(5)
        predictor = rng.normal(size=9973)
        response = rng.normal(size=9973)
        cells = _binned_calibration(predictor, response)
        self.assertEqual(sum(cell['count'] for cell in cells), 9973)

    def test_identical_values_counted_once(self):
        predictor = np.ones(100)
        response = np.arange(100.0)
        cells = _binned_calibration(predictor, response)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]['count'], 100)

    def test_tied_boundaries_not_double_counted(self):
        predictor = np.array([1., 1., 2., 3., 3., 3.])
        response = np.arange(6.0)
        cells = _binned_calibration(predictor, response, bins=3)
        self.assertEqual(sum(cell['count'] for cell in cells), 6)
        counts = sorted(cell['count'] for cell in cells)
        self.assertEqual(counts, [2, 4])

    def test_empty_input(self):
        self.assertIsNone(_binned_calibration(np.array([]), np.array([])))


class FirstVisitMoments(unittest.TestCase):
    def _moments(self, strategy, resource, size, nbr, deg):
        node_count = size * size
        first = second = 0.0
        for u in range(node_count):
            for k in range(deg[u]):
                v = nbr[u, k]
                if strategy[u] == strategy[v]:
                    continue
                kc_u = sum(strategy[nbr[u, kk]] for kk in range(deg[u]))
                kc_v = sum(strategy[nbr[v, kk]] for kk in range(deg[v]))

                def payoff(node, cooperators):
                    if strategy[node] == 1:
                        value = 1.0 + INTERFACE_PARAMS['coupling'] * INTERFACE_PARAMS['strength'] * (
                            2.0 * resource[node] - 1.0)
                        return (value if value > 0 else 0.0) * cooperators
                    return INTERFACE_PARAMS['temptation'] * cooperators

                Pu, Pv = payoff(u, kc_u), payoff(v, kc_v)
                dw = (Pu - Pv) / INTERFACE_PARAMS['selection_noise']
                dw = min(30.0, max(-30.0, dw))
                weight = 1.0 / (1.0 + np.exp(dw))
                first += weight if strategy[u] == 1 else -weight
                second += weight
        return first / (node_count * 4.0), second / (node_count * 4.0)

    def test_second_moment_identity(self):
        rng = np.random.default_rng(31)
        size = 10
        nbr, deg = build_square_lattice(size)
        for trial in range(3):
            strategy = rng.integers(0, 2, size * size).astype(np.int8)
            resource = rng.random(size * size)
            obs = interface_observables(strategy, resource, size, **INTERFACE_PARAMS)
            expected_first, expected_second = self._moments(strategy, resource, size, nbr, deg)
            self.assertAlmostEqual(obs['J'] / (size * size), expected_first, delta=1e-11)
            self.assertAlmostEqual(obs['interface_edges'] / (4.0 * size * size),
                                   expected_second, delta=1e-11)
            # Var(Z) = P/(4N) - (J/N)^2 stays non-negative for real states
            variance = obs['interface_edges'] / (4.0 * size * size) - (obs['J'] / (size * size)) ** 2
            self.assertGreaterEqual(variance, -1e-12)


if __name__ == '__main__':
    unittest.main()
