"""Confirmatory held-out models: declared ridge objective, solver correctness, baselines."""
import unittest

import numpy as np

from analyze_nucleation_confirmatory import (
    _arm_runs_digest, _calibration_rows, fit_and_evaluate, heterogeneity_diagnostic,
    penalised_gradient, penalised_objective, ridge_logistic, source_baseline)
from research_common import canonical_json


def synthetic(rows=30, seed=20261006):
    """Correlated features with fractional outcomes, as in the eta-entanglement setting."""
    rng = np.random.default_rng(seed)
    design = rng.normal(size=(rows, 3))
    design[:, 2] += 0.8 * design[:, 0]
    latent = 1.2 * design[:, 0] - 0.7 * design[:, 1] + 0.4 * design[:, 2]
    return design, 1.0 / (1.0 + np.exp(-latent))


def _reference_optimiser(design, outcome, penalty=1.0, iterations=3000, tolerance=1e-9):
    """Backtracking gradient descent on the declared objective (independent of the solver)."""
    weights = np.zeros(design.shape[1] + 1)
    for _ in range(iterations):
        gradient = penalised_gradient(weights, design, outcome, penalty)
        if np.max(np.abs(gradient)) < tolerance:
            break
        slope = float(gradient @ gradient)
        step_size = 1.0
        current = penalised_objective(weights, design, outcome, penalty)
        while step_size > 1e-18 and penalised_objective(
                weights - step_size * gradient, design, outcome, penalty) \
                > current - 1e-4 * step_size * slope:
            step_size *= 0.5
        weights -= step_size * gradient
    return weights


class RidgeLogisticMath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design, cls.outcome = synthetic()

    def test_gradient_matches_objective_by_finite_differences(self):
        weights = np.random.default_rng(11).normal(size=4)
        numeric = []
        for index in range(4):
            step = 1e-6
            plus, minus = weights.copy(), weights.copy()
            plus[index] += step
            minus[index] -= step
            numeric.append((penalised_objective(plus, self.design, self.outcome)
                            - penalised_objective(minus, self.design, self.outcome)) / (2 * step))
        np.testing.assert_allclose(penalised_gradient(weights, self.design, self.outcome),
                                   numeric, rtol=1e-5, atol=1e-7)

    def test_solution_is_stationary_point_of_declared_objective(self):
        # The pre-revision update omitted the ridge term from the gradient while
        # keeping it in the Hessian, leaving a non-zero objective gradient here.
        weights, info = ridge_logistic(self.design, self.outcome)
        self.assertLess(info['max_abs_objective_gradient'], 1e-8)
        self.assertLess(info['final_max_step'], 1e-9)

    def test_solution_is_converged_not_iteration_truncated(self):
        short, _ = ridge_logistic(self.design, self.outcome, iterations=8)
        long, _ = ridge_logistic(self.design, self.outcome, iterations=200)
        np.testing.assert_allclose(short, long, atol=1e-9)

    def test_agrees_with_independent_optimiser(self):
        # As in the pipeline, the reference optimiser sees standardised features;
        # gradient descent from zero on the declared objective must land on the
        # same optimum as the Newton solver (the objective is strictly convex).
        standard = (self.design - self.design.mean(axis=0)) / self.design.std(axis=0)
        weights, _ = ridge_logistic(standard, self.outcome)
        reference = _reference_optimiser(standard, self.outcome)
        np.testing.assert_allclose(weights, reference, atol=1e-7)

    def test_agrees_with_scipy_lbfgsb_if_available(self):
        try:
            from scipy.optimize import minimize
        except ImportError:
            self.skipTest('scipy not installed')
        weights, _ = ridge_logistic(self.design, self.outcome)
        result = minimize(lambda w: penalised_objective(w, self.design, self.outcome),
                          np.zeros(4), jac=lambda w: penalised_gradient(
                              w, self.design, self.outcome),
                          method='L-BFGS-B', options=dict(ftol=1e-15, gtol=1e-12))
        np.testing.assert_allclose(weights, result.x, atol=1e-6)

    def test_penalty_actually_shrinks_coefficients(self):
        weak, _ = ridge_logistic(self.design, self.outcome, penalty=0.1)
        strong, _ = ridge_logistic(self.design, self.outcome, penalty=1000.0)
        self.assertLess(np.linalg.norm(strong[1:]), np.linalg.norm(weak[1:]))
        self.assertLess(np.linalg.norm(strong[1:]), 0.05)

    def test_intercept_is_not_penalised(self):
        centred, _ = ridge_logistic(self.design - self.design.mean(axis=0), self.outcome,
                                    penalty=1e6)
        outcome_mean = self.outcome.mean()
        self.assertAlmostEqual(float(centred[0]), float(np.log(outcome_mean / (1 - outcome_mean))),
                               delta=0.05)


class SourceBaselineAndDiagnostics(unittest.TestCase):
    def test_source_baseline_uses_training_pooled_group_means(self):
        training = [dict(group='a', q_B=0.25, n=16), dict(group='a', q_B=0.5, n=16),
                    dict(group='b', q_B=1.0, n=16), dict(group='b', q_B=0.75, n=16)]
        heldout = [dict(group='a'), dict(group='b'), dict(group='a')]
        prediction, means = source_baseline(training, heldout)
        np.testing.assert_allclose(prediction, [0.375, 0.875, 0.375])
        self.assertAlmostEqual(means['group_means']['a'], 0.375)
        self.assertAlmostEqual(means['group_means']['b'], 0.875)
        self.assertAlmostEqual(means['overall_mean'], 0.625)

    def test_heterogeneity_diagnostic_decomposes_known_variance(self):
        rows = [dict(q_B=value, n=16) for value in (0.25, 0.5, 0.75)]
        diagnostic = heterogeneity_diagnostic(rows)
        self.assertAlmostEqual(diagnostic['sample_variance'], 0.0625)
        self.assertAlmostEqual(diagnostic['branch_noise'], 0.0138888888888889)
        self.assertAlmostEqual(diagnostic['excess_variance'], 0.0486111111111111)

    def test_heterogeneous_mothers_are_flagged_and_homogeneous_are_not(self):
        spread = [dict(q_B=value, n=16) for value in (0.125, 0.375, 0.625, 0.875)]
        flat = [dict(q_B=0.5, n=16)] * 20
        self.assertLess(heterogeneity_diagnostic(spread)['homogeneity_bootstrap_p'], 0.01)
        self.assertGreater(heterogeneity_diagnostic(flat)['homogeneity_bootstrap_p'], 0.1)

    def test_arm_runs_digest_is_order_independent_and_binding(self):
        entries = [('N06/a/seed-1', 'hash-a'), ('N06/a/seed-2', 'hash-b')]
        digest = _arm_runs_digest(entries)
        self.assertEqual(digest['n_runs'], 2)
        self.assertEqual(digest['complete_digest'],
                         _arm_runs_digest(list(reversed(entries)))['complete_digest'])
        self.assertNotEqual(digest['complete_digest'],
                            _arm_runs_digest([('N06/a/seed-1', 'hash-a'),
                                              ('N06/a/seed-2', 'hash-c')])['complete_digest'])

    def test_calibration_rows_join_residual_and_ratio_gating(self):
        by_seed = {
            '0187841521': dict(group='compact_16', seed='0187841521', J_pre_shift={
                '(2, 1)': -0.5, '(5, 31)': -17.5}),
            '0223668452': dict(group='compact_16', seed='0223668452', J_pre_shift={
                '(2, 1)': -1.5, '(5, 31)': -13.9}),
        }
        changes = {
            ('data/nucleation/N02/x/seed-0187841521/states/growth-0.05.npz',
             'shifted(2, 1)'): [-1.0, -3.0, -2.0, 0.0],
            ('data/nucleation/N02/x/seed-0187841521/states/growth-0.05.npz',
             'shifted(5, 31)'): [-20.0, -18.0],
            ('data/nucleation/N02/x/seed-0223668452/states/growth-0.05.npz',
             'shifted(5, 31)'): [-16.0, -12.0],
            ('data/nucleation/N02/x/seed-0999999999/states/growth-0.05.npz',
             'shifted(5, 31)'): [-4.0],
        }
        rows = _calibration_rows(by_seed, changes)
        self.assertEqual(len(rows), 3)  # unregistered seed is dropped
        by_key = {(row['seed'], row['arm']): row for row in rows}
        small = by_key[('0187841521', 'shifted(2, 1)')]
        self.assertAlmostEqual(small['j_pre_predicted'], -0.5)
        self.assertAlmostEqual(small['a1_observed'], -1.5)
        self.assertAlmostEqual(small['residual'], -1.0)
        self.assertIsNone(small['ratio'])  # |J_pre| < 1: ratio unstable, not reported
        self.assertAlmostEqual(small['a1_clone_se'], float(np.std([-1.0, -3.0, -2.0, 0.0],
                                                                 ddof=1) / 2.0))
        large = by_key[('0187841521', 'shifted(5, 31)')]
        self.assertAlmostEqual(large['ratio'], -19.0 / -17.5)
        self.assertAlmostEqual(large['residual'], -19.0 - (-17.5))
        canonical_json(rows)


class FitAndEvaluateExport(unittest.TestCase):
    def test_exports_source_baseline_within_group_and_json_safe_output(self):
        def rows(count, group, seed):
            rng = np.random.default_rng(seed)
            return [dict(group=group, q_B=rng.random(), n=16, descriptors=dict(
                defectors=float(rng.integers(4, 30)),
                interface_edges=float(rng.integers(10, 90)),
                n_clusters=float(rng.integers(1, 8)),
                resource_mean=float(rng.uniform(0.2, 0.9)),
                eta_mean=float(rng.uniform(0.0, 0.3)))) for _ in range(count)]

        training = rows(10, 'g1', 1) + rows(10, 'g2', 2)
        heldout = rows(9, 'g1', 3) + rows(8, 'g2', 4)
        results, intervals, predictions, within, heterogeneity = fit_and_evaluate(training, heldout)
        self.assertEqual(set(results), {'M0', 'M1', 'M2', 'SOURCE'})
        self.assertIn('group_means', results['SOURCE'])
        self.assertIn('solver', results['M2'])
        self.assertIn('standardisation', results['M2'])
        self.assertIn('pooled', intervals['M1_minus_SOURCE'])
        self.assertIn('within_g1', intervals['M1_minus_SOURCE'])
        self.assertIn('within_g2', intervals['M1_minus_SOURCE'])
        self.assertIn('mse_qB', results['M1'])
        self.assertEqual(set(within), {'g1', 'g2'})
        self.assertEqual(set(heterogeneity), {'g1', 'g2'})
        # SOURCE is constant within a group: ranking is undefined there, not NaN.
        self.assertIsNone(within['g1']['SOURCE']['spearman'])
        self.assertIsNotNone(within['g1']['SOURCE']['mse_qB'])
        canonical_json(dict(models=results, deltas=intervals, within_group=within,
                            heterogeneity=heterogeneity))


if __name__ == '__main__':
    unittest.main()
