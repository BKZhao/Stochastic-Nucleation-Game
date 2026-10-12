"""Mechanism round: sensitivity weights, rearrangement arms, coupling-0 identity."""
import hashlib
import unittest

import numpy as np

from b_eco_unified import create_simulation_state, seed_numba, step_state
from nucleation_interface import (interface_observables, interface_sensitivity,
                                  resource_substep)
from nucleation_rearrange import (boundary_cooperator_mask, rearranged_resource,
                                  MECHANISM_ARMS)
from run_nucleation_fate import apply_arm

MODEL = dict(size=10, temptation=1.16, growth=0.1, depletion=0.5, coupling=0.5,
             strength=1.0, selection_noise=0.1, mutation=0.0, diffusion=0.4)
DIGEST = 'a' * 62 + 'ab'


def _compact_defector_state(size=10, block=((3, 3), (3, 4), (4, 3), (4, 4)), seed=7):
    state = create_simulation_state(size, seed, strategy_init='all_c', resource_init=0.7)
    grid = state.strategy.reshape(size, size)
    for cell in block:
        grid[cell] = 0
    rng = np.random.default_rng(seed)
    state.resource[:] = rng.uniform(0.1, 0.95, size * size)
    return state


def _pit_state(size=16, seed=9):
    """Compact 4x4 defector block inside an anisotropic resource pit.

    The angular modulation gives the boundary-cooperator band a realistic
    value spread; on real mothers this spread is what the rearrangement arms
    redistribute (about 0.7 on the 48 compact mothers).
    """
    state = create_simulation_state(size, seed, strategy_init='all_c', resource_init=0.9)
    grid = state.strategy.reshape(size, size)
    centre = size // 2 - 0.5
    for i in range(size // 2 - 2, size // 2 + 2):
        for j in range(size // 2 - 2, size // 2 + 2):
            grid[i, j] = 0
    yy, xx = np.mgrid[0:size, 0:size]
    dy, dx = yy - centre, xx - centre
    distance = np.sqrt(dy ** 2 + dx ** 2)
    theta = np.arctan2(dy, dx)
    pit = np.clip(1.0 - 0.9 * np.exp(-distance / 3.0)
                  * (1.3 + 0.7 * np.sin(theta + 1.0)), 0.02, 0.98)
    state.resource[:] = pit.reshape(-1)
    return state


def _fixed_weight_objective(strategy, original, arranged, model):
    """sum w_original * r~(arranged): the correct linear-prediction objective.

    The weights must stay those of the ORIGINAL post-substep field - the
    quantity dJ = -sum w (r~_arm - r~_original) - and must never be recomputed
    on the arranged field.
    """
    size = model['size']
    post_original = resource_substep(strategy, original, size, model['growth'],
                                     model['depletion'], model['diffusion'])
    sensitivity = interface_sensitivity(strategy, post_original, size,
                                        model['temptation'], model['coupling'],
                                        model['strength'], model['selection_noise'])
    post_arm = resource_substep(strategy, arranged, size, model['growth'],
                                model['depletion'], model['diffusion'])
    return float((sensitivity['weights'] * post_arm).sum())


class InterfaceSensitivity(unittest.TestCase):
    def test_weights_match_numerical_derivative_of_J(self):
        state = _compact_defector_state()
        size = MODEL['size']
        sensitivity = interface_sensitivity(
            state.strategy, state.resource, size, MODEL['temptation'],
            MODEL['coupling'], MODEL['strength'], MODEL['selection_noise'])
        weights = sensitivity['weights']
        threshold = 1e-3 * weights.max()
        base = interface_observables(state.strategy, state.resource, size,
                                     MODEL['temptation'], MODEL['coupling'],
                                     MODEL['strength'], MODEL['selection_noise'])['J']
        checked = 0
        for index in np.flatnonzero(weights > threshold):
            epsilon = 1e-5
            perturbed = state.resource.copy()
            perturbed[index] += epsilon
            shifted = interface_observables(
                state.strategy, perturbed, size, MODEL['temptation'],
                MODEL['coupling'], MODEL['strength'], MODEL['selection_noise'])['J']
            linear = -weights[index] * epsilon
            self.assertAlmostEqual(shifted - base, linear,
                                   delta=max(1e-11, 1e-3 * abs(linear)))
            checked += 1
        self.assertGreater(checked, 0)

    def test_weights_vanish_off_boundary_cooperators(self):
        state = _compact_defector_state()
        size = MODEL['size']
        sensitivity = interface_sensitivity(
            state.strategy, state.resource, size, MODEL['temptation'],
            MODEL['coupling'], MODEL['strength'], MODEL['selection_noise'])
        weights = sensitivity['weights'].reshape(size, size)
        self.assertEqual(weights[state.strategy.reshape(size, size) == 0].max(), 0.0)
        mask = boundary_cooperator_mask(state.strategy, size)
        self.assertGreater(weights[mask].sum(), 0.0)
        self.assertEqual(weights[~mask].sum(), 0.0)


class RearrangementArms(unittest.TestCase):
    def test_multiset_total_and_boundary_mean_invariants(self):
        state = _compact_defector_state()
        size = MODEL['size']
        mask = boundary_cooperator_mask(state.strategy, size)
        original = state.resource.copy()
        for arm in MECHANISM_ARMS:
            arranged = rearranged_resource(state.strategy, original, MODEL, arm, DIGEST)
            self.assertAlmostEqual(arranged.sum(), original.sum(), places=12)
            np.testing.assert_array_equal(np.sort(arranged), np.sort(original))
            self.assertAlmostEqual(arranged.reshape(size, size)[mask].mean(),
                                   original.reshape(size, size)[mask].mean(), places=12)

    def test_protect_maximises_and_harm_minimises_weighted_post_sum(self):
        # the linear objective sum w_orig * r~(arrangement) decomposes exactly
        # as sum W_u phi(r_u) over arrangements of the boundary multiset, so
        # the aligned sort is the maximum and the anti-aligned sort the minimum
        # by the rearrangement inequality - the original arrangement lies
        # between the two
        state = _compact_defector_state()
        original = state.resource.copy()
        reference = _fixed_weight_objective(state.strategy, original, original, MODEL)
        protect = _fixed_weight_objective(
            state.strategy, original,
            rearranged_resource(state.strategy, original, MODEL,
                                'interface_protect', DIGEST), MODEL)
        harm = _fixed_weight_objective(
            state.strategy, original,
            rearranged_resource(state.strategy, original, MODEL,
                                'interface_harm', DIGEST), MODEL)
        self.assertGreaterEqual(protect, reference)
        self.assertLessEqual(harm, reference)
        self.assertGreater(protect, harm)

    def test_exact_J_pre_orders_protect_below_harm(self):
        # on a pit-like interface with a realistic boundary value spread the
        # exact J_pre orders the arms as designed; on a fully random toy the
        # linear objective can misorder the exact response, which is exactly
        # why the preregistration anchors every arm with its exact J_pre
        state = _pit_state()
        size = state.size
        model = dict(MODEL, size=size)
        original = state.resource.copy()

        def j_pre(resource):
            post = resource_substep(state.strategy, resource, size,
                                    model['growth'], model['depletion'],
                                    model['diffusion'])
            return interface_observables(state.strategy, post, size,
                                         model['temptation'], model['coupling'],
                                         model['strength'],
                                         model['selection_noise'])['J']

        base = j_pre(original)
        protect = j_pre(rearranged_resource(state.strategy, original, model,
                                            'interface_protect', DIGEST))
        harm = j_pre(rearranged_resource(state.strategy, original, model,
                                         'interface_harm', DIGEST))
        defector = j_pre(rearranged_resource(state.strategy, original, model,
                                             'defector_shuffle', DIGEST))
        self.assertLess(protect, base)
        self.assertGreater(harm, base)
        self.assertLess(protect, harm)
        self.assertLess(abs(defector - base), harm - protect)

    def test_defector_shuffle_linear_prediction_is_diffusion_leakage(self):
        state = _pit_state()
        size = state.size
        model = dict(MODEL, size=size)
        original = state.resource.copy()

        def linear(arrangement, diffusion):
            post = resource_substep(state.strategy, arrangement, size,
                                    model['growth'], model['depletion'], diffusion)
            post_original = resource_substep(state.strategy, original, size,
                                             model['growth'], model['depletion'],
                                             diffusion)
            sensitivity = interface_sensitivity(
                state.strategy, post_original, size, model['temptation'],
                model['coupling'], model['strength'], model['selection_noise'])
            return float(-(sensitivity['weights'] * (post - post_original)).sum())

        leakage = linear(rearranged_resource(state.strategy, original, model,
                                             'defector_shuffle', DIGEST),
                         model['diffusion'])
        protect = linear(rearranged_resource(state.strategy, original, model,
                                             'interface_protect', DIGEST),
                         model['diffusion'])
        # the defector-side permutation reaches boundary cooperators only
        # through one diffusion substep, so its linear prediction is clearly
        # weaker than a boundary rearrangement of the same value multiset
        self.assertLess(abs(leakage), 0.5 * abs(protect))
        # with diffusion switched off the D-side permutation cannot reach any
        # payoff-reading node within the substep and the prediction vanishes
        self.assertEqual(linear(rearranged_resource(state.strategy, original,
                                                    model, 'defector_shuffle',
                                                    DIGEST), 0.0), 0.0)

    def test_shuffle_is_deterministic_and_digest_dependent(self):
        state = _compact_defector_state()
        original = state.resource.copy()
        first = rearranged_resource(state.strategy, original, MODEL,
                                    'interface_shuffle', DIGEST)
        second = rearranged_resource(state.strategy, original, MODEL,
                                     'interface_shuffle', DIGEST)
        np.testing.assert_array_equal(first, second)
        other = rearranged_resource(state.strategy, original, MODEL,
                                    'interface_shuffle', 'b' * 62 + 'cd')
        self.assertFalse(np.array_equal(first, other))

    def test_degenerate_interface_returns_unchanged_field(self):
        degenerate_model = dict(MODEL, size=6)
        state = create_simulation_state(6, 3, strategy_init='all_c', resource_init=0.5)
        rng = np.random.default_rng(5)
        state.resource[:] = rng.uniform(0.2, 0.9, 36)
        for arm in MECHANISM_ARMS:
            arranged = rearranged_resource(state.strategy, state.resource,
                                           degenerate_model, arm, DIGEST)
            np.testing.assert_array_equal(arranged, state.resource)


class CouplingZeroIdentity(unittest.TestCase):
    def test_strategy_trajectories_identical_across_arms_at_zero_coupling(self):
        size, sweeps = 10, 60
        model = dict(MODEL)
        run_model = dict(MODEL)
        run_model['coupling'] = 0.0
        run_model.pop('size')
        digests = {}
        resource_digests = set()
        for arm in ('original',) + MECHANISM_ARMS:
            state = _compact_defector_state(size=size, seed=11)
            apply_arm(state, arm, [0, 0], MODEL, DIGEST)
            seed_numba(2026)
            trajectory = []
            for sweep in range(sweeps + 1):
                if sweep:
                    step_state(state, **run_model)
                if sweep % 20 == 0:
                    trajectory.append(hashlib.sha256(
                        np.ascontiguousarray(state.strategy).tobytes()).hexdigest())
            digests[arm] = trajectory
            resource_digests.add(hashlib.sha256(
                np.ascontiguousarray(
                    np.asarray(state.resource, dtype=np.float64)).tobytes()).hexdigest())
        for arm, trajectory in digests.items():
            self.assertEqual(trajectory, digests['original'])
        self.assertGreater(len(resource_digests), 1)


if __name__ == '__main__':
    unittest.main()
