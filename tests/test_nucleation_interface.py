"""Interface diagnostics: kernel-identical resource substep, J identity, runner isolation."""
import json
import math
from pathlib import Path
import tempfile
import unittest
from copy import deepcopy

import numpy as np
from numba import njit

from b_eco_unified import initialize_state, seed_numba
from lattice import build_square_lattice
from nucleation_interface import (equilibrium_observables, interface_observables,
                                  pit_observables, resource_equilibrium,
                                  resource_substep)
from nucleation_runner import run_replica, validate_config
from research_common import config_hash

INTERFACE_PARAMS = dict(temptation=1.16, coupling=0.5, strength=1.0, selection_noise=0.1)
GROWTH, DEPLETION, DIFFUSION = 0.1, 0.5, 0.4


@njit(cache=True)
def _kernel_resource_block(nbr, deg, s, Rf, sg, delta, Ddiff):
    """Verbatim copy of the resource block of b_eco_unified._step (verification oracle)."""
    N = s.shape[0]
    for i in range(N):
        si = s[i]
        Rf[i] = Rf[i] + sg * si * (1.0 - Rf[i]) - delta * (1 - si) * Rf[i]
        if Rf[i] < 0:
            Rf[i] = 0.0
        if Rf[i] > 1:
            Rf[i] = 1.0
    if Ddiff > 0:
        Rnew = Rf.copy()
        for i in range(N):
            d = deg[i]
            if d == 0:
                continue
            avg = 0.0
            for k in range(d):
                avg += Rf[nbr[i, k]]
            avg /= d
            Rnew[i] = Rf[i] + Ddiff * (avg - Rf[i])
            if Rnew[i] < 0:
                Rnew[i] = 0.0
            if Rnew[i] > 1:
                Rnew[i] = 1.0
        for i in range(N):
            Rf[i] = Rnew[i]
    return Rf


def expected_visit_change(strategy, resource, size, nbr, deg, params):
    """Brute-force expected defector-count change of one random node visit."""
    total = 0.0
    node_count = size * size
    for u in range(node_count):
        for k in range(deg[u]):
            v = nbr[u, k]
            if strategy[u] == strategy[v]:
                continue
            kc_u = 0
            for kk in range(deg[u]):
                kc_u += strategy[nbr[u, kk]]
            kc_v = 0
            for kk in range(deg[v]):
                kc_v += strategy[nbr[v, kk]]
            def payoff(node, cooperators):
                if strategy[node] == 1:
                    value = 1.0 + params['coupling'] * params['strength'] * (2.0 * resource[node] - 1.0)
                    return (value if value > 0 else 0.0) * cooperators
                return INTERFACE_PARAMS['temptation'] * cooperators
            Pu = payoff(u, kc_u)
            Pv = payoff(v, kc_v)
            dw = (Pu - Pv) / params['selection_noise']
            if dw > 30:
                dw = 30
            if dw < -30:
                dw = -30
            weight = 1.0 / (1.0 + math.exp(dw))
            total += weight if strategy[u] == 1 else -weight
    return total / (node_count * 4.0)


class ResourceSubstep(unittest.TestCase):
    def test_matches_kernel_block_bitwise(self):
        rng = np.random.default_rng(11)
        size = 12
        nbr, deg = build_square_lattice(size)
        for trial, diffusion in enumerate((0.0, 0.15, 0.4, 0.5)):
            with self.subTest(diffusion=diffusion):
                strategy = rng.integers(0, 2, size * size).astype(np.int8)
                resource = rng.random(size * size)
                kernel = resource.copy()
                _kernel_resource_block(nbr, deg, strategy, kernel, GROWTH,
                                       DEPLETION, diffusion)
                replicated = resource_substep(strategy, resource, size, GROWTH,
                                              DEPLETION, diffusion)
                np.testing.assert_array_equal(replicated, kernel)
                # inputs untouched
                np.testing.assert_array_equal(resource, resource)

    def test_extreme_resource_corners(self):
        size = 8
        nbr, deg = build_square_lattice(size)
        strategy = np.zeros(size * size, np.int8)
        strategy[::2] = 1
        for value in (0.0, 1.0):
            resource = np.full(size * size, value)
            kernel = resource.copy()
            _kernel_resource_block(nbr, deg, strategy, kernel, 0.2, 0.7, 0.4)
            replicated = resource_substep(strategy, resource, size, 0.2, 0.7, 0.4)
            np.testing.assert_array_equal(replicated, kernel)

    def test_equilibrium_is_unique_fixed_point(self):
        size = 10
        strategy = np.ones(size * size, np.int8)
        strategy[37] = 0
        strategy[64] = 0
        low, sweeps_low = resource_equilibrium(strategy, np.zeros(size * size), size,
                                               GROWTH, DEPLETION, 0.4)
        high, sweeps_high = resource_equilibrium(strategy, np.ones(size * size), size,
                                                 GROWTH, DEPLETION, 0.4)
        self.assertLessEqual(sweeps_low, 500)
        np.testing.assert_allclose(low, high, atol=1e-9)
        again = resource_substep(strategy, low, size, GROWTH, DEPLETION, 0.4)
        np.testing.assert_allclose(again, low, atol=1e-12)
        self.assertLessEqual(sweeps_high, 500)

    def test_all_cooperator_equilibrium_is_full_resource(self):
        size = 6
        strategy = np.ones(size * size, np.int8)
        equilibrium, _ = resource_equilibrium(strategy, np.zeros(size * size), size,
                                              GROWTH, 0.0, 0.4)
        np.testing.assert_allclose(equilibrium, 1.0, atol=1e-12)


class InterfaceDiagnostic(unittest.TestCase):
    def test_J_equals_expected_first_visit_change(self):
        rng = np.random.default_rng(23)
        size = 12
        nbr, deg = build_square_lattice(size)
        for trial in range(3):
            strategy = rng.integers(0, 2, size * size).astype(np.int8)
            resource = rng.random(size * size)
            obs = interface_observables(strategy, resource, size, **INTERFACE_PARAMS)
            expected = expected_visit_change(strategy, resource, size, nbr, deg, INTERFACE_PARAMS)
            self.assertAlmostEqual(obs['J'] / (size * size), expected, delta=1e-11)
            self.assertEqual(obs['defectors'], int((strategy == 0).sum()))

    def test_defector_interface_edges_and_uniform_states(self):
        size = 6
        strategy = np.ones(size * size, np.int8)
        obs = interface_observables(strategy, np.ones(size * size), size, **INTERFACE_PARAMS)
        self.assertEqual(obs['interface_edges'], 0)
        self.assertEqual(obs['J'], 0.0)
        self.assertIsNone(obs['boundary_c_mean'])
        strategy = np.zeros(size * size, np.int8)
        obs = interface_observables(strategy, np.ones(size * size), size, **INTERFACE_PARAMS)
        self.assertEqual(obs['interface_edges'], 0)
        self.assertEqual(obs['J'], 0.0)
        strategy = np.zeros(size * size, np.int8)
        strategy[14] = 1
        obs = interface_observables(strategy, np.ones(size * size), size, **INTERFACE_PARAMS)
        self.assertEqual(obs['interface_edges'], 4)
        self.assertEqual(obs['boundary_c_mean'], 1.0)

    def test_equilibrium_observables_report_adapted_field(self):
        size = 8
        strategy = np.ones(size * size, np.int8)
        strategy[27] = 0
        obs = equilibrium_observables(strategy, np.ones(size * size), size,
                                      **INTERFACE_PARAMS, growth=GROWTH,
                                      depletion=DEPLETION, diffusion=0.4)
        self.assertGreater(obs['equilibrium_sweeps'], 1)
        self.assertLess(obs['resource_mean_eq'], 1.0)
        self.assertEqual(obs['defectors'], 1)


class PitObservables(unittest.TestCase):
    @staticmethod
    def _periodic_distance_grid(size):
        """Independent periodic distance-to-origin via explicit offset enumeration."""
        rows, cols = np.indices((size, size))
        best = np.full((size, size), np.inf)
        for row_shift in (-size, 0, size):
            for col_shift in (-size, 0, size):
                shifted_r = (rows + row_shift).astype(float)
                shifted_c = (cols + col_shift).astype(float)
                best = np.minimum(best, np.sqrt(shifted_r ** 2 + shifted_c ** 2))
        return best

    def test_pit_profile_half_recovery_on_linear_ramp(self):
        size = 16
        strategy = np.ones((size, size), dtype=np.int8)
        strategy[0, 0] = 0  # single defector at the ramp origin
        distance = self._periodic_distance_grid(size)
        resource = 0.1 + 0.8 * np.clip(distance / 3.0, 0.0, 1.0)
        observables = pit_observables(strategy.reshape(-1), resource.reshape(-1), size)
        self.assertEqual(observables['defectors'], 1)
        self.assertAlmostEqual(observables['pit_floor'], 0.1)
        self.assertAlmostEqual(observables['background'], 0.9)
        # the ramp crosses halfway at distance 2; bin 2 must be the first above it
        self.assertEqual(observables['half_recovery_distance'], 2)
        self.assertLess(observables['pit_profile']['1'], observables['pit_profile']['2'])
        self.assertLess(observables['pit_profile']['2'], observables['pit_profile']['3'])

    def test_periodic_wrap_puts_diagonal_corner_in_bin_one(self):
        size = 8
        strategy = np.ones((size, size), dtype=np.int8)
        strategy[0, 0] = 0
        resource = np.full((size, size), 0.9)
        resource[7, 7] = 0.2  # wraps to distance sqrt(2) from the defector
        observables = pit_observables(strategy.reshape(-1), resource.reshape(-1), size,
                                       max_distance=4)
        # bin 1: four axis neighbours and four diagonal neighbours (one at 0.2)
        self.assertAlmostEqual(observables['pit_profile']['1'], (7 * 0.9 + 0.2) / 8)

    def test_no_defectors_gives_empty_pit(self):
        size = 8
        strategy = np.ones(size * size, dtype=np.int8)
        resource = np.full(size * size, 0.5)
        observables = pit_observables(strategy, resource, size)
        self.assertEqual(observables['defectors'], 0)
        self.assertIsNone(observables['pit_floor'])
        self.assertIsNone(observables['half_recovery_distance'])

    def test_observables_leave_inputs_untouched(self):
        rng = np.random.default_rng(3)
        size = 10
        strategy = (rng.random(size * size) < 0.2).astype(np.int8)
        resource = rng.random(size * size)
        strategy_before, resource_before = strategy.copy(), resource.copy()
        pit_observables(strategy, resource, size)
        np.testing.assert_array_equal(strategy, strategy_before)
        np.testing.assert_array_equal(resource, resource_before)


class RunnerIntegration(unittest.TestCase):
    def base_config(self):
        campaign = json.loads(Path('configs/nucleation/benchmark.json').read_text())
        config = campaign['base']
        config['experiment_id'] = 'N00'
        config['model']['size'] = 12
        config['model']['mutation'] = 0.
        config['protocol'].update(t_max_sweeps=6, sample_every=6, snapshot_every=6)
        config['damage'] = dict(count=4, shape='random', resource_value=0.)
        return config

    def test_interface_observation_leaves_physics_untouched(self):
        base = self.base_config()
        tracked = dict(deepcopy(base), interface=dict(equilibrium_times=[0, 3]))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'nucleation'
            plain = run_replica((deepcopy(base), 11, str(root), {'source_hash': 'test'}))
            observed = run_replica((tracked, 11, str(root), {'source_hash': 'test'}))
            self.assertEqual(plain['final_cooperation'], observed['final_cooperation'])
            self.assertEqual(plain['final_resource'], observed['final_resource'])
            self.assertEqual(plain['t_confirm_B'], observed['t_confirm_B'])
            plain_final = root / 'N00' / config_hash(base) / 'seed-0000000011' / 'states' / 'final.npz'
            tracked_final = root / 'N00' / config_hash(tracked) / 'seed-0000000011' / 'states' / 'final.npz'
            with np.load(plain_final) as one, np.load(tracked_final) as two:
                np.testing.assert_array_equal(one['strategy'], two['strategy'])
                np.testing.assert_array_equal(one['resource'], two['resource'])
            directory = root / 'N00' / config_hash(tracked) / 'seed-0000000011'
            with np.load(directory / 'interface.npz') as interface:
                values = interface['values']
                self.assertEqual(values.shape[0], 7)  # sweeps 0..6 inclusive
                self.assertEqual(values[0, 0], 0.0)
                self.assertEqual(values[-1, 0], 6.0)
                equilibrium = interface['equilibrium_values']
                self.assertEqual(equilibrium[:, 0].tolist(), [0.0, 3.0])
                self.assertTrue(np.isfinite(values[:, 3]).all())
            self.assertFalse((root / 'N00' / config_hash(base)
                              / 'seed-0000000011' / 'interface.npz').exists())

    def test_invalid_interface_sections_rejected(self):
        base = self.base_config()
        for section in (dict(equilibrium_times=[3, 3]), dict(equilibrium_times=[-1]),
                        dict(equilibrium_times=[9]), dict(equilibrium_times='all'),
                        dict(equilibrium_times=[0], extra=1), [0, 1]):
            config = deepcopy(base)
            config['interface'] = section
            with self.subTest(section=section):
                with self.assertRaises(ValueError):
                    validate_config(config)

    def test_empty_equilibrium_times_allowed(self):
        base = self.base_config()
        base['interface'] = dict(equilibrium_times=[])
        validate_config(base)

    def test_equilibrium_seed_field_is_fixed_point(self):
        base = self.base_config()
        base['damage'] = dict(count=6, shape='compact', resource_value=0.,
                              resource_field='equilibrium')
        base['interface'] = dict(equilibrium_times=[0])
        with tempfile.TemporaryDirectory() as tmp:
            run_replica((base, 13, str(Path(tmp) / 'nucleation'), {'source_hash': 'test'}))
            directory = (Path(tmp) / 'nucleation' / 'N00' / config_hash(base)
                         / 'seed-0000000013')
            with np.load(directory / 'states' / 'initial.npz') as state:
                strategy, resource = state['strategy'].copy(), state['resource'].copy()
            size = base['model']['size']
            growth, depletion = base['model']['growth'], base['model']['depletion']
            diffusion = base['model']['diffusion']
            again = resource_substep(strategy, resource, size, growth, depletion, diffusion)
            np.testing.assert_allclose(again, resource, atol=1e-10)
            self.assertEqual(int((strategy == 0).sum()), 6)
            self.assertLess(resource[strategy == 0].max(), 0.3)
            self.assertLess(resource[strategy == 0].min(), resource[strategy == 1].max())
            # the equilibrium field is a spatially non-uniform adaptation dented
            # around the damage, not a uniform field
            self.assertLess(resource.std(), 0.2)
            self.assertGreater(resource.std(), 1e-4)

    def test_invalid_resource_field_rejected(self):
        base = self.base_config()
        base['damage'] = dict(count=4, shape='random', resource_value=0.,
                              resource_field='adaptive')
        with self.assertRaises(ValueError):
            validate_config(base)


if __name__ == '__main__':
    unittest.main()
