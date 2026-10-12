"""Microscopic spatial eco-evolutionary game with sign-tunable feedback."""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit

from lattice import build_square_lattice


@dataclass(frozen=True)
class Trajectory:
    """Macroscopic cooperation and resource trajectories."""

    cooperation: np.ndarray
    resource: np.ndarray


@dataclass
class SimulationState:
    """Mutable microscopic state for continuation, perturbation, and escape experiments."""

    size: int
    neighbors: np.ndarray
    degree: np.ndarray
    strategy: np.ndarray
    resource: np.ndarray
    permutation: np.ndarray


@njit(cache=True)
def _R_eff(r, lam, sigma):
    val = 1.0 + lam * sigma * (2.0 * r - 1.0)
    if val < 0.0:
        val = 0.0
    return val


@njit(cache=True)
def seed_numba(seed):
    """Seed the random-number state used inside compiled update functions."""
    np.random.seed(seed)


@njit(cache=True)
def _step(nbr, deg, s, Rf, b, sg, delta, lam, sigma, K, perm, mu, Ddiff=0.0):
    N = s.shape[0]
    # 资源更新
    for i in range(N):
        si = s[i]
        Rf[i] = Rf[i] + sg * si * (1.0 - Rf[i]) - delta * (1 - si) * Rf[i]
        if Rf[i] < 0: Rf[i] = 0.0
        if Rf[i] > 1: Rf[i] = 1.0
    # 资源扩散(让 D 的低 r 泄漏到邻居)
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
            if Rnew[i] < 0: Rnew[i] = 0.0
            if Rnew[i] > 1: Rnew[i] = 1.0
        for i in range(N):
            Rf[i] = Rnew[i]
    # 随机序
    for i in range(N):
        perm[i] = i
    for i in range(N - 1, 0, -1):
        ri = np.random.randint(0, i + 1)
        perm[i], perm[ri] = perm[ri], perm[i]
    # 策略更新
    for idx in range(N):
        i = perm[idx]
        d = deg[i]
        if d == 0:
            continue
        if np.random.random() < mu:
            s[i] = 1 - s[i]
        si = s[i]
        Ri = _R_eff(Rf[i], lam, sigma)
        Pi = 0.0
        for k in range(d):
            jn = nbr[i, k]
            Pi += (s[jn] * Ri) if si else (b * s[jn])
        j = nbr[i, np.random.randint(0, d)]
        sj = s[j]
        Rj = _R_eff(Rf[j], lam, sigma)
        Pj = 0.0
        for k in range(deg[j]):
            jn = nbr[j, k]
            Pj += (s[jn] * Rj) if sj else (b * s[jn])
        dw = (Pi - Pj) / K
        if dw > 30: dw = 30
        if dw < -30: dw = -30
        if np.random.random() < 1.0 / (1.0 + math.exp(dw)):
            s[i] = s[j]


def initialize_state(size, seed, strategy_init="random", resource_init=0.5):
    """Create a lattice and an explicitly named microscopic initial state."""
    neighbors, degree = build_square_lattice(size)
    node_count = size * size
    rng = np.random.default_rng(seed)
    if strategy_init == "random":
        strategy = rng.integers(0, 2, node_count).astype(np.int8)
    elif strategy_init == "all_c":
        strategy = np.ones(node_count, dtype=np.int8)
    elif strategy_init == "all_d":
        strategy = np.zeros(node_count, dtype=np.int8)
    else:
        raise ValueError(
            "strategy_init must be one of: 'random', 'all_c', or 'all_d'"
        )
    resource = np.full(node_count, resource_init, dtype=np.float64)
    return neighbors, degree, strategy, resource


def create_simulation_state(
    size,
    seed,
    strategy_init="random",
    resource_init=0.5,
):
    """Create and seed a mutable state whose random stream persists across advances."""
    neighbors, degree, strategy, resource = initialize_state(
        size,
        seed,
        strategy_init,
        resource_init,
    )
    seed_numba(seed)
    return SimulationState(
        size=size,
        neighbors=neighbors,
        degree=degree,
        strategy=strategy,
        resource=resource,
        permutation=np.zeros(size * size, dtype=np.int32),
    )


def advance_state(
    state,
    temptation,
    growth,
    depletion,
    coupling,
    strength,
    steps,
    selection_noise=0.1,
    mutation=0.0,
    diffusion=0.0,
):
    """Advance one microscopic state without reinitializing it or reseeding its RNG."""
    cooperation_trace = np.empty(steps, dtype=np.float64)
    resource_trace = np.empty(steps, dtype=np.float64)
    for time in range(steps):
        _step(
            state.neighbors,
            state.degree,
            state.strategy,
            state.resource,
            temptation,
            growth,
            depletion,
            coupling,
            strength,
            selection_noise,
            state.permutation,
            mutation,
            diffusion,
        )
        cooperation_trace[time] = state.strategy.mean()
        resource_trace[time] = state.resource.mean()
    return Trajectory(cooperation_trace, resource_trace)


def step_state(
    state,
    temptation,
    growth,
    depletion,
    coupling,
    strength,
    selection_noise=0.1,
    mutation=0.0,
    diffusion=0.0,
):
    """Advance one Monte Carlo step and expose the mutated microscopic fields."""
    _step(
        state.neighbors,
        state.degree,
        state.strategy,
        state.resource,
        temptation,
        growth,
        depletion,
        coupling,
        strength,
        selection_noise,
        state.permutation,
        mutation,
        diffusion,
    )
    return float(state.strategy.mean()), float(state.resource.mean())


def run_trajectory(
    size,
    temptation,
    growth,
    depletion,
    coupling,
    strength,
    steps,
    seed,
    selection_noise=0.1,
    mutation=0.0,
    diffusion=0.0,
    strategy_init="random",
    resource_init=0.5,
):
    """Run the microscopic model and return reproducible macro trajectories."""
    state = create_simulation_state(
        size,
        seed,
        strategy_init,
        resource_init,
    )
    return advance_state(
        state,
        temptation,
        growth,
        depletion,
        coupling,
        strength,
        steps,
        selection_noise,
        mutation,
        diffusion,
    )


def build(L, seed=0, r0=0.5):
    """Compatibility wrapper for legacy random-initial-condition experiments."""
    return initialize_state(L, seed, "random", r0)


def run_traj(L, b, sg, delta, lam, sigma, steps, seed, K=0.1, mu=0.0, r0=0.5):
    """记录 rho_c(t) 和 r(t) 全轨迹, 用于判断振荡 vs 双稳态."""
    trajectory = run_trajectory(
        size=L,
        temptation=b,
        growth=sg,
        depletion=delta,
        coupling=lam,
        strength=sigma,
        steps=steps,
        seed=seed,
        selection_noise=K,
        mutation=mu,
        resource_init=r0,
    )
    return trajectory.cooperation, trajectory.resource


def run_final(
    L,
    b,
    sg,
    delta,
    lam,
    sigma,
    steps,
    warmup,
    seed,
    K=0.1,
    mu=0.0,
    r0=0.5,
    strategy_init="random",
):
    nbr, deg, s, Rf = initialize_state(L, seed, strategy_init, r0)
    perm = np.zeros(L * L, dtype=np.int32)
    seed_numba(seed)
    rec = []
    for t in range(steps):
        _step(nbr, deg, s, Rf, b, sg, delta, lam, sigma, K, perm, mu)
        if t >= warmup:
            rec.append((float(s.mean()), float(Rf.mean())))
    arr = np.array(rec)
    return arr[:, 0].mean(), arr[:, 1].mean(), arr[:, 0].std(), arr[:, 1].std()
