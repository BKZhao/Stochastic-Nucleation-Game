"""Read-only strategy-interface diagnostics for the unified eco-evolutionary kernel.

Provides a NumPy replication of the resource substep of ``b_eco_unified._step``
(local regrowth/depletion followed by synchronous diffusion), the ``J`` interface
drift diagnostic, and the fixed-strategy resource equilibrium. These functions
never advance a simulation, never consume random numbers, and never mutate a
``SimulationState``; the resource replication is verified bit-for-bit against a
verbatim copy of the kernel block in the test suite.

Payoff conventions follow ``b_eco_unified._step`` exactly: with strategy values
1 = cooperator and 0 = defector, the payoff of node ``i`` is
``P_C(i) = R_eff(r_i) * k_C(i)`` when cooperating and ``P_D(i) = b * k_C(i)``
when defecting, where ``k_C(i)`` counts cooperating neighbours and
``R_eff(r) = max(0, 1 + coupling * strength * (2r - 1))``.

``J`` definition: for every undirected interface edge ``{u=C, v=D}`` the term is
``tanh((P_D(v) - P_C(u)) / (2K))`` and ``J = (1/4) * sum(terms)``. Its exact
meaning: for the frozen state ``(s, r)``, the expected change of the defector
count caused by one random node visit (uniform node, one uniformly chosen
neighbour, kernel Fermi rule with zero mutation) equals ``J / N``. The same
first visit also satisfies the second-moment identity

``E[Z^2] = interface_edges / (4 N)``

because the two opposite imitation directions along one interface edge have
Fermi probabilities summing to one; hence ``Var(Z) = P/(4N) - (J/N)^2`` with
``P`` the interface edge count. Both identities hold for the frozen first visit
only; a full sweep interleaves N visits with changing strategies, so per-sweep
moments are empirical objects, not derivations. The kernel truncates the Fermi
exponent at +-30, which preserves the second-moment identity exactly in real
arithmetic; the exact first-visit mean uses clipped gaps, while this diagnostic
uses smooth tanh. ``J`` is an instantaneous first-visit
diagnostic; ``E[delta D per sweep] ~ slope * J`` is an empirical association
whose slope must be re-estimated per regime, not a general law.
"""
import numpy as np


def resource_substep(strategy, resource, size, growth, depletion, diffusion):
    """Apply one resource substep (local update then synchronous diffusion) on copies.

    Replicates the resource block of ``b_eco_unified._step`` operation-for-operation;
    inputs are 1-D arrays of length ``size**2`` and are not modified.
    """
    s = strategy.reshape(size, size)
    r = resource.reshape(size, size).copy()
    r = r + growth * s * (1.0 - r) - depletion * (1.0 - s) * r
    r = np.clip(r, 0.0, 1.0)
    if diffusion > 0:
        base = r.copy()
        average = (np.roll(base, 1, 0) + np.roll(base, -1, 0)
                   + np.roll(base, 1, 1) + np.roll(base, -1, 1)) / 4.0
        r = base + diffusion * (average - base)
        r = np.clip(r, 0.0, 1.0)
    return r.reshape(-1)


def resource_equilibrium(strategy, resource, size, growth, depletion, diffusion,
                        max_sweeps=500, tolerance=1e-13):
    """Fixed-strategy resource equilibrium reached by iterating ``resource_substep``.

    For fixed strategies the map is a sup-norm contraction with factor
    ``max(1 - growth, 1 - depletion)`` per sweep followed by a non-expansive
    convex average, so the equilibrium is unique. Returns ``(r_eq, sweeps_used)``.
    """
    current = np.asarray(resource, dtype=np.float64).reshape(-1).copy()
    for sweep in range(1, max_sweeps + 1):
        nxt = resource_substep(strategy, current, size, growth, depletion, diffusion)
        if np.max(np.abs(nxt - current)) <= tolerance:
            return nxt, sweep
        current = nxt
    return current, max_sweeps


def interface_observables(strategy, resource, size, temptation, coupling, strength,
                          selection_noise):
    """Return ``J`` and supporting interface statistics for one microscopic state.

    ``J`` uses the resource field supplied by the caller; pass the end-of-sweep
    field for the end-of-sweep diagnostic, or the resource substep applied to it
    for the exact first-visit predictor of the next sweep's strategy phase.
    """
    s = np.asarray(strategy).reshape(size, size)
    r = np.asarray(resource, dtype=np.float64).reshape(size, size)
    r_eff = 1.0 + coupling * strength * (2.0 * r - 1.0)
    np.maximum(r_eff, 0.0, out=r_eff)
    cooperators = (np.roll(s, 1, 0) + np.roll(s, -1, 0)
                   + np.roll(s, 1, 1) + np.roll(s, -1, 1))
    payoff_c = r_eff * cooperators
    payoff_d = temptation * cooperators
    two_k = 2.0 * selection_noise
    total = 0.0
    edges = 0
    for axis in (0, 1):
        s_j = np.roll(s, -1, axis)
        payoff_c_j = np.roll(payoff_c, -1, axis)
        payoff_d_j = np.roll(payoff_d, -1, axis)
        i_c_j_d = (s == 1) & (s_j == 0)
        i_d_j_c = (s == 0) & (s_j == 1)
        total += np.where(i_c_j_d, np.tanh((payoff_d_j - payoff_c) / two_k), 0.0).sum()
        total += np.where(i_d_j_c, np.tanh((payoff_d - payoff_c_j) / two_k), 0.0).sum()
        edges += int(i_c_j_d.sum() + i_d_j_c.sum())
    boundary = (s == 1) & (cooperators < 4)
    return dict(
        defectors=int(np.count_nonzero(s == 0)),
        interface_edges=edges,
        J=float(total) / 4.0,
        boundary_c_mean=float(r[boundary].mean()) if boundary.any() else None,
        boundary_c_min=float(r[boundary].min()) if boundary.any() else None,
    )


def interface_sensitivity(strategy, resource, size, temptation, coupling, strength,
                          selection_noise):
    """Linear response weights ``w_u`` of ``J`` to the supplied resource field.

    For a boundary cooperator ``u`` (C with at least one D neighbour),
    ``dJ/d r_u = -w_u`` with

    ``w_u = 2 * coupling * strength * k_C(u) / (8K)
            * sum_{v in N_D(u)} sech^2((P_D(v) - P_C(u)) / (2K))``

    and ``w_u = 0`` for every node that is not a boundary cooperator (a C node
    without D neighbours has no tanh term; a D node's own resource never enters
    its payoff ``P_D = b k_C``). Pass the post-substep field for the weights
    that apply to ``J_pre``. These are derivatives of the smooth diagnostic
    wherever ``R_eff > 0``;
    at the studied coupling, ``R_eff = 0.5 + r`` stays positive. Derivatives
    of the clipped transition rule can differ outside its unsaturated branch.
    Read-only: inputs are never modified
    and no random numbers are consumed.

    Returns a dict with the flat weights array (length ``size**2``), the
    boundary-node count and summary statistics.
    """
    s = np.asarray(strategy).reshape(size, size)
    r = np.asarray(resource, dtype=np.float64).reshape(size, size)
    r_eff = 1.0 + coupling * strength * (2.0 * r - 1.0)
    np.maximum(r_eff, 0.0, out=r_eff)
    cooperators = (np.roll(s, 1, 0) + np.roll(s, -1, 0)
                   + np.roll(s, 1, 1) + np.roll(s, -1, 1))
    payoff_c = r_eff * cooperators
    two_k = 2.0 * selection_noise
    weights = np.zeros_like(r_eff)
    for axis in (0, 1):
        for direction in (1, -1):
            s_j = np.roll(s, -direction, axis)
            payoff_d_j = np.roll(cooperators, -direction, axis) * temptation
            interface_pair = (s == 1) & (s_j == 0)
            gap = (payoff_d_j - payoff_c) / two_k
            sech_squared = 1.0 - np.tanh(gap) ** 2
            weights += np.where(interface_pair,
                                2.0 * coupling * strength * cooperators
                                * sech_squared / (4.0 * two_k), 0.0)
    boundary = (s == 1) & (cooperators < 4)
    return dict(weights=weights.reshape(-1),
                n_boundary=int(boundary.sum()),
                total_weight=float(weights.sum()),
                weighted_resource_sum=float(
                    (weights * r).sum()),
                note='dJ/d r_u = -w_u on the supplied field; zero off boundary '
                     'cooperators; see interface_sensitivity docstring for validity')


def pit_observables(strategy, resource, size, max_distance=8):
    """Resource-pit geometry around defector regions on the periodic lattice.

    Bin ``d`` of the radial profile is the mean resource of cells whose minimum
    periodic Euclidean distance to a defector rounds to ``d`` (bin 0 is the
    defector cells themselves). ``pit_floor`` is the on-defector mean resource,
    ``background`` the mean resource at distance ``>= max_distance - 0.5`` and
    ``half_recovery_distance`` the smallest bin whose profile reaches the
    halfway point between floor and background (``None`` when the field has no
    pit or never recovers). Read-only: inputs are never modified and no random
    numbers are consumed.
    """
    s = np.asarray(strategy).reshape(size, size)
    r = np.asarray(resource, dtype=np.float64).reshape(size, size)
    defectors = np.argwhere(s == 0)
    if defectors.size == 0:
        return dict(defectors=0, pit_profile={str(d): None for d in range(max_distance + 1)},
                    pit_floor=None, background=None, half_recovery_distance=None,
                    max_distance=max_distance)
    cells = np.argwhere(np.ones_like(s, dtype=bool))
    delta_x = np.abs(cells[:, None, 0] - defectors[None, :, 0])
    delta_y = np.abs(cells[:, None, 1] - defectors[None, :, 1])
    delta_x = np.minimum(delta_x, size - delta_x)
    delta_y = np.minimum(delta_y, size - delta_y)
    distance = np.rint(np.sqrt(delta_x ** 2 + delta_y ** 2)).astype(int)
    nearest = distance.min(axis=1)
    values = r.reshape(-1)
    profile = {}
    for d in range(max_distance + 1):
        mask = nearest == d
        profile[str(d)] = float(values[mask].mean()) if mask.any() else None
    floor = profile['0']
    far = nearest >= max_distance - 0.5
    background = float(values[far].mean()) if far.any() else None
    if far.any() and floor is not None and background - floor > 1e-9:
        halfway = floor + 0.5 * (background - floor)
        recovered = [d for d in range(max_distance + 1) if profile[str(d)] is not None
                     and profile[str(d)] >= halfway]
    else:
        recovered = []
    return dict(defectors=int(defectors.shape[0]), pit_profile=profile,
                pit_floor=floor, background=background,
                half_recovery_distance=int(recovered[0]) if recovered else None,
                max_distance=max_distance)


def equilibrium_observables(strategy, resource, size, temptation, coupling, strength,
                             selection_noise, growth, depletion, diffusion,
                             max_sweeps=500, tolerance=1e-13):
    """Interface diagnostics on the fixed-strategy resource equilibrium field.

    The equilibrium field solves ``r_eq = resource_substep(s, r_eq)`` and measures
    how the interface tendency would read if resources had fully adapted to the
    current strategy pattern. It generally differs from the actual field in both
    total resource and spatial arrangement.
    """
    r_eq, sweeps = resource_equilibrium(strategy, resource, size, growth, depletion,
                                        diffusion, max_sweeps, tolerance)
    obs = interface_observables(strategy, r_eq, size, temptation, coupling, strength,
                               selection_noise)
    obs['resource_mean_eq'] = float(r_eq.mean())
    obs['equilibrium_sweeps'] = sweeps
    return obs
