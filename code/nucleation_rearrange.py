"""Controlled interface-resource rearrangement arms (mechanism round).

All arms preserve the resource value multiset on the rearranged node set
exactly: the total resource, the full value distribution and the pre-substep
boundary-cooperator mean are unchanged by construction; only WHICH node holds
WHICH value changes.

Objective. The exact target is ``J_pre`` of the arranged state, and to first
order ``dJ = -sum_u w_u * (r~_arm - r~_original)_u`` with the weights
``w_u = -dJ/d r~_u`` of ``nucleation_interface.interface_sensitivity``
computed on the mother's original post-substep field. The rearranged PRE-
substep values pass through the resource substep (local update ``phi`` then
synchronous diffusion), so placing a value at boundary node ``u`` contributes
to the objective through ``phi(value)`` with the EFFECTIVE weight

``W_u = (1 - diffusion) * w_u + (diffusion / 4) * sum_{v ~ u} w_v``,

because ``r~_u = (1-D) phi(r_u) + (D/4) sum_{v~u} phi(r_v)`` and ``phi`` is
monotone (no clipping inside [0,1] for the studied parameters). Sorting the
boundary value multiset by ``W`` therefore maximises / minimises the linear
objective by the rearrangement inequality.

Arm semantics (``dJ/d r~_u = -w_u``):

- ``interface_protect``: largest boundary values to the largest ``W_u``
  (most negative ``dJ``, drift towards recovery);
- ``interface_harm``: the reverse pairing (most positive ``dJ``, drift
  towards defectors);
- ``interface_shuffle``: uniformly random permutation of the same boundary
  value multiset (frozen seed);
- ``defector_shuffle``: the same permutation logic on defector nodes. A D
  node's own resource never enters payoffs (``P_D = b * k_C``), but the
  diffusion substep carries the change into boundary cooperator cells within
  one substep with prefactor ``diffusion / 4``, so the composite response can be
  nonzero. Its magnitude depends on
  the geometry and weights; no universal response ratio follows.

Functions never consume random numbers except through the frozen per-mother
permutation seed derived from the mother file digest.
"""
import numpy as np

from nucleation_interface import interface_sensitivity, resource_substep

MECHANISM_ARMS = ('interface_protect', 'interface_harm',
                  'interface_shuffle', 'defector_shuffle')
_PERMUTATION_OFFSETS = {'interface_shuffle': 1, 'defector_shuffle': 2}


def boundary_cooperator_mask(strategy, size):
    """Cooperators with at least one defector neighbour (the interface band)."""
    grid = np.asarray(strategy).reshape(size, size)
    neighbours = (np.roll(grid, 1, 0) + np.roll(grid, -1, 0)
                  + np.roll(grid, 1, 1) + np.roll(grid, -1, 1))
    return (grid == 1) & (neighbours < 4)


def mechanism_permutation_seed(mother_digest, arm):
    """Frozen, outcome-blind permutation seed from the mother file digest."""
    base = int(mother_digest[:8], 16)
    return (base + _PERMUTATION_OFFSETS.get(arm, 0)) % (2 ** 32)


def effective_weights(strategy, post_resource, model):
    """Substep-mixed weights ``W`` = how a pre-substep value at u moves the objective.

    ``W_u = (1 - D) w_u + (D/4) sum_{v ~ u} w_v`` with ``w`` the linear
    sensitivity on the post-substep field; zero wherever ``w`` and all its
    neighbours vanish.
    """
    size = model['size']
    diffusion = float(model['diffusion'])
    w = interface_sensitivity(strategy, post_resource, size,
                              model['temptation'], model['coupling'],
                              model['strength'], model['selection_noise']
                              )['weights'].reshape(size, size)
    if diffusion == 0.0:
        return w
    neighbour_sum = (np.roll(w, 1, 0) + np.roll(w, -1, 0)
                     + np.roll(w, 1, 1) + np.roll(w, -1, 1))
    return (1.0 - diffusion) * w + (diffusion / 4.0) * neighbour_sum


def rearranged_resource(strategy, resource, model, arm, mother_digest):
    """Return the rearranged flat resource field for one mechanism arm.

    The input arrays are never modified; the returned field is a new array.
    Degenerate node sets (fewer than two members) return the field unchanged.
    """
    if arm not in MECHANISM_ARMS:
        raise ValueError(f'unknown mechanism arm: {arm}')
    size = model['size']
    strategy = np.asarray(strategy)
    field = np.asarray(resource, dtype=np.float64).reshape(size, size).copy()
    grid = strategy.reshape(size, size)
    if arm in ('interface_protect', 'interface_harm'):
        mask = boundary_cooperator_mask(strategy, size)
        if int(mask.sum()) < 2:
            return field.reshape(-1)
        post = resource_substep(strategy, np.asarray(resource, dtype=np.float64),
                                size, model['growth'], model['depletion'],
                                model['diffusion'])
        weights = effective_weights(strategy, post, model)[mask]
        values = field[mask]
        order_w = np.argsort(weights, kind='stable')
        order_v = np.argsort(values, kind='stable')
        assigned = np.empty_like(values)
        assigned[order_w] = values[order_v]
        if arm == 'interface_harm':
            assigned[order_w] = values[order_v][::-1]
        field[mask] = assigned
    else:
        mask = (grid == 0) if arm == 'defector_shuffle' \
            else boundary_cooperator_mask(strategy, size)
        if int(mask.sum()) < 2:
            return field.reshape(-1)
        values = field[mask]
        rng = np.random.default_rng(
            mechanism_permutation_seed(mother_digest, arm))
        field[mask] = values[rng.permutation(len(values))]
    return field.reshape(-1)
