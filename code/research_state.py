"""Lossless array copies and provenance; these do not capture Numba RNG state."""
import hashlib
from dataclasses import fields
import numpy as np
from b_eco_unified import SimulationState


def clone_state(state: SimulationState) -> SimulationState:
    return SimulationState(**{field.name: (getattr(state,field.name).copy()
                            if isinstance(getattr(state,field.name),np.ndarray)
                            else getattr(state,field.name)) for field in fields(state)})


def state_hash(state: SimulationState) -> str:
    digest=hashlib.sha256()
    digest.update(str(state.size).encode())
    for array in (state.strategy,state.resource):
        digest.update(array.dtype.str.encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def state_arrays(state: SimulationState) -> dict[str,np.ndarray]:
    return {field.name: np.asarray(getattr(state,field.name)).copy() for field in fields(state)}
