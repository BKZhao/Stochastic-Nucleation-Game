"""Periodic square-lattice construction for eco-evolutionary simulations."""

import numpy as np
from numba import njit


@njit(cache=True)
def build_square_lattice_nb(size, neighbors, degree):
    """Fill preallocated arrays with a periodic von Neumann neighborhood."""
    for row in range(size):
        for col in range(size):
            node = row * size + col
            neighbors[node, 0] = ((row - 1) % size) * size + col
            neighbors[node, 1] = ((row + 1) % size) * size + col
            neighbors[node, 2] = row * size + (col - 1) % size
            neighbors[node, 3] = row * size + (col + 1) % size
            degree[node] = 4


def build_square_lattice(size):
    """Return neighbor and degree arrays for a periodic ``size`` square."""
    node_count = size * size
    neighbors = np.full((node_count, 4), -1, dtype=np.int32)
    degree = np.zeros(node_count, dtype=np.int32)
    build_square_lattice_nb(size, neighbors, degree)
    return neighbors, degree
