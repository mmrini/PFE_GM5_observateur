import numpy as np

def laplacian(u, dx):
    """Opérateur de Laplacien 2D (5-point stencil, Dirichlet homogène)."""
    return ((np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) +
            (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1))) / dx**2
