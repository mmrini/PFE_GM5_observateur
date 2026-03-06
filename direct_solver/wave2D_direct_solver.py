import numpy as np

def laplacian(u, dx):
    """Opérateur de Laplacien 2D (5-point stencil, Dirichlet homogène)."""
    return ((np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) +
            (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1))) / dx**2

def initialize_wave_solution(u0, v0, c, dx, dt):
    """
    Initialisation pour un schéma explicite d'ordre 2 pour l'équation des ondes 2D avec Dirichlet homogène.

    Calcule :
        u^0 = u0
        u^1 = u0 + dt v0 + 0.5 dt^2 c^2 Δu0
    """

    u_nm1 = u0.copy()
    u_n = u0.copy()

    Lu0 = laplacian(u0, dx)
    u_np1 = u0 + dt * v0 + 0.5 * dt**2 * (c**2) * Lu0

    # Dirichlet homogène
    u_np1[0, :] = u_np1[-1, :] = 0.0
    u_np1[:, 0] = u_np1[:, -1] = 0.0

    return u_nm1, u_n, u_np1

def solve_wave_2d(u0, v0, c, dx, dt, nt):
    """
    Solveur 2D explicite pour l'équation des ondes :
        u_tt = c^2 * Δu, avec conditions de Dirichlet homogènes.
    """
    nx, ny = u0.shape

    # Initialisation 
    sol = [u0.copy()]
    u_nm1, u_n, u_np1 = initialize_wave_solution(u0, v0, c, dx, dt)
    sol.append(u_np1.copy()) 

    # Boucle temporelle
    for n in range(1, nt):
        Lu = laplacian(u_np1, dx)
        u_next = 2*u_np1 - u_n + (dt**2)*(c**2)*Lu
        u_next[0,:] = u_next[-1,:] = u_next[:,0] = u_next[:,-1] = 0
        sol.append(u_next.copy())
        u_n, u_np1 = u_np1, u_next

    return sol, dt

