import numpy as np
from .laplacian import laplacian

def solve_wave_2d(u0, v0, c, dx, dt, nt, enforce_dt_safety=True):
    """
    Solveur 2D explicite pour l'équation des ondes :
        u_tt = c^2 * Δu, avec conditions de Dirichlet homogènes.
    """
    nx, ny = u0.shape
    cmax = np.max(c)
    cfl_limit = 1.0 / np.sqrt(2.0)

    if enforce_dt_safety and cmax * dt / dx > cfl_limit:
        dt = cfl_limit * dx / cmax
        print(f"[CFL] dt ajusté à {dt:.3e}")

    u_nm1, u_n = u0.copy(), u0.copy()
    sol = [u0.copy()]

    # Premier pas
    L_u0 = laplacian(u0, dx)
    u_np1 = u0 + dt * v0 + 0.5 * (dt**2) * (c**2) * L_u0
    u_np1[0,:] = u_np1[-1,:] = u_np1[:,0] = u_np1[:,-1] = 0
    sol.append(u_np1.copy())

    # Boucle temporelle
    for n in range(1, nt):
        Lu = laplacian(u_np1, dx)
        u_next = 2*u_np1 - u_n + (dt**2)*(c**2)*Lu
        u_next[0,:] = u_next[-1,:] = u_next[:,0] = u_next[:,-1] = 0
        sol.append(u_next.copy())
        u_n, u_np1 = u_np1, u_next

    return sol, dt
