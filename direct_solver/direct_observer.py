import numpy as np
from .wave2D_direct_solver import laplacian, initialize_wave_solution

def solve_observer_wave_2d(u0_hat, v0_hat, c, dx, dt, nt, mask_obs, v_obs, gamma, enforce_dt_safety=True):
    """
    Observateur direct pour l'équation des ondes 2D :

        (1/c^2) u_tt - Δu + gamma * χ_Dobs (u_t - d) = 0

    Calcule:
    sol_hat : list of ndarray
        Solution estimée û
    """

    nx, ny = u0_hat.shape
    sol_hat = []

    cmax = np.max(c)
    cfl_limit = 1.0 / np.sqrt(2.0)

    if enforce_dt_safety and cmax * dt / dx > cfl_limit:
        dt = cfl_limit * dx / cmax
        print(f"[CFL] dt ajusté à {dt:.3e}")
    
    # Initialisation 
    sol_hat.append(u0_hat.copy())
    u_nm1, u_n, u_np1 = initialize_wave_solution(u0_hat, v0_hat, c, dx, dt)
    sol_hat.append(u_np1.copy())

    for n in range(1, nt-1):

        # vitesse estimée de l'observateur
        v_hat_n = (u_np1 - u_n) / dt

        # données observées remises sur la grille
        d_n = np.zeros_like(u_np1)
        d_n[mask_obs] = v_obs[n-1]

        # terme de rétroaction
        feedback = gamma * mask_obs * (v_hat_n - d_n)

        # mise à jour onde + feedback
        Lu = laplacian(u_np1, dx)
        u_next = (2*u_np1 - u_n + dt**2 * (c**2) * Lu - dt * feedback)

        # Dirichlet homogène
        u_next[0,:] = u_next[-1,:] = 0
        u_next[:,0] = u_next[:,-1] = 0

        sol_hat.append(u_next.copy())
        u_n, u_np1 = u_np1, u_next

    return sol_hat

def gradient_sq(u, dx):
    ux = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2*dx)
    uy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2*dx)
    return ux**2 + uy**2

def compute_error_energy(sol_true, sol_hat, c, dx, dt):
    """
    Calcul de l'énergie discrète de l'erreur pour chaque pas temporel pour vérifier sa décroissance.

    Calcule:
    E : ndarray
        Énergie de l'erreur à chaque pas (nt éléments)
    """
    nt = len(sol_hat)
    nx, ny = sol_hat[0].shape
    E = np.zeros(nt)

    for n in range(1, nt):
        u_tilde = sol_hat[n] - sol_true[n]
        # dérivée temporelle centrée
        if n < nt-1:
            du_dt = (sol_hat[n+1] - sol_hat[n-1] - (sol_true[n+1] - sol_true[n-1])) / (2*dt)
        else:
            du_dt = (u_tilde - (sol_hat[n-1] - sol_true[n-1])) / dt  # dernier pas
        grad_sq = gradient_sq(u_tilde, dx)
        E[n] = 0.5 * np.sum((du_dt**2)/c**2 + grad_sq) * dx**2

    return E