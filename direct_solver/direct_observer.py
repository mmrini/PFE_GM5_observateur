import numpy as np
from .wave2D_direct_solver import laplacian

def solve_direct_observer(u0_hat, v0_hat, c, dx, dt, nt, mask_obs, v_obs, gamma):
    nx, ny = u0_hat.shape
    sol_hat = [None] * (nt + 1)
    
    # Initialisation u^0 et u^1 (Taylor standard)
    u_nm1 = u0_hat.copy()
    Lu0 = laplacian(u0_hat, dx)
    u_n = u0_hat + dt * v0_hat + 0.5 * (dt**2) * (c**2) * Lu0 
    
    sol_hat[0] = u_nm1.copy()
    sol_hat[1] = u_n.copy()

    # Pré-calcul du dénominateur et des coefficients
    coeff_gamma = (gamma * (c**2) * dt) / 2.0
    mask_obs = mask_obs.astype(bool)
    denom = 1.0 + coeff_gamma * mask_obs
    
    for n in range(1, nt):
        # Mesure d_n
        d_n = np.zeros_like(u_n)
        d_n[mask_obs] = v_obs[n] # Mesure au temps n

        Lu = laplacian(u_n, dx)
        
        term_past = (1.0 - coeff_gamma * mask_obs) * u_nm1
        numerateur = 2*u_n - term_past + (dt**2) * (c**2) * (Lu + gamma * mask_obs * d_n)
        
        u_np1 = numerateur / denom

        # Dirichlet
        u_np1[0,:] = u_np1[-1,:] = u_np1[:,0] = u_np1[:,-1] = 0

        sol_hat[n+1] = u_np1.copy()
        u_nm1, u_n = u_n, u_np1

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
        if n < nt-1:
            #u_tilde = sol_hat[n+1] - sol_true[n+1] 
            du_dt = (sol_hat[n+1] - sol_hat[n-1] - (sol_true[n+1] - sol_true[n-1])) / (2*dt)
        else:
            du_dt = (u_tilde - (sol_hat[n-1] - sol_true[n-1])) / dt  # dernier pas
        grad_sq = gradient_sq(u_tilde, dx)
        E[n] = 0.5 * np.sum((du_dt**2)/c**2 + grad_sq) * dx**2

    return E