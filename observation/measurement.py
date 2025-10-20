import numpy as np

def compute_velocity(sol, dt):
    """Calcule la dérivée temporelle ∂t u par différences finies centrées."""
    nt = len(sol)
    v_all = np.zeros_like(sol)
    for n in range(1, nt-1):
        v_all[n] = (sol[n+1] - sol[n-1]) / (2*dt)
    return v_all

def extract_observed_velocity(v_all, mask_obs):
    """Extrait la vitesse observée uniquement dans D_obs."""
    nt = len(v_all)
    v_obs = np.array([v_all[n][mask_obs] for n in range(1, nt-1)])
    return v_obs
