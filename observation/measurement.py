"""
Mesure de la vitesse dans D_obs.
"""
import numpy as np

def compute_velocity(sol, dt):
    """Calcule la dérivée temporelle ∂t u par différences finies centrées."""
    nt = len(sol)
    v_all = np.zeros_like(sol)
    v_all[0] = (sol[1] - sol[0]) / dt  # Forward difference pour le premier instant
    v_all[-1] = (sol[-1] - sol[-2]) / dt # Backward difference pour le dernier instant
    for n in range(1, nt-1): # schéma centré pour les instants intermédiaires
        v_all[n] = (sol[n+1] - sol[n-1]) / (2*dt)
    return v_all

def extract_observed_velocity(v_all, mask_obs):
    """Extrait la vitesse observée uniquement dans D_obs."""
    nt = len(v_all)
    mask_obs = mask_obs.astype(bool)
    v_obs = np.array([v_all[n][mask_obs] for n in range(0, nt)]) #range(1, nt-1)])
    return v_obs
