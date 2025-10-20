import numpy as np
from config import Lx, Ly, nx, ny, T, nt, c0
from direct_solver.wave_solver import solve_wave_2d
from observation.mask import create_observation_mask
from observation.measurement import compute_velocity, extract_observed_velocity
from visualization.plot_fields import (
    plot_results,
    plot_field,
    plot_mask,
    plot_velocity_field,
)
from visualization.animate import animate_wave

# ======================================================
# Définition du maillage et des paramètres
# ======================================================
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
dx = x[1] - x[0]; dy = y[1] - y[0]
X, Y = np.meshgrid(x, y, indexing='ij')

dt = T / nt

print(f"Maillage : {nx}x{ny} points")
print(f"dx = {dx:.3e}, dt = {dt:.3e}")

# ======================================================
# Condition initiale et vitesse initiale
# ======================================================
# Onde gaussienne centrée, nulle sur les bords
sigma = 0.05 * Lx
x0, y0 = 0.5 * Lx, 0.5 * Ly
u0 = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
u0[0,:] = 0; u0[-1,:] = 0; u0[:,0] = 0; u0[:,-1] = 0
v0 = np.zeros_like(u0)

# ======================================================
# Résolution de l’équation d’onde
# ======================================================
print("\nRésolution du problème direct...")
sol, used_dt = solve_wave_2d(u0, v0, c0, dx, dt, nt)
print(f"Simulation terminée avec dt = {used_dt:.3e}")

# ======================================================
# Visualisation statique (maillage, u0, u_final)
# ======================================================
print("\nAffichage et sauvegarde des résultats...")
plot_results(x, y, X, Y, u0, sol, used_dt, Lx, Ly)

# ======================================================
# Animation GIF de la propagation
# ======================================================
print("\nCréation du GIF d'animation...")
animate_wave(X, Y, sol, used_dt)

# ======================================================
# Domaine d’observation D_obs 
# ======================================================
mask_obs = create_observation_mask(X, Y, Lx, Ly)
plot_mask(X, Y, mask_obs, save=True)

# ======================================================
# Calcul de la dérivée temporelle ∂t u
# ======================================================
print("\nCalcul de la dérivée temporelle ∂t u...")
v_all = compute_velocity(sol, used_dt)
v_obs = extract_observed_velocity(v_all, mask_obs)
print("Extraction dans D_obs terminée.")

# Visualisation à un instant choisi
t_index = len(v_all)//2
plot_velocity_field(X, Y, v_all[t_index], t_index * used_dt)

print("\nSimulation complète. Résultats disponibles dans le dossier 'data/'.")
