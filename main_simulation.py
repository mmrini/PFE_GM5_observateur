import numpy as np
import matplotlib.pyplot as plt

from config import Lx, Ly, nx, ny, T, nt, c0
from direct_solver.wave2D_direct_solver import solve_wave_2d
from direct_solver.direct_observer import solve_observer_wave_2d, compute_error_energy
from observation.mask import create_observation_mask
from observation.measurement import compute_velocity, extract_observed_velocity
from visualization.save_utils import ensure_data_folder, save_figure
from inverse_solver.bfn import run_bfn_algorithm

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

# ======================================================
# Observateur direct
# ======================================================
print("\nLancement de l'observateur direct...")

# Conditions initiales volontairement erronées
u0_hat = np.zeros_like(u0)
v0_hat = np.zeros_like(u0)

gamma = 0.2 # gain de rétroaction

sol_hat = solve_observer_wave_2d(u0_hat, v0_hat, c0, dx, used_dt, nt, mask_obs, v_obs, gamma)

plot_field(X, Y, sol[-1], title="Solution réelle")
plot_field(X, Y, sol_hat[-1], title="Solution observée")

# ======================================================
# Calcul de l'énergie de l'erreur et visualisation
# ======================================================
print("\nCalcul de l'énergie de l'erreur...")
E_error = compute_error_energy(sol, sol_hat, c0, dx, used_dt)[1:]

plt.figure(figsize=(7,4))
plt.plot(np.arange(len(E_error))*used_dt, E_error, 'b-', lw=2)
plt.xlabel("Temps [s]")
plt.ylabel("Énergie de l'erreur")
plt.title("Décroissance de l'énergie de l'erreur de l'observateur")
plt.grid(True)
plt.show()

plt.savefig("data/error_energy_decay.png", dpi=300)  

# ======================================================
# Test de l'algorithme BFN
# ======================================================
print("\nLancement de l'algorithme BFN complet...")

# On part de conditions initiales nulles (ou bruitées)
u0_guess = np.zeros_like(u0)
v0_guess = np.zeros_like(v0)

# Nombre d'itérations BFN
n_iter = 5
gamma_bfn = 0.5  # Parfois on prend un gamma un peu plus fort

u0_rec, v0_rec, conv_hist = run_bfn_algorithm(
    u0_guess, v0_guess, c0, dx, used_dt, nt, 
    mask_obs, v_obs, gamma_bfn, num_iterations=n_iter
)

# Visualisation de la reconstruction
plot_field(X, Y, u0, title="Condition Initiale Réelle (u0)")
plot_field(X, Y, u0_rec, title=f"u0 Reconstruit (BFN {n_iter} iters)")

# Erreur de reconstruction
err_rec = u0 - u0_rec
plot_field(X, Y, err_rec, title="Erreur de reconstruction u0", cmap='seismic')

plt.figure()
plt.plot(conv_hist, 'o-')
plt.title("Convergence des itérations BFN")
plt.xlabel("Itération")
plt.ylabel("Norme de la mise à jour (||u_{k+1} - u_k||)")
plt.grid(True)
plt.savefig("data/bfn_convergence.png")
plt.show()