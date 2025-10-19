"""
On considère l'équation des ondes u_tt = c^2 * (u_xx + u_yy) avec u=0 sur le bord.
Problème direct : en considérant u_0, v_0 et c connus, on essaie de trouver u.
- > Solveur par différences finies en 2D pour l'équation des ondes. 
On utilise les différences finies pour discrétiser le schémaa : modèle centré en temps et en espace (2D, 5-point stencil).

- Conditions de Dirichlet sur les bords : u = 0
- Célérité c : dépend de (x,y) -> Matrice
- Vérification de la condition de CFL (stabilité)

- > Visualisation du maillage 2D et de la solution
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.animation as animation

def laplacian(u, dx):
    """5-point stencil, conditions de Dirichlet sur les bords, u = 0 dehors.
    On calcule que les points intérieurs (valeur nulle sur les bords)."""
    return (np.roll(u, -1, axis=0) - 2*u + np.roll(u, 1, axis=0)) / dx**2 + \
           (np.roll(u, -1, axis=1) - 2*u + np.roll(u, 1, axis=1)) / dx**2

def solve_wave_2d(u0, v0, c, dx, dt, nt, enforce_dt_safety=True):
    """
    /!\ on stocke u_0, v_0 et c dans des matrices de taille (nx,ny).
    Params:
        dx: pas spatial (le même pour x et y) 
        dt: pas temporel 
        nt: nombre de pas temporels

    Returns:
        u : vecteur solution de taille nt
    """
    nx, ny = u0.shape
    cmax = np.max(c)

    # Condition CFL pour les schémas centrés explicites en 2D : cmax * dt / dx <= 1/sqrt(2)
    cfl_limit = 1.0/np.sqrt(2.0)
    if enforce_dt_safety and cmax*dt/dx > cfl_limit:
        suggested_dt = cfl_limit * dx / cmax
        print(f"ATTENTION: le pas dt={dt:.3e} choisi ne respect pas la condition de stablité CFL (cmax*dt/dx <= 1/sqrt(2)).")
        print(f"Ajustement de dt -> {suggested_dt:.6e}")
        dt = suggested_dt

    u_nm1 = u0.copy()          # u^{n-1}  
    u_n = u0.copy()            # u^n
    sol = [u0.copy()]

    # Calculer u1 avec le développement de Taylor: u1 = u0 + dt*v0 + 0.5*dt^2*c^2*L(u0)
    L_u0 = laplacian(u0, dx)
    u_np1 = u0 + dt*v0 + 0.5*(dt**2)*(c**2)*L_u0 # u^{n+1}

    # Condition de Dirichlet sur les bords (u = 0)
    u_np1[0,:] = 0; u_np1[-1,:] = 0; u_np1[:,0] = 0; u_np1[:,-1] = 0
    sol.append(u_np1.copy())

    # Boucle principale
    for n in range(1, nt):
        Lu = laplacian(u_np1, dx)  
        u_next = 2*u_np1 - u_n + (dt**2)*(c**2)*Lu
        u_next[0,:] = 0; u_next[-1,:] = 0; u_next[:,0] = 0; u_next[:,-1] = 0 # Dirichlet homogène
        sol.append(u_next.copy())
        u_n, u_np1 = u_np1, u_next

    return sol, dt

def create_observation_mask(X, Y, Lx, Ly, center_frac):
    x_min = (0.5 - center_frac/2) * Lx
    x_max = (0.5 + center_frac/2) * Lx
    y_min = (0.5 - center_frac/2) * Ly
    y_max = (0.5 + center_frac/2) * Ly
    mask_center = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
    return ~mask_center  # négation booléenne

Lx = 1.0
Ly = 1.0

Nx = 101
Ny = 101

dx = Lx/(Nx-1)
dy = Ly/(Ny-1)

assert abs(dx-dy) < 1e-12   # on a supposé dx = dy

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)

# Domaine [0,1]x[0,1], Maillage Nx x Ny
X, Y = np.meshgrid(x, y, indexing='ij')

# Célérité de l'onde c(x,y)
c0 = 1.0
c = c0 * (1.0 - 0.3*np.exp(-((X-0.5)**2 + (Y-0.5)**2)/0.02))

# Déplacement inital u0 : nul sur les bords, gaussien au centre
sigma = 0.05
u0 = np.exp(-((X-0.5)**2 + (Y-0.5)**2)/(2*sigma**2))
u0[0,:] = 0; u0[-1,:] = 0; u0[:,0] = 0; u0[:,-1] = 0

# Vitesse initiale v0 : nulle
v0 = np.zeros_like(u0)

# Choix du pas temporel (/!\ en respectant la condition de stabilité CFL)
cmax = np.max(c)
dt = 0.4 * dx / cmax / np.sqrt(2)  

# Nombre de pas temporels
nt = 120  

# Solution du problème
sol, used_dt = solve_wave_2d(u0, v0, c, dx, dt, nt)
print(f"Shape of the solution sol : {np.size(sol)}")

print(f"Simulation: Maillage {Nx}x{Ny}, dt={used_dt:.3e}, nt={nt}, temps T={used_dt*nt:.3f}s")

fig, axes = plt.subplots(1, 3, figsize=(15,4))

# 1) Grille du maillage
ax = axes[0]
for xi in x[::10]:
    ax.plot([xi]*2, [0, Ly], linewidth=0.5)
for yi in y[::10]:
    ax.plot([0, Lx], [yi]*2, linewidth=0.5)
ax.set_title("Grille du maillage")
ax.set_aspect('equal', adjustable='box')
ax.set_xlim(0, Lx); ax.set_ylim(0, Ly)
ax.set_xlabel('x'); ax.set_ylabel('y')

# 2) Condition initiale u0
ax = axes[1]
pcm = ax.pcolormesh(X, Y, u0, shading='auto')
ax.set_title("Déplacement initial $u_0$")
ax.set_aspect('equal', adjustable='box')
fig.colorbar(pcm, ax=ax, shrink=0.9)

# 3) Déplacement final
ax = axes[2]
u_last = sol[-1]
pcm2 = ax.pcolormesh(X, Y, u_last, shading='auto')
ax.set_title(f"Solution à t = {used_dt*nt:.3f} s")
ax.set_aspect('equal', adjustable='box')
fig.colorbar(pcm2, ax=ax, shrink=0.9)

plt.tight_layout()

# Enregistrer l'image
image_output_path = f"data/onde2D_resultats_.png"
plt.savefig(image_output_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"Image sauvegardée : {image_output_path}")

# Animation de la propagation de l'onde 
fig, ax = plt.subplots(figsize=(6,5))
pcm = ax.pcolormesh(X, Y, sol[0], shading='auto', cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_aspect('equal')
ax.set_title("Propagation de l'onde")
fig.colorbar(pcm, ax=ax, label='u(x,y,t)')
ax.set_xlabel('x')
ax.set_ylabel('y')

def update(frame):
    pcm.set_array(sol[frame].ravel())
    ax.set_title(f"t = {frame*used_dt:.3f} s")
    return [pcm]

ani = animation.FuncAnimation(fig, update, frames=len(sol), interval=30, blit=True)

# Sauvegarde en GIF 
gif_output_path = 'data/onde2d.gif'
ani.save(gif_output_path, writer='pillow', fps=30)
print(f"Animation GIF sauvegardée : {gif_output_path}")

# Zone d'observation modélisée en masque booléen
mask_obs = create_observation_mask(X, Y, Lx, Ly, center_frac=0.2)

# Calculer la dérivée temporelle centrée 
nt = len(sol)
v_obs = []  # pour stocker la vitesse observée dans D_obs

for n in range(1, nt-1):
    v_n = (sol[n+1] - sol[n-1]) / (2*used_dt)
    v_obs.append(v_n[mask_obs])  # on ne garde que les points dans D_obs

v_obs = np.array(v_obs)  # shape = (nt-2, nb_points_obs)
print(f"Shape of v_obs : {v_obs.shape}")

# Vitesse complète sur tout le domaine 
v_all = np.zeros_like(sol)
for n in range(1, nt-1):
    v_all[n] = (sol[n+1] - sol[n-1]) / (2*used_dt)
print(f"Shape of v_all : {v_all.shape}")

# Affichage de la région observée 
plt.figure(figsize=(6,5))
plt.pcolormesh(X, Y, mask_obs, shading='auto', cmap='Greens')
plt.title("Domaine d'observation D_obs (en vert)")
plt.xlabel('x'); plt.ylabel('y')
plt.show()

# Visualisation de la vitess à un instant donné 
n_show = 50
plt.figure(figsize=(6,5))
plt.pcolormesh(X, Y, v_all[n_show], shading='auto', cmap='RdBu_r')
plt.title(f"Vitesse ∂t u à t = {n_show*used_dt:.3f} s")
plt.colorbar(label='∂t u')
plt.show()

