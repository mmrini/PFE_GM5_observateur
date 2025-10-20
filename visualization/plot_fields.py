import os
import matplotlib.pyplot as plt
from .save_utils import ensure_data_folder, save_figure

def plot_results(x, y, X, Y, u0, sol, used_dt, Lx, Ly, filename="onde2D_resultats.png"):
    """
    Affiche et sauvegarde :
    1) le maillage
    2) la condition initiale u0
    3) la solution finale u(T)
    """
    folder = ensure_data_folder()
    path = os.path.join(folder, filename)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) Grille du maillage
    ax = axes[0]
    for xi in x[::10]:
        ax.plot([xi]*2, [0, Ly], linewidth=0.5)
    for yi in y[::10]:
        ax.plot([0, Lx], [yi]*2, linewidth=0.5)
    ax.set_title("Grille du maillage")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

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
    ax.set_title(f"Solution à t = {used_dt*len(sol):.3f} s")
    ax.set_aspect('equal', adjustable='box')
    fig.colorbar(pcm2, ax=ax, shrink=0.9)

    plt.tight_layout()
    save_figure(fig, filename)
    plt.show()

    return path

def plot_field(X, Y, field, title, cmap='RdBu_r', save=False, filename_prefix=None):
    """
    Affiche et (optionnellement) sauvegarde un champ scalaire 2D.
    """
    fig, ax = plt.subplots(figsize=(6,5))
    pcm = ax.pcolormesh(X, Y, field, shading='auto', cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel('x'); ax.set_ylabel('y')
    fig.colorbar(pcm, ax=ax, label=title)
    plt.tight_layout()
    
    if save:
        prefix = filename_prefix or title.replace(" ", "_").replace("$", "")
        save_figure(fig, prefix)
    plt.show()

def plot_mask(X, Y, mask_obs, save=False, filename_prefix="mask_Dobs"):
    """
    Affiche la zone d'observation et peut la sauvegarder.
    """
    fig, ax = plt.subplots(figsize=(6,5))
    pcm = ax.pcolormesh(X, Y, mask_obs, shading='auto', cmap='Greens')
    ax.set_title("Domaine d'observation D_obs (en vert)")
    ax.set_xlabel('x'); ax.set_ylabel('y')
    plt.tight_layout()

    if save:
        save_figure(fig, filename_prefix)
    plt.show()

def plot_velocity_field(X, Y, v_field, t):
    """
    Visualise la vitesse ∂t u(x,y,t) à un instant donné.
    """
    title = f"Vitesse ∂t u à t = {t:.3f} s"
    fig, ax = plt.subplots(figsize=(6,5))
    pcm = ax.pcolormesh(X, Y, v_field, shading='auto', cmap='RdBu_r')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(pcm, ax=ax, label='∂t u')
    plt.tight_layout()

    save_figure(fig, f"velocity_t{t:.3f}.png")

    plt.show()