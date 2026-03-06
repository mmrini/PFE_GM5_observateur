import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def plot_field(X, Y, field, title, cmap='RdBu_r', save=True, filename_prefix=None):
    """
    Affiche et sauvegarde (optionnellement) un champ scalaire 2D.
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

def plot_mask(X, Y, mask_obs, save=False, filename_prefix="mask_Dobs", type_mask=""):
    """
    Affiche la zone d'observation et peut la sauvegarder.
    """
    fig, ax = plt.subplots(figsize=(6,5))
    pcm = ax.pcolormesh(X, Y, mask_obs, shading='auto', cmap='Greens')
    ax.set_title(f"Domaine d'observation D_obs, {type_mask}")
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


def plot_comparison_3d(X, Y, u_true, v_true, u_rec, v_rec, title="", save_path=None):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Comparaison Théorie vs BFN - {title}", fontsize=16)

    # --- TOP LEFT: u0 True ---
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X, Y, u_true, cmap='viridis', edgecolor='none')
    ax1.set_title("u0 Théorie (Position)")
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # --- TOP RIGHT: u0 Rec ---
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    surf2 = ax2.plot_surface(X, Y, u_rec, cmap='viridis', edgecolor='none')
    ax2.set_title("u0 Reconstruction (BFN)")
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    # --- BOTTOM LEFT: v0 True ---
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    surf3 = ax3.plot_surface(X, Y, v_true, cmap='RdBu_r', edgecolor='none')
    ax3.set_title("v0 Théorie (Vitesse)")
    ax3.set_zlim(-1, 1) # Force l'échelle car la théorie est à 0
    fig.colorbar(surf3, ax=ax3, shrink=0.5, aspect=10)

    # --- BOTTOM RIGHT: v0 Rec ---
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    # On laisse l'échelle libre pour voir l'amplitude du résidu
    surf4 = ax4.plot_surface(X, Y, v_rec, cmap='RdBu_r', edgecolor='none')
    ax4.set_title("v0 Reconstruction (BFN)")
    fig.colorbar(surf4, ax=ax4, shrink=0.5, aspect=10)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.show() 
    plt.close(fig)


def plot_overlay_3d(X, Y, u_true, v_true, u_rec, v_rec, title="", save_path=None):
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle(f"Superposition 3D : Théorie (Gris) vs Reconstruction (Couleur)\n{title}", fontsize=14)

    # --- SUBPLOT 1: Position u0 ---
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    # 1. On trace la THÉORIE en filaire gris (référence)
    ax1.plot_wireframe(X, Y, u_true, color='black', alpha=0.2, linewidth=0.5, label='Théorie')
    
    # 2. On trace la RECONSTRUCTION en surface colorée semi-transparente
    surf1 = ax1.plot_surface(X, Y, u_rec, cmap='viridis', alpha=0.7, edgecolor='none')
    
    ax1.set_title("Superposition u0 (Position)")
    ax1.set_zlim(-0.2, 1.2) # Ajuste selon l'amplitude de ta gaussienne
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=10)

    # --- SUBPLOT 2: Vitesse v0 ---
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    
    # 1. On trace la THÉORIE (v0 = 0) comme un plan gris
    ax2.plot_wireframe(X, Y, v_true, color='black', alpha=0.2, linewidth=0.5)
    
    # 2. On trace la RECONSTRUCTION v0
    # On utilise 'coolwarm' pour bien voir les erreurs positives/négatives
    surf2 = ax2.plot_surface(X, Y, v_rec, cmap='coolwarm', alpha=0.7, edgecolor='none')
    
    ax2.set_title("Superposition v0 (Vitesse)")
    # On ne fixe pas de zlim ici pour voir l'amplitude réelle du bruit de vitesse
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=10)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=200)

    plt.show()
    plt.close(fig)