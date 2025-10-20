import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from .save_utils import ensure_data_folder

def animate_wave(X, Y, sol, dt, filename="onde2D.gif", fps=30, cmap='RdBu_r', vmin=-1, vmax=1):
    """
    Crée, affiche et sauvegarde un GIF de la propagation de l'onde.
    """
    folder = ensure_data_folder()
    path = os.path.join(folder, filename)

    fig, ax = plt.subplots(figsize=(6,5))
    pcm = ax.pcolormesh(X, Y, sol[0], shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_aspect('equal')
    ax.set_title("Propagation de l'onde")
    fig.colorbar(pcm, ax=ax, label='u(x,y,t)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    def update(frame):
        pcm.set_array(sol[frame].ravel())
        ax.set_title(f"t = {frame*dt:.3f} s")
        return [pcm]

    ani = animation.FuncAnimation(fig, update, frames=len(sol), interval=30, blit=True)
    ani.save(path, writer='pillow', fps=fps)
    print(f"Animation sauvegardée : {path}")

    plt.show()  # Afficher le GIF
    return path
