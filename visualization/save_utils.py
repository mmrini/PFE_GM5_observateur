import os

def ensure_data_folder():
    """Crée le dossier 'data/' s'il n'existe pas."""
    os.makedirs("data", exist_ok=True)
    return "data"

def save_figure(fig, filename, dpi=300):
    """Sauvegarde une figure matplotlib dans data/."""
    folder = ensure_data_folder()
    path = os.path.join(folder, filename)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"Image sauvegardée : {path}")
    return path
