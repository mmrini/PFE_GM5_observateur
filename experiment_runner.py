import numpy as np
import matplotlib.pyplot as plt
from direct_solver.wave2D_direct_solver import solve_wave_2d
from observation.mask import get_mask 
from observation.measurement import compute_velocity, extract_observed_velocity
from inverse_solver.bfn import run_bfn_algorithm

def run_single_experiment(config, params):
    """
    Exécute une simulation complète avec des paramètres spécifiques.
    params: dictionnaire contenant les surcharges (T, mask_type, gamma, noise_level)
    """
    # 1. Récupération des paramètres (surcharge ou défaut)
    T_exp = params.get("T", config.T)
    nt_exp = int(T_exp / config.dt) # Recalcul du nombre de pas de temps
    mask_type = params.get("mask_type", "full")
    gamma_bfn = params.get("gamma_bfn", config.gamma_bfn)
    noise_level = params.get("noise_level", 0.0)
    
    print(f"--- Lancement: Masque={mask_type}, T={T_exp}, Gamma={gamma_bfn}, Bruit={noise_level} ---")

    # 2. Simulation DIRECTE (La Vérité)
    # On utilise les mêmes conditions initiales u0, v0 définies globalement ou passées en arg
    # Ici on suppose que u0 et v0 sont générés comme dans votre main original
    # (Je recrée u0 ici pour l'exemple, mais idéalement passez-le en argument)
    sol, used_dt = solve_wave_2d(config.u0, config.v0, config.c0, config.dx, config.dt, nt_exp)
    
    # 3. OBSERVATION
    mask = get_mask(config.X, config.Y, config.Lx, config.Ly, mask_type, ratio=params.get("ratio", 0.1))

    # Conversion explicite pour l'extraction
    mask_bool = mask.astype(bool) 

    v_all = compute_velocity(sol, used_dt)

    # Utiliser mask_bool pour l'extraction
    v_obs = extract_observed_velocity(v_all, mask_bool)
    
    # Ajout du BRUIT (Expérience 4)
    if noise_level > 0:
        amplitude = np.max(np.abs(v_all))
        noise = noise_level * amplitude * np.random.normal(size=v_all.shape)
        v_all = v_all + noise
        
    #v_obs = extract_observed_velocity(v_all, mask)
    v_obs = extract_observed_velocity(v_all, mask_bool)
    
    # 4. RÉSOLUTION INVERSE (BFN)
    u0_guess = np.zeros_like(config.u0)
    v0_guess = np.zeros_like(config.v0)
    
    u0_rec, v0_rec, conv_hist = run_bfn_algorithm(
        u0_guess, v0_guess, config.c0, config.dx, used_dt, nt_exp,
        mask, v_obs, gamma_bfn, num_iterations=config.n_iter
    )
    
    # 5. CALCUL D'ERREUR
    # Erreur relative L2 sur la condition initiale
    error_L2 = np.linalg.norm(config.u0 - u0_rec) / np.linalg.norm(config.u0)
    
    return {
        "params": params,
        "conv_hist": conv_hist,
        "error_L2": error_L2,
        "u0_rec": u0_rec,
        "mask": mask
    }

# --- Configuration de base (classe simple pour stocker les constantes) ---
class Config:
    def __init__(self):
        self.Lx, self.Ly = 1.0, 1.0
        self.nx, self.ny = 101, 101
        self.T = 0.5
        self.c0 = 1.0
        self.dt = self.T / 400
        self.dx = self.Lx / (self.nx - 1)
        self.gamma_bfn = 0.5
        self.n_iter = 15 # Un peu plus pour bien voir la convergence
        
        # Maillage
        x = np.linspace(0, self.Lx, self.nx)
        y = np.linspace(0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        # Condition initiale (Gaussienne)
        sigma = 0.05 * self.Lx
        x0, y0 = 0.5 * self.Lx, 0.5 * self.Ly
        self.u0 = np.exp(-((self.X - x0)**2 + (self.Y - y0)**2) / (2 * sigma**2))
        self.v0 = np.zeros_like(self.u0)

cfg = Config()

# ==========================================
# DÉFINITION DES SCÉNARIOS
# ==========================================
scenarios = [
    # --- Groupe 1 : Références & Géométries Classiques ---
    # Référence absolue : on voit tout, pas de bruit
    {"name": "Ref Full",      "T": 1.0, "mask_type": "full",      "ratio": 0.0, "noise_level": 0.0},
    # Uniquement un cadre autour (épaisseur 10%)
    {"name": "Geo Borders",   "T": 1.0, "mask_type": "borders",   "ratio": 0.1, "noise_level": 0.0},
    # Uniquement le mur de gauche (10% de la largeur)
    {"name": "Geo LeftWall",  "T": 1.0, "mask_type": "left_wall", "ratio": 0.1, "noise_level": 0.0},

    # --- Groupe 2 : "Trou au Centre" ---
    # Votre cas de base : on voit tout sauf 20% au centre. (Devrait converger très vite)
    {"name": "Hole Standard", "T": 1.0, "mask_type": "hole_center", "ratio": 0.2, "noise_level": 0.0},
    # Cas difficile : on ne voit RIEN sur 60% du centre. (Convergence plus lente attendue)
    {"name": "Hole Large",    "T": 1.0, "mask_type": "hole_center", "ratio": 0.6, "noise_level": 0.0},
    
    # --- Groupe 3 : Horizon Temporel (Test sur un cas difficile 'LeftWall') ---
    # Temps trop court : l'onde n'a pas le temps d'atteindre le capteur gauche -> Échec
    {"name": "Time Short",    "T": 0.4, "mask_type": "left_wall", "ratio": 0.1, "noise_level": 0.0},
    # Temps long : l'onde a le temps de faire des allers-retours -> Succès
    {"name": "Time Long",     "T": 2.0, "mask_type": "left_wall", "ratio": 0.1, "noise_level": 0.0},
    
    # --- Groupe 4 : Robustesse au Bruit (sur config 'Hole Standard') ---
    # Bruit faible (10%) sur votre config standard
    {"name": "Hole Noise 10%", "T": 1.0, "mask_type": "hole_center", "ratio": 0.2, "noise_level": 0.1},
    # Bruit fort (50%) sur votre config standard
    {"name": "Hole Noise 50%", "T": 1.0, "mask_type": "hole_center", "ratio": 0.2, "noise_level": 0.5},
]

results = {}

# ==========================================
# BOUCLE D'EXÉCUTION
# ==========================================
print(f"Démarrage de {len(scenarios)} expériences...")

for s in scenarios:
    name = s["name"]
    # On fusionne les params spécifiques avec la config par défaut
    res = run_single_experiment(cfg, s)
    results[name] = res

# ==========================================
# VISUALISATION COMPARATIVE
# ==========================================
plt.figure(figsize=(10, 6))

for name, res in results.items():
    hist = res["conv_hist"]
    plt.semilogy(hist, label=f"{name} (Err Fin: {res['error_L2']:.2e})", linewidth=2)

plt.title("Comparaison de la convergence BFN")
plt.xlabel("Itérations")
plt.ylabel("Log(Norme de mise à jour)")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("data/benchmark_convergence.png")
plt.show()

# Affichage des erreurs géométriques pour le cas "Left Wall"
if "Geo LeftWall" in results:
    from visualization.plot_fields import plot_field
    err = cfg.u0 - results["Geo LeftWall"]["u0_rec"]
    plot_field(cfg.X, cfg.Y, err, title="Erreur Spatiale - Mur Gauche")