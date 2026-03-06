"""
Création du masque d'observation.
"""
import numpy as np

def get_mask(X, Y, Lx, Ly, mask_type="full", ratio=0.1, radius=None):
    mask = np.zeros_like(X)
    
    # Gestion du rayon pour les cercles
    r = radius if radius is not None else ratio * min(Lx, Ly)
    
    # 1. CAS IDÉAL (Vérifie GCC)
    if mask_type == "full":
        mask[:] = 1.0
        
    # 2. CAS ROBUSTES (Vérifient GCC via les bords)
    elif mask_type == "two_walls_l":
        # Observation sur le bord gauche (x < epaisseur) OU bord bas (y < epaisseur)
        epaisseur = ratio * min(Lx, Ly)
        mask[(X <= epaisseur) | (Y <= epaisseur)] = 1.0
        
    elif mask_type == "thick_border":
        # Observation de tout le contour (cadre)
        ep = ratio * min(Lx, Ly)
        mask[(X <= ep) | (X >= Lx - ep) | (Y <= ep) | (Y >= Ly - ep)] = 1.0
        
    # 3. CAS D'ÉCHEC (Ne vérifient PAS la GCC - Observation localisée)
    elif mask_type == "circle_center":
        cx, cy = 0.5 * Lx, 0.5 * Ly
        dist2 = (X - cx)**2 + (Y - cy)**2
        mask = (dist2 <= r**2).astype(float)

    elif mask_type == "circle_excenter":
        # Position décalée pour éviter les symétries parfaites
        cx, cy = 0.75 * Lx, 0.75 * Ly
        dist2 = (X - cx)**2 + (Y - cy)**2
        mask = (dist2 <= r**2).astype(float)

    return mask