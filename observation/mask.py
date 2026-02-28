"""
Création du masque d'observation.
"""
import numpy as np

def create_observation_mask(X, Y, Lx, Ly, center_frac=0.2):
    """Crée un masque booléen : D_obs = tout sauf un carré central."""
    x_min = (0.5 - center_frac/2) * Lx
    x_max = (0.5 + center_frac/2) * Lx
    y_min = (0.5 - center_frac/2) * Ly
    y_max = (0.5 + center_frac/2) * Ly
    mask_center = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
    return ~mask_center

def get_mask(X, Y, Lx, Ly, mask_type="full", ratio=0.1):
    """
    Génère différents types de masques d'observation.
    ratio : épaisseur des bords relative à la taille du domaine
    """
    mask = np.zeros_like(X)
    
    if mask_type == "full":
        mask[:] = 1.0
        
    elif mask_type == "hole_center":
        center_frac = ratio
        
        x_min = (0.5 - center_frac/2) * Lx
        x_max = (0.5 + center_frac/2) * Lx
        y_min = (0.5 - center_frac/2) * Ly
        y_max = (0.5 + center_frac/2) * Ly
        
        # On définit le centre
        mask_center = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
        
        # On inverse : 1 partout SAUF au centre
        mask = np.logical_not(mask_center).astype(float)

    elif mask_type == "borders":
        # Cadre autour du domaine
        mask[X < ratio*Lx] = 1
        mask[X > (1-ratio)*Lx] = 1
        mask[Y < ratio*Ly] = 1
        mask[Y > (1-ratio)*Ly] = 1
        
    elif mask_type == "left_wall":
        # Uniquement le mur de gauche
        mask[X < ratio*Lx] = 1
        
    elif mask_type == "corners":
        # Juste les 4 coins
        mask[(X < ratio*Lx) & (Y < ratio*Ly)] = 1
        mask[(X > (1-ratio)*Lx) & (Y < ratio*Ly)] = 1
        mask[(X < ratio*Lx) & (Y > (1-ratio)*Ly)] = 1
        mask[(X > (1-ratio)*Lx) & (Y > (1-ratio)*Ly)] = 1

    return mask