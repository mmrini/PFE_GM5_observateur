import numpy as np

def create_observation_mask(X, Y, Lx, Ly, center_frac=0.2):
    """Crée un masque booléen : D_obs = tout sauf un carré central."""
    x_min = (0.5 - center_frac/2) * Lx
    x_max = (0.5 + center_frac/2) * Lx
    y_min = (0.5 - center_frac/2) * Ly
    y_max = (0.5 + center_frac/2) * Ly
    mask_center = (X >= x_min) & (X <= x_max) & (Y >= y_min) & (Y <= y_max)
    return ~mask_center
