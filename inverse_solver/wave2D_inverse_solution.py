import numpy as np
from direct_solver.wave2D_direct_solver import laplacian

def solve_backward_observer(u_final, v_final, c, dx, dt, nt, mask_obs, v_obs, gamma):
    """
    Observateur rétrograde résolu directement en variable t (de T vers 0).
    
    Équation : (1/c^2) u_tt - Δu - gamma * χ_Dobs (u_t - d) = 0
    
    Nous cherchons u^{n-1} en fonction de u^n et u^{n+1}.
    """
    nx, ny = u_final.shape
    
    # On prépare une liste pour stocker la solution.
    # Attention : on va la remplir de la fin vers le début, ou la remplir et l'inverser à la fin.
    # Pour simplifier l'indexation par la suite, on va créer un tableau complet vide et le remplir.
    sol_bwd = [None] * nt
    
    # --- 1. Initialisation à t=T (indice nt-1) ---
    # u^{nt-1} = u_final
    u_n = u_final.copy()      # u à l'instant T
    sol_bwd[nt-1] = u_n.copy()
    
    # --- 2. Premier pas en arrière (calcul de u^{nt-2}) ---
    # Taylor inverse : u(T - dt) approx u(T) - dt*v(T) + 0.5*dt^2 * acc(T)
    # L'accélération à T dépend de l'équation rétrograde : acc = c^2 * Δu + c^2 * gamma * (v - d)
    # Note : Le terme gamma a un signe inversé dans l'équation globale (terme source), 
    # donc acc = c^2 * (Δu + gamma * (v-d))
    
    Lu_n = laplacian(u_n, dx)
    
    # Mesure à l'instant T (si disponible, sinon on prend la dernière)
    d_n = np.zeros_like(u_n)
    if len(v_obs) > 0:
        d_n[mask_obs] = v_obs[-1] # Dernière mesure
    
    # Terme de rétroaction (positif ici car on l'a passé de l'autre côté de l'égalité pour isoler u_tt)
    # u_tt = c^2 Δu + c^2 * gamma * (u_t - d)
    feedback_term = gamma * mask_obs * (v_final - d_n)
    
    u_nm1 = u_n - dt * v_final + 0.5 * (dt**2) * (c**2) * (Lu_n + feedback_term)
    
    # Conditions aux limites
    u_nm1[0,:] = u_nm1[-1,:] = u_nm1[:,0] = u_nm1[:,-1] = 0
    
    sol_bwd[nt-2] = u_nm1.copy()
    
    # Mise à jour des pointeurs pour la boucle
    # Pour la boucle, on décale tout d'un cran vers le "futur" (indices plus élevés)
    # u_np1 (futur) devient l'ancien u_n
    # u_n (présent) devient l'ancien u_nm1
    u_np1 = u_n
    u_n = u_nm1
    
    # --- 3. Boucle temporelle inversée (de nt-2 vers 0) ---
    # On cherche u_{n-1} connaissant u_n et u_{n+1}
    # Indice k représente l'instant où l'on est (n), on cherche k-1
    for k in range(nt-2, 0, -1):
        
        # Vitesse estimée à l'instant k. 
        # Pour être stable explicitement, on peut utiliser (u_{k+1} - u_k) / dt ou centré.
        # Utilisons (u_{k+1} - u_{k-1}) / 2dt -> cela rend le schéma implicite pour u_{k-1}.
        # Simplification explicite : on utilise la vitesse "future" connue ou décentrée (u_{k+1} - u_k)/dt
        v_hat_k = (u_np1 - u_n) / dt 
        
        # Donnée observée à l'instant k
        # v_obs est indexé de 0 à nt-2 généralement. L'indice correspondant est k-1 ou k.
        # Assumons que v_obs[i] correspond au temps t_{i+1}.
        d_k = np.zeros_like(u_n)
        idx_obs = max(0, min(k-1, len(v_obs)-1))
        d_k[mask_obs] = v_obs[idx_obs]
        
        # Terme de Feedback
        # Équation: u_tt = c^2 Δu + c^2 * gamma * (u_t - d)
        # Discret: u_{k-1} = 2u_k - u_{k+1} + dt^2 * c^2 * (Lu_k + gamma*(v_hat - d))
        
        Lu_k = laplacian(u_n, dx)
        feedback = gamma * mask_obs * (v_hat_k - d_k)
        
        u_nm1 = 2*u_n - u_np1 + (dt**2) * (c**2) * (Lu_k + feedback)
        
        # Dirichlet
        u_nm1[0,:] = u_nm1[-1,:] = u_nm1[:,0] = u_nm1[:,-1] = 0
        
        # Stockage
        sol_bwd[k-1] = u_nm1.copy()
        
        # Décalage vers le passé
        u_np1 = u_n
        u_n = u_nm1

    return sol_bwd