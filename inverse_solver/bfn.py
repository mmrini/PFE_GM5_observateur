import numpy as np
from direct_solver.direct_observer import solve_direct_observer
from direct_solver.wave2D_direct_solver import laplacian
from inverse_solver.inverse_observer import solve_backward_observer

def run_bfn_algorithm(u0_guess, v0_guess, u0_orig, v0_orig, c, dx, dt, nt, mask_obs, v_obs, gamma, num_iterations=10):
    """
    Algorithme BFN avec résolution explicite rétrograde en t.
    """
    print(f"--- Démarrage BFN ({num_iterations} itérations) ---")
    
    uk = u0_guess.copy()
    vk = v0_guess.copy()
    
    history_u = []
    history_v = []

    for k in range(num_iterations):
        # 1. Forward (Observateur Direct) -> de 0 à T
        sol_fwd = solve_direct_observer(
            uk, vk, c, dx, dt, nt, 
            mask_obs, v_obs, gamma
        )
        
        # État final à T
        u_T = sol_fwd[nt] #[-1]
        u_T_m1 = sol_fwd[nt-1] #[-2]
        
        # 2. Backward (Observateur Rétrograde) -> de T à 0
        sol_bwd = solve_backward_observer(
            u_T, u_T_m1, c, dx, dt, nt, 
            mask_obs, v_obs, gamma
        )
        
        # État initial 'reconstruit' à t=0 (c'est le dernier élément calculé par la boucle inverse)
        u_0_new = sol_bwd[0]
        u_1_new = sol_bwd[1]

        # Vitesse initiale reconstruite
        Lu0 = laplacian(u_0_new, dx)
        v_0_new = (u_1_new - u_0_new - 0.5 * (dt**2) * (c**2) * Lu0) / dt

        # Convergence monitoring
        diff_u = np.linalg.norm(u_0_new - u0_orig) 
        diff_v = np.linalg.norm(v_0_new - v0_orig)

        history_u.append(diff_u)
        history_v.append(diff_v)
        print(f"Iter {k+1}: update norm = {diff_u:.4e}, v = {diff_v:.4e}")
        
        # Mise à jour pour la prochaine itération
        uk = u_0_new
        vk = v_0_new
        
    return uk, vk, history_u, history_v