import numpy as np
from direct_solver.direct_observer import solve_observer_wave_2d
from inverse_solver.wave2D_inverse_solution import solve_backward_observer

def run_bfn_algorithm(u0_guess, v0_guess, c, dx, dt, nt, mask_obs, v_obs, gamma, num_iterations=10):
    """
    Algorithme BFN avec résolution explicite rétrograde en t.
    """
    print(f"--- Démarrage BFN ({num_iterations} itérations) ---")
    
    uk = u0_guess.copy()
    vk = v0_guess.copy()
    
    history = []

    for k in range(num_iterations):
        # 1. Forward (Observateur Direct) -> de 0 à T
        # On utilise ta fonction existante
        sol_fwd = solve_observer_wave_2d(
            uk, vk, c, dx, dt, nt, 
            mask_obs, v_obs, gamma, 
            enforce_dt_safety=False
        )
        
        # État final à T
        u_T = sol_fwd[-1]
        v_T = (sol_fwd[-1] - sol_fwd[-2]) / dt # Approx vitesse finale
        
        # 2. Backward (Observateur Rétrograde) -> de T à 0
        # On utilise la nouvelle fonction qui boucle à l'envers
        sol_bwd = solve_backward_observer(
            u_T, v_T, c, dx, dt, nt, 
            mask_obs, v_obs, gamma
        )
        
        # État initial 'reconstruit' à t=0 (c'est le dernier élément calculé par la boucle inverse)
        u_0_new = sol_bwd[0]
        # Vitesse initiale reconstruite : (u(dt) - u(0)) / dt
        v_0_new = (sol_bwd[1] - sol_bwd[0]) / dt
        
        # Convergence monitoring
        diff = np.linalg.norm(u_0_new - uk)
        history.append(diff)
        print(f"Iter {k+1}: update norm = {diff:.4e}")
        
        # Mise à jour pour la prochaine itération
        uk = u_0_new
        vk = v_0_new
        
    return uk, vk, history