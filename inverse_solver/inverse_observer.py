import numpy as np
from direct_solver.wave2D_direct_solver import laplacian

def solve_backward_observer(u_N, u_Nm1, c, dx, dt, nt, mask_obs, v_obs, gamma):
    # u_N est sol_fwd[nt], u_Nm1 est sol_fwd[nt-1]
    sol_bwd = [None] * (nt + 1)
    
    sol_bwd[nt] = u_N.copy()
    sol_bwd[nt-1] = u_Nm1.copy()
    
    u_np1 = u_N
    u_n = u_Nm1

    coeff_gamma = (gamma * (c**2) * dt) / 2.0
    mask_obs = mask_obs.astype(bool)
    denom = 1.0 + coeff_gamma * mask_obs

    # On part de nt-1 car on connaît déjà nt et nt-1
    for k in range(nt-1, 0, -1): 
        d_k = np.zeros_like(u_n)
        d_k[mask_obs] = v_obs[k]

        Lu = laplacian(u_n, dx)
        
        # Formule Backward
        term_future = (1.0 - coeff_gamma * mask_obs) * u_np1
        numerateur = 2*u_n - term_future + (dt**2) * (c**2) * (Lu - gamma * mask_obs * d_k)
        
        u_nm1 = numerateur / denom
        u_nm1[0,:] = u_nm1[-1,:] = u_nm1[:,0] = u_nm1[:,-1] = 0
        
        sol_bwd[k-1] = u_nm1.copy()
        u_np1, u_n = u_n, u_nm1

    return sol_bwd

