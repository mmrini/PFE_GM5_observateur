import os
import time
import argparse

import numpy as np

from config import Lx, Ly, nx, ny, T, nt, c0
from direct_solver.wave2D_direct_solver import solve_wave_2d
from direct_solver.direct_observer import solve_direct_observer
from direct_solver.direct_observer import compute_error_energy
from inverse_solver.bfn import run_bfn_algorithm
from observation.mask import get_mask
from observation.measurement import compute_velocity, extract_observed_velocity
from visualization.plot_fields import plot_mask, plot_overlay_3d
from visualization.save_utils import ensure_data_folder
import matplotlib.pyplot as plt


def ensure_experiments_folder():
    base = ensure_data_folder()
    exp_dir = os.path.join(base, "experiments")
    os.makedirs(exp_dir, exist_ok=True)
    return exp_dir


def run_experiments(mask_types, gammas, ratios, n_iter, num_iterations_bfn, save_dir, verbose=False):
    results = []

    # Mesh
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    dx = x[1] - x[0]
    X, Y = np.meshgrid(x, y, indexing='ij')

    dt = T / nt

    # True initial conditions 
    sigma = 0.05 * Lx
    x0, y0 = 0.5 * Lx, 0.5 * Ly
    u0 = np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2))
    u0[0,:] = 0; u0[-1,:] = 0; u0[:,0] = 0; u0[:,-1] = 0
    v0 = np.zeros_like(u0)

    # Solve forward once
    if verbose: print("Running forward (true) wave simulation...")
    sol, used_dt = solve_wave_2d(u0, v0, c0, dx, dt, nt)
    v_all = compute_velocity(sol, used_dt)

    exp_dir = save_dir

    for mask_type in mask_types:
        for ratio in ratios:
            # Build mask
            mask_obs = get_mask(X, Y, Lx, Ly, mask_type=mask_type, ratio=ratio)

            mask_obs = mask_obs.astype(bool)

            plot_mask(X, Y, mask_obs, save=True, type_mask=mask_type)

            # Extract observed velocity
            v_obs = extract_observed_velocity(v_all, mask_obs)

            for gamma in gammas:
                if verbose:
                    print(f"Experiment: mask={mask_type}, ratio={ratio}, gamma={gamma}")

                # start timer
                t0 = time.time()

                # initial guesses (zeros)
                u0_guess = np.zeros_like(u0)
                v0_guess = np.zeros_like(v0)

                # Compute direct observer solution (for energy decay plot)
                sol_hat = solve_direct_observer(u0_guess, v0_guess, c0, dx, used_dt, nt, mask_obs, v_obs, gamma)
                try:
                    E_error = compute_error_energy(sol, sol_hat, c0, dx, used_dt)[1:]
                except Exception:
                    E_error = None

                # Run BFN
                u0_rec, v0_rec, history_u, history_v = run_bfn_algorithm(
                    u0_guess, v0_guess, u0, v0, c0, dx, used_dt, nt,
                    mask_obs, v_obs, gamma, num_iterations=num_iterations_bfn
                )

                timestamp = int(time.time())
                if verbose:
                    f_overlay = os.path.join(exp_dir, f"3D_overlay_{mask_type}_g{gamma}_{timestamp}.png")
                    plot_overlay_3d(X, Y, u0, v0, u0_rec, v0_rec, 
                                    title=f"Masque: {mask_type}, Gamma: {gamma}", 
                    save_path=f_overlay)
                duration = time.time() - t0

                # Errors
                u0_err = float(np.linalg.norm(u0_rec - u0))
                v0_err = float(np.linalg.norm(v0_rec - v0))

                # Save reconstructions
                ufile = os.path.join(exp_dir, f"u0_rec_{mask_type}_r{ratio}_g{gamma}_{timestamp}.npy")
                vfile = os.path.join(exp_dir, f"v0_rec_{mask_type}_r{ratio}_g{gamma}_{timestamp}.npy")
                np.save(ufile, u0_rec)
                np.save(vfile, v0_rec)

                # --- Visualizations ---
                # True initial
                try:
                    fig = plt.figure(figsize=(6,5))
                    plt.imshow(u0, origin='lower', cmap='viridis')
                    plt.colorbar(); plt.title('True initial u0')
                    f1 = os.path.join(exp_dir, f"u0_true_{mask_type}_r{ratio}_g{gamma}_{timestamp}.png")
                    fig.savefig(f1, bbox_inches='tight', dpi=200)
                    plt.close(fig)

                    # Reconstructed u0
                    fig = plt.figure(figsize=(6,5))
                    plt.imshow(u0_rec, origin='lower', cmap='viridis')
                    plt.colorbar(); plt.title('Reconstructed u0 (BFN)')
                    f2 = os.path.join(exp_dir, f"u0_rec_img_{mask_type}_r{ratio}_g{gamma}_{timestamp}.png")
                    fig.savefig(f2, bbox_inches='tight', dpi=200)
                    plt.close(fig)

                    # Error map
                    fig = plt.figure(figsize=(6,5))
                    plt.imshow(u0 - u0_rec, origin='lower', cmap='seismic')
                    plt.colorbar(); plt.title('Reconstruction error (u0 - u0_rec)')
                    f3 = os.path.join(exp_dir, f"u0_error_{mask_type}_r{ratio}_g{gamma}_{timestamp}.png")
                    fig.savefig(f3, bbox_inches='tight', dpi=200)
                    plt.close(fig)

                    # Reconstructed v0
                    fig = plt.figure(figsize=(6,5))
                    plt.imshow(v0_rec, origin='lower', cmap='viridis')
                    plt.colorbar(); plt.title('Reconstructed v0 (BFN)')
                    f4 = os.path.join(exp_dir, f"v0_rec_img_{mask_type}_r{ratio}_g{gamma}_{timestamp}.png")
                    fig.savefig(f4, bbox_inches='tight', dpi=200)
                    plt.close(fig)

                    # Convergence history
                    fig = plt.figure(figsize=(6,4))
                    plt.plot(history_u, 'o-')
                    plt.xlabel('BFN iteration'); plt.ylabel('||update||'); plt.grid(True)
                    plt.title('BFN convergence history (u0)')
                    f5 = os.path.join(exp_dir, f"bfn_history_{mask_type}_r{ratio}_g{gamma}_{timestamp}.png")
                    fig.savefig(f5, bbox_inches='tight', dpi=200)
                    plt.close(fig)

                    # Convergence history for v0
                    fig = plt.figure(figsize=(6,4))
                    plt.plot(history_v, 'o-')
                    plt.xlabel('BFN iteration'); plt.ylabel('||update||'); plt.grid(True)
                    plt.title('BFN convergence history (v0)')
                    f7 = os.path.join(exp_dir, f"bfn_history_v0_{mask_type}_r{ratio}_g{gamma}_{timestamp}.png")
                    fig.savefig(f7, bbox_inches='tight', dpi=200)
                    plt.close(fig)

                    # Energy decay from direct observer 
                    if E_error is not None:
                        fig = plt.figure(figsize=(6,4))
                        times = np.arange(len(E_error)) * used_dt
                        plt.plot(times, E_error, '-'); plt.xlabel('Time [s]'); plt.ylabel('Error energy')
                        plt.title('Direct observer error energy decay')
                        plt.grid(True)
                        f6 = os.path.join(exp_dir, f"energy_decay_{mask_type}_r{ratio}_g{gamma}_{timestamp}.png")
                        fig.savefig(f6, bbox_inches='tight', dpi=200)
                        plt.close(fig)

                except Exception as e:
                    if verbose:
                        print('Plotting failed:', e)

                results.append({
                    'mask_type': mask_type,
                    'ratio': ratio,
                    'gamma': gamma,
                    'u0_err': u0_err,
                    'v0_err': v0_err,
                    'history_u': history_u,
                    'history_v': history_v,
                    'duration_s': duration,
                    'u_file': ufile,
                    'v_file': vfile,
                })

                if verbose:
                    print(f" -> u0 err: {u0_err:.4e}, v0 err: {v0_err:.4e}, time: {duration:.2f}s")

    return results


def parse_list(arg, cast=float):
    parts = arg.split(',')
    return [cast(p) for p in parts]


def main():
    parser = argparse.ArgumentParser(description='Run automated BFN experiments.')
    parser.add_argument('--masks', type=str, default='full', help='Comma-separated mask types')
    parser.add_argument('--ratios', type=str, default='0.1', help='Comma-separated ratios used by masks (float)')
    parser.add_argument('--radius', type=str, default='0.1', help='Comma-separated radiuses used by circular masks (float)')
    parser.add_argument('--gammas', type=str, default='1.0', help='Comma-separated gamma values')
    parser.add_argument('--n_iter', type=int, default=1, help='Number of times to repeat each experiment')
    parser.add_argument('--bfn_iters', type=int, default=10, help='Number of BFN iterations')
    parser.add_argument('--out', type=str, default=None, help='Directory to save experiment data (default: data/experiments)')
    parser.add_argument('--verbose', action='store_true')

    args = parser.parse_args()

    mask_types = args.masks.split(',')
    ratios = parse_list(args.ratios, float)
    gammas = parse_list(args.gammas, float)

    save_dir = args.out if args.out is not None else ensure_experiments_folder()

    all_results = []
    for _ in range(args.n_iter):
        res = run_experiments(mask_types, gammas, ratios, nt, args.bfn_iters, save_dir, verbose=args.verbose)
        all_results.extend(res)

    print("All done. Results folder:", save_dir)


if __name__ == '__main__':
    main()
