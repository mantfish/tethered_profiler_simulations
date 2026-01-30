import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import moordyn
from tqdm import tqdm
from scipy.integrate import solve_ivp

from sim.ArgoMover import ArgoMover
from sim.helpers import get_linear_current_at_depth, edit_dat_file
from sim.waves import WaveRecord  # keep your existing helpers


@contextlib.contextmanager
def silence_output(enabled: bool = True):
    """Silence stdout/stderr at the OS level (for MoorDyn C output)."""
    if not enabled:
        yield
        return

    # Save the original file descriptors
    save_stdout = os.dup(1)
    save_stderr = os.dup(2)

    # Open null device
    null_fd = os.open(os.devnull, os.O_RDWR)

    try:
        # Replace file descriptors for stdout and stderr with the null device
        os.dup2(null_fd, 1)
        os.dup2(null_fd, 2)
        yield
    finally:
        # Restore original file descriptors
        os.dup2(save_stdout, 1)
        os.dup2(save_stderr, 2)

        # Close all the file descriptors we opened
        os.close(null_fd)
        os.close(save_stdout)
        os.close(save_stderr)


@contextlib.contextmanager
def moordyn_system(dat_file: str, x0: np.ndarray, xd0: np.ndarray, quiet: bool = True):
    """Create + init + always close a MoorDyn system."""
    # Silence the Create call as well
    with silence_output(quiet):
        md = moordyn.Create(dat_file)
    try:
        with silence_output(quiet):
            moordyn.Init(md, x0.tolist(), xd0.tolist())
            moordyn.ExternalWaveKinInit(md)
        yield md
    finally:
        with contextlib.suppress(Exception):
            with silence_output(quiet):  # Also silence the Close call
                moordyn.Close(md)


@dataclass
class Histories:
    t: list
    state: list
    tension: list
    line_pos: list
    wave: list | None


def _safe_line_info(md_sys, line_id: int = 1):
    line = moordyn.GetLine(md_sys, line_id)
    n_nodes = moordyn.GetLineNumberNodes(line)
    return line, n_nodes


def _get_line_tension(line):
    return moordyn.GetLineNodeTen(line, 0)


def _get_line_positions(line, n_nodes: int):
    return np.array([moordyn.GetLineNodePos(line, i) for i in range(n_nodes)], dtype=float)


def _set_external_kin(md_sys, t: float, current_speed: float, depth: float, current_profile: str):
    coords = moordyn.ExternalWaveKinGetCoordinates(md_sys)

    if current_profile == "linear":
        u = [[get_linear_current_at_depth(c[2], current_speed, depth), 0.0, 0.0] for c in coords]
    else:
        u = [[current_speed, 0.0, 0.0] for _ in coords]

    du = [[0.0, 0.0, 0.0] for _ in coords]

    u_arr = np.asarray(u, dtype=float)
    du_arr = np.asarray(du, dtype=float)
    if not np.isfinite(u_arr).all():
        raise RuntimeError(f"non-finite velocity at t={t:.3f}")
    if not np.isfinite(du_arr).all():
        raise RuntimeError(f"non-finite accel at t={t:.3f}")

    moordyn.ExternalWaveKinSet(md_sys, u, du, t)
    return u  # list-of-vecs; u[0] corresponds to the float coordinate you assume


def _hydro_forces(argo: ArgoMover, x: np.ndarray, xd_rel: np.ndarray, wave: WaveRecord | None, t: float,
                  dt: float):

    # Hydrostatic / linear restoring (+ optional wave excitation in heave)
    F_k = -argo.C @ x
    if wave is not None:
        F_k = F_k + argo.C[:, 2] * wave.height(t)

    # Clip (keep your original)
    F_k = np.array([
        F_k[0], F_k[1],
        np.clip(F_k[2], -3.8, 3.8),
        np.clip(F_k[3], -20.0, 20.0),
        np.clip(F_k[4], -20.0, 20.0),
    ], dtype=float)

    Dv = argo.estimated_viscous_damping()

    # Keep your “quadratic in surge/sway, linear elsewhere” approach
    F_viscous =  - np.concatenate((
        (Dv @ (xd_rel * np.abs(xd_rel)))[0:2],
        (Dv @ xd_rel)[2:]
    ))

    return F_k + F_viscous


def _state_derivatives(t, state, md_sys, argo, current_speed, depth, current_profile, wave, quiet_moordyn, diag=None):
    """Compute the derivatives of the state vector for the ODE solver."""
    if np.any(np.isnan(state)):
        raise RuntimeError(f"NaN detected at t={t:.3f}")

    # 1) External kinematics (current profile)
    with silence_output(quiet_moordyn):
        u = _set_external_kin(md_sys, t, current_speed, depth, current_profile)
        current_at_float = u[0]

    # 2) Mooring force
    x, xd = state[:5], state[5:]
    x6 = np.hstack((x, 0.0))
    xd6 = np.hstack((xd, 0.0))
    with silence_output(quiet_moordyn):
        # Note: MoorDyn.Step takes a timestep, but we're using it just to get forces
        # We'll use a small dt for force calculation, the actual integration is handled by solve_ivp
        small_dt = 1e-4  # Small dt for MoorDyn force calculation
        F_moor = np.array(moordyn.Step(md_sys, x6.tolist(), xd6.tolist(), t, small_dt), dtype=float)[:5]

    # 3) Hydrodynamics
    xd_rel = argo.get_relative_flow(x, xd, current_at_float)
    F_hydro = _hydro_forces(argo, x, xd_rel, wave, t, small_dt)

    # 4) Acceleration
    xdd = np.linalg.solve(argo.M_tot, F_hydro + F_moor)

    if diag is not None:
        diag["current_at_float"] = current_at_float
        diag["u_rel_body"] = xd_rel
        diag["F_moor"] = F_moor
        diag["F_hydro"] = F_hydro
        diag["F_tot"] = F_hydro + F_moor

    # Return the derivatives [xd, xdd]
    return np.hstack((xd, xdd))


def run_simulation(
        dat_file: str,
        wamit_file: str,
        current_speed: float,
        depth: float,
        simulation_time: float,
        x0: np.ndarray,
        current_profile: str = "linear",
        steady_state_tol: float = 1e-5,
        dt: float = 1e-4,  # Now used as max_step for adaptive timestep
        save_interval: float = 0.1,
        verbose: bool = False,
        waves=None,
        quiet_moordyn: bool = True,
        safety_factor: float = 0.1,  # Safety factor for first timestep
        plot: bool = False,
        plot_every: float | None = None,
):
    if not os.path.exists(dat_file):
        raise FileNotFoundError(f"DAT file not found: {dat_file}")
    if not os.path.exists(wamit_file):
        raise FileNotFoundError(f"WAMIT file not found: {wamit_file}")

    wave = WaveRecord(waves) if waves is not None else None

    argo = ArgoMover(wamit_file, "argo")

    x = x0.astype(float).copy()
    if x.shape[0] != 5:
        raise ValueError(f"x0 must have length 5 (x, y, z, roll, pitch). Got {x.shape[0]}.")
    xd = np.zeros(5, dtype=float)

    state = np.hstack((x, xd))
    x6 = np.hstack((x, 0.0))
    xd6 = np.hstack((xd, 0.0))

    with moordyn_system(dat_file, x6, xd6, quiet=quiet_moordyn) as md_sys:
        with silence_output(quiet_moordyn):
            float_line, n_nodes = _safe_line_info(md_sys, line_id=1)

        h = Histories(
            t=[0.0],
            state=[state.copy()],
            tension=[],
            line_pos=[],
            wave=[wave.height(0.0)] if wave is not None else None,
        )

        # Seed line outputs (best-effort)
        with contextlib.suppress(Exception):
            with silence_output(quiet_moordyn):
                h.tension.append(_get_line_tension(float_line))
        with contextlib.suppress(Exception):
            with silence_output(quiet_moordyn):
                h.line_pos.append(_get_line_positions(float_line, n_nodes))

        next_save = save_interval
        last_line_pos = None
        plot_interval = save_interval if plot_every is None else plot_every
        if plot_interval <= 0:
            plot_interval = save_interval
        next_plot = plot_interval
        prev_tension = h.tension[-1] if h.tension else None
        prev_state = h.state[-1]

        # Create a progress bar for this simulation
        # Only show if verbose is True
        sim_desc = f"Depth={depth}m, Speed={current_speed:.2f}m/s"
        if waves is not None:
            wave_name = Path(waves).stem if isinstance(waves, (str, Path)) else "wave"
            sim_desc += f", Wave={wave_name}"

        # Configure solve_ivp parameters for adaptive timesteps
        # rtol, atol, and safety_factor are passed from config

        # Setup for tracking progress and saving data
        t_eval = None  # We'll handle data saving manually
        last_update_time = 0.0
        next_verbose_print = save_interval
        steady_state_reached = False

        # Define event function to check for steady state
        def steady_state_event(t, y, md_sys, argo, current_speed, depth, current_profile, wave, quiet_moordyn):
            # Only check after 20 seconds of simulation time
            if t < 20.0 or not h.tension or prev_tension is None:
                return 1.0

            # Get current tension
            x_curr, xd_curr = y[:5], y[5:]
            x6 = np.hstack((x_curr, 0.0))
            xd6 = np.hstack((xd_curr, 0.0))

            # Update MoorDyn state
            with silence_output(quiet_moordyn):
                _set_external_kin(md_sys, t, current_speed, depth, current_profile)
                moordyn.Step(md_sys, x6.tolist(), xd6.tolist(), t, dt)

            # Get tension
            with contextlib.suppress(Exception):
                with silence_output(quiet_moordyn):
                    current_tension = _get_line_tension(float_line)

                    # Calculate changes
                    tension_change = np.linalg.norm(np.asarray(current_tension) - np.asarray(prev_tension))
                    state_change = np.linalg.norm(y - prev_state)

                    # Return negative value when steady state is reached
                    if tension_change < steady_state_tol and state_change < steady_state_tol:
                        return -1.0

            return 1.0

        steady_state_event.terminal = True
        steady_state_event.direction = -1

        # Define a callback function for solve_ivp to track progress and save data
        def save_data_callback(t, y):
            nonlocal last_update_time, next_save, prev_tension, prev_state, last_line_pos, next_plot

            # Save data at specified intervals
            if t >= next_save:
                h.t.append(t)
                h.state.append(y.copy())
                if wave is not None:
                    h.wave.append(wave.height(t))

                # Line outputs (best-effort, but don't hide NaNs)
                with contextlib.suppress(Exception):
                    with silence_output(quiet_moordyn):
                        h.tension.append(_get_line_tension(float_line))
                with contextlib.suppress(Exception):
                    with silence_output(quiet_moordyn):
                        pos = _get_line_positions(float_line, n_nodes)
                        if not np.isfinite(pos).all():
                            raise RuntimeError(f"MoorDyn NaN in node positions at t={t:.3f}")
                        h.line_pos.append(pos)
                        last_line_pos = pos

                next_save += save_interval

                # Update for steady state check
                if h.tension:
                    prev_tension = h.tension[-1]
                    prev_state = y.copy()

                # Verbose state printout at save intervals
                if verbose:
                    x_curr = y[:5]
                    xd_curr = y[5:]
                    tqdm.write(
                        "t={:.2f}s x=[{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}] "
                        "xd=[{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}]".format(
                            t,
                            x_curr[0], x_curr[1], x_curr[2], x_curr[3], x_curr[4],
                            xd_curr[0], xd_curr[1], xd_curr[2], xd_curr[3], xd_curr[4],
                        )
                    )

                if plot and last_line_pos is not None and t >= next_plot:
                    line_plot.set_data(last_line_pos[:, 0], last_line_pos[:, 2])
                    ax.relim()
                    ax.autoscale_view()
                    ax.set_title(f"Tether position at t={t:.1f}s")
                    fig.canvas.draw_idle()
                    plt.pause(0.001)
                    next_plot += plot_interval

            return True  # Continue integration

        if plot:
            import matplotlib.pyplot as plt
            plt.ion()
            fig, ax = plt.subplots()
            line_plot, = ax.plot([], [], "-o", lw=1)
            ax.set_xlabel("x [m]")
            ax.set_ylabel("z [m]")
            plt.show(block=False)

        with tqdm(
                total=simulation_time,
                desc=sim_desc,
                disable=not verbose,
                unit="s",
                bar_format='{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}]'
        ) as pbar:
            # Run the integration with solve_ivp
            def rhs_with_verbose(t, y):
                nonlocal last_update_time, next_verbose_print, next_plot
                diag = {} if verbose else None

                if verbose:
                    if t - last_update_time >= 1.0:
                        pbar.update(t - last_update_time)
                        last_update_time = t

                        # Add useful info to progress bar
                        x_curr = y[:5]
                        heave_ref = x_curr[2] + (wave.height(t) if wave is not None else 0.0)
                        pbar.set_postfix({
                            "x": f"{x_curr[0]:.2f}m",
                            "z": f"{x_curr[2]:.2f}m",
                            "pitch": f"{np.degrees(x_curr[4]):.1f}°",
                            "rel_z": f"{heave_ref:.2f}m" if wave is not None else None
                        })

                deriv = _state_derivatives(
                    t, y, md_sys, argo, current_speed, depth, current_profile, wave, quiet_moordyn, diag
                )

                if verbose and t >= next_verbose_print:
                    x_curr = y[:5]
                    xd_curr = y[5:]
                    u_rel = diag.get("u_rel_body")
                    F_moor = diag.get("F_moor")
                    F_hydro = diag.get("F_hydro")
                    F_tot = diag.get("F_tot")
                    tqdm.write(
                        "t={:.2f}s x=[{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}] "
                        "xd=[{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}] "
                        "u_rel=[{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}] "
                        "F_moor=[{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}] "
                        "F_hydro=[{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}] "
                        "F_tot=[{:.3f}, {:.3f}, {:.3f}, {:.3f}, {:.3f}]".format(
                            t,
                            x_curr[0], x_curr[1], x_curr[2], x_curr[3], x_curr[4],
                            xd_curr[0], xd_curr[1], xd_curr[2], xd_curr[3], xd_curr[4],
                            u_rel[0], u_rel[1], u_rel[2], u_rel[3], u_rel[4],
                            F_moor[0], F_moor[1], F_moor[2], F_moor[3], F_moor[4],
                            F_hydro[0], F_hydro[1], F_hydro[2], F_hydro[3], F_hydro[4],
                            F_tot[0], F_tot[1], F_tot[2], F_tot[3], F_tot[4],
                        )
                    )
                    next_verbose_print += save_interval

                if plot and t >= next_plot:
                    with contextlib.suppress(Exception):
                        with silence_output(quiet_moordyn):
                            pos = _get_line_positions(float_line, n_nodes)
                        if not np.isfinite(pos).all():
                            raise RuntimeError(f"MoorDyn NaN in node positions at t={t:.3f}")
                        line_plot.set_data(pos[:, 0], pos[:, 2])
                        ax.relim()
                        ax.autoscale_view()
                        ax.set_title(f"Tether position at t={t:.1f}s")
                        fig.canvas.draw_idle()
                        plt.pause(0.001)
                        next_plot += plot_interval

                return deriv

            # Create a wrapper for the steady_state_event function to work with solve_ivp
            # Create a wrapper for the steady_state_event function to work with solve_ivp
            def steady_state_event_wrapper(t, y, *_args):
                return steady_state_event(
                    t, y, md_sys, argo, current_speed, depth, current_profile, wave, quiet_moordyn
                )

            sol = solve_ivp(
                fun=rhs_with_verbose,
                t_span=(0, simulation_time),
                y0=state,
                method='RK45',  # 4th order Runge-Kutta method
                max_step=dt,  # Conservative max step size
                events=steady_state_event_wrapper,
                dense_output=True,  # Enable dense output for callback
                vectorized=False,
                first_step=dt * safety_factor,  # Start with a small first step based on safety factor
            )

            # Process the solution with our callback to collect data at desired intervals
            t_dense = np.linspace(0, sol.t[-1], int(sol.t[-1] / save_interval) + 1)
            sol_dense = sol.sol(t_dense)
            for i, t in enumerate(t_dense):
                if t > 0:  # Skip t=0 as we already have it
                    save_data_callback(t, sol_dense[:, i])

            # Process the solution
            if sol.status == 1:  # Terminated due to event (steady state)
                if verbose:
                    pbar.set_postfix({"status": "Steady state reached"})
                steady_state_reached = True

            # Ensure the progress bar is updated to the final time
            if verbose:
                final_t = sol.t[-1]
                if final_t > last_update_time:
                    pbar.update(final_t - last_update_time)

            # Add the final state if it's not already saved
            if h.t[-1] < sol.t[-1]:
                h.t.append(sol.t[-1])
                h.state.append(sol.y[:, -1].copy())
                if wave is not None:
                    h.wave.append(wave.height(sol.t[-1]))

                # Final line outputs
                with contextlib.suppress(Exception):
                    with silence_output(quiet_moordyn):
                        h.tension.append(_get_line_tension(float_line))
                with contextlib.suppress(Exception):
                    with silence_output(quiet_moordyn):
                        pos = _get_line_positions(float_line, n_nodes)
                        h.line_pos.append(pos)

    results = {
        "time_hist": np.asarray(h.t),
        "state_hist": np.asarray(h.state),
        "tension_hist": np.asarray(h.tension) if h.tension else np.empty((0, 3)),
        "line_pos_hist": np.asarray(h.line_pos) if h.line_pos else np.empty((0, 0, 3)),
    }
    if wave is not None:
        results["waves"] = np.asarray(h.wave)

    return results


if __name__ == "__main__":
    import pickle

    depth = 30
    n = 2
    density = 1025
    cur_speed = 0.25

    replacements = {
        "DEPTH": str(-depth),
        "DENSITY": str(density),
        "LENGTH": str(n * depth),
        "WTR_DEPTH": str(depth),
    }

    dat_file = f"./data/dat_files/single_tether4mm_{n}_{depth}m_{cur_speed}ms.dat"
    edit_dat_file("./data/dat_files/template.dat", dat_file, replacements)

    wamit_file = "/home/ddyob/Documents/tethered_argo/tethered_profiler_simulations/data/wamit/ArgoBoxStiffness"  # <-- you must set this

    x0 = np.zeros(5)
    x0[0] = np.sqrt((n ** 2 - 1)) * depth

    results = run_simulation(
        dat_file=dat_file,
        wamit_file=wamit_file,
        current_speed=cur_speed,
        depth=depth,
        simulation_time=500,
        x0=x0,
        dt=5e-4,
        verbose=True,
        waves=None,
        quiet_moordyn=False,
        plot=True,
        plot_every=1,
    )

    with open("wave_test.pkl", "wb") as f:
        pickle.dump(results, f)
