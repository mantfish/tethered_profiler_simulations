import contextlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import moordyn
from tqdm import tqdm
from scipy.integrate import solve_ivp

from sim.ArgoMover import ArgoMover
from sim.helpers import get_linear_current_at_depth
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


def _hydro_forces(argo: ArgoMover, x: np.ndarray, xd: np.ndarray, current_at_float, wave: WaveRecord | None, t: float,
                  dt: float):
    # Relative flow
    xd_rel = argo.get_relative_flow(x, xd, current_at_float)

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
        F_k[5],
    ], dtype=float)

    Dv = argo.estimated_viscous_damping()

    # Keep your “quadratic in surge/sway, linear elsewhere” approach
    F_viscous = - np.concatenate((
        (Dv @ (xd_rel * np.abs(xd_rel)))[0:2],
        (Dv @ xd_rel)[2:]
    ))

    return F_k + F_viscous


def _state_derivatives(t, state, md_sys, argo, current_speed, depth, current_profile, wave, quiet_moordyn):
    """Compute the derivatives of the state vector for the ODE solver."""
    if np.any(np.isnan(state)):
        raise RuntimeError(f"NaN detected at t={t:.3f}")

    # 1) External kinematics (current profile)
    with silence_output(quiet_moordyn):
        u = _set_external_kin(md_sys, t, current_speed, depth, current_profile)
        current_at_float = u[0]

    # 2) Mooring force
    x, xd = state[:6], state[6:]
    with silence_output(quiet_moordyn):
        # Note: MoorDyn.Step takes a timestep, but we're using it just to get forces
        # We'll use a small dt for force calculation, the actual integration is handled by solve_ivp
        small_dt = 1e-4  # Small dt for MoorDyn force calculation
        F_moor = np.array(moordyn.Step(md_sys, x.tolist(), xd.tolist(), t, small_dt), dtype=float)

    # 3) Hydrodynamics
    F_hydro = _hydro_forces(argo, x, xd, current_at_float, wave, t, small_dt)

    # 4) Acceleration
    xdd = np.linalg.solve(argo.M_tot, F_hydro + F_moor)

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
):
    if not os.path.exists(dat_file):
        raise FileNotFoundError(f"DAT file not found: {dat_file}")
    if not os.path.exists(wamit_file):
        raise FileNotFoundError(f"WAMIT file not found: {wamit_file}")

    wave = WaveRecord(waves) if waves is not None else None

    argo = ArgoMover(wamit_file, "argo")

    x = x0.astype(float).copy()
    xd = np.zeros(6, dtype=float)

    state = np.hstack((x, xd))

    with moordyn_system(dat_file, x, xd, quiet=quiet_moordyn) as md_sys:
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
        steady_state_reached = False

        # Define event function to check for steady state
        def steady_state_event(t, y, md_sys, argo, current_speed, depth, current_profile, wave, quiet_moordyn):
            # Only check after 20 seconds of simulation time
            if t < 20.0 or not h.tension or prev_tension is None:
                return 1.0

            # Get current tension
            x_curr, xd_curr = y[:6], y[6:]

            # Update MoorDyn state
            with silence_output(quiet_moordyn):
                _set_external_kin(md_sys, t, current_speed, depth, current_profile)
                moordyn.Step(md_sys, x_curr.tolist(), xd_curr.tolist(), t, dt)

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
            nonlocal last_update_time, next_save, prev_tension, prev_state

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

                next_save += save_interval

                # Update for steady state check
                if h.tension:
                    prev_tension = h.tension[-1]
                    prev_state = y.copy()

            # Update progress bar
            if verbose and (t - last_update_time >= 1.0):
                pbar.update(t - last_update_time)
                last_update_time = t

                # Add useful info to progress bar
                x_curr = y[:6]
                heave_ref = x_curr[2] + (wave.height(t) if wave is not None else 0.0)
                pbar.set_postfix({
                    "x": f"{x_curr[0]:.2f}m",
                    "z": f"{x_curr[2]:.2f}m",
                    "pitch": f"{np.degrees(x_curr[4]):.1f}°",
                    "rel_z": f"{heave_ref:.2f}m" if wave is not None else None
                })

            return True  # Continue integration

        with tqdm(
                total=simulation_time,
                desc=sim_desc,
                disable=not verbose,
                unit="s",
                bar_format='{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}]'
        ) as pbar:
            # Run the integration with solve_ivp
            # Create a wrapper for the callback to work with solve_ivp
            def solve_ivp_callback(t, y):
                save_data_callback(t, y)
                return False  # Don't terminate integration

            # Create a wrapper for the steady_state_event function to work with solve_ivp
            # Create a wrapper for the steady_state_event function to work with solve_ivp
            def steady_state_event_wrapper(t, y, *_args):
                return steady_state_event(
                    t, y, md_sys, argo, current_speed, depth, current_profile, wave, quiet_moordyn
                )

            sol = solve_ivp(
                fun=_state_derivatives,
                t_span=(0, simulation_time),
                y0=state,
                method='RK45',  # 4th order Runge-Kutta method
                args=(md_sys, argo, current_speed, depth, current_profile, wave, quiet_moordyn),
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
    from helpers import edit_dat_file

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

    dat_file = f"./dat_files/single_tether4mm_{n}_{depth}m_{cur_speed}ms.dat"
    edit_dat_file("./dat_files/template.dat", dat_file, replacements)

    wamit_file = "./path/to/your.wamit"  # <-- you must set this

    x0 = np.zeros(6)
    x0[0] = np.sqrt((n ** 2 - 1)) * depth / 2

    results = run_simulation(
        dat_file=dat_file,
        wamit_file=wamit_file,
        current_speed=cur_speed,
        depth=depth,
        simulation_time=500,
        x0=x0,
        dt=1e-4,
        verbose=True,
        waves=None,
    )

    with open("wave_test.pkl", "wb") as f:
        pickle.dump(results, f)
