import contextlib
import os
from dataclasses import dataclass

import numpy as np
import moordyn

from sim.ArgoMover import ArgoMover
from sim.helpers import get_linear_current_at_depth
from sim.waves import WaveRecord  # keep your existing helpers


@contextlib.contextmanager
def silence_output(enabled: bool = True):
    """Optionally silence stdout/stderr (MoorDyn can be chatty)."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


@contextlib.contextmanager
def moordyn_system(dat_file: str, x0: np.ndarray, xd0: np.ndarray, quiet: bool = True):
    """Create + init + always close a MoorDyn system."""
    md = moordyn.Create(dat_file)
    try:
        with silence_output(quiet):
            moordyn.Init(md, x0.tolist(), xd0.tolist())
            moordyn.ExternalWaveKinInit(md)
        yield md
    finally:
        with contextlib.suppress(Exception):
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


def _hydro_forces(argo: ArgoMover, x: np.ndarray, xd: np.ndarray, current_at_float, wave: WaveRecord | None, t: float, dt: float):
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
    F_viscous = np.concatenate((
        (Dv @ (xd_rel * np.abs(xd_rel)))[0:2],
        (Dv @ xd_rel)[2:]
    ))

    return F_k + F_viscous


def run_simulation(
    dat_file: str,
    wamit_file: str,
    current_speed: float,
    depth: float,
    simulation_time: float,
    x0: np.ndarray,
    current_profile: str = "linear",
    steady_state_tol: float = 1e-5,
    dt: float = 1e-4,
    save_interval: float = 0.1,
    verbose: bool = False,
    waves=None,
    quiet_moordyn: bool = True,
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
            h.tension.append(_get_line_tension(float_line))
        with contextlib.suppress(Exception):
            h.line_pos.append(_get_line_positions(float_line, n_nodes))

        nt = int(np.ceil(simulation_time / dt))
        t = 0.0
        next_save = 0.0

        prev_tension = h.tension[-1] if h.tension else None
        prev_state = h.state[-1]

        for k in range(nt):
            t += dt

            if np.any(np.isnan(state)):
                raise RuntimeError(f"NaN detected at t={t:.3f}")

            # 1) External kinematics (current profile)
            with silence_output(quiet_moordyn):
                u = _set_external_kin(md_sys, t, current_speed, depth, current_profile)
                current_at_float = u[0]

            # 2) Mooring force
            x, xd = state[:6], state[6:]
            with silence_output(quiet_moordyn):
                F_moor = np.array(moordyn.Step(md_sys, x.tolist(), xd.tolist(), t, dt), dtype=float)

            # 3) Hydrodynamics
            F_hydro = _hydro_forces(argo, x, xd, current_at_float, wave, t, dt)

            # 4) Acceleration + integrate
            xdd = np.linalg.solve(argo.M_tot, F_hydro + F_moor)
            xd = xd + dt * xdd
            x = x + dt * xd
            state = np.hstack((x, xd))

            # 5) Save
            if t >= next_save:
                h.t.append(t)
                h.state.append(state.copy())
                if wave is not None:
                    h.wave.append(wave.height(t))

                # line outputs (best-effort, but don’t hide NaNs)
                with contextlib.suppress(Exception):
                    h.tension.append(_get_line_tension(float_line))
                with contextlib.suppress(Exception):
                    pos = _get_line_positions(float_line, n_nodes)
                    if not np.isfinite(pos).all():
                        raise RuntimeError(f"MoorDyn NaN in node positions at t={t:.3f}")
                    h.line_pos.append(pos)

                next_save += save_interval

                # 6) steady-state check
                if prev_tension is not None and h.tension and t > 20.0:
                    tension_change = np.linalg.norm(np.asarray(h.tension[-1]) - np.asarray(prev_tension))
                    state_change = np.linalg.norm(h.state[-1] - prev_state)
                    if tension_change < steady_state_tol and state_change < steady_state_tol:
                        if verbose:
                            print(f"Steady state reached at t={t:.1f}s")
                        break
                    prev_tension = h.tension[-1]
                    prev_state = h.state[-1]

            # 7) progress
            if verbose and (k % max(1, int(5.0 / dt)) == 0):
                heave_ref = x[2] + (wave.height(t) if wave is not None else 0.0)
                wtxt = f", wave={wave.height(t):.3f}, rel={heave_ref:.3f}" if wave is not None else f", rel={heave_ref:.3f}"
                print(f"t={t:.1f}s, Surge={x[0]:.3f}m, Heave={x[2]:.3f}m, Pitch={np.degrees(x[4]):.1f}{wtxt}")

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
    x0[0] = np.sqrt((n**2 - 1)) * depth / 2

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
