from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import contextlib
import pickle
import time

import numpy as np

from sim.simulate import run_simulation
from sim.helpers import edit_dat_file  # your existing function


@dataclass(frozen=True)
class Job:
    depth: int
    current_speed: float
    wave_path: Path | None
    dt: float
    simulation_time: float
    save_interval: float


def dt_for_speed(cs: float, table: dict[float, float], default: float = 1e-4) -> float:
    return table.get(round(cs, 2), default)


def sim_time_for_speed_depth(cs: float, depth: float, base: float, multiplier: float) -> float:
    # base + (sqrt(3)/2)*depth/cs if multiplier==sqrt(3)/2
    return base + multiplier * depth / cs


def run_job(
        job: Job,
        *,
        dat_template: Path,
        dat_outdir: Path,
        output_dir: Path,
        wamit_file: Path,
        n: int,
        density: float,
        clean_temp_files: bool,
        skip_existing: bool,
        quiet_moordyn: bool,
        show_progress: bool = True,
        steady_state_tol: float,
) -> tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dat_outdir.mkdir(parents=True, exist_ok=True)

    wave_name = job.wave_path.stem if job.wave_path is not None else "None"
    wave_file_name = str(job.wave_path) if job.wave_path is not None else None

    # Output paths
    dat_file = dat_outdir / f"single_tether4mm_{n}_{job.depth}m_{job.current_speed:.2f}ms.dat"
    out_pkl = (
            output_dir
            / f"{job.depth}m_{job.current_speed:.2f}ms_dt{job.dt:.0e}_T{job.simulation_time:.0f}s__{wave_name}.pkl"
    )
    log_path = out_pkl.with_suffix(".log")

    def log(msg: str) -> None:
        # Always write a timestamped line; flush so it appears even if the job hangs later.
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
            f.flush()

    if skip_existing and out_pkl.exists():
        # Log skips too; otherwise “nothing happened” is ambiguous.
        log("SKIP (output exists)")
        return True, f"SKIP {out_pkl.name}"

    # Initial condition
    x0 = np.zeros(6)
    x0[0] = np.sqrt(n ** 2 - 1) * job.depth / 2.0

    # Template replacements
    replacements = {
        "DEPTH": str(-job.depth),
        "DENSITY": str(density),
        "LENGTH": str(n * job.depth),
        "WTR_DEPTH": str(job.depth),
    }

    t0 = time.perf_counter()
    log(
        "START "
        f"depth={job.depth} cs={job.current_speed:.2f} "
        f"dt={job.dt:g} T={job.simulation_time:g} save_interval={job.save_interval:g} "
        f"wave={wave_name}"
    )

    try:
        edit_dat_file(str(dat_template), str(dat_file), replacements)
        log(f"DAT written: {dat_file.name}")

        results = run_simulation(
            dat_file=str(dat_file),
            wamit_file=str(wamit_file),
            current_speed=job.current_speed,
            depth=job.depth,
            simulation_time=job.simulation_time,
            x0=x0,
            dt=job.dt,
            save_interval=job.save_interval,
            waves=wave_file_name,
            steady_state_tol=steady_state_tol,
            verbose=show_progress,
            quiet_moordyn=quiet_moordyn,
        )

        if not results:
            log("FAIL (empty results)")
            return False, f"EMPTY {job.depth}m {job.current_speed:.2f} {wave_name}"

        with open(out_pkl, "wb") as f:
            pickle.dump(results, f)

        elapsed = time.perf_counter() - t0
        log(f"OK wrote {out_pkl.name} elapsed={elapsed:.1f}s")
        return True, f"OK {out_pkl.name}"

    except Exception as e:
        elapsed = time.perf_counter() - t0
        log(f"FAIL elapsed={elapsed:.1f}s error={type(e).__name__}: {e}")
        return False, f"FAIL depth={job.depth} cs={job.current_speed:.2f} wave={wave_name}: {e}"

    finally:
        if clean_temp_files:
            with contextlib.suppress(Exception):
                dat_file.unlink(missing_ok=True)
            with contextlib.suppress(Exception):
                for p in dat_outdir.glob(f"{dat_file.stem}*.out"):
                    p.unlink(missing_ok=True)
            log("CLEANUP done")
