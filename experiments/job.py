from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pickle
import numpy as np
import contextlib

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
    steady_state_tol: float,
) -> tuple[bool, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dat_outdir.mkdir(parents=True, exist_ok=True)

    wave_name = job.wave_path.stem if job.wave_path is not None else "None"
    wave_file_name = str(job.wave_path) if job.wave_path is not None else None

    x0 = np.zeros(6)
    x0[0] = np.sqrt(n**2 - 1) * job.depth / 2.0

    replacements = {
        "DEPTH": str(-job.depth),
        "DENSITY": str(density),
        "LENGTH": str(n * job.depth),
        "WTR_DEPTH": str(job.depth),
    }

    dat_file = dat_outdir / f"single_tether4mm_{n}_{job.depth}m_{job.current_speed:.2f}ms.dat"
    out_pkl = output_dir / f"{job.depth}m_{job.current_speed:.2f}ms_dt{job.dt:.0e}_T{job.simulation_time:.0f}s__{wave_name}.pkl"

    if skip_existing and out_pkl.exists():
        return True, f"SKIP {out_pkl.name}"

    edit_dat_file(str(dat_template), str(dat_file), replacements)

    try:
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
            verbose=False,
            quiet_moordyn=quiet_moordyn,
        )
        if not results:
            return False, f"EMPTY {job.depth}m {job.current_speed:.2f} {wave_name}"
        with open(out_pkl, "wb") as f:
            pickle.dump(results, f)
        return True, f"OK {out_pkl.name}"
    except Exception as e:
        return False, f"FAIL depth={job.depth} cs={job.current_speed:.2f} wave={wave_name}: {e}"
    finally:
        if clean_temp_files:
            with contextlib.suppress(Exception):
                dat_file.unlink(missing_ok=True)
            with contextlib.suppress(Exception):
                for p in dat_outdir.glob(f"{dat_file.stem}*.out"):
                    p.unlink(missing_ok=True)
