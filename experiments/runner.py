from __future__ import annotations
from multiprocessing import Pool
from functools import partial
from tqdm import tqdm  # Add this import

from experiments.config import Config
from experiments.job import Job, dt_for_speed, sim_time_for_speed_depth, run_job
from sim.waves import find_wave_files


def build_jobs(cfg: Config) -> list[Job]:
    # wave set
    if cfg.sweep.waves.mode == "none":
        waves = [None]
    else:
        waves = find_wave_files(cfg.paths.wave_dir, cfg.sweep.waves.fp_tokens, cfg.sweep.waves.hs_tokens)
        if cfg.sweep.waves.limit is not None:
            waves = waves[: cfg.sweep.waves.limit]
        if not waves:
            # if you asked for files but found none, fail fast rather than silently running nothing
            raise RuntimeError(f"No wave files found in {cfg.paths.wave_dir}")

    jobs: list[Job] = []
    for depth in cfg.sweep.depths:
        for cs in cfg.sweep.current_speeds:
            dt = dt_for_speed(cs, cfg.timing.dt_table, default=1e-4)

            if cfg.timing.sim_time_mode == "fixed":
                sim_time = cfg.timing.sim_time_fixed
            else:
                sim_time = sim_time_for_speed_depth(
                    cs=cs,
                    depth=depth,
                    base=cfg.timing.sim_time_base,
                    multiplier=cfg.timing.sim_time_multiplier,
                )

            for w in waves:
                jobs.append(Job(
                    depth=depth,
                    current_speed=cs,
                    wave_path=w,
                    dt=dt,
                    simulation_time=sim_time,
                    save_interval=cfg.timing.save_interval,
                ))
    return jobs


def run_job_wrapper(job, common_args):
    return run_job(job, **common_args)


def run_all(cfg: Config) -> int:
    jobs = build_jobs(cfg)

    # Print total number of jobs
    print(f"Running {len(jobs)} simulation jobs")

    # pack constant kwargs once
    common = dict(
        dat_template=cfg.paths.dat_template,
        dat_outdir=cfg.paths.dat_outdir,
        output_dir=cfg.paths.output_dir,
        wamit_file=cfg.system.wamit_file,
        n=cfg.system.n,
        density=cfg.system.density,
        clean_temp_files=cfg.system.clean_temp_files,
        skip_existing=cfg.runner.skip_existing,
        quiet_moordyn=cfg.runner.quiet_moordyn,
        steady_state_tol=cfg.timing.steady_state_tol,
        show_progress=True,
    )

    ok_count = 0
    fail_count = 0

    if cfg.runner.workers <= 1:
        # Single process mode with progress bar
        for job in tqdm(jobs, desc="Progress", unit="job"):
            ok, msg = run_job(job, **common)
            print(msg)
            ok_count += int(ok)
            fail_count += int(not ok)
    else:
        # Multi-process mode with progress bar
        with Pool(processes=cfg.runner.workers) as pool:
            worker_func = partial(run_job_wrapper, common_args=common)

            # Use tqdm with imap to show progress
            for ok, msg in tqdm(
                    pool.imap_unordered(worker_func, jobs, chunksize=1),
                    total=len(jobs),
                    desc="Progress",
                    unit="job"
            ):
                print(msg)
                ok_count += int(ok)
                fail_count += int(not ok)

    print(f"Done. OK={ok_count}, FAIL={fail_count}, TOTAL={len(jobs)}")
    return 0 if fail_count == 0 else 1
