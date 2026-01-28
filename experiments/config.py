from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathsCfg:
    dat_template: Path
    dat_outdir: Path
    output_dir: Path
    wave_dir: Path


@dataclass(frozen=True)
class SystemCfg:
    n: int
    density: float
    clean_temp_files: bool
    wamit_file: Path


@dataclass(frozen=True)
class WavesCfg:
    mode: str  # "files" or "none"
    fp_tokens: list[str]
    hs_tokens: list[str]
    limit: int | None


@dataclass(frozen=True)
class TimingCfg:
    save_interval: float
    steady_state_tol: float
    dt_table: dict[float, float]
    sim_time_mode: str      # "fixed" or "speed_depth"
    sim_time_fixed: float
    sim_time_base: float
    sim_time_multiplier: float
    safety_factor: float    # Safety factor for first timestep


@dataclass(frozen=True)
class SweepCfg:
    depths: list[int]
    current_speeds: list[float]
    waves: WavesCfg


@dataclass(frozen=True)
class RunnerCfg:
    workers: int
    skip_existing: bool
    quiet_moordyn: bool


@dataclass(frozen=True)
class Config:
    paths: PathsCfg
    system: SystemCfg
    sweep: SweepCfg
    timing: TimingCfg
    runner: RunnerCfg


def load_config(path: str | Path) -> Config:
    path = Path(path)
    data: dict[str, Any] = yaml.safe_load(path.read_text())

    paths = PathsCfg(
        dat_template=Path(data["paths"]["dat_template"]),
        dat_outdir=Path(data["paths"]["dat_outdir"]),
        output_dir=Path(data["paths"]["output_dir"]),
        wave_dir=Path(data["paths"]["wave_dir"]),
    )
    system = SystemCfg(
        n=int(data["system"]["n"]),
        density=float(data["system"]["density"]),
        clean_temp_files=bool(data["system"]["clean_temp_files"]),
        wamit_file=Path(data["system"]["wamit_file"]),
    )
    waves = WavesCfg(
        mode=str(data["sweep"]["waves"]["mode"]),
        fp_tokens=list(data["sweep"]["waves"].get("fp_tokens", [])),
        hs_tokens=list(data["sweep"]["waves"].get("hs_tokens", [])),
        limit=data["sweep"]["waves"].get("limit", None),
    )

    dt_table_raw = data["timing"].get("dt_table", {})
    dt_table = {float(k): float(v) for k, v in dt_table_raw.items()}

    timing = TimingCfg(
        save_interval=float(data["timing"]["save_interval"]),
        steady_state_tol=float(data["timing"]["steady_state_tol"]),
        dt_table=dt_table,
        sim_time_mode=str(data["timing"]["sim_time"]["mode"]),
        sim_time_fixed=float(data["timing"]["sim_time"]["fixed"]),
        sim_time_base=float(data["timing"]["sim_time"]["base"]),
        sim_time_multiplier=float(data["timing"]["sim_time"]["multiplier"]),
        safety_factor=float(data["timing"].get("safety_factor", 0.1)),  # Default to 0.1 if not specified
    )
    sweep = SweepCfg(
        depths=[int(x) for x in data["sweep"]["depths"]],
        current_speeds=[float(x) for x in data["sweep"]["current_speeds"]],
        waves=waves,
    )
    runner = RunnerCfg(
        workers=int(data["runner"]["workers"]),
        skip_existing=bool(data["runner"]["skip_existing"]),
        quiet_moordyn=bool(data["runner"]["quiet_moordyn"]),
    )

    return Config(paths=paths, system=system, sweep=sweep, timing=timing, runner=runner)
