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
    wave_dirs: list[Path]


@dataclass(frozen=True)
class SystemCfg:
    n: int
    density: float
    clean_temp_files: bool
    wamit_dirs: list[Path]


@dataclass(frozen=True)
class WavesCfg:
    mode: str  # "files" or "none"
    fp_tokens: list[str]
    hs_tokens: list[str]
    limit: int | None
    use_all: bool


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
    start_pos: str | None


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

    wave_dir_raw = data["paths"].get("wave_dirs", data["paths"].get("wave_dir"))
    if isinstance(wave_dir_raw, list):
        wave_dirs = [Path(x) for x in wave_dir_raw]
    else:
        wave_dirs = [Path(wave_dir_raw)]

    paths = PathsCfg(
        dat_template=Path(data["paths"]["dat_template"]),
        dat_outdir=Path(data["paths"]["dat_outdir"]),
        output_dir=Path(data["paths"]["output_dir"]),
        wave_dirs=wave_dirs,
    )
    wamit_raw = data["system"].get("wamit_dir", data["system"].get("wamit_file"))
    if isinstance(wamit_raw, list):
        wamit_dirs = [Path(x) for x in wamit_raw]
    else:
        wamit_dirs = [Path(wamit_raw)]

    system = SystemCfg(
        n=int(data["system"]["n"]),
        density=float(data["system"]["density"]),
        clean_temp_files=bool(data["system"]["clean_temp_files"]),
        wamit_dirs=wamit_dirs,
    )

    waves_data = data["sweep"].get("waves", {})
    waves = WavesCfg(
        mode=str(waves_data.get("mode", "files")),
        fp_tokens=list(waves_data.get("fp_tokens", [])),
        hs_tokens=list(waves_data.get("hs_tokens", [])),
        limit=waves_data.get("limit", None),
        use_all=bool(waves_data.get("use_all", False)),
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
        safety_factor=float(data["timing"].get("safety_factor", 0.1)),
        start_pos=str(data["timing"].get("start_pos", None)),
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
