from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union
import re
import numpy as np

Array = np.ndarray


@dataclass
class WaveGenerator:
    """
    Generate irregular sea-surface elevation records η(t) using a JONSWAP spectrum.

    Typical usage:
        wg = WaveGenerator(outdir="real_outputs", seed=123)
        wg.generate_grid(Hs_list=[0.1, 0.2], Fp_list=[0.1, 0.2])  # writes csvs

        # single-case in-memory:
        t, eta, (f, S) = wg.generate(Hs=0.8, Fp=0.4, write=True)
    """

    outdir: Union[str, Path] = Path("real_outputs")

    # time settings
    dt: float = 0.01
    t_total: float = 1000.0

    # spectrum settings
    gamma: float = 3.3
    fmin: float = 0.03
    fmax: float = 1.0
    nf: int = 4000
    g: float = 9.81

    # reproducibility
    seed: Optional[int] = None

    # filename formatting
    float_fmt: str = ".2f"  # used for Hs/Fp tags

    def __post_init__(self) -> None:
        self.outdir = Path(self.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self._rng = np.random.default_rng(self.seed)

        if self.dt <= 0 or self.t_total <= 0:
            raise ValueError("dt and t_total must be positive.")
        if self.fmin <= 0 or self.fmax <= 0 or self.fmax <= self.fmin:
            raise ValueError("Require 0 < fmin < fmax.")
        if self.nf < 2:
            raise ValueError("nf must be >= 2.")

    # ----------------------------
    # Public API
    # ----------------------------
    def generate(
            self,
            *,
            Hs: float,
            Fp: float,
            write: bool = True,
            write_spectrum: bool = True,
            prefix: str = "",
    ) -> Tuple[Array, Array, Tuple[Array, Array]]:
        """
        Generate one (t, eta) record for given Hs and Fp.

        Args:
            Hs: significant wave height [m]
            Fp: peak frequency [Hz]
            write: whether to write eta CSV to outdir
            write_spectrum: whether to also write spectrum CSV
            prefix: optional prefix for filenames (e.g. "caseA_")

        Returns:
            (t, eta, (f, S))
        """
        if Hs <= 0:
            raise ValueError("Hs must be > 0.")
        if Fp <= 0:
            raise ValueError("Fp must be > 0.")

        Tp = 1.0 / Fp
        t, eta, (f, S) = self._synth_wave(Hs=Hs, Tp=Tp)

        tag = self._tag(Hs=Hs, Fp=Fp)
        if write:
            self._write_eta_csv(prefix + tag, t, eta)
            if write_spectrum:
                self._write_spec_csv(prefix + tag, f, S)

        return t, eta, (f, S)

    def generate_grid(
            self,
            *,
            Hs_list: Iterable[float],
            Fp_list: Iterable[float],
            write_spectrum: bool = True,
            prefix: str = "",
            verbose: bool = True,
    ) -> None:
        """
        Generate a grid of files, one per (Hs, Fp).
        """
        for Hs in Hs_list:
            for Fp in Fp_list:
                self.generate(
                    Hs=float(Hs),
                    Fp=float(Fp),
                    write=True,
                    write_spectrum=write_spectrum,
                    prefix=prefix,
                )
                if verbose:
                    print(f"Wrote {self._tag(Hs=float(Hs), Fp=float(Fp))}")

        if verbose:
            print("\nAll combinations done – files in:", self.outdir.resolve())

    def set_seed(self, seed: Optional[int]) -> None:
        """Reset RNG seed (affects phases)."""
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

    # ----------------------------
    # Core math
    # ----------------------------
    def _jonswap_spectrum(self, f: Array, Hs: float, Tp: float) -> Array:
        """One-sided JONSWAP S(f) normalized to integrate to Hs^2 / 16."""
        fp = 1.0 / Tp
        sigma = np.where(f <= fp, 0.07, 0.09)
        r = np.exp(-((f - fp) ** 2) / (2 * sigma ** 2 * fp ** 2))

        s0 = (f ** -5) * np.exp(-1.25 * (fp / f) ** 4) * (self.gamma ** r)

        target = (Hs ** 2) / 16.0
        alpha = target / np.trapz(s0, f)
        return alpha * s0

    def _synth_wave(self, *, Hs: float, Tp: float) -> Tuple[Array, Array, Tuple[Array, Array]]:
        """Return (t, eta, (f, S))."""
        f = np.linspace(self.fmin, self.fmax, self.nf)
        df = f[1] - f[0]
        s = self._jonswap_spectrum(f, Hs, Tp)

        n = int(np.round(self.t_total / self.dt))
        t = np.arange(n) * self.dt

        phi = 2.0 * np.pi * self._rng.random(len(f))
        amp = np.sqrt(2.0 * s * df)

        eta = np.sum(
            amp[:, None] * np.cos(2.0 * np.pi * f[:, None] * t[None, :] + phi[:, None]),
            axis=0,
        )
        return t, eta, (f, s)

    # ----------------------------
    # IO helpers
    # ----------------------------
    def _tag(self, *, Hs: float, Fp: float) -> str:
        hs = format(Hs, self.float_fmt)
        fp = format(Fp, self.float_fmt)
        return f"Hs_{hs}m_Fp_{fp}Hz"

    def _write_eta_csv(self, tag: str, t: Array, eta: Array) -> Path:
        path = self.outdir / f"{tag}_eta.csv"
        np.savetxt(
            path,
            np.column_stack([t, eta]),
            delimiter=",",
            header="t_s, eta_m",
            comments="",
        )
        return path

    def _write_spec_csv(self, tag: str, f: Array, s: Array) -> Path:
        path = self.outdir / f"{tag}_spec.csv"
        np.savetxt(
            path,
            np.column_stack([f, s]),
            delimiter=",",
            header="f_Hz, S_m2/Hz",
            comments="",
        )
        return path


class WaveRecord:
    """Read a wave elevation time series and provide circular lookup."""

    def __init__(self, filepath: Union[str, Path], dt: Optional[float] = None) -> None:
        filepath = Path(filepath)

        with filepath.open("r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        self.dt = float(dt) if dt is not None else self._infer_dt(lines)
        self.eta = self._parse_single_column_series(lines)

        self.N = int(self.eta.size)
        self.T_total = float(self.N * self.dt)

    @staticmethod
    def _infer_dt(lines: list[str]) -> float:
        # Try to infer dt from header line like: "0 0.01 1"
        try:
            hdr = lines[1].split()
            return float(hdr[1])
        except Exception:
            return 0.01

    @staticmethod
    def _parse_single_column_series(lines: list[str]) -> Array:
        vals: list[float] = []
        for s in lines:
            parts = s.split()
            if len(parts) != 1:
                continue
            try:
                vals.append(float(parts[0]))
            except ValueError:
                continue

        if not vals:
            raise ValueError("No single-column numeric data found in file.")

        return np.asarray(vals, dtype=float)

    def height(self, t: Union[float, Array], *, interp: bool = False) -> Union[float, Array]:
        """
        Circular lookup.

        Args:
            t: scalar or array-like (seconds)
            interp: if True, linear interpolation between samples
        """
        t_arr = np.asarray(t)
        tw = np.mod(t_arr, self.T_total)
        idxf = tw / self.dt

        if interp:
            i0 = np.floor(idxf).astype(int) % self.N
            i1 = (i0 + 1) % self.N
            w = idxf - np.floor(idxf)
            out = (1.0 - w) * self.eta[i0] + w * self.eta[i1]
        else:
            idx = idxf.astype(int) % self.N
            out = self.eta[idx]

        return float(out.item()) if np.isscalar(t) else out



def find_wave_files(
    wave_dir: Path,
    fp_tokens: list[float] | None = None,
    hs_tokens: list[float] | None = None,
    suffixes: tuple[str, ...] = (".dat", ".csv"),
) -> list[Path]:
    """
    Match files like: Hs_0.25m_Fp_0.33Hz_spec.csv

    fp_tokens / hs_tokens:
      - if None -> don't filter on that parameter
      - otherwise -> keep only files whose parsed values match (within rounding)
    """
    HS_RE = re.compile(r"Hs[_-]?([0-9]*\.?[0-9]+)m")
    FP_RE = re.compile(r"Fp[_-]?([0-9]*\.?[0-9]+)Hz")

    fp_set = {round(x, 3) for x in fp_tokens} if fp_tokens is not None else None
    hs_set = {round(x, 3) for x in hs_tokens} if hs_tokens is not None else None

    out: list[Path] = []
    for p in wave_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in suffixes:
            continue

        name = p.name

        m_hs = HS_RE.search(name)
        m_fp = FP_RE.search(name)
        if not m_hs or not m_fp:
            continue

        hs = round(float(m_hs.group(1)), 3)
        fp = round(float(m_fp.group(1)), 3)

        if hs_set is not None and hs not in hs_set:
            continue
        if fp_set is not None and fp not in fp_set:
            continue

        out.append(p)

    return sorted(out)


if __name__ == "__main__":
    waves = WaveRecord("/home/ddyob/Documents/tethered_argo/tethered_profiler_simulations/data/synthetic_waves/Hs_0.25m_Fp_0.29Hz_eta.dat")