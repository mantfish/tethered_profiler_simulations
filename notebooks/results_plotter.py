from __future__ import annotations

from pathlib import Path
import pickle
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
import re

BASE_DIR = Path("/home/ddyob/Documents/tethered_argo/tethered_profiler_simulations/results/config_1")  # adjust if your files live elsewhere
Z_THRESHOLD = -0.20   # "below 20 cm under the water line" (assuming z>0 is up)

# ---------- HELPERS ----------

fname_re = re.compile(
    r"(?P<depth>\d+)m_"
    r"(?P<current>[\d.]+)ms_"
    r"dt(?P<dt>[\deE+-]+)_"
    r"T(?P<T>\d+)s__"
    r"Hs_(?P<Hs>[\d.]+)m_"
    r"Fp_(?P<Fp>[\d.]+)Hz_eta\.pkl$"
)



def final_heave_perc(res, z_thres=-0.2):
    """Return fraction of last 60 s with elevation < z_thres (0–1)."""
    time = res["time_hist"]  # (N,)
    state = res["state_hist"]  # (N, 12)
    try:
        waves = res["waves"]
    except KeyError:
        waves = np.zeros_like(time)

    if np.max(time) <= 100:
        return None

    #print(time)
    # find index where last 60 s starts
    last_60_idx = np.argmin(np.abs(time - (time[-1] - 60.0)))
    print("max time: ", time[-1])
    print("index of closest time", last_60_idx)
    print("time at index: ", time[last_60_idx])
    #print(last_60_idx, print(time[last_60_idx:]))

    heave = state[:, 2]
    last_heaves = heave[last_60_idx:]
    last_waves = waves[last_60_idx:]
    last_elevations = last_heaves - last_waves

    frac_under = np.count_nonzero(last_elevations < z_thres) / last_heaves.shape[0]
    return frac_under  # 0..1


def get_depth_cs_t_hs(file_name):
    """Parse depth, current, Tp, Hs from file name."""
    parts = file_name.split("_")
    depth = float(parts[0][:-1])   # '40m' -> 40
    cs = float(parts[1][:-2])      # '0.10ms' -> 0.10

    if "None" in file_name:
        # No waves case
        Hs = 0.0
        T = 0.0
    else:
        # parts example:
        # ['40m','0.10ms','dt1e-04','T846s','','Hs','1.00m','Fp','0.20Hz','eta.pkl']
        T = 1.0 / float(parts[-2][:-2])   # '0.20Hz' -> 0.20 -> Tp = 1/Fp
        Hs = float(parts[-4][:-1])        # '1.00m'  -> 1.00

    return {
        "depth": depth,
        "current": cs,
        "Tp": T,
        "Hs": Hs,
    }

def _write_depth_summary_csv(
    *,
    out_path: Path,
    depth: float,
    records: list[dict],
) -> None:
    """
    Writes one CSV per depth with columns:
    depth_m,current_ms,Hs_m,Tp_s,Fp_Hz,frac_submerged
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header = "depth_m,current_ms,Hs_m,Tp_s,Fp_Hz,frac_submerged\n"
    lines = [header]

    for r in records:
        Hs = float(r.get("Hs", np.nan))
        Tp = float(r.get("Tp", np.nan))
        # Convert period -> frequency (Hz). If Tp is 0 or missing, set Fp=0.
        Fp = 0.0 if (not np.isfinite(Tp) or Tp <= 0.0) else 1.0 / Tp

        line = (
            f"{float(depth):.6g},"
            f"{float(r.get('current', np.nan)):.6g},"
            f"{Hs:.6g},"
            f"{Tp:.6g},"
            f"{Fp:.6g},"
            f"{float(r.get('frac', np.nan)):.6g}\n"
        )
        lines.append(line)

    out_path.write_text("".join(lines), encoding="utf-8")


def plot_submergence_heatmap_for_depth(depth: float, base_dir: Path):
    files = sorted(base_dir.glob(f"{int(depth)}m_*.pkl"))
    if not files:
        print(f"No files found for depth={depth} m in {base_dir}")
        return None

    records: list[dict] = []
    for path in files:
        meta = get_depth_cs_t_hs(path.name)
        print(meta)

        with open(path, "rb") as f:
            D = pickle.load(f)

        frac = final_heave_perc(D, z_thres=Z_THRESHOLD)
        if frac is None:
            print(f"No waves found for {path}")
            continue

        meta["frac"] = float(frac)  # 0..1
        records.append(meta)

    if not records:
        print(f"No usable records for depth={depth}")
        return None

    # ---- write per-depth summary ----
    summary_path = base_dir / f"{int(depth)}m_summary.csv"
    _write_depth_summary_csv(out_path=summary_path, depth=depth, records=records)

    # ---- build scatter grid ----
    currents = sorted({r["current"] for r in records})
    wave_keys = sorted({(r["Hs"], r["Tp"]) for r in records}, key=lambda k: (k[0], k[1]))

    n_c, n_w = len(currents), len(wave_keys)
    data = np.full((n_c, n_w), np.nan, dtype=float)

    c_index = {c: i for i, c in enumerate(currents)}
    w_index = {k: j for j, k in enumerate(wave_keys)}

    for r in records:
        data[c_index[r["current"]], w_index[(r["Hs"], r["Tp"])]] = r["frac"]

    valid = np.isfinite(data)
    if not np.any(valid):
        print(f"No finite data for depth={depth}")
        return None

    iy, ix = np.where(valid)
    z = data[iy, ix]

    fig, ax = plt.subplots(figsize=(10, 5))
    sc = ax.scatter(
        ix,
        iy,
        c=z,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        s=60,
        edgecolors="k",
        linewidths=0.5,
    )

    xlabels = [
        "No waves" if (Hs == 0.0 and Tp == 0.0) else f"Hs={Hs:.2g} m\nTp={Tp:.1f} s"
        for (Hs, Tp) in wave_keys
    ]
    ax.set_xticks(np.arange(n_w))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")

    ax.set_yticks(np.arange(n_c))
    ax.set_yticklabels([f"{c:.2f}" for c in currents])

    ax.set_xlabel("Wave state (Hs / Tp)")
    ax.set_ylabel("Current speed [m/s]")
    ax.set_title(f"Depth = {depth:.0f} m")

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Fraction of time submerged (> 0.20 m)")

    plt.tight_layout()

    # ---- save figure per depth ----
    fig_path = base_dir / f"{int(depth)}m_heatmap.png"
    fig.savefig(fig_path, dpi=300)

    # Jupyter-friendly display (separate output per depth)
    display(fig)
    plt.close(fig)

    return {"fig_path": fig_path, "summary_path": summary_path}


def plot_all_depths(base_dir: Path):
    depths = sorted({float(p.name.split("_")[0][:-1]) for p in base_dir.glob("*.pkl")})
    if not depths:
        raise FileNotFoundError(f"No .pkl files found in {base_dir}")

    outputs = []
    for d in depths:
        print(f"Plotting depth {d} m")
        out = plot_submergence_heatmap_for_depth(d, base_dir)
        if out is not None:
            outputs.append(out)
    return outputs


# run
outputs = plot_all_depths(BASE_DIR)
outputs