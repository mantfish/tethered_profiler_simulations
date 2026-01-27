"""Argo float equations of motion driven by WAMIT output files."""

import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp


class ArgoEOM:
    """Equation of motion model for an Argo float using WAMIT outputs."""

    def __init__(self, folder_path, name):
        """Initialize the model from a WAMIT results folder."""
        self.folder_path = folder_path

        if not os.path.exists(self.folder_path):
            raise FileNotFoundError(f"Folder path does not exist: {self.folder_path}")

        self.float_name = name

        self.float_properties = self.parse_wamit_header()
        self.float_properties.update(self._parse_json())

        self.M = self.create_mass_matrix()
        self.C = self.create_c_matrix()
        self.M_res = self.create_added_mass_at_res()
        self.M_tot = self.M + self.M_res

    def parse_wamit_header(self):
        """Parse WAMIT .out header for geometry and hydrostatic metadata.

        Returns
        -------
        dict
            Parsed values including length scale, gravity, depth, centers of
            buoyancy/gravity, radii of gyration, and hydrostatic C matrix.
        """
        HEADER_KEYS = {
            "length": r"Length scale:\s*([0-9.Ee+-]+)",
            "gravity": r"Gravity:\s*([0-9.Ee+-]+)",
            "depth": r"Water depth:\s*([0-9.Ee+-]+)",
        }
        file_name = self.float_name + ".out"
        out_path = os.path.join(self.folder_path, file_name)

        data = {"Cstar": np.zeros((6, 6))}
        with open(out_path, "r", errors="ignore") as f:
            lines = f.readlines()

        txt = "\n".join(lines)
        for k, pat in HEADER_KEYS.items():
            m = re.search(pat, txt)
            if m:
                data[k] = float(m.group(1))

        for tag in ("Buoyancy", "Gravity"):
            pat = (
                fr"Center of {tag}\s*\(Xg?,Yg?,Zg?\):\s*"
                r"([0-9.Ee+-]+\s+[0-9.Ee+-]+\s+[0-9.Ee+-]+)"
            )
            m = re.search(pat, txt)
            if m:
                data[f"C{tag[0:2]}"] = np.fromstring(m.group(1), sep=" ")

        for line in lines:
            if "C(" in line:
                nums = re.findall(r'C\((\d),(\d)\)|([0-9.\-+E]+)', line)
                idx = [(int(a) - 1, int(b) - 1) for (a, b, _) in nums if a and b]
                vals = [float(v) for (_, _, v) in nums if v]
                for (i, j), v in zip(idx, vals):
                    data["Cstar"][i, j] = data["Cstar"][j, i] = v

        rg_pat = r"Radii of gyration:"
        for k, line in enumerate(lines):
            if re.search(rg_pat, line):
                numbers = re.findall(r"[-+]?\d*\.\d+", str(lines[k:k + 4]))
                data["r_gyr"] = np.array(numbers, dtype=float).reshape(3, 3)
                break

        return data

    def _parse_json(self):
        """Parse JSON configuration for physical constants.

        Returns
        -------
        dict
            JSON content containing physical properties (e.g., rho).
        """
        file_name = self.float_name + ".json"
        fname = os.path.join(self.folder_path, file_name)
        with open(fname) as f:
            return json.load(f)

    def _read_hst_file(self):
        """Reads hydrostatic restoring coefficients from .hst file."""
        file_name = self.float_name + ".hst"
        fname = os.path.join(self.folder_path, file_name)
        Cstar = np.zeros((6, 6))
        with open(fname, "r", errors="ignore") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    continue
                try:
                    i, j = int(parts[0]) - 1, int(parts[1]) - 1
                    value = float(parts[2])
                    Cstar[i, j] = Cstar[j, i] = value
                except (ValueError, IndexError):
                    continue
        return Cstar

    def _read_mmx_file(self):
        """Reads normalized mass matrix from .mmx file."""
        file_name = self.float_name + ".mmx"
        fname = os.path.join(self.folder_path, file_name)
        mass_matrix = np.zeros((6, 6))
        with open(fname, "r", errors="ignore") as f:
            for line in f.readlines()[13:]:
                parts = line.split()
                if len(parts) == 3:
                    i, j = int(parts[0]) - 1, int(parts[1]) - 1
                    value = float(parts[2])
                    mass_matrix[i, j] = mass_matrix[j, i] = value
        return mass_matrix

    def create_mass_matrix(self):
        """Creates the Mass matrix."""
        M_nd = self._read_mmx_file()
        M = self.float_properties["rho"] * M_nd
        return M

    def create_c_matrix(self):
        """Create the dimensional hydrostatic restoring matrix."""
        C = self._read_hst_file()

        rho = self.float_properties["rho"]
        g = self.float_properties["gravity"]
        L = self.float_properties["length"]

        C[2, 2] *= rho * g * L**3
        C[2, 3] = C[3, 2] = C[2, 3] * rho * g * L**3
        C[2, 4] = C[4, 2] = C[2, 4] * rho * g * L**3
        C[3, 3] *= rho * g * L**4
        C[3, 4] = C[4, 3] = C[3, 4] * rho * g * L**4
        C[3, 5] = C[5, 3] = C[3, 5] * rho * g * L**4
        C[4, 4] *= rho * g * L**4
        C[4, 5] = C[5, 4] = C[4, 5] * rho * g * L**4

        return C

    def read_all_A(self, fname):
        """Read all added mass matrices A(ω) for ω >= 0 from a .1 file.

        Parameters
        ----------
        fname : str
            Path to the WAMIT .1 file.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Sorted frequencies (rad/s) and the stacked added mass matrices.
        """
        data = {}
        with open(fname, "r", errors="ignore") as f:
            for ln in f.readlines()[1:]:
                parts = ln.split()
                if len(parts) < 4:
                    continue
                try:
                    T = float(parts[0])
                except ValueError:
                    continue
                if T < 0:
                    continue  # Skip T=-1 (A_infinity)
                w = 0.0 if T == 0 else 2 * np.pi / T
                try:
                    i, j = int(parts[1]) - 1, int(parts[2]) - 1
                except ValueError:
                    continue
                if w not in data:
                    data[w] = np.zeros((6, 6))
                data[w][i, j] = data[w][j, i] = float(parts[3])
        omega = np.array(sorted(data.keys()))
        A = np.array([data[w] for w in omega])
        return omega, A

    def read_added_mass_file(self):
        """Reads added mass at resonance (max A_{33}(ω))."""
        file_name = self.float_name + ".1"
        fname = os.path.join(self.folder_path, file_name)
        omega, A = self.read_all_A(fname)
        idx_max = np.argmax(A[:, 2, 2])  # A_{33} is at index [2,2]
        A_res = A[idx_max]
        print(
            "Using added mass at ω = "
            f"{omega[idx_max]:.4f} rad/s where A_33 is maximum: {A_res[2, 2]:.6e}"
        )
        return A_res

    def create_added_mass_at_res(self):
        """Create dimensional added mass at the resonance frequency.

        Returns
        -------
        np.ndarray
            6x6 added mass matrix scaled by rho and length.
        """
        A_res = self.read_added_mass_file()
        rho, L = self.float_properties["rho"], self.float_properties["length"]
        A_res[:3, :3] *= rho * L**3  # Translational-Translational
        A_res[:3, 3:] *= rho * L**5  # Translational-Rotational
        A_res[3:, :3] *= rho * L**5  # Rotational-Translational
        A_res[3:, 3:] *= rho * L**5  # Rotational-Rotational
        return A_res

    def estimated_viscous_damping(self):
        """Estimate viscous drag coefficients for each DOF.

        Returns
        -------
        np.ndarray
            6x6 diagonal viscous damping matrix.
        """
        C_viscous = np.zeros((6, 6))
        C_viscous[0, 0] = 82.5
        C_viscous[1, 1] = 82.5
        C_viscous[2, 2] = 2.9969
        C_viscous[3, 3] = 4.4342
        C_viscous[4, 4] = 4.4342
        C_viscous[5, 5] = 1
        return C_viscous

    def rhs(self, t, y):
        """Right-hand side of the equation of motion.

        Parameters
        ----------
        t : float
            Time (s).
        y : np.ndarray
            State vector [x, y, z, roll, pitch, yaw, xdot...].

        Returns
        -------
        np.ndarray
            Time derivative of the state.
        """
        x, xdot = y[:6], y[6:]

        viscous = self.estimated_viscous_damping()
        F_rest = -self.C @ x
        F_visc = -viscous @ xdot
        F_tot = F_rest + F_visc

        xddot = np.linalg.solve(self.M_tot, F_tot)

        return np.concatenate((xdot, xddot))


if __name__ == "__main__":
    argo = ArgoEOM("/home/ddyob/Documents/tethered_argo/WHAMIT/ArgoFloater_WAMIT_v2/", "argo")
    print(argo.M_tot)
    print("heave stiffness C33 =", argo.C[2, 2], "N/m")
    print("Pitch stiffness C44 =", argo.C[3, 3], "N/m")
    print("heave damping C33 =", argo.C[4, 4], "N/m")
    print("total heave mass    =", argo.M_tot[2, 2], "kg")
    print("total pitch mass    =", argo.M_tot[3, 3], "kg")

    y0 = np.zeros(12)
    y0[2] = 0.5  # heave is index 2 (0‑based)
    y0[3] = 0.5

    tf = 20.0
    tout = np.linspace(0, tf, 2000)

    sol = solve_ivp(
        fun=lambda t, y: argo.rhs(t, y),
        t_span=(0, tf),
        y0=y0,
        t_eval=tout,
        rtol=1e-6,
        atol=1e-8
    )

    plt.plot(sol.t, sol.y[2], label="heave")
    plt.plot(sol.t, sol.y[3], label="roll")  # heave response
    plt.xlabel("time [s]")
    plt.legend()
    plt.ylabel("heave z [m]")
    plt.title("Free decay – conservative model")
    plt.show()
