"""
Bloch-state decomposition for ultracold atoms in an optical lattice.

Solves the band structure via plane-wave expansion, then projects the
simulated wavefunction onto Bloch states to obtain a pseudomomentum-and-band
representation.

Usage (after extended sweep data exists):
    python bloch_analysis.py --l3 5e-39 1e-38 5e-38 --as -15 -5 0 5 10 20 50 100
    python bloch_analysis.py --all          # decompose every CSV row

References:
    - Kittel, "Introduction to Solid State Physics" (Bloch theorem)
    - Kohn, "Analytic Properties of Bloch Waves and Wannier Functions" (1959)
    - Cold-atom band-mapping: Greiner et al., Nature 415, 39 (2002)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import toml
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import linalg
from scipy.constants import hbar

# Paths (resolve from repo root)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SNAPSHOT_DIR = os.path.join(_REPO_ROOT, "results", "snapshots")
EXPERIMENT_RUN = os.path.join(_REPO_ROOT, "input", "exp_fig2a_run.toml")
RESULT_DIR = os.path.join(_REPO_ROOT, "results", "bloch")
MEDIA_DIR = os.path.join(_REPO_ROOT, "media", "bloch")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Physical parameters (dimensionless units used in simulation)
# ---------------------------------------------------------------------------

def load_lattice_params(experiment_file=EXPERIMENT_RUN):
    """Return dimensionless lattice parameters (V0, dl, G, and 1BZ)."""
    exp = toml.load(experiment_file)
    l_perp = np.sqrt(hbar / (exp["omega_perp"] * exp["m"]))
    e_perp = hbar * exp["omega_perp"]
    e_recoil = (np.pi * hbar / exp["d"]) ** 2 / (2 * exp["m"])

    dl = exp["d"] / l_perp                    # dimensionless lattice spacing
    v0_phys = exp["v_0"] * e_recoil / e_perp  # dimensionless lattice depth
    G = 2.0 * np.pi / dl                      # primitive reciprocal vector
    return {
        "dl": dl,
        "V0": v0_phys,
        "G": G,
        "q_max": G / 2.0,                    # 1BZ boundary
        "l_perp": l_perp,
        "t_perp": 1.0 / exp["omega_perp"],
    }


# ---------------------------------------------------------------------------
# Band structure (plane-wave expansion)
# ---------------------------------------------------------------------------

class BandStructure:
    """Solve for Bloch states of V(z) = -V0 * cos(G*z) via plane-wave basis.

    Bloch state:  φ_{n,q}(z) = e^{iqz} u_{n,q}(z)
    Expansion:    u_{n,q}(z) = ∑_m c_m^{(n)}(q) e^{imGz}
    """

    def __init__(self, V0, G, n_bands=6, n_q=101, n_plane_waves=21):
        self.V0 = V0
        self.G = G
        self.n_bands = n_bands
        self.n_q = n_q
        self.N_m = n_plane_waves if n_plane_waves % 2 == 1 else n_plane_waves + 1

        half = (self.N_m - 1) // 2
        self.m_range = np.arange(-half, half + 1)

        self.q_points = None
        self.energies = None      # shape (n_bands, n_q)
        self.coefficients = None  # shape (n_bands, n_q, N_m)

        self._solve()

    def _solve(self):
        self.q_points = np.linspace(-self.G / 2, self.G / 2, self.n_q)
        n_bands = min(self.n_bands, self.N_m)
        energies = np.zeros((n_bands, self.n_q))
        coeffs = np.zeros((n_bands, self.n_q, self.N_m), dtype=complex)

        for iq, q in enumerate(self.q_points):
            H = np.zeros((self.N_m, self.N_m), dtype=complex)
            for idx, m in enumerate(self.m_range):
                k = q + m * self.G
                H[idx, idx] = 0.5 * k ** 2
                if m - 1 in self.m_range:
                    jdx = np.where(self.m_range == m - 1)[0][0]
                    H[idx, jdx] = -self.V0 / 2.0
                if m + 1 in self.m_range:
                    jdx = np.where(self.m_range == m + 1)[0][0]
                    H[idx, jdx] = -self.V0 / 2.0

            evals, evecs = linalg.eigh(H)
            energies[:, iq] = evals[:n_bands]
            coeffs[:, iq, :] = evecs[:, :n_bands].T

        self.energies = energies
        self.coefficients = coeffs

    def evaluate_bloch_state(self, n, q, z):
        """Evaluate the Bloch state φ_{n,q}(z) on grid z.

        φ_{n,q}(z) = e^{iqz} * ∑_m c_m e^{imGz}
        """
        iq = np.argmin(np.abs(self.q_points - q))
        if np.abs(self.q_points[iq] - q) > 1e-6:
            raise ValueError(f"q={q:.4f} not in q-point grid")
        c_m = self.coefficients[n, iq, :]
        z_reshaped = z[:, np.newaxis]
        m_reshaped = self.m_range[np.newaxis, :]
        u = np.sum(c_m[np.newaxis, :] * np.exp(1j * m_reshaped * self.G * z_reshaped), axis=1)
        phi = np.exp(1j * q * z) * u
        # Normalize
        dz = z[1] - z[0]
        norm = np.sqrt(np.sum(np.abs(phi) ** 2) * dz)
        if norm > 0:
            phi /= norm
        return phi

    def pseudomomentum_folded(self, k):
        """Fold a momentum k into the 1st Brillouin zone; return (q, band_offset)."""
        q = k % self.G
        if q > self.G / 2:
            q -= self.G
        band_offset = int(np.round((k - q) / self.G))
        return q, band_offset


# ---------------------------------------------------------------------------
# Wavefunction loading
# ---------------------------------------------------------------------------

def load_1d_complex_wavefunction(title, snap_dir=SNAPSHOT_DIR):
    """Load the full complex wavefunction from a complex HDF5 snapshot.

    Returns dict with 'z' (grid), 'psi' (complex array), 'density',
    'has_phase' flag.
    """
    path = os.path.join(snap_dir, f"{title}_1d.h5")
    if not os.path.exists(path):
        print(f"  [WARN] {path} not found")
        return None

    with h5py.File(path, "r") as f:
        z = np.array(f["l"])
        psi_sq = np.array(f["psi_squared"])
        has_phase = "psi_re" in f and "psi_im" in f
        if has_phase:
            psi_re = np.array(f["psi_re"])
            psi_im = np.array(f["psi_im"])
            psi = psi_re + 1j * psi_im
        else:
            psi = np.sqrt(psi_sq)  # phase-less reconstruction

    dz = z[1] - z[0]
    return {
        "z": z,
        "psi": psi,
        "density": psi_sq,
        "dz": dz,
        "has_phase": has_phase,
        "title": title,
        "path": path,
    }


def load_1d_axial_profile_3d(title, snap_dir=SNAPSHOT_DIR):
    """Project a 3D snapshot onto the axial (x) axis. Returns None for 1D."""
    path = os.path.join(snap_dir, f"{title}_3d.h5")
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        l_x = np.array(f["l_x"])
        l_y = np.array(f["l_y"])
        l_z = np.array(f["l_z"])
        psi2 = np.array(f["psi_squared"])
    dy = l_y[1] - l_y[0]
    dz = l_z[1] - l_z[0]
    profile = np.sum(psi2, axis=(1, 2)) * dy * dz
    dx = l_x[1] - l_x[0]
    norm = np.sum(profile) * dx
    if norm > 0:
        profile /= norm
    return {
        "z": l_x,
        "density": profile,
        "dz": dx,
        "has_phase": False,
        "title": title,
    }


# ---------------------------------------------------------------------------
# Projection onto Bloch states
# ---------------------------------------------------------------------------

def project_onto_bloch(wf_data, band_struct, n_bands=4):
    """Project wavefunction onto Bloch states.

    Returns dict with:
      - q_points: array
      - weights: (n_bands, n_q) matrix of |c_n(q)|^2
      - band_populations: (n_bands,) total population per band
      - momentum_dist: (n_k,) momentum distribution n(k)
      - k_grid: (n_k,) corresponding momentum grid
    """
    z = wf_data["z"]
    psi = wf_data["psi"]
    dz = wf_data["dz"]
    n_q = band_struct.n_q
    q_points = band_struct.q_points
    n_bands_actual = min(n_bands, band_struct.n_bands)

    weights = np.zeros((n_bands_actual, n_q))
    for n in range(n_bands_actual):
        for iq in range(n_q):
            phi = band_struct.evaluate_bloch_state(n, q_points[iq], z)
            overlap = np.sum(np.conj(phi) * psi) * dz
            weights[n, iq] = np.abs(overlap) ** 2

    band_pop = np.sum(weights, axis=1)  # total per band

    # Momentum distribution n(k) = |FT[psi](k)|^2
    n_z = len(z)
    psi_ft = np.fft.fftshift(np.fft.fft(psi, norm="ortho"))
    dk = 2.0 * np.pi / (n_z * dz)
    k_grid = np.fft.fftshift(np.fft.fftfreq(n_z, d=dz)) * 2.0 * np.pi
    momentum_dist = np.abs(psi_ft) ** 2

    return {
        "q_points": q_points,
        "weights": weights,
        "band_populations": band_pop,
        "momentum_dist": momentum_dist,
        "k_grid": k_grid,
        "n_bands": n_bands_actual,
        "has_phase": wf_data["has_phase"],
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_band_structure(band_struct, outpath, title=None):
    """Plot the band structure E_n(q) for the optical lattice."""
    fig, ax = plt.subplots(figsize=(5, 4))
    q = band_struct.q_points / band_struct.G * 2  # units of G/2 (1BZ edge)
    for n in range(min(4, band_struct.n_bands)):
        ax.plot(q, band_struct.energies[n], lw=1.5, label=f"n={n}")
    ax.set_xlabel(r"$q$ [$G/2$]")
    ax.set_ylabel(r"$E_n(q)$ [$\hbar\omega_\perp$]")
    ax.legend(fontsize=8)
    ax.axvline(-1, color="gray", ls=":", lw=0.5)
    ax.axvline(1, color="gray", ls=":", lw=0.5)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  Saved band structure: {outpath}")


def plot_pseudomomentum_map(wf_data, proj, band_struct, outpath, title=None):
    """Pseudomomentum-and-band colormap.

    Shows |c_n(q)|^2 as a colour intensity on a vertical stack of bands.
    Overlaid with the band energies as guide lines.
    """
    n_bands_plot = min(proj["n_bands"], 6)
    G = band_struct.G
    q_norm = proj["q_points"] / (G / 2)  # normalize to ±1

    fig, ax = plt.subplots(figsize=(6, 5))

    # Colormap: for each band, plot weight as a horizontal strip
    extent = [q_norm[0], q_norm[-1], -0.5, n_bands_plot - 0.5]
    image_data = np.zeros((n_bands_plot, len(q_norm)))
    for n in range(n_bands_plot):
        image_data[n, :] = proj["weights"][n, :]

    vmax = np.max(image_data) if np.max(image_data) > 0 else 1.0
    ax.imshow(
        image_data, aspect="auto", origin="lower", extent=extent,
        cmap="hot_r", norm=Normalize(vmin=0, vmax=vmax),
        interpolation="bicubic",
    )

    # Band-energy guide lines
    for n in range(n_bands_plot):
        e_scaled = band_struct.energies[n] / np.max(band_struct.energies[:4]) * (n_bands_plot - 1)
        ax.plot(q_norm, e_scaled, "w--", lw=0.5, alpha=0.4)

    ax.set_xlabel(r"Pseudomomentum $q$ [$G/2$]")
    ax.set_ylabel("Band index $n$")
    ax.set_yticks(range(n_bands_plot))
    ax.set_yticklabels([str(n) for n in range(n_bands_plot)])
    ax.axvline(-1, color="cyan", ls=":", lw=0.5, alpha=0.5)
    ax.axvline(1, color="cyan", ls=":", lw=0.5, alpha=0.5)

    cbar = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=Normalize(vmin=0, vmax=vmax), cmap="hot_r"
        ), ax=ax, fraction=0.04, pad=0.02,
    )
    cbar.set_label(r"$|c_n(q)|^2$")

    phase_tag = " [no phase]" if not proj["has_phase"] else ""
    ax.set_title(f"{title or ''}{phase_tag}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    fig.savefig(outpath.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved pseudomomentum map: {outpath}")


def plot_momentum_distribution(wf_data, proj, band_struct, outpath, title=None):
    """Momentum distribution n(k) = |FT[psi]|^2 with 1BZ overlays."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    k = proj["k_grid"] / band_struct.G  # units of reciprocal lattice vector
    n_k = proj["momentum_dist"]

    ax.plot(k, n_k / np.max(n_k), "b-", lw=0.8)
    ax.set_xlabel(r"$k$ [$G$]")
    ax.set_ylabel(r"$n(k)$ [norm.]")
    ax.set_xlim(k[0], k[-1])

    # Mark 1BZ boundaries
    for bz in range(-5, 6):
        edge = bz + 0.5
        ax.axvline(edge, color="gray", ls=":", lw=0.3, alpha=0.5)
        ax.axvline(bz - 0.5, color="gray", ls=":", lw=0.3, alpha=0.5)

    phase_tag = " [no phase]" if not proj["has_phase"] else ""
    ax.set_title(f"{title or ''}{phase_tag}")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  Saved momentum distribution: {outpath}")


def plot_band_populations_sweep(csv_files, outpath):
    """Band populations vs a_s from multiple CSV files (one per L3)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {5e-39: "#1f77b4", 1e-38: "#ff7f0e", 5e-38: "#d62728"}
    markers = ["o", "s", "^", "D"]

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        l3 = df["L3"].iloc[0]
        sub = df.sort_values("a_s")
        for n in range(min(4, 6)):
            col = f"band_{n}_pop"
            if col not in df.columns:
                continue
            ax.plot(
                sub["a_s"], sub[col], marker=markers[n], ms=3, lw=0.8,
                color=colors.get(l3, "gray"), label=f"L3={l3:.0e}, n={n}",
            )

    ax.set_xlabel(r"$a_s / a_0$")
    ax.set_ylabel("Band population")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"  Saved band-population sweep: {outpath}")


# ---------------------------------------------------------------------------
# Full decomposition pipeline
# ---------------------------------------------------------------------------

def decompose_snapshot(title, band_struct, n_bands=4):
    """Load and decompose a single snapshot into Bloch states."""
    # Try complex 1D first
    wf = load_1d_complex_wavefunction(title)
    if wf is None:
        wf = load_1d_axial_profile_3d(title)
    if wf is None:
        print(f"  [SKIP] {title}: no snapshot found")
        return None

    proj = project_onto_bloch(wf, band_struct, n_bands=n_bands)
    return {**wf, **proj}


def decompose_and_plot(row, band_struct, outdir, n_bands=4):
    """Decompose one snapshot and generate figures."""
    l3 = row["L3"]
    a_s = row["a_s"]
    title = f"ss-L3-{l3:.0e}".replace("-", "m").replace("+", "") + f"-as{str(a_s).replace('-','m').replace('.','p')}"

    result = decompose_snapshot(title, band_struct, n_bands=n_bands)
    if result is None:
        return None

    tag = f"L3-{l3:.0e}-as{a_s:+.0f}".replace("+", "p").replace("-", "m").replace(".", "p")
    plot_pseudomomentum_map(
        result, result, band_struct,
        os.path.join(outdir, f"bloch_map_{tag}.pdf"),
        title=f"a_s={a_s}, L3={l3:.0e}",
    )
    plot_momentum_distribution(
        result, result, band_struct,
        os.path.join(outdir, f"momentum_{tag}.pdf"),
        title=f"a_s={a_s}, L3={l3:.0e}",
    )
    return result


def run_sweep_decomposition(csv_path, band_struct, n_bands=4, a_s_subset=None):
    """Run Bloch decomposition for every row in a sweep CSV."""
    df = pd.read_csv(csv_path)
    if a_s_subset is not None:
        df = df[df["a_s"].isin(a_s_subset)]
    outdir = os.path.join(MEDIA_DIR, f"bloch_{os.path.basename(csv_path).replace('.csv','')}")
    os.makedirs(outdir, exist_ok=True)

    results = []
    for _, row in df.iterrows():
        tag = (
            f"L3-{row['L3']:.0e}-as{row['a_s']:+.0f}"
            .replace("+", "p").replace("-", "m").replace(".", "p")
        )
        print(f"  [{tag}]", end="", flush=True)
        result = decompose_and_plot(row, band_struct, outdir, n_bands=n_bands)
        if result is not None:
            results.append({
                "L3": row["L3"],
                "a_s": row["a_s"],
                **{f"band_{n}_pop": result["band_populations"][n]
                   if n < len(result["band_populations"]) else 0.0
                   for n in range(n_bands)},
            })
        print()

    if results:
        df_out = pd.DataFrame(results)
        csv_out = os.path.join(RESULT_DIR, "bloch_populations.csv")
        df_out.to_csv(csv_out, index=False)
        print(f"  Saved populations: {csv_out}")
        # Band-population sweep plot
        plot_band_populations_sweep(
            [csv_out],
            os.path.join(MEDIA_DIR, "band_populations_sweep.pdf"),
        )
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Bloch analysis of self-trapping wavefunctions")
    parser.add_argument("--csv", default=None, help="Path to sweep CSV")
    parser.add_argument("--l3", type=float, nargs="+", default=None, help="L3 values to process")
    parser.add_argument("--as", dest="a_s", type=float, nargs="+", default=None,
                        help="a_s values to process (subset)")
    parser.add_argument("--all", action="store_true", help="Decompose every row in CSV")
    parser.add_argument("--n-bands", type=int, default=4, help="Number of bands")
    parser.add_argument("--band-structure-only", action="store_true",
                        help="Only plot band structure and exit")
    args = parser.parse_args()

    # Lattice parameters
    lp = load_lattice_params()
    print(f"Lattice: V0 = {lp['V0']:.4f}, dl = {lp['dl']:.4f}, G = {lp['G']:.4f}")
    bs = BandStructure(V0=lp["V0"], G=lp["G"], n_bands=args.n_bands)

    if args.band_structure_only:
        plot_band_structure(bs, os.path.join(MEDIA_DIR, "band_structure.pdf"))
        return

    # Determine CSV files to process
    st_root = os.path.join(_REPO_ROOT, "results", "self_trapping", "3d")
    if args.csv:
        csv_files = [args.csv]
    elif args.l3 is not None:
        csv_files = []
        for l3 in args.l3:
            tag_l3 = "L3-" + f"{l3:.0e}".replace("-", "m")
            path = os.path.join(st_root, tag_l3, "ss_sweep_extended_3D_v0-1p3.csv")
            if os.path.exists(path):
                csv_files.append(path)
            else:
                print(f"  [WARN] {path} not found, trying original sweep...")
                path2 = os.path.join(st_root, tag_l3, "ss_sweep_3D_v0-1p3.csv")
                if os.path.exists(path2):
                    csv_files.append(path2)
    elif args.all:
        import glob
        csv_files = glob.glob(os.path.join(st_root, "*", "ss_sweep_*.csv"))
    else:
        parser.print_help()
        return

    for csv_path in csv_files:
        print(f"\nProcessing: {csv_path}")
        run_sweep_decomposition(csv_path, bs, n_bands=args.n_bands, a_s_subset=args.a_s)


if __name__ == "__main__":
    main()
