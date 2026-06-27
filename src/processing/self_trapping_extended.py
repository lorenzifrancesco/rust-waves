import os
import h5py
import numpy as np
import pandas as pd
import toml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.constants import hbar

from launch.rust_launcher import Simulation
from launch.rw import write_from_experiment, Params
from plot_widths import width_from_wavefunction, central_site_fraction

plt.rcParams["text.usetex"] = False

# --- CONFIGURATION -----------------------------------------------------------
RECOMPUTE = False
PLOT_EVOLUTION = True
D = 3
N_ATOMS = 1800
V0_VALUES = [1.3]
# Dense near transition (−15..+15 step 1), coarser for repulsive (20..100 step 5)
A_S_VALUES = list(range(-15, 16)) + list(range(20, 101, 5))
L3_VALUES = [5e-39, 1e-38, 5e-38]
CENTRAL_SITES = np.arange(-4, 5)  # 9 central sites

EXPERIMENT_PRE = "input/exp_fig2a_pre.toml"
EXPERIMENT_RUN = "input/exp_fig2a_run.toml"
SNAPSHOT_DIR = "results/snapshots"
RESULT_ROOT = f"results/self_trapping/{D}d"
MEDIA_ROOT = f"media/self_trapping/{D}d"


def l3_tag(l3):
    return "L3-" + f"{l3:.0e}".replace("-", "m")


def as_tag(a_s):
    return str(a_s).replace("-", "m").replace(".", "p")


def simulation_title(l3, a_s):
    # Legacy naming (L3=5e-39 only, from self_trapping.py)
    if l3 == L3_VALUES[0]:
        legacy = f"ss-as{as_tag(a_s)}"
        if os.path.exists(f"{SNAPSHOT_DIR}/{legacy}_{D}d.h5"):
            return legacy
    return f"ss-{l3_tag(l3)}-as{as_tag(a_s)}"


def lattice_spacing_dimensionless():
    exp = toml.load(EXPERIMENT_RUN)
    l_perp = np.sqrt(hbar / (exp["omega_perp"] * exp["m"]))
    return exp["d"] / l_perp


def central_sites_fraction(title, dimensions=D, sites=CENTRAL_SITES):
    """Fraction of atoms in central lattice sites (default −4..+4)."""
    filename = f"{SNAPSHOT_DIR}/{title}_{dimensions}d.h5"
    params = Params.read("input/_params.toml")
    dl = params.dl
    lower = (sites[0] - 0.5) * dl
    upper = (sites[-1] + 0.5) * dl

    if dimensions == 1:
        with h5py.File(filename, "r") as f:
            l = np.array(f["l"])
            psi2 = np.array(f["psi_squared"])
        dx = l[1] - l[0]
    else:
        with h5py.File(filename, "r") as f:
            l_x = np.array(f["l_x"])
            l_y = np.array(f["l_y"])
            l_z = np.array(f["l_z"])
            psi2 = np.array(f["psi_squared"])
        dx = l_x[1] - l_x[0]
        dy = l_y[1] - l_y[0]
        dz = l_z[1] - l_z[0]
        dV = dx * dy * dz

    if dimensions == 1:
        n_total = np.sum(psi2) * dx
        mask = (l >= lower) & (l <= upper)
        n_central = np.sum(psi2[mask]) * dx
    else:
        n_total = np.sum(psi2) * dV
        mask = (l_x >= lower) & (l_x <= upper)
        n_central = np.sum(psi2[mask, :, :]) * dV

    return n_central / n_total if n_total > 0 else 0.0


def site_occupancies(title, dimensions=D, sites=CENTRAL_SITES):
    """Return per-site fractions and total fraction outside the range."""
    filename = f"{SNAPSHOT_DIR}/{title}_{dimensions}d.h5"
    dl = lattice_spacing_dimensionless()
    with h5py.File(filename, "r") as f:
        l_x = np.array(f["l_x"])
        psi2 = np.array(f["psi_squared"])
        dy = f["l_y"][1] - f["l_y"][0]
        dz = f["l_z"][1] - f["l_z"][0]

    dx = l_x[1] - l_x[0]
    profile = np.sum(psi2, axis=(1, 2)) * dy * dz
    total = np.sum(profile) * dx

    occ = []
    for site in sites:
        lower = (site - 0.5) * dl
        upper = (site + 0.5) * dl
        mask = (l_x >= lower) & (l_x < upper)
        occ.append(np.sum(profile[mask]) * dx / total)

    return np.array(occ), 1.0 - np.sum(occ)


# ---------------------------------------------------------------------------
#  Plotting helpers
# ---------------------------------------------------------------------------

def save_colormap(df, outpath, title=None):
    """Site-occupancy colormap (extended a_s range)."""
    pivot = np.zeros((len(CENTRAL_SITES), len(A_S_VALUES)))
    for j, a_s in enumerate(A_S_VALUES):
        row = df[df["a_s"] == a_s].iloc[0]
        pivot[:, j] = [row[f"site_{site}"] for site in CENTRAL_SITES]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    extent = [A_S_VALUES[0] - 0.5, A_S_VALUES[-1] + 0.5,
              CENTRAL_SITES[0] - 0.5, CENTRAL_SITES[-1] + 0.5]
    image = ax.imshow(
        pivot, cmap="hot_r",
        norm=Normalize(vmin=0.0, vmax=np.max(pivot)),
        aspect="auto", origin="lower", extent=extent, interpolation="nearest",
    )
    ax.set_xlabel(r"$a_s / a_0$")
    ax.set_ylabel(r"$z / d$")
    ax.set_xticks(list(range(-15, 16, 3)) + list(range(20, 101, 10)))
    ax.set_yticks(CENTRAL_SITES)
    if title:
        ax.set_title(title)

    for site in CENTRAL_SITES - 0.5:
        ax.axhline(site, color="white", lw=0.3, alpha=0.4)
    for a_s in A_S_VALUES:
        ax.axvline(a_s + 0.5, color="white", lw=0.3, alpha=0.4)

    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(r"$n_s / N$")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_summary_plots(all_results, outdir):
    """Four-panel summary: width_full, width_central, N_c/N, N_central/N."""
    colors = {5e-39: "#1f77b4", 1e-38: "#ff7f0e", 5e-38: "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("width_full",      r"Width $w$ [sites]", "full-space width"),
        ("width_central",   r"Width $w_c$ [sites]", "central-sites width"),
        ("N_c/N",           r"$N_c / N$", "central-site fraction"),
        ("N_central/N",     r"$N_{\mathrm{central}} / N$", "central-sites fraction"),
    ]

    for ax, (col, ylabel, _title) in zip(axes.flat, panels):
        for l3, df in all_results.items():
            sub = df.sort_values("a_s")
            ax.plot(sub["a_s"], sub[col], marker="o", ms=3, lw=1,
                    color=colors[l3], label=f"L3 = {l3:.0e}")
        ax.axvline(0, color="gray", ls=":", lw=0.7)
        ax.set_xlabel(r"$a_s / a_0$")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)

    fig.suptitle(f"Extended sweep — N = {N_ATOMS}, V0 = {V0_VALUES[0]} Er", fontsize=13)
    fig.tight_layout()
    outpath = os.path.join(outdir, "ss_sweep_extended_summary.pdf")
    fig.savefig(outpath, dpi=300)
    fig.savefig(outpath.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved summary: {outpath}")


def save_combined_observable(ax, df_dict, col, ylabel, outpath=None):
    """Helper: single-panel plot of `col` vs a_s for all L3."""
    colors = {5e-39: "#1f77b4", 1e-38: "#ff7f0e", 5e-38: "#d62728"}
    for l3, df in df_dict.items():
        sub = df.sort_values("a_s")
        ax.plot(sub["a_s"], sub[col], marker="o", ms=3, lw=1,
                color=colors[l3], label=f"{l3:.0e}")
    ax.axvline(0, color="gray", ls=":", lw=0.7)
    ax.set_xlabel(r"$a_s / a_0$")
    ax.set_ylabel(ylabel)
    ax.legend(title=r"L3 [m$^6$/s]", fontsize=7)


# ---------------------------------------------------------------------------
#  Simulation pipeline
# ---------------------------------------------------------------------------

def ensure_ground_state():
    gs_file = f"{SNAPSHOT_DIR}/pre-quench_{D}d.h5"
    write_from_experiment(
        EXPERIMENT_PRE, "input/_params.toml", "pre-quench",
        a_s=20.0, load_gs=False, n_atoms=N_ATOMS, dimension=D,
    )
    launcher = Simulation(
        input_params="input/_params.toml", output_file="results/",
        rust="./target/release/rust_waves",
    )
    launcher.compile("release")
    if not os.path.exists(gs_file) or RECOMPUTE:
        launcher.run()
    return launcher


def run_l3_sweep(l3):
    tag = l3_tag(l3)
    out_dir = os.path.join(RESULT_ROOT, tag)
    fig_dir = os.path.join(MEDIA_ROOT, tag)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "ss_sweep_extended_3D_v0-1p3.csv")

    rows = []
    for v0 in V0_VALUES:
        for a_s in A_S_VALUES:
            title = simulation_title(l3, a_s)
            snapshot = f"{SNAPSHOT_DIR}/{title}_{D}d.h5"
            print(f"  a_s={a_s:+3d}  title={title}", end="", flush=True)

            write_from_experiment(
                EXPERIMENT_RUN, "input/_params.toml", title,
                a_s=a_s, v_0=v0, load_gs=True, n_atoms=N_ATOMS,
                l_3=l3, dimension=D,
            )
            if not os.path.exists(snapshot) or RECOMPUTE:
                Simulation(
                    input_params="input/_params.toml", output_file="results/",
                    rust="./target/release/rust_waves",
                ).run()
            else:
                print("  [snapshot exists]", end="")

            # --- Observables ---
            pf_full, width_full = width_from_wavefunction(
                title, dimensions=D, harmonium=False)
            pf_central, width_central = width_from_wavefunction(
                title, dimensions=D, harmonium=True)
            nc = central_site_fraction(title, dimensions=D)
            ncentral = central_sites_fraction(title, dimensions=D)
            occ, missing = site_occupancies(title)

            row = {
                "L3": l3,
                "v0": v0,
                "a_s": a_s,
                "width_full": width_full,
                "width_central": width_central,
                "N_c/N": nc,
                "N_central/N": ncentral,
                "particle_fraction": pf_full,
                "missing_fraction_outside": missing,
            }
            for site, value in zip(CENTRAL_SITES, occ):
                row[f"site_{site}"] = value
            rows.append(row)

            print(
                f"  w_full={width_full:.3f}  w_cent={width_central:.3f}"
                f"  Nc/N={nc:.3f}  Ncent/N={ncentral:.3f}"
            )
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows).sort_values(["v0", "a_s"])
    df.to_csv(csv_path, index=False)

    # --- Static plots ---
    save_colormap(df, os.path.join(fig_dir, "colormap_extended.pdf"),
                  title=f"L3 = {l3:.0e} m^6/s")
    save_summary_plots({l3: df}, fig_dir)

    # --- Dynamical evolutions ---
    if PLOT_EVOLUTION:
        from projections_evolution import plot_heatmap_h5_3d
        heatmap_dir = os.path.join(fig_dir, "heatmaps_extended")
        os.makedirs(heatmap_dir, exist_ok=True)
        for a_s in A_S_VALUES:
            title = simulation_title(l3, a_s)
            dyn_file = f"{SNAPSHOT_DIR}/dyn_{title}_{D}d.h5"
            if os.path.exists(dyn_file):
                try:
                    plot_heatmap_h5_3d(
                        dyn_file, a_s, override_n_atoms=N_ATOMS,
                        experiment_file=EXPERIMENT_RUN,
                        outpath=os.path.join(heatmap_dir, f"{title}_heatmap.pdf"),
                    )
                except Exception as e:
                    print(f"    [heatmap failed: {e}]")

    return df, csv_path, fig_dir


def main():
    print("=" * 72)
    print("Extended self-trapping sweep")
    print(f"L3 values: {[f'{x:.0e}' for x in L3_VALUES]}")
    print(f"a_s range: {A_S_VALUES[0]}..{A_S_VALUES[-1]} ({len(A_S_VALUES)} points)")
    print(f"Central sites: {CENTRAL_SITES[0]}..{CENTRAL_SITES[-1]}")
    print("=" * 72)

    ensure_ground_state()

    all_results = {}
    for l3 in L3_VALUES:
        df, csv_path, fig_dir = run_l3_sweep(l3)
        all_results[l3] = df
        print(f"  Saved CSV: {csv_path}")
        print(f"  Saved media: {fig_dir}")

    # Combined plots across L3 values
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("width_full",      r"Width $w$ [sites]"),
        ("width_central",   r"Width $w_c$ [sites]"),
        ("N_c/N",           r"$N_c / N$"),
        ("N_central/N",     r"$N_{\mathrm{central}} / N$"),
    ]
    for ax, (col, ylabel) in zip(axes.flat, panels):
        save_combined_observable(ax, all_results, col, ylabel)
    fig.suptitle(f"Extended sweep combined — N = {N_ATOMS}", fontsize=13)
    fig.tight_layout()
    combined_path = os.path.join(MEDIA_ROOT, "L3_comparison_extended.pdf")
    fig.savefig(combined_path, dpi=300)
    fig.savefig(combined_path.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)
    print(f"  Saved combined: {combined_path}")


if __name__ == "__main__":
    main()
