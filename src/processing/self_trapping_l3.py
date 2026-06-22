import os

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
from matplotlib.colors import Normalize
from scipy.constants import hbar

from launch.rust_launcher import Simulation
from launch.rw import write_from_experiment
from plot_widths import central_site_fraction, width_from_wavefunction
from projections_evolution import plot_heatmap_h5_3d

plt.rcParams["text.usetex"] = False


RECOMPUTE = False
PLOT_EVOLUTION = True
D = 3
N_ATOMS = 1800
V0_VALUES = [1.3]
A_S_VALUES = list(range(-15, 16))
L3_VALUES = [5e-39, 1e-38, 5e-38]
SITES = np.arange(-3, 4)

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
    legacy_title = f"ss-as{as_tag(a_s)}"
    legacy_file = f"{SNAPSHOT_DIR}/{legacy_title}_{D}d.h5"
    if l3 == L3_VALUES[0] and os.path.exists(legacy_file):
        return legacy_title
    return f"ss-{l3_tag(l3)}-as{as_tag(a_s)}"


def lattice_spacing_dimensionless():
    exp = toml.load(EXPERIMENT_RUN)
    l_perp = np.sqrt(hbar / (exp["omega_perp"] * exp["m"]))
    return exp["d"] / l_perp


def site_occupancies(title, dimensions=D):
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
    for site in SITES:
        lower = (site - 0.5) * dl
        upper = (site + 0.5) * dl
        mask = (l_x >= lower) & (l_x < upper)
        occ.append(np.sum(profile[mask]) * dx / total)

    return np.array(occ), 1.0 - np.sum(occ)


def save_fig2a_colormap(df, outpath, title=None):
    pivot = np.zeros((len(SITES), len(A_S_VALUES)))
    for j, a_s in enumerate(A_S_VALUES):
        row = df[df["a_s"] == a_s].iloc[0]
        pivot[:, j] = [row[f"site_{site}"] for site in SITES]

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    extent = [A_S_VALUES[0] - 0.5, A_S_VALUES[-1] + 0.5, SITES[0] - 0.5, SITES[-1] + 0.5]
    image = ax.imshow(
        pivot,
        cmap="hot_r",
        norm=Normalize(vmin=0.0, vmax=np.max(pivot)),
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest",
    )

    ax.set_xlabel("as / a0")
    ax.set_ylabel("z / d")
    ax.set_xticks(range(-15, 16, 3))
    ax.set_yticks(SITES)
    if title:
        ax.set_title(title)

    for site in SITES - 0.5:
        ax.axhline(site, color="white", lw=0.3, alpha=0.4)
    for a_s in A_S_VALUES:
        ax.axvline(a_s + 0.5, color="white", lw=0.3, alpha=0.4)

    cbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("ns / N")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    fig.savefig(outpath.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_summary_plot(df, outpath, l3):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for v0 in V0_VALUES:
        sub = df[df["v0"] == v0].sort_values("a_s")
        ax1.plot(sub["a_s"], sub["width"], marker="o", ms=3, label=f"V0 = {v0} Er")
        ax2.plot(sub["a_s"], sub["N_c/N"], marker="s", ms=3, label=f"V0 = {v0} Er")

    for ax in (ax1, ax2):
        ax.axvline(0, color="gray", ls=":", lw=0.7)
        ax.axvline(7, color="gray", ls=":", lw=0.7)
        ax.set_xlabel("as / a0")
        ax.legend(fontsize=8)

    ax1.set_ylabel("Width wm [sites]")
    ax2.set_ylabel("Nc / N")
    fig.suptitle(f"L3 = {l3:.0e} m^6/s", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def save_combined_nc_plot(all_results, outpath):
    fig, ax = plt.subplots(figsize=(6, 4))
    for l3, df in all_results.items():
        sub = df.sort_values("a_s")
        ax.plot(sub["a_s"], sub["N_c/N"], marker="o", ms=3, lw=1, label=f"{l3:.0e}")
    ax.set_xlabel("as / a0")
    ax.set_ylabel("Nc / N")
    ax.axvline(0, color="gray", ls=":", lw=0.7)
    ax.legend(title="L3 [m6/s]", fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    fig.savefig(outpath.replace(".pdf", ".png"), dpi=300)
    plt.close(fig)


def ensure_ground_state():
    gs_file = f"{SNAPSHOT_DIR}/pre-quench_{D}d.h5"
    write_from_experiment(
        EXPERIMENT_PRE,
        "input/_params.toml",
        "pre-quench",
        a_s=20.0,
        load_gs=False,
        n_atoms=N_ATOMS,
        dimension=D,
    )
    launcher = Simulation(input_params="input/_params.toml", output_file="results/", rust="./target/release/rust_waves")
    launcher.compile("release")
    if not os.path.exists(gs_file) or RECOMPUTE:
        launcher.run()
    return launcher


def run_l3_sweep(l3):
    tag = l3_tag(l3)
    out_dir = f"{RESULT_ROOT}/{tag}"
    fig_dir = f"{MEDIA_ROOT}/{tag}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    csv_path = f"{out_dir}/ss_sweep_3D_v0-1p3.csv"

    rows = []
    for v0 in V0_VALUES:
        print(f"\n---- L3={l3:.0e}, V0={v0} Er ----")
        for a_s in A_S_VALUES:
            title = simulation_title(l3, a_s)
            snapshot = f"{SNAPSHOT_DIR}/{title}_{D}d.h5"
            print(f"  a_s={a_s:+3d}  title={title}", end="", flush=True)
            write_from_experiment(
                EXPERIMENT_RUN,
                "input/_params.toml",
                title,
                a_s=a_s,
                v_0=v0,
                load_gs=True,
                n_atoms=N_ATOMS,
                l_3=l3,
                dimension=D,
            )
            if not os.path.exists(snapshot) or RECOMPUTE:
                Simulation(input_params="input/_params.toml", output_file="results/", rust="./target/release/rust_waves").run()
            else:
                print("  [snapshot exists]", end="")

            pf, width = width_from_wavefunction(title, dimensions=D, harmonium=False)
            nc = central_site_fraction(title, dimensions=D)
            occ, missing = site_occupancies(title)
            row = {
                "L3": l3,
                "v0": v0,
                "a_s": a_s,
                "width": width,
                "N_c/N": nc,
                "particle_fraction": pf,
                "missing_fraction_outside_pm3": missing,
            }
            for site, value in zip(SITES, occ):
                row[f"site_{site}"] = value
            rows.append(row)
            print(f"  width={width:.3f}  N_c/N={nc:.3f}  missing_pm3={missing:.3f}")
            pd.DataFrame(rows).to_csv(csv_path, index=False)

    df = pd.DataFrame(rows).sort_values(["v0", "a_s"])
    df.to_csv(csv_path, index=False)
    save_summary_plot(df, f"{fig_dir}/ss_sweep_3D_v0-1p3.pdf", l3)
    save_fig2a_colormap(df, f"{fig_dir}/fig2a_colormap.pdf", title=f"L3 = {l3:.0e} m^6/s")

    if PLOT_EVOLUTION:
        heatmap_dir = f"{fig_dir}/heatmaps"
        os.makedirs(heatmap_dir, exist_ok=True)
        for a_s in A_S_VALUES:
            title = simulation_title(l3, a_s)
            dyn_file = f"{SNAPSHOT_DIR}/dyn_{title}_{D}d.h5"
            if os.path.exists(dyn_file):
                plot_heatmap_h5_3d(
                    dyn_file,
                    a_s,
                    override_n_atoms=N_ATOMS,
                    experiment_file=EXPERIMENT_RUN,
                    outpath=f"{heatmap_dir}/{title}_heatmap.pdf",
                )

    return df, csv_path, fig_dir


def main():
    print("=" * 72)
    print("Fig 2a L3 sweep")
    print(f"L3 values: {[f'{x:.0e}' for x in L3_VALUES]}")
    print(f"a_s range: {A_S_VALUES[0]}..{A_S_VALUES[-1]} ({len(A_S_VALUES)} points)")
    print("=" * 72)
    ensure_ground_state()

    all_results = {}
    for l3 in L3_VALUES:
        df, csv_path, fig_dir = run_l3_sweep(l3)
        all_results[l3] = df
        print(f"Saved CSV: {csv_path}")
        print(f"Saved media: {fig_dir}")

    save_combined_nc_plot(all_results, f"{MEDIA_ROOT}/L3_comparison_Nc.pdf")
    print(f"Saved combined comparison: {MEDIA_ROOT}/L3_comparison_Nc.pdf")


if __name__ == "__main__":
    main()
