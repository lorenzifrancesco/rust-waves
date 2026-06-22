import numpy as np
import pandas as pd
from launch.rust_launcher import Simulation
from launch.rw import write_from_experiment
from plot_widths import width_from_wavefunction, central_site_fraction
from projections_evolution import plot_heatmap_h5, plot_heatmap_h5_3d
import os

# --- CONFIGURATION -----------------------------------------------------------
recompute = False
plotting_evolution = True
d = 3                     # Dimension: 1 for NPSE (fast exploration), 3 for full 3D GPE
model_str = {1: "1D NPSE (nonpolynomial Schrödinger equation)", 3: "3D GPE (full Gross-Pitaevskii)"}
n_atoms = 1800            # Fig 2a: N ≈ 1800
prefix = "ss-"

# a_s sweep covering all three regimes of Fig 2a
a_s_values = [*range(-15, 16), 100, 150]

# Optional V_0 sweep to reproduce Fig 2b
# Set to a single-element list for a 1D sweep, or expand to map the 2D phase diagram
v0_values = [1.3]

# ---- file paths -------------------------------------------------------------
gs_title = "pre-quench"
gs_file = f"results/snapshots/{gs_title}_{d}d.h5"
model_tag = model_str[d].split()[0]
out_dir = f"results/self_trapping/{d}d"
fig_dir = f"media/self_trapping/{d}d"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)
csv_path = f"{out_dir}/ss_sweep_{model_tag}_v0-{'_'.join(str(v).replace('.','p') for v in v0_values)}.csv"

# ---- explanatory banner -----------------------------------------------------
print("=" * 72)
print(f"  Solver: {model_str[d]}")
print(f"  N_atoms: {n_atoms},  V_0 = {v0_values}")
print(f"  a_s sweep: [{a_s_values[0]}, {a_s_values[-1]}]  ({len(a_s_values)} points)")
print(f"  Output CSV: {csv_path}")
if d == 1:
    print("  NOTE: NPSE integrates out radial degrees of freedom. Reliable for qualitative")
    print("        regime mapping, but barrier heights and transition boundaries may shift")
    print("        by ~10-20% vs the full 3D GPE used in the paper.")
    print("        For publication-quality results matching Fig 2a, set d=3 below and re-run.")
elif d == 3:
    print("  Full 3D GPE — quantitative results expected.")
print("=" * 72)

# --- STEP 1: GENERATE GROUND STATE -------------------------------------------
print(f"\n_____ [{model_str[d]}] Generating Ground State (imaginary time, a_s = +20) _____")
write_from_experiment(
    "input/exp_fig2a_pre.toml",
    "input/_params.toml",
    gs_title,
    a_s=20.0,
    load_gs=False,
    n_atoms=n_atoms,
    dimension=d,
)
l_gs = Simulation(
    input_params="input/_params.toml",
    output_file="results/",
    rust="./target/release/rust_waves",
)
l_gs.compile("release")
if not os.path.exists(gs_file) or recompute:
    print(f"  Running {model_str[d]} ground-state simulation...")
    l_gs.run()
else:
    print(f"  {gs_file} found, skipping.")

# --- STEP 2: SYSTEMATIC SWEEP ------------------------------------------------
print("\n_____ Sweeping a_s over the repulsive-localization regimes _____")

# Load existing results to resume if interrupted
results = []
if os.path.exists(csv_path):
    existing = pd.read_csv(csv_path)
    for _, row in existing.iterrows():
        results.append(row.to_dict())
    print(f"  Loaded {len(results)} existing results from {csv_path}")

done = {(r["v0"], r["a_s"]) for r in results}

for v0 in v0_values:
    print(f"\n  ---- V_0 = {v0} Er ----")
    for a_s in a_s_values:
        if (v0, a_s) in done and not recompute:
            print(f"    a_s = {a_s:+3d}  [skipped, already done]")
            continue

        sim_name = f"{prefix}as{str(a_s).replace('-','m').replace('.','p')}"
        outfile = f"results/snapshots/{sim_name}_{d}d.h5"

        print(f"    a_s = {a_s:+3d}", end="", flush=True)

        write_from_experiment(
            "input/exp_fig2a_run.toml",
            "input/_params.toml",
            sim_name,
            a_s=a_s,
            v_0=v0,
            load_gs=True,
            n_atoms=n_atoms,
            dimension=d,
        )

        l_run = Simulation(
            input_params="input/_params.toml",
            output_file="results/",
            rust="./target/release/rust_waves",
        )
        if not os.path.exists(outfile) or recompute:
            l_run.run()

        # Measure observables
        try:
            pf, width = width_from_wavefunction(sim_name, dimensions=d, harmonium=False)
        except Exception as e:
            print(f"  [width failed: {e}]", end="")
            pf, width = 0.0, np.nan
        try:
            nc = central_site_fraction(sim_name, dimensions=d)
        except Exception as e:
            print(f"  [N_c failed: {e}]", end="")
            nc = np.nan

        results.append({"v0": v0, "a_s": a_s, "width": width,
                        "N_c/N": nc, "particle_fraction": pf})
        print(f"  width={width:.3f}  N_c/N={nc:.3f}")

        # Save incrementally
        pd.DataFrame(results).to_csv(csv_path, index=False)

# --- STEP 2b: TIME-EVOLUTION COLORMAPS --------------------------------------
if plotting_evolution:
    print("\n_____ Generating time-evolution colormaps _____")
    heatmap_dir = f"{fig_dir}/heatmaps"
    os.makedirs(heatmap_dir, exist_ok=True)
    for v0 in v0_values:
        for a_s in a_s_values:
            sim_name = f"{prefix}as{str(a_s).replace('-','m').replace('.','p')}"
            dyn_file = f"results/snapshots/dyn_{sim_name}_{d}d.h5"
            heatmap_out = f"{heatmap_dir}/{sim_name}_heatmap.pdf"
            if not os.path.exists(dyn_file):
                continue
            if d == 1:
                plot_heatmap_h5(dyn_file, a_s, experiment_file="input/exp_fig2a_run.toml",
                                outpath=heatmap_out)
            elif d == 3:
                plot_heatmap_h5_3d(dyn_file, a_s, override_n_atoms=n_atoms,
                                   experiment_file="input/exp_fig2a_run.toml",
                                   outpath=heatmap_out)

# --- STEP 3: PLOT RESULTS ----------------------------------------------------
print("\n_____ Plotting _____")
import matplotlib.pyplot as plt

df = pd.DataFrame(results).dropna(subset=["width"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

for v0 in v0_values:
    sub = df[df["v0"] == v0].sort_values("a_s")
    ax1.plot(sub["a_s"], sub["width"], marker="o", ms=3,
             label=f"$V_0 = {v0}$ Er")
    ax2.plot(sub["a_s"], sub["N_c/N"], marker="s", ms=3,
             label=f"$V_0 = {v0}$ Er")

ax1.axvline(0, color="gray", ls=":", lw=0.7)
ax1.axvline(7, color="gray", ls=":", lw=0.7)
ax1.set(xlabel=r"$a_s / a_0$", ylabel=r"Width $w_m$ [sites]")
ax1.legend(fontsize=8)

ax2.axvline(0, color="gray", ls=":", lw=0.7)
ax2.axvline(7, color="gray", ls=":", lw=0.7)
ax2.set(xlabel=r"$a_s / a_0$", ylabel=r"$N_c / N$")
ax2.legend(fontsize=8)

plt.tight_layout()
plot_path = f"{fig_dir}/ss_sweep_{model_tag}_v0-{'_'.join(str(v).replace('.','p') for v in v0_values)}.pdf"
plt.savefig(plot_path, dpi=300)
print(f"  Saved {plot_path}")

# --- SUMMARY ----------------------------------------------------------------
print("\n_____ Summary _____")
print(f"  Solver: {model_str[d]}")
print(df.groupby("v0").agg(
    a_s_min=("a_s", "min"), a_s_max=("a_s", "max"),
    width_range=("width", lambda x: f"{x.min():.2f}–{x.max():.2f}"),
    Nc_range=("N_c/N", lambda x: f"{x.min():.2f}–{x.max():.2f}"),
))
print(f"\n  CSV: {csv_path}")
print(f"  PDF: {plot_path}")
if d == 1:
    print("  >>> Results are from the NPSE (1D reduction). For quantitative 3D GPE,")
    print("      set d=3 at the top of this script and re-run.")
elif d == 3:
    print("  >>> Full 3D GPE results. Compare with Fig 2a of Cruikshank et al.")
