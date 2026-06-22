import argparse
import os
import numpy as np
import pandas as pd
from launch.rust_launcher import Simulation
from launch.rw import Params, write_from_experiment
from plot_widths import width_from_wavefunction
from projections_evolution import *
from projections_volumetric import *


def main():
    parser = argparse.ArgumentParser(description="Unified sweep runner")
    parser.add_argument("--prefix", default="idx-",
                        help="Prefix for output files (default: idx-)")
    parser.add_argument("--a_s", type=float, nargs="+", default=None,
                        help="Scattering length(s) to sweep (a_0).")
    parser.add_argument("--a_s_range", type=float, nargs=3, default=None,
                        metavar=("START", "STOP", "NUM"),
                        help="a_s range: start stop num_points")
    parser.add_argument("--n_atoms", type=float, nargs="+", default=None,
                        help="Atom number(s) to sweep.")
    parser.add_argument("--n_atoms_range", type=float, nargs=3, default=None,
                        metavar=("START", "STOP", "NUM"),
                        help="n_atoms range: start stop num_points")
    parser.add_argument("--from_csv", type=str, default=None,
                        help="Read a_s and n_atoms from CSV (cols: a_s, width, number)")
    parser.add_argument("--experiment", default="experiment",
                        help="Experiment file stem in input/ (default: experiment)")
    parser.add_argument("--dim", type=int, default=None,
                        help="Dimension (1 or 3). Overrides _default.toml.")
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute even if output exists")
    parser.add_argument("--skip-gs", action="store_true",
                        help="Skip ground-state recomputation")
    parser.add_argument("--plot", action="store_true",
                        help="Plot evolution after each run")
    parser.add_argument("--harmonium", action="store_true",
                        help="Apply harmonic oscillator correction")
    parser.add_argument("--pre-quench-a-s", type=float, default=20.0,
                        help="a_s for pre-quench GS (default: 20.0)")
    parser.add_argument("--gs-n-atoms", type=float, default=2900.0,
                        help="n_atoms for pre-quench GS (default: 2900)")
    parser.add_argument("--case", type=float, default=None,
                        help="Append _<case> to output filenames")
    args = parser.parse_args()

    if args.from_csv is not None:
        data = pd.read_csv(args.from_csv, header=None, names=["a_s", "width", "number"])
        a_s_list = data["a_s"].values
        n_atoms_list = data["number"].values
    elif args.a_s is not None and args.n_atoms is not None:
        a_s_list = np.array(args.a_s)
        n_atoms_list = np.array(args.n_atoms)
        if len(a_s_list) != len(n_atoms_list):
            n_atoms_list = np.full(len(a_s_list), args.n_atoms[0])
    elif args.a_s_range is not None and args.n_atoms is not None:
        a_s_list = np.linspace(*args.a_s_range)
        n_atoms_list = np.full(len(a_s_list), args.n_atoms[0])
    elif args.a_s is not None and args.n_atoms_range is not None:
        a_s_list = np.full(int(args.n_atoms_range[2]), args.a_s[0])
        n_atoms_list = np.linspace(*args.n_atoms_range)
    elif args.a_s_range is not None and args.n_atoms_range is not None:
        a_s_list = np.linspace(*args.a_s_range)
        n_atoms_list = np.linspace(*args.n_atoms_range)
        if len(a_s_list) != len(n_atoms_list):
            n_atoms_list = np.full(len(a_s_list), n_atoms_list[0])
    else:
        parser.print_help()
        return

    default = Params.read("input/_default.toml")
    d = args.dim if args.dim is not None else default.dimension
    case = f"_{args.case}" if args.case is not None else ""
    gs_title = "pre-quench"

    if not args.skip_gs:
        print("Computing ground state...")
        write_from_experiment(f"input/{args.experiment}_pre_quench.toml",
                              "input/_params.toml",
                              gs_title,
                              a_s=args.pre_quench_a_s,
                              load_gs=False,
                              t_imaginary=20.0,
                              n_atoms=args.gs_n_atoms)
        sim = Simulation(input_params="input/_params.toml",
                          output_file="results/",
                          rust="./target/release/rust_waves")
        sim.compile("release")
        gs_path = f"results/snapshots/{gs_title}_{d}d.h5"
        if not os.path.exists(gs_path) or args.recompute:
            sim.run()
        if args.plot:
            if d == 1:
                plot_heatmap_h5(f"results/snapshots/dyn_{gs_title}_{d}d.h5")
                plot_snap(gs_path)
            elif d == 3:
                plot_heatmap_h5_3d(f"results/snapshots/dyn_{gs_title}_{d}d.h5",
                                   -1, override_n_atoms=args.gs_n_atoms)
        pf0, w0 = width_from_wavefunction(gs_title,
                                          dimensions=d,
                                          harmonium=args.harmonium)
        print(f"pre-quench: fraction = {pf0:3.2f}, width = {w0:3.2f}")

    result_widths = np.zeros(len(a_s_list))
    remaining_fraction = np.zeros(len(a_s_list))

    for idx, (a_s_val, n_at_val) in enumerate(zip(a_s_list, n_atoms_list)):
        run_title = f"{args.prefix}idx-{idx}{case}"
        write_from_experiment(f"input/{args.experiment}.toml",
                              "input/_params.toml",
                              run_title,
                              a_s=a_s_val,
                              load_gs=True,
                              n_atoms=n_at_val)
        sim = Simulation(input_params="input/_params.toml",
                          output_file="results/",
                          rust="./target/release/rust_waves")
        name = f"results/snapshots/{run_title}_{d}d.h5"
        print(f"  [{idx+1}/{len(a_s_list)}] a_s={a_s_val:.2f}, N={n_at_val:.0f} -> {name}")
        if not os.path.exists(name) or args.recompute:
            sim.run()
        if args.plot:
            if d == 1:
                plot_heatmap_h5(f"results/snapshots/dyn_{run_title}_{d}d.h5", idx)
                plot_snap(name, idx)
            elif d == 3:
                plot_heatmap_h5_3d(f"results/snapshots/dyn_{run_title}_{d}d.h5",
                                   idx, override_n_atoms=n_at_val)
        try:
            remaining_fraction[idx], result_widths[idx] = width_from_wavefunction(
                run_title,
                dimensions=d,
                harmonium=args.harmonium,
                particle_threshold=0.05)
        except Exception:
            remaining_fraction[idx], result_widths[idx] = 0.0, 0.0
        print(f"    fraction={remaining_fraction[idx]:3.2f}, width={result_widths[idx]:3.2f}")

    df = pd.DataFrame({
        "a_s": a_s_list,
        "n_atoms": n_atoms_list,
        "width": result_widths,
        "remaining_fraction": remaining_fraction,
    })
    os.makedirs("results/sweeps", exist_ok=True)
    csv_name = f"results/sweeps/sweep_{args.prefix.strip('-')}_{d}d{case}.csv"
    df.to_csv(csv_name, index=False)
    print(f"Saved {csv_name}")


if __name__ == "__main__":
    main()
