use rust_waves::io::{
    load_1d_wavefunction, load_3d_wavefunction, save_1d_dynamics, save_1d_wavefunction,
    save_3d_wavefunction,
};
use rust_waves::propagate::*;
use rust_waves::tools::*;
use rust_waves::types::*;

use std::time::Instant;

// IO
use std::fs;
use std::path::Path;

// Logging
use log::{debug, info, Level};
use simple_logger;

fn main() {
    simple_logger::init_with_level(Level::Debug).unwrap();
    let input: &Path = Path::new("input/params.toml");
    let contents = fs::read_to_string(input).expect("Failed to read the TOML file");
    let params: Params = toml::from_str(&contents).expect("Failed to load the config");
    if params.physics.dimension == 3 {
        simulation_3d(&params);
    } else {
        simulation_1d(&params);
    }
}

/**
 *
 *
 */
fn simulation_1d(params: &Params) {
    // Define the path to your TOML file
    info!("1D simulation... \n Parsing the input TOML file...");
    let output = format!("results/{}_1d.h5", params.title);
    let output_dyn = format!("results/dyn_{}_1d.h5", params.title);

    let n_l = params.numerics.n_l;
    let l = params.numerics.l;
    assert!(l > 2.0 * params.initial.w, "The domain is too small");
    let h_l: f64 = (l) / ((n_l - 1) as f64);
    let l_range = symmetric_range(l, h_l);
    assert!(l_range.len() == n_l, "The range is not correct");
    let mut initial_wave;
    if params.initial.w == -1.0 {
        initial_wave = load_1d_wavefunction("results/pre-quench_1d.h5").unwrap();
    } else {
        initial_wave = Wavefunction1D {
            field: l_range
                .iter()
                .map(|x| gaussian_normalized(params.initial.w, &(x)))
                .collect(),
            l: l_range.clone(),
        };
    }
    // let initial_wave_for_save = initial_wave.clone();
    assert!(
        normalization_factor_1d(&initial_wave) - 1.0 < 1e-10,
        "The wavefunction is not normalized"
    );

    let time_start = Instant::now();
    let saved_psi = propagate_1d(&mut initial_wave, &params, params.physics.im_t);
    let time_elapsed = time_start.elapsed();
    info!(
        "Propagation done in {:?}. Saving the results to a HDF5 file...",
        time_elapsed
    );
    // let target_psi = Wavefunction1D {
    //     field: l_range
    //         .iter()
    //         .map(|x| ndrustfft::Complex::new(sech_normalized(1. / 2. * params.physics.g, *x), 0.0))
    //         .collect(),
    //     l: l_range.clone(),
    // };
    // save_1d_wavefunction(&target_psi, "results/1d_psi_0.h5")
    //     .expect("Failed to save the wavefunction");
    save_1d_dynamics(&saved_psi, &output_dyn).expect("Failed to save the dynamics");
    save_1d_wavefunction(&saved_psi.psi[saved_psi.psi.len() - 1], &output)
        .expect("Failed to save the wavefunction");
    // save_1d_wavefunction(&Wavefunction1D::new(k_squared, initial_wave.l.clone()), "results/1d_psi_end.h5")
    //     .expect("Failed to save the wavefunction");
    info!("Done!");
}

/**
 *
 *
*/
fn simulation_3d(params: &Params) {
    // Define the path to your TOML file
    info!(
        "3D simulation...);
  info!(Parsing the input TOML file..."
    );
    let output = format!("results/{}_3d.h5", params.title);
    let output_dyn = format!("results/dyn_{}_3d.h5", params.title);
    // let n_l = params.numerics.n_l;
    let l = params.numerics.l;
    assert!(l > 2.0 * params.initial.w, "The domain is too small");
    // let h_l_x: f64 = (params.numerics.l) / (params.numerics.n_l as f64);
    // let h_l_y: f64 = (params.numerics.l_y) / (params.numerics.n_l_y as f64);
    // let h_l_z: f64 = (params.numerics.l_z) / (params.numerics.n_l_z as f64);
    let l_range_x = symmetric_range(
        params.numerics.l,
        params.numerics.l / (params.numerics.n_l as f64 - 1.0),
    );
    let l_range_y = symmetric_range(
        params.numerics.l_y,
        params.numerics.l_y / (params.numerics.n_l_y as f64 - 1.0),
    );
    let l_range_z = symmetric_range(
        params.numerics.l_z,
        params.numerics.l_z / (params.numerics.n_l_z as f64 - 1.0),
    );
    // let k_range = k_vector(&l_range);

    let mut initial_wave;
    if params.initial.w == -1.0 {
        info!("Loading the pre-quench wavefunction...");
        initial_wave = load_3d_wavefunction("results/pre-quench_3d.h5").unwrap();
    } else {
        info!("Creating the initial wavefunction as a Gaussian...");
        initial_wave = Wavefunction3D {
            field: gaussian_3d(
                (&l_range_x, &l_range_y, &l_range_z),
                (0.0, 0.0, 0.0),
                (params.initial.w, params.initial.w_y, params.initial.w_z),
            ),
            l_x: l_range_x.clone(),
            l_y: l_range_y.clone(),
            l_z: l_range_z.clone(),
        };
    }
    let ns = normalization_factor_3d(&initial_wave);
    initial_wave.field.iter_mut().for_each(|x| *x /= ns);
    assert!(
        (normalization_factor_3d(&initial_wave) - 1.0).abs() < 1e-10,
        "The wavefunction is not normalized"
    );
    debug!(
        "normalization factor: {:10.5e}",
        normalization_factor_3d(&initial_wave)
    );

    let time_start = Instant::now();
    let saved_psi: View3D = propagate_3d(&mut initial_wave, &params, params.physics.im_t);
    let time_elapsed: std::time::Duration = time_start.elapsed();
    info!(
        "Propagation done in {:?}. Saving the results to a HDF5 file...",
        time_elapsed
    );

    saved_psi.save_to_hdf5(&output_dyn).unwrap();
    save_3d_wavefunction(&initial_wave, &output).unwrap();
    info!("Done!");
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    #[test]
    fn test_io_1d() {
        let psi: Wavefunction1D = Wavefunction1D {
            field: vec![I; 10],
            l: vec![0.0; 10],
        };
        save_1d_wavefunction(&psi, "results/test.h5").unwrap();
    }

    #[test]
    fn test_io_3d() {
        let psi = Wavefunction3D {
            field: Array3::<ndrustfft::Complex<f64>>::ones((10, 10, 10)),
            l_x: vec![0.0; 10],
            l_y: vec![0.0; 10],
            l_z: vec![0.0; 10],
        };
        save_3d_wavefunction(&psi, "results/psi_3d.h5").unwrap();
    }
}
