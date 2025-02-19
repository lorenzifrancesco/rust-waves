// use env_logger::init;
// use rustfft::num_complex::Complex;
// use rustfft::{Fft, Length};
use std::time::Instant;
// use std::vec;

// IO
use std::fs;
use std::path::Path;

pub mod io;
pub mod propagate;
pub mod tools;
pub mod types;
pub mod trotterized_operators;

use crate::io::*;
use crate::propagate::*;
use crate::tools::*;
use crate::types::*;

// Logging
use log::{debug, info, warn, Level};
use simple_logger;

fn main() {
    simple_logger::init_with_level(Level::Debug).unwrap();
    simulation_3d();
  }

fn simulation_1d() {
  // Define the path to your TOML file
  info!("1D simulation... \n Parsing the input TOML file...");
  let input = Path::new("input/params.toml");
  // let output = Path::new("results/output.csv");
  let contents = fs::read_to_string(input).expect("Failed to read the TOML file");
  let params: Params = toml::from_str(&contents).expect("Failed to load the config");

  let n_l = params.numerics.n_l;
  let l = params.numerics.l;
  assert!(l > 2.0 * params.initial.w, "The domain is too small");
  let h_l: f64 = (l) / (n_l as f64);
  let l_range = symmetric_range(l, h_l);

  let mut initial_wave = Wavefunction1D {
      field: l_range
          .iter()
          .map(|x| gaussian(params.initial.a, params.initial.w, &(x)))
          .collect(),
      l: l_range.clone(),
  };
  assert!(normalization_factor_1d(&initial_wave) - 1.0 < 1e-10, "The wavefunction is not normalized");

  let time_start = Instant::now();
  let saved_psi = propagate_1d(&mut initial_wave, &params, false);
  let time_elapsed = time_start.elapsed();
  info!(
      "Propagation done in {:?}. Saving the results to a CSV file...",
      time_elapsed
  );
  save_1d_wavefunction(&saved_psi.psi[0], "results/psi.h5")
      .expect("Failed to save the wavefunction");
  info!("Done!");
}

fn simulation_3d() {
  // Define the path to your TOML file
  info!("3D simulation...);
  info!(Parsing the input TOML file...");
  let input = Path::new("input/params.toml");
  // let output = Path::new("results/output.csv");
  let contents = fs::read_to_string(input).expect("Failed to read the TOML file");
  let params: Params = toml::from_str(&contents).expect("Failed to load the config");

  // let n_l = params.numerics.n_l;
  let l = params.numerics.l;
  assert!(l > 2.0 * params.initial.w, "The domain is too small");
  // let h_l_x: f64 = (params.numerics.l) / (params.numerics.n_l as f64);
  // let h_l_y: f64 = (params.numerics.l_y) / (params.numerics.n_l_y as f64);
  // let h_l_z: f64 = (params.numerics.l_z) / (params.numerics.n_l_z as f64);
  let l_range_x = symmetric_range(params.numerics.l,   params.numerics.l/(params.numerics.n_l as f64 - 1.0));
  let l_range_y = symmetric_range(params.numerics.l_y, params.numerics.l_y/(params.numerics.n_l_y as f64 - 1.0));
  let l_range_z = symmetric_range(params.numerics.l_z, params.numerics.l_z/(params.numerics.n_l_z as f64 - 1.0));
  // let k_range = k_vector(&l_range);

  
  let mut initial_wave: Wavefunction3D = Wavefunction3D {
      field: gaussian_3d((&l_range_x, 
                                    &l_range_y, 
                                    &l_range_z), 
                                    (0.0, 0.0, 0.0), 
                                    (params.initial.w, params.initial.w_y, params.initial.w_z)
                                ),
      l_x: l_range_x.clone(),
      l_y: l_range_y.clone(),
      l_z: l_range_z.clone(),
  };
  let ns = normalization_factor_3d(&initial_wave);
  initial_wave.field.iter_mut().for_each(|x| *x /= ns);
  assert!((normalization_factor_3d(&initial_wave) - 1.0).abs() < 1e-10, "The wavefunction is not normalized");
  debug!("normalization factor: {:10.5e}", normalization_factor_3d(&initial_wave));

  let time_start = Instant::now();
  let saved_psi: Dynamics3D = propagate_3d(&mut initial_wave, &params);
  let time_elapsed: std::time::Duration = time_start.elapsed();
  info!(
      "Propagation done in {:?}. Saving the results to a CSV file...",
      time_elapsed
  );

  save_3d_wavefunction(&saved_psi.psi[0], "results/psi_3d.h5")
      .expect("Failed to save the wavefunction");
  save_3d_wavefunction(&saved_psi.psi[params.options.n_saves-1], "results/psi_3d_2.h5")
      .expect("Failed to save the wavefunction");
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