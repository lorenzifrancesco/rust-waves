use rust_waves::io::{save_1d_wavefunction, save_3d_wavefunction};
use rust_waves::propagate::{propagate_1d, propagate_3d};
use rust_waves::types::{Dynamics3D, Params, Wavefunction3D, Wavefunction1D};
use rust_waves::tools::{
  gaussian, gaussian_3d, gaussian_normalized, normalization_factor_1d, normalization_factor_3d, sech_normalized, symmetric_range};
use std::fs::File;
use std::io::Write;
use std::time::Instant;
use std::fs;
use std::path::Path;  
use hdf5_metno;
use log::{info, Level};
use log::debug;
use toml;

#[test]
pub fn basic_1d_soliton() {
  // TODO put standard params here
  let input = Path::new("input/params.toml");
  let contents = fs::read_to_string(input).expect("Failed to read the TOML file");
  let mut params: Params = toml::from_str(&contents).expect("Failed to load the config");
  params.physics.t = 100.0;
  let n_l = params.numerics.n_l;
  let l = params.numerics.l;
  assert!(l > 2.0 * params.initial.w, "The domain is too small");
  let h_l: f64 = (l) / (n_l as f64);
  let l_range = symmetric_range(l, h_l);

  let mut initial_wave = Wavefunction1D {
      field: l_range
          .iter()
          .map(|x| gaussian_normalized(params.initial.w, &(x)))
          .collect(),
      l: l_range.clone(),
  };
  for (i, x) in initial_wave.l.iter().enumerate() {
    println!("{}: {}", i, x);
  }
  print!("norm = {}", normalization_factor_1d(&initial_wave));
  assert!(normalization_factor_1d(&initial_wave) - 1.0 < 1e-10, "The wavefunction is not normalized");

  let saved_psi = propagate_1d(&mut initial_wave, &params, true);
  let target_psi = Wavefunction1D {
    field: l_range
        .iter()
        .map(|x| ndrustfft::Complex::new(sech_normalized(params.physics.g, *x), 0.0))
        .collect(),
    l: l_range.clone(),
  };
  assert_eq!(saved_psi.psi[n_l].field, target_psi.field);
}

pub fn can_you_even_collapse() {

}