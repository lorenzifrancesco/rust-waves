use rustfft::{num_complex::Complex, FftPlanner};
use crate::types::Sim;

/** Propagation function containing iteration loop.
 propagate the wavefunction using specified simulation
 parameters  */
pub fn propagate(psi0: Vec<Complex<f64>>, s: Sim) {

  // Plan the FFT transforms

  // Run the propagation
  // for tt in 0..s.nt {
  //   linear_step(vec, dt);
  //   Tkx * vec;
  //   nonlinear_step(vec, dt);
  //   Txk * vec;
  // }
  // vec
}

/** Perform the linear propagation step in reciprocal space
 this include 
  - kinetic operator */
fn linear_step(kvec: &mut Vec<Complex<f64>>, dt: f64) {
    // let mut planner = FftPlanner::new();
    // let fft = planner.plan_fft_forward(50);
    // return kvec;
    return 
}

/** Perform the nonlinear step in direct space
 this includes: 
  - nonlinearity 
  - external potentials */
fn nonlinear_step(xvec: &mut Vec<Complex<f64>>, dt: f64) {
  return 
}

