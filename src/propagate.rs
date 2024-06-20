use crate::{k_vector, types::*};
use log::*;
use rustfft::{num_complex::Complex, FftPlanner};
const I: Complex<f64> = Complex::new(0.0, 1.0);
use crate::tools::k_squared;

/** Propagation function containing iteration loop.
propagate the wavefunction using specified simulation
parameters  */
pub fn propagate(
    l_range: &Vec<f64>,
    psi0: &mut Vec<Complex<f64>>,
    params: &Params,
) -> Vec<Vec<Complex<f64>>> {
    let mu_max = 1000.0;
    let h_t = 1. / mu_max;
    let n_t = ((*params).physics.t / h_t).round() as u32;
    // let n_t = 2;
    debug!("Running using n_t = {}", n_t);
    let g = (*params).physics.g;

    let n_l = psi0.len();
    let k_range_squared = k_squared(&k_vector(l_range));

    // Plan the transforms
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_l);
    let ifft = planner.plan_fft_inverse(n_l);
    warn!("Using unscaled transforms");
    // let mut dummy: Complex<f64> = I;
    // Run the propagation
    let mut saved_psi: Vec<Vec<Complex<f64>>> = vec![vec![I; n_l]; params.options.n_saves];
    let mut save_interval = (n_t as f64 / params.options.n_saves as f64).round() as u32;

    if save_interval == 0 {
        save_interval = 1;
        info!("Using save_interval = 1");
    }
    let mut cnt = 0;
    for idt in 0..n_t {
        fft.process(psi0);
        linear_step(psi0, &k_range_squared, h_t);
        ifft.process(psi0);
        psi0.iter_mut().for_each(|x| *x = *x / (n_l as f64));
        nonlinear_step(psi0, h_t, g);
        // println!("{}", psi0[1]/dummy);
        // dummy = psi0[1];
        // for idl in 0..n_l {
        //   println!("{:3.2e} ", psi0[idl]);
        // }
        // println!("=======")
        if idt % save_interval == 0 {
            for i in 0..n_l {
                saved_psi[cnt][i] = psi0[i];
            }
            cnt += 1;
        }
    }
    saved_psi
}

/**
 Perform the linear propagation step in reciprocal space
 this include
  - kinetic operator
*/
fn linear_step(kvec: &mut Vec<Complex<f64>>, k_range: &Vec<f64>, dt: f64) {
    kvec.iter_mut()
        .zip(k_range.iter())
        .for_each(|(x, y)| *x *= (-I *100.0* dt * 1. / 2. * y).exp());
}

/**
 Perform the nonlinear step in direct space
 this includes:
  - nonlinearity
  - external potentials
*/
fn nonlinear_step(xvec: &mut Vec<Complex<f64>>, dt: f64, g: f64) {
    xvec.iter_mut()
        .for_each(|x| *x *= (I * dt * g * x.powf(2.0)).exp());
}
