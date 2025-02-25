use crate::tools::*;
use crate::types::*;
use log::*;
use ndarray::{Array2, Array3};
use ndrustfft::Complex;
use ndrustfft::{ndfft_par, ndifft_par, FftHandler};
use rustfft;
use std::f64::INFINITY;
use std::time::Instant;
use std::vec;

// use ndrustfft;
pub const I: rustfft::num_complex::Complex<f64> = rustfft::num_complex::Complex::new(0.0, 1.0);
use crate::trotterized_operators::*;

/** Propagation function containing iteration loop.
propagate the wavefunction using specified simulation
parameters  */
pub fn propagate_1d(
    psi0: &mut Wavefunction1D,
    params: &Params,
    imaginary_time: bool,
) -> Dynamics1D {
    let mut h_t = ndrustfft::Complex::new(0.0, 0.0);
    if imaginary_time {
        info!("Imaginary time propagation");
        h_t.im = -params.numerics.dt;
    } else {
        h_t.re = params.numerics.dt;
    }
    let n_t = ((*params).physics.t / params.numerics.dt).round() as u32;
    debug!("Running using n_t = {}, and ht = {}", n_t, h_t.norm());
    let g = (*params).physics.g;
    let mut ns = normalization_factor_1d(psi0);
    assert!(
        (ns - 1.0).abs() < 1e-10,
        "The wavefunction is not normalized"
    );
    let n_l = psi0.field.len();
    let k_range_squared = k_squared(&k_vector(&psi0.l));
    // Plan the transforms
    let mut planner: rustfft::FftPlanner<f64> = rustfft::FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_l);
    let ifft = planner.plan_fft_inverse(n_l);
    // let mut dummy: Complex<f64> = I;
    // Run the propagation
    let mut saved_psi: Dynamics1D = Dynamics1D {
        psi: vec![Wavefunction1D::new(vec![I; n_l], vec![]); params.options.n_saves],
        t: vec![0.0; params.options.n_saves],
    };
    let dt_save = params.physics.t / params.options.n_saves as f64;
    let t_axis: Vec<f64> = (0..params.options.n_saves)
        .map(|i| dt_save * (i as f64))
        .collect();
    for it in 0..params.options.n_saves {
        saved_psi.t[it] = t_axis[it];
        saved_psi.psi[it] = Wavefunction1D::new(vec![I; n_l], psi0.l.clone());
    }
    ns = normalization_factor_1d(psi0);
    assert!(ns - 1.0 < 1e-10, "The wavefunction is not normalized");
    let mut save_interval = (n_t as f64 / params.options.n_saves as f64).round() as u32;

    if save_interval == 0 {
        save_interval = 1;
        info!("Using save_interval = 1");
    }
    warn!("we are overwriting the first element in saved_psi");
    let mut cnt = 0;
    for idt in 0..n_t {
        fft.process(&mut psi0.field);
        psi0.field
            .iter_mut()
            .for_each(|x: &mut Complex<f64>| *x = *x / (n_l as f64));
        linear_step_1d(psi0, &k_range_squared, h_t);
        ifft.process(&mut psi0.field);
        if params.physics.npse {
            nonlinear_npse(psi0, h_t, g);
        } else {
            nonlinear_step_1d(psi0, h_t, g);
        }
        if imaginary_time {
            ns = normalization_factor_1d(psi0);
            debug!("Normalization factor = {:10.5e}", ns);
            psi0.field.iter_mut().for_each(|x| *x = *x / ns);
        }
        ns = normalization_factor_1d(psi0);
        assert!(ns - 1.0 < 1e-10, "The wavefunction is not normalized");
        // println!("{}", psi0[1]/dummy);
        // dummy = psi0[1];
        // for idl in 0..n_l {
        //   println!("{:3.2e} ", psi0[idl]);
        // }
        // println!("=======")
        if idt % save_interval == 0 {
            for i in 0..n_l {
                saved_psi.psi[cnt].field[i] = psi0.field[i];
            }
            cnt += 1;
        }
    }
    saved_psi
}

/** Propagation function containing iteration loop.
propagate the wavefunction using specified simulation
parameters
how to perform the three dimensional fft:
iterate over the three axis and do the fft on each 1D slice.
This may be accelerated using Rayon?
*/
pub fn propagate_3d(
    mut psi0: &mut Wavefunction3D,
    params: &Params,
    imaginary_time: bool,
) -> View3D {
    let mut ns = normalization_factor_3d(&psi0);
    assert!(
        (ns - 1.0).abs() < 1e-10,
        "The wavefunction is not normalized"
    );
    let k_x = k_vector(&psi0.l_x);
    let k_y = k_vector(&psi0.l_y);
    let k_z = k_vector(&psi0.l_z);
    let k_squared = k_squared_3d(&k_x, &k_y, &k_z);
    let v0 = v0_harmonic(&psi0.l_x, &psi0.l_y, &psi0.l_z, params.physics.l_harm_x)
        + v0_optical_lattice(
            &psi0.l_x,
            &psi0.l_y,
            &psi0.l_z,
            params.physics.v0,
            params.physics.dl,
        );
    // let v0 = Array3::zeros((psi0.l_x.len(), psi0.l_y.len(), psi0.l_z.len()));
    let mut h_t = ndrustfft::Complex::new(0.0, 0.0);
    if imaginary_time {
        info!("Imaginary time propagation");
        h_t.im = -params.numerics.dt;
    } else {
        h_t.re = params.numerics.dt;
    }
    // Cloning is actually better... Still taking 3x the time of the debug version of rust-marangon with the same dimensions
    let mut buffer_1 = psi0.field.clone();
    let mut buffer_2 = psi0.field.clone();
    let size_x = psi0.l_x.len();
    let size_y = psi0.l_y.len();
    let size_z = psi0.l_z.len();
    let handler_x = FftHandler::new(size_x);
    let handler_y = FftHandler::new(size_y);
    let handler_z = FftHandler::new(size_z);
    // let buffer_2 = psi0.field.clone();

    let n_l = psi0.field.len();
    assert!(n_l == size_x * size_y * size_z);
    let n_t = ((*params).physics.t / params.numerics.dt).round() as u32;

    let dt_save = params.physics.t / params.options.n_saves as f64;
    let t_axis: Vec<f64> = (0..params.options.n_saves)
        .map(|i| dt_save * (i as f64))
        .collect();
    let mut saved_psi: View3D = View3D {
        movie: vec![
            Projections3D {
                xz: Array2::<f64>::zeros((1, 1)),
                yz: Array2::<f64>::zeros((1, 1))
            };
            params.options.n_saves
        ],
        l_x: psi0.l_x.clone(),
        l_y: psi0.l_y.clone(),
        l_z: psi0.l_z.clone(),
        t: t_axis,
    };

    let mut save_interval = (n_t as f64 / params.options.n_saves as f64).round() as u32;
    if save_interval == 0 {
        save_interval = 1;
    }
    debug!("Using save_interval = {}", save_interval);
    let t_start = Instant::now();
    warn!("we are overwriting the first element in saved_psi");
    let mut cnt = 0;
    info!(
        "Starting the propagation. Time steps = {}, save steps = {}",
        n_t, params.options.n_saves
    );
    for idt in 0..n_t {
        ndfft_par(&psi0.field, &mut buffer_1, &handler_x, 0);
        ndfft_par(&buffer_1, &mut buffer_2, &handler_y, 1);
        ndfft_par(&buffer_2, &mut psi0.field, &handler_z, 2);
        // psi0.field.iter_mut().for_each(|x| *x = *x / (n_l as f64));
        linear_step_3d(&mut psi0, &k_squared, h_t);
        ndifft_par(&psi0.field, &mut buffer_1, &handler_x, 0);
        ndifft_par(&buffer_1, &mut buffer_2, &handler_y, 1);
        ndifft_par(&buffer_2, &mut psi0.field, &handler_z, 2);
        // psi0.field.iter_mut().for_each(|x| *x = *x / (n_l as f64)); already normalized
        nonlinear_step_3d(&mut psi0, &v0, h_t, params.physics.g);

        if imaginary_time {
            ns = normalization_factor_3d(psi0);
            debug!("Normalization factor = {:10.5e}", ns);
            psi0.field.iter_mut().for_each(|x| *x = *x / ns);
        }
        ns = normalization_factor_3d(psi0);
        assert!(ns - 1.0 < 1e-10, "The wavefunction is not normalized");

        if idt % save_interval == 0 {
            debug!("saving step {}", cnt);
            saved_psi.movie[cnt] = Projections3D::new(psi0);
            cnt += 1;
        }
    }
    let t_elapsed = t_start.elapsed();
    ns = normalization_factor_3d(&psi0);
    if (ns - 1.0).abs() < 1e-6 {}
    warn!(
        "The wavefunction is not normalized. Normalization factor = {:10.5e}",
        ns
    );
    info!("Propagation done in {:?}", t_elapsed);

    saved_psi
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;
    use rand::Rng;
    use rustfft::num_complex;
    // TODO create some actually testable functions for the transforms
    #[test]
    fn involution_1d() {
        let n_l = 100;
        let l: Vec<f64> = (0..n_l).map(|x| x as f64).collect();
        let mut psi_gaussian: Wavefunction1D = Wavefunction1D {
            field: l.iter().map(|x| gaussian_normalized(1.0, x)).collect(),
            l: l.clone(),
        };
        let original_psi_field = psi_gaussian.field.clone();
        let mut planner: rustfft::FftPlanner<f64> = rustfft::FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(n_l);
        let ifft = planner.plan_fft_inverse(n_l);
        fft.process(&mut psi_gaussian.field);
        ifft.process(&mut psi_gaussian.field);
        psi_gaussian
            .field
            .iter_mut()
            .for_each(|x| *x = *x / ((n_l) as f64));
        for i in 0..n_l {
            print!(
                "orig {:3.2e}, invol {:3.2e} \n",
                original_psi_field[i].norm_sqr(),
                psi_gaussian.field[i].norm_sqr()
            );
        }
        psi_gaussian
            .field
            .iter()
            .zip(original_psi_field.iter())
            .for_each(|(x, y)| {
                assert!((x - y).norm_sqr() < 1e-16);
            });
    }

    #[test]
    fn involution_3d() {}
}
