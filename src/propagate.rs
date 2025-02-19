use crate::tools::*;
use crate::types::*;
use hdf5_metno::Group;
use log::*;
use ndarray::Array3;
use ndrustfft::{ndfft_par, FftHandler};
use rustfft;
use std::time::Instant;
use std::vec;
// use ndrustfft;
pub const I: rustfft::num_complex::Complex<f64> = rustfft::num_complex::Complex::new(0.0, 1.0);
use crate::trotterized_operators::*;

/** Propagation function containing iteration loop.
propagate the wavefunction using specified simulation
parameters  */
pub fn propagate_1d(psi0: &mut Wavefunction1D, params: &Params, imaginary_time: bool) -> Dynamics1D {
    let mu_max = 1000.0;
    let h_t = 1. / mu_max;
    if imaginary_time {
        info!("Imaginary time propagation");
        let h_t = ndrustfft::Complex::new(0.0, h_t);
    }
    let n_t = ((*params).physics.t / h_t).round() as u32;
    debug!("Running using n_t = {}", n_t);
    let g = (*params).physics.g;
    let mut ns = normalization_factor_1d(psi0);
    assert!((ns-1.0).abs() < 1e-10, "The wavefunction is not normalized");
    let n_l = psi0.field.len();
    let k_range_squared = k_squared(&k_vector(&psi0.l));

    // Plan the transforms
    let mut planner: rustfft::FftPlanner<f64> = rustfft::FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n_l);
    let ifft = planner.plan_fft_inverse(n_l);
    warn!("Using unscaled transforms");
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
        if it == 0 {
            saved_psi.psi[it] = Wavefunction1D::new(vec![I; n_l], psi0.l.clone());
        } else {
            saved_psi.psi[it] = Wavefunction1D::new(vec![I; n_l], vec![]);
        }
    }
    ns = normalization_factor_1d(psi0);
    if (ns-1.0).abs()>1e-10 {
      warn!("The wavefunction is not normalized. Normalization factor = {:10.5e}", ns);
    }
    let mut save_interval = (n_t as f64 / params.options.n_saves as f64).round() as u32;

    if save_interval == 0 {
        save_interval = 1;
        info!("Using save_interval = 1");
    }
    warn!("we are overwriting the first element in saved_psi");
    let mut cnt = 0;
    for idt in 0..n_t {
        fft.process(&mut psi0.field);
        linear_step_1d(psi0, &k_range_squared, h_t);
        ifft.process(&mut psi0.field);
        psi0.field.iter_mut().for_each(|x| *x = *x / (n_l as f64));
        nonlinear_step_1d(psi0, h_t, g);
        if imaginary_time {
            ns = normalization_factor_1d(psi0);
            psi0.field.iter_mut().for_each(|x| *x /= ns);
        }
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
pub fn propagate_3d(mut psi0: &mut Wavefunction3D, params: &Params) -> Dynamics3D {
    let mut ns = normalization_factor_3d(&psi0);
    assert!((ns-1.0).abs() < 1e-10, "The wavefunction is not normalized");
    let k_x = k_vector(&psi0.l_x);
    let k_y = k_vector(&psi0.l_y);
    let k_z = k_vector(&psi0.l_z);
    let k_squared = k_squared_3d(&k_x, &k_y, &k_z);

    let dt = params.numerics.dt;
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
    let n_t = ((*params).physics.t / dt).round() as u32;
    let mut saved_psi: Dynamics3D = Dynamics3D {
        psi: vec![
            Wavefunction3D::new(
                buffer_1.clone(),
                psi0.l_x.clone(),
                psi0.l_y.clone(),
                psi0.l_z.clone()
            );
            params.options.n_saves
        ],
        t: vec![0.0; params.options.n_saves],
    };

    let dt_save = params.physics.t / params.options.n_saves as f64;
    let t_axis: Vec<f64> = (0..params.options.n_saves)
        .map(|i| dt_save * (i as f64))
        .collect();
    for it in 0..params.options.n_saves {
        saved_psi.t[it] = t_axis[it];
        if it == 0 {
            saved_psi.psi[it] = Wavefunction3D::new(
                buffer_1.clone(),
                psi0.l_x.clone(),
                psi0.l_y.clone(),
                psi0.l_z.clone(),
            );
        } else {
            saved_psi.psi[it] = Wavefunction3D::new(
                Array3::zeros((size_x, size_y, size_z)),
                psi0.l_x.clone(),
                psi0.l_y.clone(),
                psi0.l_z.clone(),
            );
        }
    }
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
        linear_step_3d(&mut psi0, &k_squared, dt);
        ndfft_par(&psi0.field, &mut buffer_1, &handler_x, 0);
        ndfft_par(&buffer_1, &mut buffer_2, &handler_y, 1);
        ndfft_par(&buffer_2, &mut psi0.field, &handler_z, 2);
        psi0.field.iter_mut().for_each(|x| *x = *x / (n_l as f64));
        nonlinear_step_3d(&mut psi0, dt, params.physics.g);
        if idt % save_interval == 0 {
            debug!("saving step {}", cnt);
            saved_psi.psi[cnt].field = psi0.field.clone();
            cnt += 1;
        }
    }
    let t_elapsed = t_start.elapsed();
    ns = normalization_factor_3d(&psi0);
    if (ns-1.0).abs()<1e-10 {
    }
    warn!("The wavefunction is not normalized. Normalization factor = {:10.5e}", ns);
    info!("Propagation done in {:?}", t_elapsed);

    saved_psi
}