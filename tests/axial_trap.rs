//! Convention of the axial trap parameter `l_harm_x`.
//!
//! The kernels add V = (x / l_harm_x)^2 / 2 (units: hbar omega_perp, l_perp), i.e. the
//! trap (omega_x/omega_perp)^2 x^2 / 2 for l_harm_x = omega_perp/omega_x. The ground
//! state of the linear 1D equation in that trap has <x^2> = omega_perp/(2 omega_x).
//! With the oscillator length sqrt(omega_perp/omega_x) in place of l_harm_x (as the
//! launcher rw.py wrote before this test) <x^2> would be sqrt(omega_perp/omega_x)/2.
use ndrustfft::Complex;
use rust_waves::propagate::propagate_1d;
use rust_waves::tools::{gaussian_normalized, symmetric_range, v0_axial_1d, v0_harmonic};
use rust_waves::types::{Gaussian, Numerics, Options, Params, Physics, Wavefunction1D};

fn params(l_harm_x: f64, l: f64, n_l: usize) -> Params {
    Params {
        title: "axial_trap_test".to_string(),
        numerics: Numerics { n_l, n_l_y: 1, n_l_z: 1, l, l_y: 1.0, l_z: 1.0, dt: 1e-2 },
        physics: Physics {
            g: 0.0,
            g5: 0.0,
            l_harm_x,
            v0: 0.0,
            dl: 1.0,
            t: 60.0,
            dimension: 1,
            npse: false,
            im_t: true,
        },
        initial: Gaussian { w: 1.0, w_y: 1.0, w_z: 1.0 },
        options: Options { n_saves: 4 },
    }
}

#[test]
fn ground_state_width_sets_trap_frequency() {
    let (l, n_l) = (40.0, 512);
    for &ratio in [0.1, 0.25, 0.5].iter() {
        // ratio = omega_x / omega_perp; gap to the first excited state = ratio
        let p = params(1.0 / ratio, l, n_l);
        let h = l / (n_l - 1) as f64;
        let x = symmetric_range(l, h);
        assert_eq!(x.len(), n_l);
        let mut psi = Wavefunction1D {
            field: x.iter().map(|xi| gaussian_normalized(1.0, xi)).collect(),
            l: x.clone(),
        };
        let out = propagate_1d(&mut psi, &p, true);
        let gs = &out.psi[out.psi.len() - 1];
        let norm: f64 = gs.field.iter().map(|f| f.norm_sqr()).sum::<f64>() * h;
        let x2: f64 = gs.field.iter().zip(x.iter()).map(|(f, xi)| f.norm_sqr() * xi * xi).sum::<f64>() * h / norm;
        let expected = 1.0 / (2.0 * ratio);
        let rel = (x2 - expected).abs() / expected;
        // first-order splitting in imaginary time: O(dt) relative error
        assert!(rel < 5e-3, "omega_x/omega_perp = {ratio}: <x^2> = {x2}, expected {expected}");
        let wrong = (1.0 / ratio).sqrt() / 2.0;
        assert!((x2 - wrong).abs() / wrong > 0.1, "the test does not separate the conventions");
    }
}

#[test]
fn potentials_use_the_same_convention() {
    let x: Vec<f64> = vec![-3.0, -0.5, 0.0, 1.0, 2.5];
    let l_harm_x = 4.0;
    let v1 = v0_axial_1d(&x, l_harm_x, 0.0, 1.0);
    let y = vec![0.0];
    let v3 = v0_harmonic(&x, &y, &y, l_harm_x);
    for (i, xi) in x.iter().enumerate() {
        let expected = 0.5 * (xi / l_harm_x).powi(2);
        assert!((v1[i] - Complex::new(expected, 0.0)).norm() < 1e-14);
        assert!((v3[[i, 0, 0]] - Complex::new(expected, 0.0)).norm() < 1e-14);
    }
}
