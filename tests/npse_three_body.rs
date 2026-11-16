//! End-to-end test of the three-body loss in the NPSE through `propagate_1d`.
//!
//! A homogeneous condensate in a periodic box stays homogeneous (the kinetic step
//! acts only on k = 0), so its density obeys the ODE of the NPSE loss,
//!     dn/dt = -2 c n^3 / (1 + g n),   c = g5 / (3 pi^2),
//! which follows from the 3D term -i g5 |psi|^4 psi integrated over the NPSE
//! Gaussian of width sigma^2 = sqrt(1 + g n). Separating variables,
//!     1/(2 n^2) + g/n = 1/(2 n0^2) + g/n0 + 2 c t,
//! a quadratic in 1/n. The split-step solution must converge to it at first order
//! in dt (the loss factor is evaluated with the density at the start of the step).
use rust_waves::propagate::propagate_1d;
use rust_waves::tools::{normalization_factor_1d, symmetric_range};
use rust_waves::types::{Gaussian, Numerics, Options, Params, Physics, Wavefunction1D};
use ndrustfft::Complex;
use std::f64::consts::PI;

const BOX: f64 = 3.0;
const N_L: usize = 32;

fn params(g: f64, g5: f64, t: f64, dt: f64) -> Params {
    Params {
        title: "npse_three_body_test".to_string(),
        numerics: Numerics { n_l: N_L, n_l_y: 1, n_l_z: 1, l: BOX, l_y: 1.0, l_z: 1.0, dt },
        physics: Physics {
            g,
            g5,
            l_harm_x: 1e300,
            v0: 0.0,
            dl: 1.0,
            t,
            dimension: 1,
            npse: true,
            im_t: false,
        },
        initial: Gaussian { w: 1.0, w_y: 1.0, w_z: 1.0 },
        options: Options { n_saves: 4 },
    }
}

/// Uniform, normalised field on the solver grid; returns it and its density.
fn homogeneous() -> (Wavefunction1D, f64) {
    let h = BOX / (N_L - 1) as f64;
    let l = symmetric_range(BOX, h);
    assert_eq!(l.len(), N_L);
    let n0 = 1.0 / (h * N_L as f64);
    let psi = Wavefunction1D { field: vec![Complex::new(n0.sqrt(), 0.0); N_L], l };
    assert!((normalization_factor_1d(&psi) - 1.0).abs() < 1e-12);
    (psi, n0)
}

fn exact_density(n0: f64, g: f64, g5: f64, t: f64) -> f64 {
    let c = g5 / (3.0 * PI.powi(2));
    let k = 1.0 / (2.0 * n0 * n0) + g / n0 + 2.0 * c * t;
    1.0 / (-g + (g * g + 2.0 * k).sqrt())
}

fn simulated_density(g: f64, g5: f64, t: f64, dt: f64) -> f64 {
    let (mut psi, _) = homogeneous();
    let p = params(g, g5, t, dt);
    let out = propagate_1d(&mut psi, &p, false);
    let last = &out.psi[out.psi.len() - 1];
    let mean = last.field.iter().map(|x| x.norm_sqr()).sum::<f64>() / N_L as f64;
    let spread = last
        .field
        .iter()
        .map(|x| (x.norm_sqr() - mean).abs())
        .fold(0.0, f64::max);
    assert!(spread < 1e-12 * mean, "the homogeneous state did not stay homogeneous");
    mean
}

#[test]
fn npse_homogeneous_three_body_decay() {
    let (_, n0) = homogeneous();
    let (t, g5) = (2.0, 60.0);
    for &g in [-1.0, 0.0, 2.0, 6.0].iter() {
        let exact = exact_density(n0, g, g5, t);
        // the loss is not a small perturbation over this time
        assert!(exact < 0.9 * n0, "g = {g}: decay too weak to test ({exact} vs {n0})");
        let e1 = (simulated_density(g, g5, t, 2e-3) - exact).abs() / exact;
        let e2 = (simulated_density(g, g5, t, 1e-3) - exact).abs() / exact;
        assert!(e2 < 2e-4, "g = {g}: relative error {e2:e} at dt = 1e-3");
        let order = (e1 / e2).log2();
        assert!(
            (0.8..1.2).contains(&order),
            "g = {g}: convergence order {order:.3} (errors {e1:e}, {e2:e})"
        );
    }
}

/// The NPSE loss with the transverse factor sigma^-2 instead of sigma^-4
/// (rust-waves 0f53a52 to this fix) changes the final density by more than 2%,
/// i.e. 100 times the tolerance of the test above, for g = 2 and g = 6.
#[test]
fn wrong_transverse_power_is_detectable() {
    let (_, n0) = homogeneous();
    let (t, g5) = (2.0, 60.0);
    for &g in [2.0, 6.0].iter() {
        // dn/dt = -2 c n^3 / sqrt(1 + g n), integrated by RK4
        let c = g5 / (3.0 * PI.powi(2));
        let f = |n: f64| -2.0 * c * n.powi(3) / (1.0 + g * n).sqrt();
        let (mut n, h): (f64, f64) = (n0, 1e-4);
        for _ in 0..(t / h).round() as usize {
            let k1 = f(n);
            let k2 = f(n + 0.5 * h * k1);
            let k3 = f(n + 0.5 * h * k2);
            let k4 = f(n + h * k3);
            n += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
        let exact = exact_density(n0, g, g5, t);
        assert!((n - exact).abs() / exact > 100.0 * 2e-4, "g = {g}: {n} vs {exact}");
    }
}
