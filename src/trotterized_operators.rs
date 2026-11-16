use crate::propagate::I;
use crate::types::{Wavefunction1D, Wavefunction3D};
use ndarray::{Array1, Array3, Zip};
use ndrustfft::Complex;
use std::f64::consts::PI;

// Each step operator comes in two forms: `*_with` takes the precomputed diagonal
// propagators and is what the time loops call; the plain form computes them and calls
// `*_with`, so that both forms run the same arithmetic. Splitting order and the
// frozen-density evaluation of the nonlinear factor are those of the original code.

/// exp(-i dt k^2/2) * scale for each k^2 (scale = 1/n for an unnormalised inverse FFT).
pub fn kinetic_propagator(k_squared: &[f64], dt: Complex<f64>, scale: f64) -> Vec<Complex<f64>> {
    k_squared.iter().map(|k| (-I * dt * 0.5 * k).exp() * scale).collect()
}

/// exp(-i dt k^2/2) on the 3D grid.
pub fn kinetic_propagator_3d(k_squared: &Array3<f64>, dt: Complex<f64>) -> Array3<Complex<f64>> {
    k_squared.mapv(|k| (-I * dt * 0.5 * k).exp())
}

/// exp(-i dt V) for a static potential V.
pub fn potential_propagator_1d(v0: &Array1<Complex<f64>>, dt: Complex<f64>) -> Array1<Complex<f64>> {
    v0.mapv(|v| (-I * dt * v).exp())
}

/// exp(-i dt V) for a static potential V on the 3D grid.
pub fn potential_propagator_3d(v0: &Array3<Complex<f64>>, dt: Complex<f64>) -> Array3<Complex<f64>> {
    v0.mapv(|v| (-I * dt * v).exp())
}

/**
 Perform the linear propagation step in reciprocal space
 this include
  - kinetic operator
*/
pub fn linear_step_1d(kvec: &mut Wavefunction1D, k_range_squared: &Vec<f64>, dt: Complex<f64>) {
    linear_step_1d_with(kvec, &kinetic_propagator(k_range_squared, dt, 1.0));
}

pub fn linear_step_1d_with(kvec: &mut Wavefunction1D, kinetic: &[Complex<f64>]) {
    kvec.field.iter_mut().zip(kinetic.iter()).for_each(|(x, p)| *x *= *p);
}

/**
Perform the nonlinear step in direct space
this includes:
- nonlinearity
- external potentials
*/
pub fn nonlinear_step_1d(
    xvec: &mut Wavefunction1D,
    v0: &Array1<Complex<f64>>,
    dt: Complex<f64>,
    g: f64,
    g5_1d: f64,
) {
    nonlinear_step_1d_with(xvec, &potential_propagator_1d(v0, dt), dt, g, g5_1d);
}

pub fn nonlinear_step_1d_with(
    xvec: &mut Wavefunction1D,
    potential: &Array1<Complex<f64>>,
    dt: Complex<f64>,
    g: f64,
    g5_1d: f64,
) {
    xvec.field.iter_mut().zip(potential.iter()).for_each(|(x, p)| {
        let n = x.norm_sqr();
        *x *= (-I * dt * (-I * g5_1d * n + g) * n).exp() * *p;
    });
}

/**
 * Perform the nonlinear propagation step using the NPSE equation.
 *
 * Units: lengths in a_perp, times in 1/omega_perp, n = |psi|^2 the axial density
 * (normalised to 1). The transverse profile is the Gaussian
 * |phi|^2 = exp(-rho^2/sigma^2)/(pi sigma^2) with sigma^2 = sqrt(1 + g n).
 *
 * Three-body loss: the 3D term -i g5 |psi|^4 psi, integrated over the transverse
 * Gaussian, gives -i g5 n^2 INT|phi|^6 d^2rho = -i g5 n^2 / (3 pi^2 sigma^4)
 *                                          = -i g5_1d n^2 / (1 + g n),
 * with g5_1d = g5 / (3 pi^2) (see propagate_1d). For g = 0 this is the loss of
 * `nonlinear_step_1d`.
 */
pub fn nonlinear_npse(
    xvec: &mut Wavefunction1D,
    v0: &Array1<Complex<f64>>,
    dt: Complex<f64>,
    g: f64,
    g5_1d: f64,
) {
    nonlinear_npse_with(xvec, &potential_propagator_1d(v0, dt), dt, g, g5_1d);
}

pub fn nonlinear_npse_with(
    xvec: &mut Wavefunction1D,
    potential: &Array1<Complex<f64>>,
    dt: Complex<f64>,
    g: f64,
    g5_1d: f64,
) {
    xvec.field.iter_mut().zip(potential.iter()).for_each(|(x, p)| {
        let n = x.norm_sqr();
        let sigma2 = (1.0 + g * n).sqrt(); // sigma^2 / a_perp^2
        *x *= (-I
            * dt
            * (g * n / sigma2 + 1.0 / 2.0 * (sigma2 + 1.0 / sigma2)
                - I * g5_1d * n.powi(2) / sigma2.powi(2)))
        .exp()
            * *p;
    });
}

/**
 Perform the linear propagation step in reciprocal space
 this include
  - kinetic operator
*/
pub fn linear_step_3d(kvec: &mut Wavefunction3D, k_range_squared: &Array3<f64>, dt: Complex<f64>) {
    linear_step_3d_with(kvec, &kinetic_propagator_3d(k_range_squared, dt));
}

pub fn linear_step_3d_with(kvec: &mut Wavefunction3D, kinetic: &Array3<Complex<f64>>) {
    Zip::from(&mut kvec.field).and(kinetic).par_for_each(|x, p| *x *= *p);
}

/**
Perform the nonlinear step in direct space
this includes:
- nonlinearity
- external potentials
*/
pub fn nonlinear_step_3d(
    xvec: &mut Wavefunction3D,
    v0: &Array3<ndrustfft::Complex<f64>>,
    dt: Complex<f64>,
    g: f64,
    g5: f64,
) {
    nonlinear_step_3d_with(xvec, &potential_propagator_3d(v0, dt), dt, g, g5);
}

pub fn nonlinear_step_3d_with(
    xvec: &mut Wavefunction3D,
    potential: &Array3<Complex<f64>>,
    dt: Complex<f64>,
    g: f64,
    g5: f64,
) {
    Zip::from(&mut xvec.field).and(potential).par_for_each(|x, p| {
        let n = x.norm_sqr();
        *x *= (-I * dt * ((-I * g5 * n + 2.0 * PI * g) * n)).exp() * *p;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_nonlinear() {
        let mut psi: Wavefunction1D = Wavefunction1D {
            field: vec![I; 10],
            l: vec![0.0; 10],
        };
        let g = 0.0;
        let g5 = 0.0;
        let h_t = Complex::new(0.01, 0.0);
        let v0 = Array1::zeros(10);
        nonlinear_step_1d(&mut psi, &v0, h_t * 100.0, g, g5);
        assert_eq!(psi.field, vec![I; 10]);
        // these tests are broken
        // nonlinear_npse(&mut psi, h_t * 100.0, g);
        // assert_eq!(psi.field, vec![I; 10]);
        // let mut psi3: Wavefunction3D = Wavefunction3D {
        //     field: Array3::ones((10, 10, 10)),
        //     l_x: vec![0.0; 10],
        //     l_y: vec![0.0; 10],
        //     l_z: vec![0.0; 10],
        // };
        // let v0 = Array3::ones((10, 10, 10));
        // nonlinear_step_3d(&mut psi3, &v0, h_t * 100.0, g);
        // assert_eq!(psi3.field, Array3::ones((10, 10, 10)));
    }
}

/// Tests of the three-body loss in the NPSE step against the transverse integral
/// of the 3D loss term. The reference values are computed by quadrature, not from
/// the closed form used in `nonlinear_npse`.
#[cfg(test)]
mod npse_loss_tests {
    use super::*;

    /// INT |phi|^6 d^2rho for |phi|^2 = exp(-rho^2/sigma^2)/(pi sigma^2), by
    /// composite Simpson on [0, 12 sigma] (the integrand is < 1e-180 beyond it).
    fn transverse_sextic_integral(sigma: f64) -> f64 {
        let n_int = 20_000; // even
        let r_max = 12.0 * sigma;
        let h = r_max / n_int as f64;
        let f = |r: f64| {
            let p2 = (-r * r / (sigma * sigma)).exp() / (PI * sigma * sigma);
            2.0 * PI * r * p2.powi(3)
        };
        let mut acc = f(0.0) + f(r_max);
        for i in 1..n_int {
            let w = if i % 2 == 1 { 4.0 } else { 2.0 };
            acc += w * f(i as f64 * h);
        }
        acc * h / 3.0
    }

    fn one_point(n: f64) -> Wavefunction1D {
        Wavefunction1D {
            field: vec![Complex::new(n.sqrt(), 0.0)],
            l: vec![0.0],
        }
    }

    const G_VALUES: [f64; 7] = [-1.2, -0.6, -0.2, 0.0, 0.5, 2.0, 5.0];
    const N_VALUES: [f64; 5] = [0.02, 0.1, 0.3, 0.5, 0.8];

    /// The atom loss rate per unit length produced by one NPSE step equals
    /// g5 n^3 INT|phi|^6 d^2rho with the NPSE width sigma^2 = sqrt(1 + g n):
    /// d|psi|^2/dt = -2 g5 n^3 INT|phi|^6, i.e. the 3D loss -2 g5 INT|psi|^6 d^2rho
    /// evaluated on the NPSE transverse profile. The step is exact for the frozen
    /// density, so the comparison holds to quadrature accuracy.
    #[test]
    fn npse_loss_rate_matches_transverse_integral() {
        let g5 = 7.3;
        let g5_1d = g5 / (3.0 * PI.powi(2)); // as in propagate_1d
        let v0 = Array1::zeros(1);
        let mut checked = 0;
        for &g in G_VALUES.iter() {
            for &n in N_VALUES.iter() {
                if 1.0 + g * n < 0.05 {
                    continue; // beyond the NPSE collapse
                }
                let sigma = (1.0 + g * n).powf(0.25);
                let expected = g5 * n.powi(2) * transverse_sextic_integral(sigma);
                // the step is exact for frozen density: take a step that removes
                // about 20% of the density, so the logarithm is not cancellation-limited
                let dt = Complex::new(0.1 / expected, 0.0);
                let mut psi = one_point(n);
                nonlinear_npse(&mut psi, &v0, dt, g, g5_1d);
                let n_after = psi.field[0].norm_sqr();
                let rate = -(n_after / n).ln() / (2.0 * dt.re); // amplitude decay rate
                let rel = (rate - expected).abs() / expected;
                assert!(
                    rel < 1e-9,
                    "g = {g}, n = {n}: loss rate {rate:.12e}, transverse integral {expected:.12e} \
                     (ratio {:.6})",
                    rate / expected
                );
                checked += 1;
            }
        }
        assert!(checked >= 30, "too few (g, n) pairs checked: {checked}");
    }

    /// For g = 0 the NPSE width is a_perp and the NPSE loss equals the loss of the
    /// 1D GPE step (same g5_1d).
    #[test]
    fn npse_loss_reduces_to_1d_gpe_at_zero_coupling() {
        let g5_1d = 0.37;
        let dt = Complex::new(2e-2, 0.0);
        let v0 = Array1::zeros(1);
        for &n in N_VALUES.iter() {
            let mut a = one_point(n);
            let mut b = one_point(n);
            nonlinear_npse(&mut a, &v0, dt, 0.0, g5_1d);
            nonlinear_step_1d(&mut b, &v0, dt, 0.0, g5_1d);
            let (na, nb) = (a.field[0].norm_sqr(), b.field[0].norm_sqr());
            assert!((na - nb).abs() < 1e-14 * nb, "n = {n}: npse {na:e}, gpe1d {nb:e}");
        }
    }

    /// The loss term changes only the modulus: with g5 = 0 the step is unitary and
    /// its phase is the NPSE local energy g n/sigma^2 + (sigma^2 + 1/sigma^2)/2;
    /// with g5 > 0 the phase is the same.
    #[test]
    fn npse_conservative_part_unchanged() {
        let dt = Complex::new(1e-2, 0.0);
        let v0 = Array1::zeros(1);
        for &g in G_VALUES.iter() {
            for &n in N_VALUES.iter() {
                if 1.0 + g * n < 0.05 {
                    continue;
                }
                let s2 = (1.0 + g * n).sqrt();
                let mu = g * n / s2 + 0.5 * (s2 + 1.0 / s2);
                let mut c = one_point(n);
                nonlinear_npse(&mut c, &v0, dt, g, 0.0);
                assert!((c.field[0].norm_sqr() - n).abs() < 1e-14);
                let phase = c.field[0].arg();
                assert!((phase + mu * dt.re).abs() < 1e-12, "g = {g}, n = {n}");
                let mut d = one_point(n);
                nonlinear_npse(&mut d, &v0, dt, g, 0.5);
                assert!((d.field[0].arg() - phase).abs() < 1e-12, "g = {g}, n = {n}");
                assert!(d.field[0].norm_sqr() < n);
            }
        }
    }
}
