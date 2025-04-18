use crate::propagate::I;
use crate::types::{Wavefunction1D, Wavefunction3D};
use ndarray::{Array1, Array3, Zip};
use ndrustfft::Complex;
use std::f64::consts::PI;

/**
 Perform the linear propagation step in reciprocal space
 this include
  - kinetic operator
*/
pub fn linear_step_1d(kvec: &mut Wavefunction1D, k_range_squared: &Vec<f64>, dt: Complex<f64>) {
    kvec.field
        .iter_mut()
        .zip(k_range_squared.iter())
        .for_each(|(x, y)| *x *= (-I * dt * 1. / 2. * y).exp());
    // debug!("{}", k_range_squared[]);
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
    xvec.field
        .iter_mut()
        .for_each(|x| *x *= (-I * dt * (-I * g5_1d * x.norm_sqr() + g) * x.norm_sqr()).exp());
    xvec.field
        .iter_mut()
        .zip(v0.iter())
        .for_each(|(x, y)| *x *= (-I * dt * y).exp());
}
/**
 * Perform the nonlinear propagation step using the NPSE equation
 */
pub fn nonlinear_npse(
    xvec: &mut Wavefunction1D,
    v0: &Array1<Complex<f64>>,
    dt: Complex<f64>,
    g: f64,
    g5_1d: f64,
) {
    // TODO check
    xvec.field.iter_mut().for_each(|x| {
        *x *= (-I
            * dt
            * (g * x.norm_sqr() / (1.0 + g * x.norm_sqr()).sqrt()
                + 1.0 / 2.0
                    * ((1.0 + g * x.norm_sqr()).sqrt() + 1.0 / (1.0 + g * x.norm_sqr()).sqrt())
                - I * g5_1d * x.norm_sqr().powi(2) / ((1.0 + g * x.norm_sqr()).sqrt())))
        .exp()
    });
    xvec.field
        .iter_mut()
        .zip(v0.iter())
        .for_each(|(x, y)| *x *= (-I * dt * y).exp());
} // predice il collasso corretto!

/**
 Perform the linear propagation step in reciprocal space
 this include
  - kinetic operator
*/
pub fn linear_step_3d(kvec: &mut Wavefunction3D, k_range_squared: &Array3<f64>, dt: Complex<f64>) {
    // kvec.field.iter_mut()
    //     .zip(k_range_squared.iter())
    //     .for_each(|(x, y)| *x *= (-I * dt * 1. / 2. * y).exp());
    for ((ix, iy, iz), psi) in kvec.field.indexed_iter_mut() {
        *psi *= (-I * dt * 1. / 2. * k_range_squared[[ix, iy, iz]])
                .exp();
    }
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
    xvec.field.iter_mut().for_each(|x| {
        *x *= (-I * dt * ((-I * g5 * x.norm_sqr() + 2.0 * PI * g) * x.norm_sqr()))
              .exp();
    });
    Zip::indexed(&mut xvec.field)
    .for_each(|(i, j, k), x| *x *= (-I * dt * v0[[i, j, k]])
    .exp());
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
        let v0 = Array1::ones(10);
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
