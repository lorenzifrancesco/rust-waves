use crate::types::{Wavefunction1D, Wavefunction3D};
use log::warn;
use ndarray::Array3;
use rustfft::num_complex::Complex;
use std::f64::{self, consts::PI};

pub fn gaussian(a: f64, w: f64, x: &f64) -> Complex<f64> {
    Complex::new(a * (-(x / w).powf(2.) / 2.).exp(), 0.0)
}

pub fn sech_normalized(g: f64, x: f64) -> f64 {
    (g / 2.0).sqrt() * 2.0 / ((g * x).exp() + (-g * x).exp())
}

// pub fn sech_1d_normalized() {

// }

pub fn coordinate_to_index(x: f64, l: f64, n_l: usize) -> usize {
    let h_l = l / (n_l as f64);
    let index = ((x + l / 2.0) / h_l).round() as usize;
    if index >= n_l {
        warn!("Index out of bounds: {} >= {}", index, n_l);
        return n_l - 1;
    }
    index
}

pub fn index_to_coordinate(index: usize, l: f64, n_l: usize) -> f64 {
    let h_l = l / (n_l as f64);
    let x = (index as f64) * h_l - l / 2.0;
    x
}

pub fn gaussian_3d(
    l_domain: (&Vec<f64>, &Vec<f64>, &Vec<f64>),
    mean: (f64, f64, f64),
    sigma: (f64, f64, f64),
) -> Array3<ndrustfft::Complex<f64>> {
    let (nx, ny, nz) = (l_domain.0.len(), l_domain.1.len(), l_domain.2.len());
    let mut array = Array3::<ndrustfft::Complex<f64>>::zeros((nx, ny, nz));
    for (ix, x) in l_domain.0.iter().enumerate() {
        for (iy, y) in l_domain.1.iter().enumerate() {
            for (iz, z) in l_domain.2.iter().enumerate() {
                let dx = x - mean.0;
                let dy = y - mean.1;
                let dz = z - mean.2;
                let gaussian_value = (1.0 / ((2.0 * PI).sqrt())).powi(3)
                    * (-((dx / sigma.0).powi(2) + (dy / sigma.1).powi(2) + (dz / sigma.2).powi(2))
                        / (2.0))
                        .exp();
                array[[ix, iy, iz]].re = gaussian_value;
                array[[ix, iy, iz]].im = 0.0;
            }
        }
    }
    array
}

pub fn float2complex(f: f64) -> Complex<f64> {
    Complex::new(f, 0.0)
}

pub fn symmetric_range(length: f64, step: f64) -> Vec<f64> {
    let steps_l = ((length / step).round() + 1.0) as usize;
    let mut vec = vec![0.0; steps_l];
    let mut current = -length / 2.0;
    for i in 0..steps_l {
        vec[i] = current;
        current += step
    }
    vec
}

pub fn k_vector(x_vector: &Vec<f64>) -> Vec<f64> {
    let steps_l = x_vector.len();
    let mut vec = vec![0.0; steps_l];
    let dx = x_vector[1] - x_vector[0];
    // let k_n = 2.0 * PI / (2.0 * dx);
    // N * dx
    let dk = 2.0 * PI / (dx * (steps_l as f64));
    let mut current = 0.0;
    for i in 0..steps_l {
        vec[i] = current;
        current += dk;
    }
    vec
}

pub fn k_squared(k_vector: &Vec<f64>) -> Vec<f64> {
    let n = k_vector.len();
    let k_folding = k_vector[1] * (n as f64) / 2.0;
    let mut vec = vec![0.0; n];
    println!("{}", k_folding);

    for i in 1..n {
        if k_vector[i] < k_folding {
            vec[i] = k_vector[i].powi(2);
        } else {
            vec[i] = (2.0 * k_folding - k_vector[i]).powi(2);
        }
    }
    vec
}

pub fn k_squared_3d(k_x: &Vec<f64>, k_y: &Vec<f64>, k_z: &Vec<f64>) -> Array3<f64> {
    let (nx, ny, nz) = (k_x.len(), k_y.len(), k_z.len());

    // Calculate folding thresholds
    let k_folding_x = k_x[1] * (nx as f64) / 2.0;
    let k_folding_y = k_y[1] * (ny as f64) / 2.0;
    let k_folding_z = k_z[1] * (nz as f64) / 2.0;

    let mut k_squared = Array3::<f64>::zeros((nx, ny, nz));

    for (ix, &x) in k_x.iter().enumerate() {
        let folded_x = if x < k_folding_x {
            x
        } else {
            2.0 * k_folding_x - x
        };

        for (iy, &y) in k_y.iter().enumerate() {
            let folded_y = if y < k_folding_y {
                y
            } else {
                2.0 * k_folding_y - y
            };

            for (iz, &z) in k_z.iter().enumerate() {
                let folded_z = if z < k_folding_z {
                    z
                } else {
                    2.0 * k_folding_z - z
                };

                // Compute squared magnitude with folding
                k_squared[[ix, iy, iz]] = folded_x.powi(2) + folded_y.powi(2) + folded_z.powi(2);
            }
        }
    }

    k_squared
}

/**
 * Compute the normalization factor for the 1D wavefunction.
 * We assume to have the integral of the wavefunction squared set to 1.
 * - in the future, better to implement it in a smart way through traits
 */
pub fn normalization_factor_1d(psi: &Wavefunction1D) -> f64 {
    let h = psi.l[1] - psi.l[0];
    let norm = psi.field.iter().fold(0.0, |acc, x| acc + x.norm_sqr());
    (h * norm).sqrt()
}

/**
 * Compute the normalization factor for the 3D wavefunction.
 * We assume to have the integral of the wavefunction squared set to 1.
 * - in the future, better to implement it in a smart way through traits
 */
pub fn normalization_factor_3d(psi: &Wavefunction3D) -> f64 {
    let dv = (psi.l_x[1] - psi.l_x[0]) * (psi.l_y[1] - psi.l_y[0]) * (psi.l_z[1] - psi.l_z[0]);
    let norm = psi.field.iter().fold(0.0, |acc, x| acc + x.norm_sqr());
    (dv * norm).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn involution_1d() {}

    #[test]
    fn involution_3d() {}
}
