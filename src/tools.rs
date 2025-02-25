use crate::propagate::I;
use crate::types::{Wavefunction1D, Wavefunction3D};
use log::warn;
use ndarray::{Array1, Array3, Zip};
use rustfft::num_complex::Complex;
use std::f64::{self, consts::PI};

pub fn gaussian(a: f64, w: f64, x: &f64) -> Complex<f64> {
    // TODO relax the complex type and only return f64
    Complex::new(a * (-(x / w).powi(2) / 4.).exp(), 0.0)
}

/**
 * the width is chosen such that the probability density is a gaussian
 * with width w (sigma=w)
 */
pub fn gaussian_normalized(w: f64, x: &f64) -> Complex<f64> {
    gaussian(1.0 / ((2.0 * PI).sqrt() * w).sqrt(), w, x)
}

pub fn sech_normalized(g: f64, x: f64) -> f64 {
    assert!(g < 0.0, "The nonlinearity should be negative");
    (-g / 2.0).sqrt() * 2.0 / ((g * x).exp() + (-g * x).exp())
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

pub fn v0_axial_1d(
  l: &Vec<f64>, 
  l_harm_x: f64, 
  amplitude:f64, 
  lattice_constant: f64) -> Array1<ndrustfft::Complex<f64>> {
    assert!(amplitude >= 0.0);
    assert!(lattice_constant > 0.0);
    let mut v0 = Array1::<ndrustfft::Complex<f64>>::zeros(l.len());
    Zip::indexed(&mut v0).for_each(|i, x| {
        *x += 1.0 / 2.0 * (l[i] / l_harm_x).powi(2) - amplitude * (2.0 * PI * l[i] / lattice_constant).cos();
    });
    v0
}

pub fn v0_harmonic(
    l_x: &Vec<f64>,
    l_y: &Vec<f64>,
    l_z: &Vec<f64>,
    l_harm_x: f64,
) -> Array3<ndrustfft::Complex<f64>> {
    assert!(l_harm_x > 0.0);
    let mut v0 = Array3::<ndrustfft::Complex<f64>>::zeros((l_x.len(), l_y.len(), l_z.len()));
    Zip::indexed(&mut v0).for_each(|(i, j, k), x| {
        *x += 1.0 / 2.0 * ((l_x[i] / l_harm_x).powi(2) + (l_y[j]).powi(2) + (l_z[k]).powi(2))
    });
    v0
}

pub fn v0_optical_lattice(
    l_x: &Vec<f64>,
    l_y: &Vec<f64>,
    l_z: &Vec<f64>,
    amplitude: f64,
    lattice_constant: f64,
) -> Array3<ndrustfft::Complex<f64>> {
    assert!(amplitude >= 0.0);
    assert!(lattice_constant > 0.0);
    let mut v0 = Array3::<ndrustfft::Complex<f64>>::zeros((l_x.len(), l_y.len(), l_z.len()));
    Zip::indexed(&mut v0)
        .for_each(|(i, _j, _k), x| *x += -amplitude * (2.0 * PI * l_x[i] / lattice_constant).cos());
    v0
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
mod test {
    use super::*;

    #[test]
    fn normalization_of_basic_pulses() {
        let n_l = 100;
        let l = 30.0;
        let h_l: f64 = l / ((n_l - 1) as f64);
        let l_range = symmetric_range(l, h_l);
        assert!(l_range.len() == n_l, "The range is not correct");
        let psi_gaussian = Wavefunction1D {
            field: l_range
                .iter()
                .map(|x| gaussian_normalized(0.812345, x))
                .collect(),
            l: l_range.clone(),
        };
        let psi_sech = Wavefunction1D {
            field: l_range
                .iter()
                .map(|x| Complex::new(sech_normalized(-1.12345, *x), 0.0))
                .collect(),
            l: l_range.clone(),
        };
        let ns_gaussian = normalization_factor_1d(&psi_gaussian);
        let ns_sech = normalization_factor_1d(&psi_sech);
        print!("Gaussian: {}, Sech: {}", ns_gaussian, ns_sech);
        assert!(
            (ns_gaussian - 1.0).abs() < 1e-10,
            "Gaussian wavefunction is not normalized"
        );
        assert!(
            (ns_sech - 1.0).abs() < 1e-10,
            "Sech wavefunction is not normalized"
        );
    }
}

// pub fn simple_chemical_potential(psi: &Array1<Complex64>, sim: &Sim) -> f64 {
//     let mut mu = (0.5 / sim.Vol) * Zip::from(&sim.ksquared)
//         .and(psi)
//         .map_collect(|&k2, &psi| k2 * psi.norm_sqr())
//         .sum();

//     let tmp = xspace(psi, sim);

//     mu += sim.dV * Zip::from(&sim.V0)
//         .and(&tmp)
//         .map_collect(|&V0, &tmp| (V0 + sim.g * tmp.norm_sqr()) * tmp.norm_sqr())
//         .sum();

//     mu + 1.0 // add one transverse energy unit (1D-GPE case)
// }
