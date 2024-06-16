use rustfft::num_complex::Complex;
use std::f64::consts::PI;
use log::warn;

pub fn gaussian(a: f64, w: f64, t: &f64) -> Complex<f64> {
    Complex::new(a * (-(t / w).powf(2.) / 2.).exp(), 0.0)
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
    let dk = 2.0 * PI / x_vector[steps_l - 1];
    let k_n = 2.0 * PI / dx;
    let mut current = -k_n / 2.0;
    warn!("Need to check again the k_vector construction");
    for i in 0..steps_l {
        vec[i] = current;
        current += dk;
        // println!("{:3.2e}", vec[i]);
    }
    vec
}
