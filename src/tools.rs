use log::warn;
use rustfft::num_complex::Complex;
use std::f64::consts::PI;

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
    // let k_n = 2.0 * PI / (2.0 * dx);
    // N * dx
    let dk = 2.0 * PI / (dx * ((steps_l as f64)));
    let mut current = 0.0;
    for i in 0..steps_l {
        vec[i] = current;
        current += dk;
    }
    vec
}

pub fn k_squared(k_vector: &Vec<f64>) -> Vec<f64> {
    let n = k_vector.len() ;
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