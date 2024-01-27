extern crate nalgebra;
extern crate rustfft;
use std::fmt;
use std::time::Instant;
use std::vec;

use rustfft::{num_complex::Complex, FftPlanner, Length};

fn main() {
    let mut vec = vec![
        Complex {
            re: 1.0f64,
            im: 0.0f64
        };
        50
    ];
    let mut vref = &mut vec;
    let start = Instant::now();
    let res = transform(vref);

    println!("The elapsed time is {:?}", start.elapsed());
    if false {
        for i in 0..49 {
            println!("{}", res[i]);
        }
    }
}

fn transform(vecref: &mut Vec<Complex<f64>>) -> &mut Vec<Complex<f64>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(50);
    let ifft = planner.plan_fft_inverse(50);
    for _ in 0..500000 {
        fft.process(vecref);
        ifft.process(vecref);
        for element in &mut *vecref {
            *element = *element / 50.0;
        }
    }
    return vecref;
}
