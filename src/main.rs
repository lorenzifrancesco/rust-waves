extern crate nalgebra;
// use std::fmt;
use rustfft::{num_complex::Complex, FftPlanner};
use std::time::Instant;
use std::vec;
// use nalgebra::*;
pub mod propagator;
pub mod types;
use crate::propagator::propagate;
use crate::types::Equation;
use crate::types::Sim;

fn main() {
    // let mut vec = vec![
    //     Complex {
    //         re: 1.0f64,
    //         im: 0.0f64
    //     };
    //     50
    // ];
    // let mut vref = &mut vec;
    let start = Instant::now();
    let s = Sim {
        nl: 10,
        l: 40.0,
        nt: 5000,
        t: 10.0,
        g: 0.65,
        equation: Equation::Gpe1D,
    };
    println!("{}", format!("{:10.5}", s.nl));
    // let res = propagate(vref, 10.0);
    // println!("The elapsed time is {:?}", start.elapsed());
    // if false {
    //     for i in 0..49 {
    //         println!("{}", res[i]);
    //     }
    // }
    // println!("Ciao {:6.3}", 0.44);
}
