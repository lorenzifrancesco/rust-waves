extern crate nalgebra;
use rustfft::num_complex::Complex;
// use rustfft::{Fft, Length};
// use std::time::Instant;
// use std::vec;

// IO
use csv::Writer;
use std::fs;
use std::path::Path;
// use std::error::Error;

// use nalgebra::*;
pub mod io;

pub mod propagate;
pub mod tools;
pub mod types;
// use crate::io::io;
use crate::propagate::*;
use crate::tools::*;
use crate::types::*;

// Logging
use log::{debug, Level};
use simple_logger;

fn main() {
    simple_logger::init_with_level(Level::Debug).unwrap();

    // Define the path to your TOML file
    debug!("Parsing the input TOML file...");
    let input = Path::new("input/params.toml");
    let output = Path::new("results/output.csv");
    let contents = fs::read_to_string(input).expect("Failed to read the TOML file");
    let config: Params = toml::from_str(&contents).expect("Failed to load the config");

    let n_l = 100;
    let steps_l = n_l + 1;
    let sigma = 5.0;
    let l_max = 2.0 * sigma * config.initial.w;
    let h_l: f64 = (2.0 * l_max) / (n_l as f64);
    let l_range = symmetric_range(2.0 * l_max, h_l);
    // let k_range = k_vector(&l_range);

    let initial_wave: Vec<Complex<f64>> = l_range
        .iter()
        .map(|x| gaussian(config.initial.a, config.initial.w, &(x)))
        .collect();

    let psi = propagate(&l_range, initial_wave, config);

    // write the waveform to csv
    let mut writer = Writer::from_path(output).expect("Failed to create a CSV Writer");
    writer
        .write_record(&["Space", "Value"])
        .expect("Failed writing");
    for i in 1..steps_l {
        // let formatted_row = format!("{:<50e}, {:<100e}", l_range[i], initial_wave[i]);
        writer
            .write_record(&[l_range[i].to_string(), psi[i].to_string()])
            .expect("Failed to write in buffer");
    }
    writer.flush().expect("Failed to write the buffer");
}
