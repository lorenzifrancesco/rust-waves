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
    let params: Params = toml::from_str(&contents).expect("Failed to load the config");

    let n_l = 300;
    let steps_l = n_l + 1;
    let sigma = 5.0;
    let l_max = 2.0 * sigma * params.initial.w;
    let h_l: f64 = (2.0 * l_max) / (n_l as f64);
    let l_range = symmetric_range(2.0 * l_max, h_l);
    // let k_range = k_vector(&l_range);

    let mut initial_wave: Vec<Complex<f64>> = l_range
        .iter()
        .map(|x| gaussian(params.initial.a, params.initial.w, &(x)))
        .collect();

    let saved_psi = propagate(&l_range, &mut initial_wave, &params);

    save_matrix_to_csv(output, &l_range, &saved_psi, steps_l, params.options.n_saves);

}

fn save_matrix_to_csv(output: &Path, l_range: &Vec<f64>, psi: &Vec<Vec<Complex<f64>>>, steps_l: usize, n_saves: usize) {
    // Create a CSV writer
    let mut writer = Writer::from_path(output).expect("Failed to create a CSV Writer");

    // Write the header
    let mut header = vec!["Space".to_string()];
    for i in 0..n_saves {
        header.push(format!("Value_{}", i + 1));
    }
    writer.write_record(&header).expect("Failed writing header");

    // Write the rows
    for i in 0..steps_l {
        let mut row = vec![l_range[i].to_string()];
        for j in 0..n_saves {
            row.push(psi[j][i].to_string());
        }
        writer.write_record(&row).expect("Failed to write row");
    }

    // Flush the writer
    writer.flush().expect("Failed to write the buffer");
} 
