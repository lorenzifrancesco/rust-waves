use serde_derive::Deserialize;
use rustfft::num_complex::Complex;
use toml::value::Array;
use ndarray::Array3;

#[derive(Deserialize)]
pub struct Params {
    pub title: String,
    pub numerics: Numerics,
    pub physics: Physics,
    pub initial: Gaussian,
    pub options: Options,
}

#[derive(Deserialize)]
pub struct Numerics {
    pub n_l: usize,
    pub n_l_y: usize,
    pub n_l_z: usize,
    pub l:  f64,
    pub l_y:  f64,
    pub l_z:  f64,
    pub dt: f64,
}

#[derive(Deserialize)]  
pub struct Initial {
    pub a: f64,
    pub w: f64,
}

#[derive(Deserialize)]
pub struct Physics {
    pub g: f64,
    pub t: f64,
    pub npse: bool,
    pub im_t: bool,
}

#[derive(Deserialize)]
pub struct Gaussian {
    pub w: f64,
    pub w_y: f64,
    pub w_z: f64,
    pub a: f64,
}

#[derive(Deserialize)]
pub struct Options {
    pub n_saves: usize,
}


// Wavefunction types
#[derive(Clone)]
pub struct Wavefunction1D {
    pub field: Vec<Complex<f64>>,
    pub l: Vec<f64>, // to have each field with its own space vector is handy for the detection of collapse
}

impl Wavefunction1D {
    pub fn new(field: Vec<Complex<f64>>, l: Vec<f64>) -> Wavefunction1D {
        Wavefunction1D {
            field,
            l,
        }
    }
}

pub struct Dynamics1D {
    pub psi: Vec<Wavefunction1D>,
    pub t: Vec<f64>,
}

#[derive(Clone)]
pub struct Wavefunction3D {
    pub field: Array3<Complex<f64>>,
    pub l_x: Vec<f64>,
    pub l_y: Vec<f64>,
    pub l_z: Vec<f64>,
}

impl Wavefunction3D {
    pub fn new(field: Array3<Complex<f64>>, l_x: Vec<f64>, l_y: Vec<f64>, l_z: Vec<f64>) -> Wavefunction3D {
        Wavefunction3D {
            field,
            l_x,
            l_y,
            l_z,
        }
    }
}
pub struct Dynamics3D {
    pub psi: Vec<Wavefunction3D>,
    pub t: Vec<f64>,
}