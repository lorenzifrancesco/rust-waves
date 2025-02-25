use serde_derive::Deserialize;
use rustfft::num_complex::Complex;
use toml::value::Array;
use ndarray::{Array3, Array2};
use crate::io::project_3d_pdf;
use hdf5_metno::File;

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
    pub l_harm_x: f64,
    pub v0: f64,
    pub dl: f64,
    pub t: f64,
    pub npse: bool,
    pub im_t: bool,
}

#[derive(Deserialize)]
pub struct Gaussian {
    pub w: f64,
    pub w_y: f64,
    pub w_z: f64,
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
#[derive(Clone)]
pub struct Projections3D {
  pub xz: Array2<f64>, 
  pub yz: Array2<f64>,
}

impl Projections3D {
  pub fn new(wf:& Wavefunction3D) -> Projections3D {
    Projections3D {
      xz: project_3d_pdf(&wf, 1),
      yz: project_3d_pdf(&wf, 0),
    }
  }
}

pub struct View3D {
  pub movie: Vec<Projections3D>,
  pub l_x: Vec<f64>,
  pub l_y: Vec<f64>,
  pub l_z: Vec<f64>,
  pub t: Vec<f64>,
}

impl View3D {
    pub fn save_to_hdf5(&self, filename: &str) -> hdf5_metno::Result<()> {
        let file = File::create(filename)?;

        // Save 1D arrays
        file.new_dataset::<f64>().shape(self.l_x.len()).create("l_x")?.write(&self.l_x)?;
        file.new_dataset::<f64>().shape(self.l_y.len()).create("l_y")?.write(&self.l_y)?;
        file.new_dataset::<f64>().shape(self.l_z.len()).create("l_z")?.write(&self.l_z)?;
        file.new_dataset::<f64>().shape(self.t.len()).create("t")?.write(&self.t)?;

        // Save 3D projections as groups
        let movie_group = file.create_group("movie")?;
        for (i, proj) in self.movie.iter().enumerate() {
            let proj_group = movie_group.create_group(&format!("frame_{}", i))?;
            proj_group.new_dataset::<f64>().shape(proj.xz.dim()).create("xz")?.write(&proj.xz)?;
            proj_group.new_dataset::<f64>().shape(proj.yz.dim()).create("yz")?.write(&proj.yz)?;
        }
        Ok(())
    }
}