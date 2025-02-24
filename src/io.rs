use crate::types::{Dynamics1D, Dynamics3D, View3D};
use crate::types::{Wavefunction1D, Wavefunction3D};
use hdf5_metno;
use log::debug;
use log::info;
use ndarray::{Array1, Array2, Array3, Dim};
use std::fs::File;
use std::io::{self, Write};
use ndarray::Axis;

pub fn save_1d_wavefunction(
    wavefunction: &Wavefunction1D,
    filename: &str,
) -> hdf5_metno::Result<()> {
    let file = hdf5_metno::File::create(filename)?;
    let field_dataset = file
        .new_dataset::<f64>()
        .shape(wavefunction.field.len())
        .create("psi_squared")?;
    // we are not using ndarrays here
    let psi_squared = wavefunction
        .field
        .clone()
        .into_iter()
        .map(|x| x.re.powi(2) + x.im.powi(2))
        .collect::<Vec<f64>>();
    field_dataset.write(&psi_squared)?;
    debug!("l length: {}", wavefunction.l.len());
    let l_dataset = file
        .new_dataset::<f64>()
        .shape(wavefunction.l.len())
        .create("l")?;
    l_dataset.write(&wavefunction.l)?;
    info!("HDF5 file created as {:?}", &filename);
    Ok(())
}

pub fn save_3d_wavefunction(
    wavefunction: &Wavefunction3D,
    filename: &str,
) -> hdf5_metno::Result<()> {
    let psi_squared = wavefunction
        .field
        .iter()
        .map(|x: &ndrustfft::Complex<f64>| x.re.powi(2) + x.im.powi(2))
        .collect::<Vec<f64>>();
    let shape = wavefunction.field.dim(); // Shape should be a 3D tuple (x, y, z)
    // Reshape the flattened vector into a 3D array
    let psi_squared_reshaped =
        Array3::from_shape_vec(shape, psi_squared).expect("Shape mismatch when reshaping");

    let file = hdf5_metno::File::create(filename)?;
    let psi_dataset = file
        .new_dataset::<f64>()
        .shape(wavefunction.field.dim())
        .create("psi_squared")?;
    psi_dataset.write(&psi_squared_reshaped)?;
    let l_x_dataset = file
        .new_dataset::<f64>()
        .shape(wavefunction.l_x.len())
        .create("l_x")?;
    l_x_dataset.write(&wavefunction.l_x)?;
    let l_y_dataset = file
        .new_dataset::<f64>()
        .shape(wavefunction.l_y.len())
        .create("l_y")?;
    l_y_dataset.write(&wavefunction.l_y)?;
    let l_z_dataset = file
        .new_dataset::<f64>()
        .shape(wavefunction.l_z.len())
        .create("l_z")?;
    l_z_dataset.write(&wavefunction.l_z)?;
    info!("HDF5 file created as {:?}", &filename);
    Ok(())
}

pub fn save_1d_dynamics(dynamics: &Dynamics1D, filename: &str) -> hdf5_metno::Result<()> {
  let file = hdf5_metno::File::create(filename)?;

  let rows = dynamics.psi.len();        // Number of time steps
  let cols = dynamics.psi[0].field.len(); // Number of spatial points

  // Flatten psi_squared using row-major order (ensure correct shape)
  let mut psi_squared = Vec::with_capacity(rows * cols);
  for psi in &dynamics.psi {
      psi_squared.extend(psi.field.iter().map(|y| y.norm_sqr()));
  }

  // Save psi_squared with correct shape
  let psi_dataset = file
      .new_dataset::<f64>()
      .shape(rows*cols)
      .create("psi_squared")?;
  psi_dataset.write(&psi_squared)?;

  // Save l (1D array)
  let l_dataset = file
      .new_dataset::<f64>()
      .shape((dynamics.psi[0].l.len(),)) // Ensure tuple shape
      .create("l")?;
  l_dataset.write(&dynamics.psi[0].l)?;

  // Save t (1D array)
  let t_dataset = file
      .new_dataset::<f64>()
      .shape((dynamics.t.len(),)) // Ensure tuple shape
      .create("t")?;
  t_dataset.write(&dynamics.t)?;

  info!("HDF5 file created as {:?}", &filename);
  Ok(())
}

// pub fn save_3d_dynamics(dynamics: &Dynamics3D, filename: &str) -> hdf5_metno::Result<()> {
//   let file = hdf5_metno::File::create(filename)?;
//   // project the wavefunction integrating over the x and y directions
  
//   let yz_


// }

pub fn project_3d_pdf(wf: & Wavefunction3D, ax: usize) -> Array2<f64> {
  let dline;
  assert!(ax == 0 || ax ==1 || ax == 2);
  if ax == 0 {
    dline= wf.l_x[1]-wf.l_x[0];
  }  else if ax == 1 {
    dline= wf.l_y[1]-wf.l_y[0];
  } else {
    dline= wf.l_z[1]-wf.l_z[0];
  }
  let axis = Axis(ax);
  let projected = wf.field.mapv(|x| x.norm_sqr()).sum_axis(axis) * dline;
  projected
}