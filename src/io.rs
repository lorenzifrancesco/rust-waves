use crate::types::{Wavefunction1D, Wavefunction3D};
use hdf5_metno;
use log::debug;
use log::info;
use ndarray::{Array1, Array3};
use std::fs::File;
use std::io::{self, Write};

pub fn save_1d_wavefunction(
    wavefunction: &Wavefunction1D,
    filename: &str,
) -> hdf5_metno::Result<()> {
    let file = hdf5_metno::File::create(filename)?;
    let field_dataset = file
        .new_dataset::<f64>()
        .shape(wavefunction.field.len())
        .create("field")?;
    // we are not using ndarrays here
    let psi_squared = wavefunction
        .field
        .clone()
        .into_iter()
        .map(|x| x.re.powi(2) + x.im.powi(2))
        .collect::<Vec<f64>>();
    field_dataset.write(&psi_squared)?;
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
        .create("field")?;
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

// fn save_1d(array: Array1<f64>, filename: &String) -> hdf5_metno::Result<()> {
//     let file = hdf5_metno::File::create(filename)?;
//     let dataset = file.new_dataset::<f64>().shape(array.dim()).create("psi")?;
//     dataset.write(&array)?;
//     info!("HDF5 file created as {:?}", &filename);
//     Ok(())
// }

// fn save_3d(array: Array3<f64>, filename: &String) -> hdf5_metno::Result<()> {
//     let file = hdf5_metno::File::create(filename)?;
//     let dataset = file.new_dataset::<f64>().shape(array.dim()).create("psi")?;
//     dataset.write(&array)?;
//     info!("HDF5 file created as {:?}", &filename);
//     Ok(())
// }
