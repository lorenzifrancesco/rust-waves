use rust_waves::io::save_1d_wavefunction;
use rust_waves::propagate::propagate_1d;
use rust_waves::tools::{
    gaussian_normalized, normalization_factor_1d,
    sech_normalized, symmetric_range,
};
use rust_waves::types::{Params, Wavefunction1D};
use std::fs;
use std::path::Path;
use toml;

#[test]
pub fn basic_1d_soliton() {
    // TODO put standard params here
    let input = Path::new("input/_params.toml");
    let contents = fs::read_to_string(input).expect("Failed to read the TOML file");
    let mut params: Params = toml::from_str(&contents).expect("Failed to load the config");
    params.physics.t = 200.0;
    params.physics.g = -2.0 * 0.65;
    let n_l = params.numerics.n_l;
    let l = params.numerics.l;
    assert!(l > 2.0 * params.initial.w, "The domain is too small");
    let h_l: f64 = (l) / ((n_l - 1) as f64);
    let l_range = symmetric_range(l, h_l);
    assert!(l_range.len() == n_l, "The range is not correct");
    let mut initial_wave = Wavefunction1D {
        field: l_range
            .iter()
            .map(|x| gaussian_normalized(params.initial.w, &(x)))
            .collect(),
        l: l_range.clone(),
    };
    assert!(
        params.physics.g < 0.0,
        "Nonlinearity needs to be attractive"
    );
    let saved_psi = propagate_1d(&mut initial_wave, &params, true);
    let target_psi = Wavefunction1D {
        field: l_range
            .iter()
            .map(|x| ndrustfft::Complex::new(sech_normalized(params.physics.g / 2.0, *x), 0.0))
            .collect(),
        l: l_range.clone(),
    };
    let n_snaps = saved_psi.psi.len();
    assert!(
        normalization_factor_1d(&saved_psi.psi[n_snaps - 1]) - 1.0 < 1e-10,
        "The wavefunction is not normalized"
    );
    assert!(
        normalization_factor_1d(&target_psi) - 1.0 < 1e-10,
        "The target wavefunction is not normalized"
    );
    // the two solutions should have a L2 distance in the density space of 1e-6
    let l2_distance = saved_psi.psi[n_snaps - 1]
        .field
        .iter()
        .zip(target_psi.field.iter())
        .map(|(x, y)| (x.norm_sqr() - y.norm_sqr()).powi(2))
        .sum::<f64>()
        .sqrt()
        / (l_range.len() as f64).sqrt()
        * params.numerics.l;
    // print the abs2 of the two fields
    // for (x, y) in saved_psi.psi[n_snaps - 1]
    //     .field
    //     .iter()
    //     .zip(target_psi.field.iter())
    // {
    //     print!("{:>15.5e} {:>15.5e}\n", x.norm_sqr(), y.norm_sqr());
    // }
    print!("L2 distance = {:>10.5e}\n", l2_distance);

    save_1d_wavefunction(&target_psi, "results/1d_psi_0.h5")
        .expect("Failed to save the wavefunction");
    save_1d_wavefunction(
        &saved_psi.psi[saved_psi.psi.len() - 1],
        "results/1d_psi_end.h5",
    )
    .expect("Failed to save the wavefunction");

    // Do the same, but with the distance induced by the sup norm:
    let lsup_distance = saved_psi.psi[n_snaps - 1]
        .field
        .iter()
        .zip(target_psi.field.iter())
        .map(|(x, y)| (x.norm_sqr() - y.norm_sqr()).abs())
        .fold(0.0, |acc, x| f64::max(acc, x));
    print!("Lsup distance = {:>10.5e}\n", lsup_distance);
    assert!(lsup_distance < 1e-4);
    assert!(l2_distance < 1e-4);
}

pub fn can_you_even_npse_collapse() {
    // TODO put standard params here
    let input = Path::new("input/_params.toml");
    let contents = fs::read_to_string(input).expect("Failed to read the TOML file");
    let mut params: Params = toml::from_str(&contents).expect("Failed to load the config");
    params.physics.t = 200.0;
    params.physics.g = -2.0 * 0.65;
    let n_l = params.numerics.n_l;
    let l = params.numerics.l;
    assert!(l > 2.0 * params.initial.w, "The domain is too small");
    let h_l: f64 = (l) / ((n_l - 1) as f64);
    let l_range = symmetric_range(l, h_l);
    assert!(l_range.len() == n_l, "The range is not correct");
    let mut initial_wave = Wavefunction1D {
        field: l_range
            .iter()
            .map(|x| gaussian_normalized(params.initial.w, &(x)))
            .collect(),
        l: l_range.clone(),
    };
    assert!(
        params.physics.g < 0.0,
        "Nonlinearity needs to be attractive"
    );
    let saved_psi = propagate_1d(&mut initial_wave, &params, true);
    let target_psi = Wavefunction1D {
        field: l_range
            .iter()
            .map(|x| ndrustfft::Complex::new(sech_normalized(params.physics.g / 2.0, *x), 0.0))
            .collect(),
        l: l_range.clone(),
    };
    let n_snaps = saved_psi.psi.len();
    assert!(
        normalization_factor_1d(&saved_psi.psi[n_snaps - 1]) - 1.0 < 1e-10,
        "The wavefunction is not normalized"
    );
    assert!(
        normalization_factor_1d(&target_psi) - 1.0 < 1e-10,
        "The target wavefunction is not normalized"
    );
    // the two solutions should have a L2 distance in the density space of 1e-6
    let l2_distance = saved_psi.psi[n_snaps - 1]
        .field
        .iter()
        .zip(target_psi.field.iter())
        .map(|(x, y)| (x.norm_sqr() - y.norm_sqr()).powi(2))
        .sum::<f64>()
        .sqrt()
        / (l_range.len() as f64).sqrt()
        * params.numerics.l;
    // print the abs2 of the two fields
    // for (x, y) in saved_psi.psi[n_snaps - 1]
    //     .field
    //     .iter()
    //     .zip(target_psi.field.iter())
    // {
    //     print!("{:>15.5e} {:>15.5e}\n", x.norm_sqr(), y.norm_sqr());
    // }
    print!("L2 distance = {:>10.5e}\n", l2_distance);

    save_1d_wavefunction(&target_psi, "results/1d_psi_0.h5")
        .expect("Failed to save the wavefunction");
    save_1d_wavefunction(
        &saved_psi.psi[saved_psi.psi.len() - 1],
        "results/1d_psi_end.h5",
    )
    .expect("Failed to save the wavefunction");

    // Do the same, but with the distance induced by the sup norm:
    let lsup_distance = saved_psi.psi[n_snaps - 1]
        .field
        .iter()
        .zip(target_psi.field.iter())
        .map(|(x, y)| (x.norm_sqr() - y.norm_sqr()).abs())
        .fold(0.0, |acc, x| f64::max(acc, x));
    print!("Lsup distance = {:>10.5e}\n", lsup_distance);
    assert!(lsup_distance < 1e-4);
    assert!(l2_distance < 1e-4);
}
