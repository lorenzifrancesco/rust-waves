use serde_derive::Deserialize;
#[derive(Deserialize)]
pub struct Params {
    pub title: String,
    pub physics: Phys,
    pub initial: Gaussian,
    pub options: Options,
}
#[derive(Deserialize)]
pub struct Phys {
    pub g: f64,
    pub t: f64,
}

#[derive(Deserialize)]
pub struct Gaussian {
    pub w: f64,
    pub a: f64,
}

#[derive(Deserialize)]
pub struct Options {
    pub n_saves: i32,
}
