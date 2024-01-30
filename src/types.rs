pub enum Equation {
  Gpe1D,
  Npse,
  NpsePlus,
  Gpe3D
}

pub struct Sim {
  pub nl: i64,
  pub l: f64,
  pub nt: i64,
  pub t: f64,
  pub g: f64,
  pub equation: Equation
}

// enum Flags {
//   Collapse
// }