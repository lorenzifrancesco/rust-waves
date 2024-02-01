#[derive(Debug)]
pub enum Equation {
  Gpe1D,
  Npse,
  NpsePlus,
  Gpe3D
}

#[derive(Debug)]
pub struct Sim {
  pub nl: i64,
  pub l: f64,
  pub nt: i64,
  pub t: f64,
  pub g: f64,
  pub equation: Equation
}

pub struct N(u64, u64, u64);
pub struct L(f64, f64, f64);

// impl N {
//   fn mesh_size(&self) -> u64 {
//     N.0 * N.1 * N.2
//   }
// } 

// impl L {
//   fn dV(&self, N: &N) -> f64 {

//     (L.1 * L.2 * L.3)/(N.1 * N.2 * N.3)
//   }
// }
// enum Flags {
//   Collapse
// }

/// other ideas
pub enum Domain {
  ThreeDim(L, N),
  OneDim(f64, u64)
}

fn d_vol(d: &Domain) -> f64 {
  match d {
    Domain::ThreeDim(L(lx, ly, lz), N(nx, ny, nz)) => 1.0,
    Domain::OneDim(l, n) => 2.0
  }
}