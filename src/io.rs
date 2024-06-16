use std::fs::File;
use std::io::{self, Write};

pub fn io(mat: &Vec<Vec<f64>>) -> io::Result<()> {
    let mut file = File::create("matrix.csv")?;
    for row in mat {
        for (i, val) in row.iter().enumerate() {
            if i != 0 {
                write!(file, ",")?;
            }
            write!(file, "{}", val)?;
        }
        writeln!(file)?;
    }
    Ok(())
}
