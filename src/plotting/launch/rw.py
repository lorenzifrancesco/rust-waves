import toml
from dataclasses import dataclass
from os import path
from scipy.constants import pi, hbar
from scipy.constants import physical_constants
import numpy as np

@dataclass
class Params:

    title: str
    # numerics
    n_l: int
    n_l_y: int
    n_l_z: int
    l: float
    l_y: float
    l_z: float
    dt: float

    # physics
    g: float
    g5: float
    l_harm_x: float
    v0: float
    dl: float
    t: float
    npse: bool
    im_t: bool

    # initial
    w: float
    w_y: float
    w_z: float

    # options
    n_saves: int

    @classmethod
    def read(cls, filepath: str) -> "Params":
        data = toml.load(filepath)
        return cls(
            title=data["title"],
            **data["numerics"],
            **data["physics"],
            **data["initial"],
            **data["options"]
        )

    def write(self, filepath: str):
        data = {
            "title": self.title,
            "numerics": {
                "n_l": self.n_l,
                "n_l_y": self.n_l_y,
                "n_l_z": self.n_l_z,
                "l": self.l,
                "l_y": self.l_y,
                "l_z": self.l_z,
                "dt": self.dt,
            },
            "physics": {
                "g":  self.g,
                "g5": self.g5,
                "l_harm_x": self.l_harm_x,
                "v0": self.v0,
                "dl": self.dl,
                "t": self.t,
                "npse": self.npse,
                "im_t": self.im_t,
            },
            "initial": {
                "w": self.w,
                "w_y": self.w_y,
                "w_z": self.w_z,
            },
            "options": {
                "n_saves": self.n_saves,
            }
        }
        with open(filepath, "w") as f:
            toml.dump(data, f)


def write_from_experiment(
  input_filename="input/experiment.toml", 
  output_filename="input/params.toml", 
  title="exp", 
  a_s=None, 
  load_gs=False):
    ex = toml.load(input_filename)
    
    # scales
    a0 = physical_constants["Bohr radius"][0]
    l_perp = np.sqrt(hbar/(ex["omega_perp"]*ex["m"]))
    e_perp = hbar * ex["omega_perp"]
    t_perp = ex["omega_perp"]**(-1) * 2 * pi 
    e_recoil = (pi * hbar / ex["d"])**2 / ex["m"]
    
    # normalized
    
    if ex["omega_x"] != 0:
      l_x = np.sqrt(hbar/(ex["omega_x"]*ex["m"])) / l_perp
    else:
      l_x = 1e300
    if a_s == None:
      a_s = ex["a_s"] 
    g = 2 * a0 * a_s * (ex["n_atoms"]-1) / l_perp
    t = ex["t_f"]/t_perp
    g5 = ex["l_3"] / l_perp**6 * t_perp * ex["n_atoms"]**3 / (6*pi**2)
    v_0 = ex["v_0"] * e_recoil / e_perp
    p = Params.read("input/default.toml")
    assert(p.title == "default")
    p.title=title
    p.g  = float(g)
    p.g5 = float(g5)
    p.l_harm_x=float(l_x)
    p.v0=float(v_0)
    p.t =float(t)
    if load_gs:
      p.im_t = False
      p.w = -1.0
    else: 
      p.im_t = True
      p.t = 50.0
    p.write(output_filename)
    return

if __name__ == "__main__":
    p = Params.read("input/params.toml")
    print(p)
    p.write("input/params2.toml")
    p2 = Params.read("input/params2.toml")
    print(p2)
    assert p == p2
    write_from_experiment()
    print("All tests passed.")