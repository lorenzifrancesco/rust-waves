import toml
from dataclasses import dataclass
from os import path

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
                "g": self.g,
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


def write_from_experiment(filename = "input/experiment.toml", title = "exp"):
    p = Params(
        title="Experiment",
        n_l=256,
        n_l_y=256,
        n_l_z=256,
        l=1.0,
        l_y=1.0,
        l_z=1.0,
        dt=0.01,
        g=1.0,
        g5=1.0,
        l_harm_x=1.0,
        v0=1.0,
        dl=0.1,
        t=0.1,
        npse=True,
        im_t=True,
        w=1.0,
        w_y=1.0,
        w_z=1.0,
        n_saves=100
    )
    p.write(filename)
    return p
  

if __name__=="__main__":
    p = Params.read("input/params.toml")
    print(p)
    p.write("input/params2.toml")
    p2 = Params.read("input/params2.toml")
    print(p2)
    # assert p == p2
    print("All tests passed.")