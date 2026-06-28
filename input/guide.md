# Choosing the configuration

The structure of the script is the following:

(experiment_<>.toml | _default.toml) ---> (_params.toml) ---> Rust code
In _default.toml, all the data for numerical grid and 1D/3D choice are set. 
In experiment_<>.toml, all the other parameters, regarding physics, are set.

Typical numerical values which are good for the physical applications in the article are
- 1D: (n_l_x = 1024)
- 3D: (n_l_x = 512, n_l_y = n_l_z = 16) takes about 1s/ms in debug mode.

### Referee response of 8/07/2025
Simulation to fit the Fig.3: multisite solitons.

New scripts 
```
optimize_initial.py
optimize_final.py
```
they are made to find a good agreement with the experimental data, 
in particular, the case of the GS is adjusted to the case of the grey line, 
and the line of results of the dynamical evolution are compared to the red and blue lines.


Parameters utilized in the simulations sent to Elmar and Robbie:

```experiment_pre_quench_fig3c.toml
# this file is made for linking the simulation
# to the experimental simulation values
# unless stated, all units are in SI

omega_x =          19.07  # 2 pi * 25 Hz
a_s =               20 # in units of a_0
l_3 =                1 # beware of units: m^6 s^-1
m =                  2.21e-25

n_atoms =          2900
omega_perp =       157.08 # 2 pi * 25 Hz
d =                  2.6e-6
v_0 =                1.3 # Er (Recoil energy)
```

```experiment_fig3c.toml
# this file is made for linking the simulation
# to the experimental simulation values
# unless stated, all units are in SI

omega_x =            0.0  # 2 pi * 25 Hz
a_s =               -5.7 # in units of a_0 ---> change to +2 for the dispersing case
l_3 =               5e-39 # beware of units: m^6 s^-1
t_f =              253e-3 # s  ATTENTION! This is done for the panels c and d
m =                  2.21e-25

n_atoms =          2900
omega_perp =       157.08 # 2 pi * 25 Hz
d =                  2.6e-6
v_0 =                1.3 # Er (Recoil energy)
```

All the simulations are made with the 3D-GPE, after some fast experiments with the NPSE.

The numerical parameters are
```_default.toml
title = "default"

[numerics]
n_l      = 512
n_l_y    = 16
n_l_z    = 16
l        = 60.0
l_y      = 16.0
l_z      = 16.0
dt       = 0.005

[physics]
g         = -1.3 # -1.34/-1.35 -> NPSE collapse threshold
g5        = 0.0
l_harm_x  = 1e300
v0        = 0.0
dl        = 2.0
t         = 4.0
dimension = 3
npse      = true
im_t      = false

[initial]
w        = 1.0
w_y      = 1.0
w_z      = 1.0

[options]
n_saves  = 100
```

The simulations run in less than a minute. The time for the selection of the GS is set to $t_f=4 \omega_\perp^{-1}$, it seems sufficient when comparing with much longer NPSE simulations.