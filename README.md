# Simple Nonlinear Schrödinger Equation solver in Rust
## Objectives of the package
- Learn Rust project management 
- Learn the actual language
- Familiarize with the numerics packages

## The physical model
In normalized units, the equation we want to solve is 
```math
u_x = -iu_{tt}+\gamma|u|^2u 
```
where we used the notation similar to nonlinear fiber optics: $x$ as space and $t$ as time variables.

## Method
We use Split Step Fourier Mehtod (SSFM) with Strang splitting.
<!-- ## Comments
We expect to have an improved reliability development experience with respect to Julia,
and a slightly slower development due to absence of dynamic types.
GPU support will be also more difficult
Performance is critical and we should  

In the case of a cylindrically symmetric potential, one can use a field transformation to keep the radial operator in the same shape in the 1D method, but changing the potential and the nonlinear term. The result is a 2D equation where the laplacian is cartesian. -->