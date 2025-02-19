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
## Comments
We expect to have an improved reliability development experience with respect to Julia,
and a slightly slower development due to absence of dynamic types.
GPU support will be also more difficult
Performance is critical and we should  

### TODO
  -[ ] check the normalization of the 3d wavefunction and the transforms. Check the range of the k_vectors and their folding 
  -[ ] check the signs of the propagators and their constants. (we could eventually use an adaptive time stepping by comparing two different operator splitting methods)
  -[ ] check that we are getting physically meaningful 1D solitons, with the right shape.
  -[ ] implement the NPSE (it should be very easy) 