---
layout: page
title: Testing ADER-DG
lesson_number: 20
thumbnail: /assets/images/waveEquation.webp
extract: Validating the ADER-DG timestepping scheme
categories: [hyperbolic, numerical]
---

This page provides test cases for validating the **ADER-DG** (Arbitrary high-order DERivatives Discontinuous Galerkin) timestepping scheme implementation in VisualPDE.

## What is ADER-DG?

ADER-DG is a high-order accurate numerical method particularly well-suited for hyperbolic PDEs like the advection and wave equations. It combines:
- **Discontinuous Galerkin (DG)** spatial discretization, which allows for high-order accuracy and handles discontinuities well
- **ADER** time integration via element-local Picard predictor + single-step corrector with Gauss-Legendre temporal quadrature

VisualPDE implements true ADER-DG with orders 1-4:
- Order 1: Forward Euler — $O(\Delta t)$
- Order 2: ADER midpoint predictor-corrector — $O(\Delta t^2)$, element-local predictor + full-flux corrector
- Order 3: ADER with 2-pt Gauss-Legendre, 2 Picard iterations — $O(\Delta t^3)$
- Order 4: ADER with 2-pt Gauss-Legendre, 3 Picard iterations — $O(\Delta t^4)$

The key innovation over RK-DG: predictor stages use element-local spatial derivatives only (no inter-element numerical flux). Only the final corrector step uses the full DG operator with inter-element communication.

## Test 1: Linear Advection (1D)

The linear advection equation is the simplest hyperbolic PDE:

$$\pd{u}{t} + c \pd{u}{x} = 0$$

where $c$ is the wave speed. An initial profile should translate to the right (for $c > 0$) without changing shape.

### Test 1a: Smooth Gaussian Profile

This test uses a smooth Gaussian initial condition that should advect without distortion:

* Load the [ADER-DG advection test (Order 2)](/sim/?preset=aderDGAdvection1D_O2)

The Gaussian should translate smoothly to the right and wrap around due to periodic boundaries. Try:
- Comparing with Euler scheme (more diffusive)
- Increasing to Order 3 or 4 for higher accuracy

### Test 1b: Sharp Profile (Discontinuity)

A more challenging test uses a square wave initial condition:

* Load the [ADER-DG square wave test](/sim/?preset=aderDGSquareWave1D)

DG methods handle discontinuities better than standard finite differences. Watch for:
- Minimal spreading of the sharp edges
- No Gibbs oscillations (ringing near discontinuities)

## Test 2: Wave Equation (1D)

The wave equation tests ADER-DG on a second-order hyperbolic system:

$$\pdd{u}{t} = D \nabla^2 u$$

We reformulate this as a first-order system with $u$ and $v = \partial u/\partial t$.

* Load the [ADER-DG wave equation test](/sim/?preset=aderDGWave1D)

The wave should split into left and right traveling components. With periodic boundaries, they recombine periodically.

## Test 3: Convergence Study

To verify the order of accuracy, we can compare different DG orders on the same problem:

| Order | Method | Expected Temporal Accuracy |
|-------|--------|---------------------------|
| 1 | Forward Euler | $O(\Delta t)$ |
| 2 | SSPRK2 | $O(\Delta t^2)$ |
| 3 | SSPRK3 | $O(\Delta t^3)$ |
| 4 | Classical RK4 | $O(\Delta t^4)$ |

Try these different orders on the advection test:
- [Order 1 (Euler)](/sim/?preset=aderDGAdvection1D_O1)
- [Order 2 (SSP-RK2)](/sim/?preset=aderDGAdvection1D_O2)
- [Order 3 (SSP-RK3)](/sim/?preset=aderDGAdvection1D_O3)
- [Order 4 (RK4)](/sim/?preset=aderDGAdvection1D_O4)

Higher orders should maintain the Gaussian shape better over long times.

## Test 4: CFL Stability

The ADER-DG scheme has a CFL condition that depends on the order:

$$\Delta t \leq \frac{C \cdot \Delta x}{c \cdot (2p+1)}$$

where $p$ is the polynomial order. Higher orders require smaller timesteps for stability.

* Load the [CFL stability test](/sim/?preset=aderDGCFLTest)

Try increasing the timestep (`dt`) under <span class='click_sequence'>{{ layout.settings }} → **Timestepping**</span> to observe instability.

## Numerical Notes

The ADER-DG implementation in VisualPDE:

1. Uses Gauss-Lobatto nodes for the DG spatial discretization
2. Applies limiter functions to prevent spurious oscillations near discontinuities
3. Handles periodic boundary conditions naturally through element connectivity

For best results with hyperbolic problems:
- Use ADER-DG with order ≥ 2
- Keep timesteps within the CFL limit
- Use the SSP (Strong Stability Preserving) variants for problems with discontinuities

## Comparison with Other Schemes

You can compare ADER-DG against other timestepping schemes by changing the scheme in:
<span class='click_sequence'>{{ layout.settings }} → **Timestepping** → **Scheme**</span>

For the advection equation:
- **Euler**: Most diffusive, smooths out sharp features
- **Midpoint**: Better than Euler, still some diffusion
- **RK4**: Very accurate for smooth solutions
- **ADER-DG**: Best for hyperbolic problems, especially with discontinuities
