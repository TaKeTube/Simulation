# Rod in Gravitational field

<img src="https://github.com/TaKeTube/Simulation/blob/main/EulerLagrangeEquation/res.gif?raw=true" alt="res" style="zoom:80%;" />

### Introduction

In this mini project, we simulate the dynamics of a rod with uniform mass distribution in a gravitational field. We first derive the Euler-Lagrange in a general coordinate system, then use forward Euler method to solve the ODE. The simulation program is written using [taichi programming language](https://github.com/taichi-dev/taichi), and we use auto diff of taichi to compute the differential of the general potential energy because the analytical form is very complex.

---

See `Simulation.pdf` for detailed derivation and analysis.