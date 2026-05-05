# kernelMO
Kernel Multiple Operator Learning

# Errors

#### Framework 1

Model | Conservation Law | Diffusion Reaction Advection | Nonlinear Klein Gordon | Parametric Diffusion Reaction | Parametric Wave
| :---        |    :----:   |   :----:   |  :----:   | :----:   |        ---: |
| Vanilla Kernel |   3.72%   | 12%   | 28.6%   | 6.78%    | 52.7% |
| Framework 1   | 0.0145% | 0.384%   | 0.209%  | 0.049%   | 2.78% |
| Framework 2   | 0.0145% | 0.787%   | 0.209%  | 0.0645%  | 2.78% |


Out-of-distribution
Model | Conservation Law | Diffusion Reaction Advection | Nonlinear Klein Gordon
| :---        |    :----:   |   :----:   |  ----:   |
| Vanilla Kernel |   8.48%   | 20.4%   | 45.2%   |
| Framework 1   | 0.772% | 3.72%   | 6.72%  |
 Framework 2   | 0.772% | 5.33%   | 6.72%  |
 
#### To do list:

1. Framework 1: out-of-distribution for Parametric Diffusion Reaction Equation and Parametric Wave Equation, I do not know which ood we should take. The coefficient functions are now sampled from Gaussian Process.

2. Framework 2: out-of-distribution, same as item 1.

3. Framework 2: out-of-distribution initial conditions.

4. Framework 2: ood $\alpha$ and ood $u_0$.

We may need to rewirte the functions to generate ood datatsets.
