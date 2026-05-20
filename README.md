# kernelMO
Kernel Multiple Operator Learning

# Errors

## Framework 1

#### In-distribution Errors:

Model | Conservation Law | Diffusion Reaction Advection | Nonlinear Klein Gordon | Parametric Diffusion Reaction | Parametric Wave
| :---        |    :----:   |   :----:   |  :----:   | :----:   |        ---: |
| Vanilla Kernel |   3.72%   | 12%   | 28.6%   | 6.78%    | 52.7% |
| Framework 1   | 0.0145% | 0.384%   | 0.209%  | 0.049%   | 2.78% |
| Framework 2   | 0.0145% | 0.787%   | 0.209%  | 0.0645%  | 2.78% |


#### Out-of-distribution
Model | Conservation Law | Diffusion Reaction Advection | Nonlinear Klein Gordon
| :---        |    :----:   |   :----:   |  ----:   |
| Vanilla Kernel |   8.48%   | 20.4%   | 45.2%   |
| Framework 1   | 0.772% | 3.72%   | 6.72%  |
 Framework 2   | 0.772% | 5.33%   | 6.72%  |

 ## Framework 2

#### In-distribution Errors:

Model | Conservation Law | Diffusion Reaction Advection | Nonlinear Klein Gordon | Parametric Diffusion Reaction | Parametric Wave
| :---        |    :----:   |   :----:   |  :----:   | :----:   |        ---: |
| Vanilla Kernel |   4.76%   | 12.6%   | 28.9%   | 6.89%    | 58.0% |
| Framework 2   | 1.76% | 2.55%   | 0.314%  | 2.88%   | 4.88% |


#### Out-of-distribution
Model | Conservation Law | Diffusion Reaction Advection | Nonlinear Klein Gordon
| :---        |    :----:   |   :----:   |  ----:   |
| Vanilla Kernel |   7.43%   | 15.8%   | 43.0%   |
| Framework 2   | 6.13% | 4.57%   | 8.36%  |


 
## To do list:

1. <s>Framework 1: out-of-distribution for Parametric Diffusion Reaction Equation and Parametric Wave Equation, I do not know which ood we should take. The coefficient functions are now sampled from Gaussian Process.</s>

2. <s>Framework 2: out-of-distribution, same as item 1.</s>

3. <s>Framework 2: out-of-distribution initial conditions.</s>

4. <s>Framework 2: ood $\alpha$ and ood $u_0$.</s>

5. <s>Try Gaussian kernel and Matern kernel for all models, and tune the hyper-parameters.</s>

<s>We may need to rewirte the functions to generate ood datatsets.</s>
