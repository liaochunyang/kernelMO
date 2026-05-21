# Framework 2

<s>One thing I need to fix: Conservation law experiment has 40 out-of-distribution parametric functions, while DiffReacAdv and Nonlinear_Klein_Gordon have 80 out-of-distribution parametric functions.</s>

## Data Description

For the training dataset and in-distribution test dataset, we follow the same setting as the MNO paper.

- Conservation Law: The components of $alpha$ are sampled from the ranges $\alpha_i \in [0.9\alpha_i^c,1.1\alpha_i^c]$ with the reference values given by $\alpha^c = [1,1,1,0.1]^\top$.
- Diffusion Reaction Advection: The first three components are sampled from the ranges $\alpha_i \in [0.9\alpha_i^c,1.1\alpha_i^c]$ with the reference values given by $\alpha^c = [0.01,1,1]^\top$, while $\apha_4,\alpha_5% are drawn uniformly from $[1,3]$.
- Nonlinear Klein Gordon: The components of $alpha$ are sampled from the ranges $\alpha_i \in [0.9\alpha_i^c,1.1\alpha_i^c]$ with the reference values given by $\alpha^c = [1,1,1]^\top$.
- In distribution Gaussian process setting: Gaussian process with variance 1, RBF kernel with scale parameter 1. 

## Out-of-distribution:

For conservation law, diffusion reaction advection equation, and nonlinear klein gordon equation, we have the following ood datasets:
- **ood_par.h5**: parameter range $\pm20\%$  (in-distribution test range $\pm10\%$). This is the same ood as the MNO paper.
- **ood_init_amp.h5**: change the amplitudes (uniform [0,1] -> uniform [-2,2]) and number of sine functions (2 -> 4)
- **ood_init_GP.h5**: initial conditions are generated from Gaussian process with variance 1 (RBF kernel with scale parameter 1)
- **ood_par_init_amp.h5**: ood parametric functions and ood initial conditions (amplitudes and number of sine functions)
- **ood_par_init_GP.h5**: ood parametric functions and ood initial conditions (GP)


For parametric diffusion reaction equation and parametric wave equations, we have the following ood datasets:
- **ood_par_var.h5**: ood parametric functions, RBF kernel, change variance 1 to 0.01
- **ood_par_scale.h5**: ood parametric functions, RBF kernel, change scale 1 to 0.1
- **ood_par_matern.h5**: ood parametric function, change RBF kernel to Matern kernel. Matern kernel: smoothness parameter 1/2 and scale 1. GP variance is still 1.
- **ood_init.h5**: change the amplitudes (uniform [0,1] -> uniform [-2,2]) and number of sine functions (2 -> 4). *We may consider GP initial condition as well, though it may not be necessary to consider the difficult case here.*
- **ood_par_init.h5**: ood parametric functions (scale) + ood initial conditions. *From function visualization, I realize that scale parameter changes the parameter functions the most, so I consider it here.*

