# 

We compare Framework 1 $G:W\to\{ G_W: U\to V\}$, and Framework 2 $G:W\times U\to V$ here. We only consider the generation of parametric function. 

## Data Description

#### Training data and in-distribution test

We fix 20 initial conditions and consider 400 parametric functions (80% for training and 20% for testing). For all PDEs, we follow the parameter settings in the MNO ppaer.



#### Out-of-distribution

For conservation law, diffusion reaction advection, and nonlinear Klein Gordon equation, we follow the ood setting in the MNO paper. 

For parametric diffusion reaction equation and parametric wave equation, the parametric functions are generated from Gaussian process. We then consider three ood datasets, which are
- ood_var.h5: Gaussian process with RBF kernel and variance = 0.01, other hyperparameters do not change.
- ood_scale.h5: Gaussian process with RBF kernel and scale = 0.1, other hyperparameters do not change.
- ood_matern.h5: Gaussian with Matern kernel with smoothness parameter 1/2 and scale parameter 1.

  






