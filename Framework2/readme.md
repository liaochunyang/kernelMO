# Framework 2

<s>One thing I need to fix: Conservation law experiment has 40 out-of-distribution parametric functions, while DiffReacAdv and Nonlinear_Klein_Gordon have 80 out-of-distribution parametric functions.</s>

## Data Description

For the training dataset and in-distribution test dataset, we follow the same setting as the MNO paper.

## Out-of-distribution:

For conservation law, diffusion reaction advection equation, and nonlinear klein gordon equation, we have the following ood datasets:
- ood_par.h5: parameter range $\pm20\%$  (in-distribution test range $\pm10\%$). This is the same ood as the MNO paper.
- ood_init_amp.h5: change the amplitudes (uniform [0,1] -> uniform [-2,2]) and number of sine functions (2 -> 4)
