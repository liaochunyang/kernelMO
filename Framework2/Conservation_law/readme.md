# Conservation Law

data_generation.ipynb: generate conservation law dataset.

main.ipynb: main file, the implementation of vanilla kernel and our proposed method.

In-distribution experiments: we set parameter vector $[\alpha_1, \alpha_2, \alpha_3, \alpha_4]$ sampled from the ranges $\alpha_i\in[0.9\alpha_i^c,1.1\alpha_i^c]$ with the reference values given by $\alpha^c = [1,1,1,0.1]$.

Out-of-distribution experiments: we set $\alpha_i\in[0.8\alpha_i^c,1.2\alpha_i^c]$.

Each parametric function has 50 initial conditions. There are 200 parametric functions for training, 80 parametric functions for in-distribution testing, and 40 parametric functions for out-of-distribution testing. 


