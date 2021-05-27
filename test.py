import numpy as np
from steepest_descent import SteepestDescent

sd = SteepestDescent(
    ndim=2,
    nu=0.8,
    sigma=0.8,
    eps=1e-5,
)

x_init = np.array([2, 2])
x = sd.steepest(x_init)
print(sd.F(x))