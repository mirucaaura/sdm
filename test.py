import numpy as np
from steepest_descent import SteepestDescent

sd = SteepestDescent(
    2,
    0.8,
    0.8,
    1e-5,
)

x_init = np.array([2, 2])
x = sd.steepest(x_init)
print(sd.F(x))