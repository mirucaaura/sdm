import numpy as np
import matplotlib.pyplot as plt
from steepest_descent import SteepestDescent

sd = SteepestDescent(
    ndim=2,
    nu=0.8,
    sigma=0.8,
    eps=1e-5,
)

np.random.seed(1)

nmax = 100
x_opt = np.zeros((nmax, 2))

for i in range(nmax):
    x_init = np.random.rand(2)
    x = sd.steepest(x_init)
    x_opt[i] = sd.F(x)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_opt[:, 0], x_opt[:, 1], c="r", alpha=0.7)
ax.set_xlabel("$f(x_1)$")
ax.set_ylabel("$f(x_2)$")
plt.show()
