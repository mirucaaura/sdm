import numpy as np
import matplotlib.pyplot as plt
from steepest_descent import SteepestDescent
from obj_func import Obj

sd = SteepestDescent(
    ndim=2,
    nu=0.8,
    sigma=0.8,
    eps=1e-5,
)

obj = Obj()

np.random.seed(10)

nmax = 100 # the number of computing the Pareto optimal
x_opt = np.zeros((nmax, 2))
f_opt = np.zeros((nmax, 2))

for i in range(nmax):
    x_init = np.random.rand(2)
    x = sd.steepest(x_init)
    x_opt[i] = x
    f_opt[i] = obj.Fs(x)

fig, ax = plt.subplots(1, 2, figsize = (12, 4))
ax[0].scatter(f_opt[:, 0], f_opt[:, 1], c="r", alpha=0.7)
ax[1].scatter(x_opt[:, 0], x_opt[:, 1], c="r", alpha=0.7)

ax[0].set_xlabel("$f(x_1)$")
ax[0].set_ylabel("$f(x_2)$")
ax[1].set_xlabel("$x_1$")
ax[1].set_ylabel("$x_2$")
ax[0].set_title("The Pareto Front")
ax[1].set_title("The Pareto Optimal set")

fig.tight_layout()
plt.show()
