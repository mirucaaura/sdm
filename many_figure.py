import numpy as np
import matplotlib.pyplot as plt
from steepest_descent import SteepestDescent
from obj_func import Obj

ndims = [2, 2, 2, 2]
nus = [0.5, 0.6, 0.7, 0.9]
sigmas = [0.4, 0.6, 0.9, 0.3]
epss = [1e-5, 1e-5, 1e-5, 1e-5]

obj = Obj()

np.random.seed(1)

nmax = 20 # the number of computing the Pareto optimal
x_opt = np.zeros((4, nmax, 2))

for i in range(4):
    sd = SteepestDescent(
        ndim=ndims[i],
        nu=nus[i],
        sigma=sigmas[i],
        eps=epss[i],
    )
    for j in range(nmax):
        x_init = np.random.rand(2)
        x = sd.steepest(x_init)
        x_opt[i][j] = obj.Fs(x)

fig, ax = plt.subplots(2, 2)
ax[0, 0].scatter(x_opt[0][:, 0], x_opt[0][:, 1], alpha=0.7)
ax[0, 1].scatter(x_opt[1][:, 0], x_opt[1][:, 1], alpha=0.7)
ax[1, 0].scatter(x_opt[2][:, 0], x_opt[2][:, 1], alpha=0.7)
ax[1, 1].scatter(x_opt[3][:, 0], x_opt[3][:, 1], alpha=0.7)

ax[0, 0].set_title(r"$\nu = 0.5, \sigma = 0.4$")
ax[0, 1].set_title(r"$\nu = 0.6, \sigma = 0.6$")
ax[1, 0].set_title(r"$\nu = 0.7, \sigma = 0.9$")
ax[1, 1].set_title(r"$\nu = 0.9, \sigma = 0.3$")

ax[0, 0].set_xlabel("$f_1(x)$")
ax[0, 0].set_ylabel("$f_2(x)$")
ax[0, 1].set_xlabel("$f_1(x)$")
ax[0, 1].set_ylabel("$f_2(x)$")
ax[1, 0].set_xlabel("$f_1(x)$")
ax[1, 0].set_ylabel("$f_2(x)$")
ax[1, 1].set_xlabel("$f_1(x)$")
ax[1, 1].set_ylabel("$f_2(x)$")

fig.tight_layout()
plt.show()
