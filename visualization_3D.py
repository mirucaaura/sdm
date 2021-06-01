import numpy as np
import matplotlib.pyplot as plt
from steepest_descent import SteepestDescent
from scaled_MMD import Obj

sd = SteepestDescent(
    ndim=2,
    nu=0.8,
    sigma=0.8,
    eps=1e-5,
)

obj = Obj()

np.random.seed(1)

nmax = 50 # the number of computing the Pareto optimal
f_opt = np.zeros((nmax, 3))
x_opt = np.zeros((nmax, 2))

for i in range(nmax):
    x_init = np.random.rand(2)
    x = sd.steepest(x_init)
    x_opt[i] = x
    f_opt[i] = obj.Fs(x)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(f_opt[:, 0], f_opt[:, 1], f_opt[:, 2], c="r", alpha=0.7, linewidth=0)
ax.set_xlabel("$f_1(x)$")
ax.set_ylabel("$f_2(x)$")
ax.set_zlabel("$f_3(x)$")
ax.set_title("The Pareto Front")
plt.show()
