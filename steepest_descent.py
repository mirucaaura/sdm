import numpy as np
from scipy.optimize import fmin

class SteepestDescent():
    def __init__(
        self,
        ndim: int,
    ):
        self.ndim = ndim
    
    def f1(self, x):
        return x[0]**2 + x[1]**2

    def f2(self, x):
        return 5 + x[1]**2 - x[0]

    def grad(self, f, x, h=1e-4):
        g = np.zeros_like(x)
        for i in range(self.ndim):
            tmp = x[i]
            x[i] = tmp + h
            yr = f(x)
            x[i] = tmp - h
            yl = f(x)
            g[i] = (yr - yl) / (2 * h)
            x[i] = tmp
        return g

if __name__ == "__main__":
    method = SteepestDescent(2)
    print(method.ndim)
    x = np.array([2., 3.], dtype=np.float64)
    print(x)
    print(method.f1(x))
    print(method.f2(x))
    print(method.grad(method.f1, x))