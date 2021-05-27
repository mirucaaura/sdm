import numpy as np
from scipy.optimize import fmin

class SteepestDescent():
    def __init__(
        self,
        ndim: int,
        nu: float,
        sigma: float,
        eps: float,
    ):
        self.ndim = ndim
        self.nu = nu
        self.sigma = sigma
        self.eps = eps
    
    def f1(self, x):
        # return x[0]**2 + x[1]**2
        return x[0]**2 + 3 * (x[1] - 1)**2

    def f2(self, x):
        # return 5 + x[1]**2 - x[0]
        return 2 * (x[0] - 1)**2 + x[1]**2

    def F(self, x):
        return np.array([self.f1(x), self.f2(x)])

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

    def nabla_F(self, x):
        return np.array([self.grad(self.f1, x), self.grad(self.f2, x)])

    def phi(self, d, x):
        nabla_F = self.nabla_F(x)
        return max(np.dot(nabla_F, d)) + 0.5 * np.linalg.norm(d) ** 2

    def theta(self, d, x):
        return self.phi(d, x) + 0.5 * np.linalg.norm(d) ** 2

    def armijo(self, d, x):
        power = 0
        t = pow(self.nu, power)
        Fl = np.array(self.F(x + t * d))
        Fr = np.array(self.F(x))
        Re = self.sigma * t * np.dot(self.nabla_F(x), d)
        # i = 0
        while np.all(Fl > Fr + Re):
            t *= self.nu
            Fl = np.array(self.F(x + t * d))
            Fr = np.array(self.F(x))
            Re = self.sigma * t * np.dot(self.nabla_F(x), d)
            # print(Fl, Fr, Re, Fl - Fr - Re)
            # i += 1
            # if i == 10:
            #     break
        return t
    
    def steepest(self, x):
        d = np.array(fmin(self.phi, x, args=(x, )))
        th = self.theta(d, x)
        while abs(th) > self.eps:
            t = self.armijo(d, x)
            x = x + t * d
            d = np.array(fmin(self.phi, x, args=(x, )))
            th = self.theta(d, x)
            print("theta = ", th)
        return x

if __name__ == "__main__":
    method = SteepestDescent(2, 0.8, 0.8, 1e-5)
    print(method.ndim)
    x = np.array([2., 3.], dtype=np.float64)
    print(x)
    print(method.f1(x))
    print(method.f2(x))
    print("F = ", method.F(x))
    print(method.grad(method.f1, x))
    print(method.nabla_F(x))
    d = np.array([1., 5.], dtype=np.float64)
    print(method.phi(d, x))
    t = method.armijo(x, d)
    print(t)
    x_init = np.array([2, 2])
    x = method.steepest(x_init)
    print(method.F(x))