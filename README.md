# Steepest Descent Method for Multi-Objective Optimization

The implementation of the method for solving multi-objective optimization probelem. The main algorithm can be found on the article [1]. Unlike methods based on evolutionary computation such as a genetic algorithm, descent methods can only provide a single solution, so the algorithm must be run iteratively to obtain a Pareto optimal set.

# Requirements

- Python 3.8 or above.

# How to use

Define the objective function to be minimized in the `obj_func.py`. We also define a list of objective functions in the same file. An example is as follows:

```python
class Obj:

    def f(self, x: NDArray[(1, ...), np.float64]) -> Any:
        return x[0]**2 + 3 * (x[1] - 1)**2

    def g(self, x: NDArray[(1, ...), np.float64]) -> Any:
        return 2 * (x[0] - 1)**2 + x[1]**2

    def Fs(self, x: NDArray[(1, ...), np.float64]) -> NDArray[(1, ...), np.float64]:
        return np.array([self.f(x), self.g(x)])

    def Fss(self):
        return np.array([self.f, self.g])
```

The main algorithm lies in `steepest_descent.py`, and we can run a test code by calling `steepest()` method. An example is as follows:

```python
import numpy as np
from steepest_descent import SteepestDescent
from obj_func import Obj

sd = SteepestDescent(
    ndim=2,
    nu=0.8,
    sigma=0.8,
    eps=1e-5,
)

obj = Obj()

x_init = np.array([1, 2])
x_opt = sd.steepest(x_init)
print(obj.Fs(x_opt)) # Pareto optimal
```

# Refferences

1. [Steepest descent methods for multicriteria optimization](https://link.springer.com/article/10.1007%2Fs001860000043)