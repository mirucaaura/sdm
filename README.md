# Steepest Descent Method for Multi-Objective Optimization

The implementation of the method for solving multi-objective optimization probelem. The main algorithm can be found on the article [1]. Unlike methods based on evolutionary computation such as a genetic algorithm, descent methods can only provide a single solution, so the algorithm must be run iteratively to obtain a Pareto optimal set.

# Requirements

- Python 3.8 or above.

# How to use

```python
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
print(sd.F(x)) # Pareto optimal
```

# Refferences

1. [Steepest descent methods for multicriteria optimization](https://link.springer.com/article/10.1007%2Fs001860000043)