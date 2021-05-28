import numpy as np
from nptyping import NDArray
from typing import Any, Callable
from dataclasses import dataclass

@dataclass
class Obj:

    def f(self, x: NDArray[(1, ...), np.float64]) -> Any:
        return x[0]**2 + 3 * (x[1] - 1)**2

    def g(self, x: NDArray[(1, ...), np.float64]) -> Any:
        return 2 * (x[0] - 1)**2 + x[1]**2

    def Fs(self, x: NDArray[(1, ...), np.float64]) -> NDArray[(1, ...), np.float64]:
        return np.array([self.f(x), self.g(x)])

    def Fss(self):
        return np.array([self.f, self.g])