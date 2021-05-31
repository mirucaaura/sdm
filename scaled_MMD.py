import numpy as np
from nptyping import NDArray
from typing import Any, Callable

class Obj:

    def f1(self, x: NDArray[(1, ...), np.float64]) -> Any:
        return x[0]**2 + 3 * (x[1] - 1)**2

    def f2(self, x: NDArray[(1, ...), np.float64]) -> Any:
        return 2 * (x[0] - 1)**2 + x[1]**2

    def f3(self, x: NDArray[(1, ...), np.float64]) -> Any:
        return 3 * (x[0] + 1)**2 + 2 * (x[1] + 1)**2

    def Fs(self, x: NDArray[(1, ...), np.float64]) -> NDArray[(1, ...), np.float64]:
        return np.array([self.f1(x), self.f2(x), self.f3(x)])

    def Fss(self):
        return np.array([self.f1, self.f2, self.f3])