from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
from overrides import overrides
from sklearn.metrics.pairwise import rbf_kernel

from .rvm import BaseRVM


class RVR(BaseRVM):
    _y_var: float
    _update_y_var: bool

    def __init__(self,
                 kernel_func: Callable[[np.ndarray, np.ndarray],
                                       np.ndarray] = partial(rbf_kernel,
                                                             gamma=None),
                 y_var: float = 1e-6,
                 update_y_var: bool = False,
                 include_bias: bool = True,
                 tol: float = 1e-3,
                 max_iter: Optional[int] = None) -> None:
        super().__init__(kernel_func=kernel_func,
                         include_bias=include_bias,
                         tol=tol,
                         max_iter=max_iter)
        self._y_var = y_var
        self._update_y_var = update_y_var

    @property
    def y_var(self) -> float:
        return self._y_var

    @property
    def update_y_var(self) -> bool:
        return self._update_y_var