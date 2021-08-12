from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from numpy.random import RandomState
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import rbf_kernel


class BaseRVM(BaseEstimator, ABC):
    kernel_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    y_var: float
    update_y_var: bool
    include_bias: bool
    tol: float
    max_iter: Optional[int]

    def __init__(self,
                 kernel_func: Callable[[np.ndarray, np.ndarray],
                                       np.ndarray] = partial(rbf_kernel,
                                                             gamma=None),
                 y_var: float = 1e-6,
                 update_y_var: bool = False,
                 include_bias: bool = True,
                 tol: float = 1e-3,
                 max_iter: Optional[int] = None) -> None:
        super().__init__()
        self.kernel_func = kernel_func
        self.y_var = y_var
        self.update_y_var = update_y_var
        self.include_bias = include_bias
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseRVM":
        phi_matrix: np.ndarray = self.kernel_func(X, X)
        
        return self

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass
