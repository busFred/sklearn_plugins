from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
from overrides.overrides import overrides
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel

from ._binary_rvc import _BinaryRVC


class RVC(ClassifierMixin, BaseEstimator):

    _kernel_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    _include_bias: bool
    _tol: float
    _max_iter: Union[int, None]
    _verbose: bool

    _relevance_vectors_: Union[np.ndarray, None]  # aka self._X_prime
    _weight_posterior_mean_: Union[np.ndarray, None]  # aka self._mu
    _weight_posterior_cov_: Union[np.ndarray, None]  # aka self._sigma

    def __init__(self,
                 kernel_func: Callable[[np.ndarray, np.ndarray],
                                       np.ndarray] = partial(rbf_kernel,
                                                             gamma=None),
                 include_bias: bool = True,
                 tol: float = 1e-3,
                 max_iter: Optional[int] = None,
                 verbose: bool = True) -> None:
        super().__init__()
        self._kernel_func = kernel_func
        self._include_bias = include_bias
        self._tol = tol
        self._max_iter = max_iter
        self._verbose = verbose

        self._relevance_vectors_ = None
        self._weight_posterior_mean_ = None
        self._weight_posterior_cov_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RVC":
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @property
    def kernel_func(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        return self._kernel_func

    @property
    def include_bias(self) -> bool:
        return self._include_bias

    @property
    def tol(self) -> float:
        return self._tol

    @property
    def max_iter(self) -> Union[int, None]:
        return self._max_iter

    @property
    def verbose(self):
        return self._verbose

    @property
    def relevance_vectors_(self) -> np.ndarray:
        if self._relevance_vectors_ is not None:
            return self._relevance_vectors_.copy()
        else:
            raise ValueError("self._relevance_vectors_ is None")

    @property
    def weight_posterior_mean_(self) -> np.ndarray:
        if self._weight_posterior_mean_ is not None:
            return self._weight_posterior_mean_.copy()
        else:
            raise ValueError("self._weight_posterior_mean_ is None")

    @property
    def weight_posterior_cov_(self):
        if self._weight_posterior_cov_ is not None:
            return self._weight_posterior_cov_.copy()
        else:
            raise ValueError("self._weight_posterior_cov_ is None")
