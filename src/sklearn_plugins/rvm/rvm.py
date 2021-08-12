from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from numpy.core.fromnumeric import shape
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import rbf_kernel


class BaseRVM(BaseEstimator, ABC):
    _kernel_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    _include_bias: bool
    _tol: float
    _max_iter: Union[int, None]

    _relevance_vectors_: Union[np.ndarray, None]

    def __init__(self,
                 kernel_func: Callable[[np.ndarray, np.ndarray],
                                       np.ndarray] = partial(rbf_kernel,
                                                             gamma=None),
                 include_bias: bool = True,
                 tol: float = 1e-3,
                 max_iter: Optional[int] = None) -> None:
        super().__init__()
        self._kernel_func = kernel_func
        self._include_bias = include_bias
        self._tol = tol
        self._max_iter = max_iter

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseRVM":
        # step 0
        phi_matrix: np.ndarray = self._compute_phi_matrix(X, X)
        # step 1
        beta_matrix: np.ndarray = self._init_beta_matrix(target=y)

        return self

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def _compute_phi_matrix(self, X: np.ndarray,
                            X_prime: np.ndarray) -> np.ndarray:
        phi_matrix: np.ndarray = self._kernel_func(X, X_prime)
        if self._include_bias == True:
            n_samples: int = phi_matrix.shape[0]
            phi_matrix = np.append(phi_matrix,
                                   values=np.ones(shape=(n_samples, 1)),
                                   axis=1)
        return phi_matrix

    @abstractmethod
    def _init_beta_matrix(self, target: np.ndarray) -> np.ndarray:
        """Initialize beta matrix.

        Args:
            y (np.ndarray): (n_samples, ) The target vector.

        Returns:
            beta_matrix (np.ndarray): (n_samples, n_samples) The beta matrix.
        """
        pass

    @abstractmethod
    def _update_beta_matrix(self, active_alpha_matrix: np.ndarray,
                            active_phi_matrix: np.ndarray,
                            target: np.ndarray) -> np.ndarray:
        """Update beta matrix.

        Args:
            active_phi_matrix (np.ndarray): [description]

        Returns:
            np.ndarray: [description]
        """
        pass

    # public properties
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
    def relevance_vectors_(self) -> np.ndarray:
        if self._relevance_vectors_ is not None:
            return self._relevance_vectors_
        else:
            raise ValueError("self._relevance_vectors_ is None")
