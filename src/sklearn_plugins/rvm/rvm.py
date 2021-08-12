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
    _weight_posterior_mean_: Union[np.ndarray, None]
    _weight_posterior_cov_: Union[np.ndarray, None]

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

        self._relevance_vectors_ = None
        self._weight_posterior_mean_ = None
        self._weight_posterior_cov_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseRVM":
        # step 0
        phi_matrix: np.ndarray = self._compute_phi_matrix(X, X)
        # step 1
        beta_matrix, init_beta = self._init_beta_matrix(target=y)
        # step 2
        alpha_matrix: np.ndarray = self._init_alpha_matrix(
            init_beta=init_beta, phi_matrix=phi_matrix, target=y)
        active_basis_mask: np.ndarray = self._get_active_basis_mask(
            alpha_matrix=alpha_matrix)
        active_alpha_matrix: np.ndarray = self._get_active_alpha_matrix(
            alpha_matrix=alpha_matrix, active_basis_mask=active_basis_mask)
        active_phi_matrix: np.ndarray = self._get_active_phi_matrix(
            phi_matrix=phi_matrix, active_basis_mask=active_basis_mask)
        # step 3

        return self

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def _compute_phi_matrix(self, X: np.ndarray,
                            X_prime: np.ndarray) -> np.ndarray:
        """Compute phi_matrix.

        Args:
            X (np.ndarray): (n_new_samples n_features) During inference, `X` is the new input vectors. 
            X_prime (np.ndarray): (n_samples, n_basis_vectors) or (N, M) in 2003 Tipphing. During inference, `X` is the `self._relevance_vectors_`.

        Returns:
            phi_matrix (np.ndarray): (n_new_samples, n_samples) The computed phi_matrix.
        """
        phi_matrix: np.ndarray = self._kernel_func(X, X_prime)
        if self._include_bias == True:
            n_samples: int = phi_matrix.shape[0]
            phi_matrix = np.append(phi_matrix,
                                   values=np.ones(shape=(n_samples, 1)),
                                   axis=1)
        return phi_matrix

    def _init_select_basis_vector(
            self, phi_matrix: np.ndarray,
            target: np.ndarray) -> Tuple[int, np.ndarray]:
        """Select the largest normalized projection onto the target vector.

        Args:
            phi_matrix (np.ndarray): (n_samples, n_basis_vectors) or (N, M) in Tipping 2003. The complete phi matrix.
            target (np.ndarray): (n_sampels, ) the target vector.

        Returns:
            idx (int): index of the selected vector
            curr_basis_vector (np.ndarray): (n_samples, 1) the selected basis vector.
        """
        proj: np.ndarray = phi_matrix.T @ target
        phi_norm: np.ndarray = np.linalg.norm(phi_matrix, axis=0)
        proj_norm: np.ndarray = np.divide(proj, phi_norm)
        idx: int = np.argmax(proj_norm)
        curr_basis_vector: np.ndarray = phi_matrix[:, idx]
        return idx, curr_basis_vector

    def _init_alpha_matrix(self, init_beta: float, phi_matrix: np.ndarray,
                           target: np.ndarray) -> np.ndarray:
        """Initialize alpha matrix.

        Args:
            init_beta (float): The initial beta value used to compute initial alpha_matrix.
            phi_matrix (np.ndarray): (n_samples, n_basis_vectors) The complete phi matrix.
            target (np.ndarray): (n_samples, ) The target vector.

        Returns:
            alpha_matrix (np.ndarray): (n_basis_vectors, n_basis_vectors) or (M, M) in Tipping 2003. The complete alpha matrix.
        """
        curr_basis_idx, curr_basis_vector = self._init_select_basis_vector(
            phi_matrix=phi_matrix, target=target)
        alpha_vector: np.ndarray = np.full(shape=(phi_matrix.shape[1]),
                                           fill_value=np.inf)
        basis_norm: float = np.linalg.norm(curr_basis_vector, axis=0)
        alpha_i: float = basis_norm**2 / ((
            (curr_basis_vector.T @ target)**2 / basis_norm**2) -
                                          (1 / init_beta))
        alpha_vector[curr_basis_idx] = alpha_i
        alpha_matrix: np.ndarray = np.diag(v=alpha_vector)
        return alpha_matrix

    def _get_active_basis_mask(self, alpha_matrix: np.ndarray) -> np.ndarray:
        """Get active basis mask

        Args:
            alpha_matrix (np.ndarray): (n_basis_vector, n_basis_vector) or (M, M). The complete alpha matrix.

        Returns:
            active_basis_mask (np.ndarray): (n_basis_vecor) or (M,) with all elements being boolean value
        """
        alpha_vector: np.ndarray = np.diagonal(a=alpha_matrix)
        active_basis_mask: np.ndarray = (alpha_vector != np.inf)
        return active_basis_mask

    @abstractmethod
    def _init_beta_matrix(self,
                          target: np.ndarray) -> Tuple[np.ndarray, float]:
        """Initialize beta matrix.

        Args:
            y (np.ndarray): (n_samples, ) The target vector.

        Returns:
            beta_matrix (np.ndarray): (n_samples, n_samples) The beta matrix.
            init_beta (float): The initial beta value for self._init_alpha_matrix  function call
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

    def _get_active_phi_matrix(self, phi_matrix: np.ndarray,
                               active_basis_mask: np.ndarray) -> np.ndarray:
        return phi_matrix[active_basis_mask, :][:, active_basis_mask]

    def _get_active_alpha_matrix(self, alpha_matrix: np.ndarray,
                                 active_basis_mask: np.ndarray) -> np.ndarray:
        return alpha_matrix[active_basis_mask, :][:, active_basis_mask]

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
