from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
from overrides import overrides
from sklearn.metrics.pairwise import rbf_kernel

from .rvm import BaseRVM


class RVR(BaseRVM):
    _y_var: Union[float, None]
    _update_y_var: bool

    _y_var_: float

    def __init__(self,
                 kernel_func: Callable[[np.ndarray, np.ndarray],
                                       np.ndarray] = partial(rbf_kernel,
                                                             gamma=None),
                 y_var: Optional[float] = None,
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
        self._y_var_ = 0.0

    @overrides
    def predict(self, X: np.ndarray) -> np.ndarray:
        phi_matrix: np.ndarray = self._compute_phi_matrix(
            X=X, X_prime=self._X_prime)
        y: np.ndarray = phi_matrix @ self._mu
        return y

    @overrides
    def _init_beta_matrix(self,
                          target: np.ndarray) -> Tuple[np.ndarray, float]:
        """Initialize beta matrix.

        Args:
            target (np.ndarray): (n_samples, ) The target vector.

        Returns:
            beta_matrix (np.ndarray): (n_samples, n_samples) The beta matrix.
            init_beta (float): The initial beta value for self._init_alpha_matrix  function call
        """
        if self._y_var is None or self._y_var <= 0:
            self._y_var_ = np.var(target)
            self._y_var_ = 1e-6 if self._y_var_ == 0.0 else self._y_var_
        init_beta: float = 1 / self._y_var_
        n_samples: int = target.shape[0]
        beta: np.ndarray = np.full(shape=(n_samples), fill_value=init_beta)
        beta_matrix: np.ndarray = np.diag(beta)
        return beta_matrix, init_beta

    @overrides
    def _update_beta_matrix(self, active_alpha_matrix: np.ndarray,
                            beta_matrix: np.ndarray,
                            active_phi_matrix: np.ndarray,
                            target: np.ndarray) -> np.ndarray:
        """Update beta matrix.

        Args:
            active_phi_matrix (np.ndarray): [description]

        Returns:
            np.ndarray: [description]
        """
        if self._update_y_var == False:
            return beta_matrix
        n_samples: int = target.shape[0]
        n_active_basis_vectors: int = active_alpha_matrix.shape[0]
        y = active_phi_matrix @ self._mu
        loss_norm: float = np.linalg.norm(y - target)
        alpha_diag_vector: np.ndarray = np.diagonal(active_alpha_matrix)
        sigma_diag_vector: np.ndarray = np.diagonal(self._sigma_matrix)
        alpha_diag_dot_cov_diag: float = alpha_diag_vector.T @ sigma_diag_vector
        beta: float = loss_norm**2 / (n_samples - n_active_basis_vectors +
                                      alpha_diag_dot_cov_diag)
        self._y_var_ = 1 / beta
        beta_matrix = np.diagonal(np.zeros_like(beta_matrix), beta)
        return beta_matrix

    @overrides
    def _compute_target_hat(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute target hat

        Args:
            X (np.ndarray): (n_samples, n_features) The input vector.
            y (np.ndarray): (n_samples, )The ground truth target.

        Returns:
            target_hat (np.ndarray): (n_samples, ) The predicted target.
        """
        pass

    @overrides
    def _compute_weight_posterior(
            self, active_alpha_matrix: np.ndarray, beta_matrix: np.ndarray,
            target_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the "most probable" or "MP" weight posterior statistics

        Args:
            target_hat (np.ndarray): (n_samples, ) The target hat vector.
            alpha_matrix_active (np.ndarray): (n_active_basis_vectors, n_active_basis_vectors) The current active alpha matrix.
            beta_matrix (np.ndarray): (n_samples, n_samples) The beta matrix

        Returns:
            weight_posterior_mean (np.ndarray): (n_active_basis_vectors, )The updated weight posterior mean
            weight_posterior_cov_matrix (np.ndarray): (n_active_basis_vectors, n_active_basis_vectors)
        """
        # force subclass to return so that the corresponding instance variables will definitely get updated.
        pass

    @property
    def y_var(self) -> float:
        return self._y_var

    @property
    def update_y_var(self) -> bool:
        return self._update_y_var