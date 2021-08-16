from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.special import expit as sigmoid
from overrides.overrides import overrides
from sklearn.utils.validation import check_X_y

from .rvm import BaseRVM


class _BinaryRVC(BaseRVM):
    def __init__(self, kernel_func: Callable[[np.ndarray, np.ndarray],
                                             np.ndarray], include_bias: bool,
                 tol: float, max_iter: Optional[int], verbose: bool) -> None:
        super().__init__(kernel_func=kernel_func,
                         include_bias=include_bias,
                         tol=tol,
                         max_iter=max_iter,
                         verbose=verbose)

    @overrides
    def predict(self, X: np.ndarray):
        pass

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
        beta_diag = target.copy().astype(np.float)
        beta_matrix: np.ndarray = np.diag(beta_diag)
        init_beta: float = np.mean(beta_diag)
        return beta_matrix, init_beta

    @overrides
    def _update_beta_matrix(self, active_alpha_matrix: np.ndarray,
                            beta_matrix: np.ndarray,
                            active_phi_matrix: np.ndarray,
                            target: np.ndarray) -> np.ndarray:
        """Update beta matrix.

        Args:
            active_phi_matrix (np.ndarray): (n_samples, n_active_basis_functions)

        Returns:
            np.ndarray: [description]
        """
        y: np.ndarray = active_phi_matrix @ self._mu
        prob: np.ndarray = sigmoid(y)
        beta_diag: np.ndarray = prob * (1 - prob)
        beta_matrix = np.diag(beta_diag)
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
    def _update_weight_posterior(
            self, active_phi_matrix: np.ndarray,
            active_alpha_matrix: np.ndarray, beta_matrix: np.ndarray,
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
