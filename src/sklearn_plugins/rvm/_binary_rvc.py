from typing import Callable, Optional, Tuple

import numpy as np
from scipy.special import expit as sigmoid
from overrides.overrides import overrides

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
        prob: np.ndarray = self._predict_training(
            active_phi_matrix=active_phi_matrix)
        beta_diag: np.ndarray = prob * (1 - prob)
        beta_matrix = np.diag(beta_diag)
        return beta_matrix

    @overrides
    def _compute_target_hat(self, active_phi_matrix: np.ndarray,
                            beta_matrix: np.ndarray,
                            y: np.ndarray) -> np.ndarray:
        """Compute target hat

        Args:
            X (np.ndarray): (n_samples, n_features) The input vector.
            y (np.ndarray): (n_samples, )The ground truth target.

        Returns:
            target_hat (np.ndarray): (n_samples, ) The predicted target.
        """
        # re-assign to match paper expression
        t: np.ndarray = y
        y = self._predict_training(active_phi_matrix=active_phi_matrix)
        beta_inv: np.ndarray = np.linalg.inv(beta_matrix)
        target_hat: np.ndarray = active_phi_matrix @ self._mu + beta_inv @ (t -
                                                                            y)
        return target_hat

    # @overrides
    # def _update_weight_posterior(
    #         self, active_phi_matrix: np.ndarray,
    #         active_alpha_matrix: np.ndarray, beta_matrix: np.ndarray,
    #         target_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    #     """Compute the "most probable" or "MP" weight posterior statistics

    #     Args:
    #         target_hat (np.ndarray): (n_samples, ) The target hat vector.
    #         alpha_matrix_active (np.ndarray): (n_active_basis_vectors, n_active_basis_vectors) The current active alpha matrix.
    #         beta_matrix (np.ndarray): (n_samples, n_samples) The beta matrix

    #     Returns:
    #         weight_posterior_mean (np.ndarray): (n_active_basis_vectors, )The updated weight posterior mean
    #         weight_posterior_cov_matrix (np.ndarray): (n_active_basis_vectors, n_active_basis_vectors)
    #     """
    #     # force subclass to return so that the corresponding instance variables will definitely get updated.
    #     phi_tr_beta: np.ndarray = active_phi_matrix.T @ beta_matrix
    #     weight_posterior_cov_matrix: np.ndarray = np.linalg.inv(
    #         phi_tr_beta @ active_phi_matrix + active_alpha_matrix)
    #     weight_posterior_mean: np.ndarray = weight_posterior_cov_matrix @ phi_tr_beta @ target_hat
    #     return weight_posterior_mean, weight_posterior_cov_matrix

    def _predict_training(self, active_phi_matrix: np.ndarray) -> np.ndarray:
        """Predict training dataset with current active_phi_matrix.

        Args:
            active_phi_matrix (np.ndarray): (n_samples, n_active_basis) The current active phi matrix.

        Returns:
            pred (np.ndarray): (n_samples, ) The prediction of training dataset given current active phi matrix.
        """
        y: np.ndarray = active_phi_matrix @ self._mu
        pred: np.ndarray = sigmoid(y)
        return pred
