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
        """Predict the input X.

        Args:
            X (np.ndarray): (n_samples, n_features) Input data.

        Returns:
            labels (np.ndarray): (n_samples, ) The predicted labels.
        """
        prob: np.ndarray = self.predict_proba(X)
        labels: np.ndarray = np.argwhere(prob > 0.5)
        return labels

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict the input X with given model

        Args:
            X (np.ndarray): (n_samples, n_features) Input data.

        Returns:
            probas (np.ndarray): (n_samples, ) The predicted probabilities.
        """
        phi_matrix: np.ndarray = self._compute_phi_matrix(
            X=X, X_prime=self._X_prime)
        prob: np.ndarray = self._predict_phi_matrix(
            active_phi_matrix=phi_matrix)
        return prob

    def predict_y(self, X: np.ndarray) -> np.ndarray:
        """Compute y with given model

        Args:
            X (np.ndarray): (n_samples, n_features) Input data.

        Returns:
            y (np.ndarray): (n_samples, ) The predicted y before applying sigmoid.
        """
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
        init_beta: float = np.mean(target)
        beta_diag: np.ndarray = np.full(shape=target.shape[0],
                                        fill_value=init_beta)
        beta_matrix: np.ndarray = np.diag(beta_diag)
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
        prob: np.ndarray = self._predict_phi_matrix(
            active_phi_matrix=active_phi_matrix)
        beta_diag: np.ndarray = prob * (1 - prob)
        beta_diag = np.where(beta_diag == 0.0, 1e-5, beta_diag)
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
        y = self._predict_phi_matrix(active_phi_matrix=active_phi_matrix)
        beta_inv: np.ndarray = np.linalg.inv(beta_matrix)
        phi_mu: np.ndarray = active_phi_matrix @ self._mu
        beta_inv_diff: np.ndarray = beta_inv @ (t - y)
        target_hat: np.ndarray = phi_mu + beta_inv_diff
        return target_hat

    def _predict_phi_matrix(self, active_phi_matrix: np.ndarray) -> np.ndarray:
        """Predict training dataset with current active_phi_matrix.

        Args:
            active_phi_matrix (np.ndarray): (n_samples, n_active_basis) The current active phi matrix.

        Returns:
            pred (np.ndarray): (n_samples, ) The prediction of training dataset given current active phi matrix.
        """
        y: np.ndarray = active_phi_matrix @ self._mu
        pred: np.ndarray = sigmoid(y)
        return pred


from skl2onnx import update_registered_converter

from ._onnx_transfrom import rvr_converter, rvr_shape_calculator

update_registered_converter(_BinaryRVC,
                            "SklearnPlugins_BinaryRVC",
                            shape_fct=rvr_shape_calculator,
                            convert_fct=rvr_converter)
