from typing import Callable, Optional, Tuple, Union

import numpy as np
from numpy.random import RandomState
from overrides import overrides

from .rvm import BaseRVM


class RVR(BaseRVM):

    y_var: Union[float, None]
    update_y_var: bool

    _beta_: float

    def __init__(
            self,
            kernel: Union[str, Callable[[np.ndarray, np.ndarray],
                                        np.ndarray]] = 'rbf',
            degree: int = 3,
            gamma: Union[str, float] = 'scale',
            coef0: float = 0.0,
            y_var: Optional[float] = 1e-6,
            update_y_var: bool = False,
            include_bias: bool = True,
            tol: float = 1e-3,
            verbose: bool = False,
            max_iter: int = -1,
            random_state: Optional[Union[int, RandomState]] = None) -> None:
        super().__init__(kernel=kernel,
                         degree=degree,
                         gamma=gamma,
                         coef0=coef0,
                         include_bias=include_bias,
                         tol=tol,
                         verbose=verbose,
                         max_iter=max_iter,
                         random_state=random_state)
        self.y_var = y_var
        self.update_y_var = update_y_var
        self._beta_ = 1 / 1e-6

    @overrides
    def predict(self, X: np.ndarray) -> np.ndarray:
        y: np.ndarray = self._apply_kernel_func(
            X=X, Y=self._phi_active_) @ self._weight_posterior_mean_
        return y

    @overrides
    def _init_beta_matrix(self, phi_matrix: np.ndarray,
                          target: np.ndarray) -> np.ndarray:
        """Compute the beta matrix or B matrix in the paper.
        Args:
            phi_matrix (np.ndarray): (n_samples, n_basis_vectors) The complete phi matrix.
            target (np.ndarray): (n_samples, )the target vector of the problem.
        Returns:
            beta_matrix (np.ndarray): (n_samples, n_samples) The beta matrix B with beta_i on the diagonal.
        """
        if self.y_var is None:
            var: float = np.var(target)
            if var == 0.0:
                var = 1e-6
            else:
                var = var * 0.1
            self.beta_ = 1 / var
        else:
            self.beta_ = 1 / self.y_var
        n_samples: int = phi_matrix.shape[0]
        beta_vector: np.ndarray = np.full(shape=n_samples,
                                          fill_value=self.beta_)
        beta_matrix: np.ndarray = np.diag(beta_vector)
        return beta_matrix

    # @overrides
    # def _init_alpha_matrix(self, phi_matrix: np.ndarray, target: np.ndarray,
    #                        beta_matrix: np.ndarray) -> np.ndarray:
    #     """Initialize alpha matrix

    #     Args:
    #         curr_basis_idx (int): current basis vector index number
    #         curr_basis_vector (np.ndarray): (n_samples, 1)current basis vector
    #         phi_matrix (np.ndarray): (n_samples, n_basis_vectors) or (N, M) in Tipping 2003. The complete phi matrix.
    #         target (np.ndarray): (n_samples) the target vector.
    #         beta_matrix (np.ndarray): (n_samples, n_samples) or (N, N) the beta matrix with beta_i on the diagonal.

    #     Returns:
    #         alpha_matrix (np.ndarray): (n_basis_vectors, n_basis_vectors) or (M, M) in Tipping 2003. The complete alpha matrix.
    #     """
    #     curr_basis_idx, curr_basis_vector = self.__init_select_basis_vector(
    #         phi_matrix=phi_matrix, target=target)
    #     # beta_i = 1 / sigma_i**2
    #     alpha_vector: np.ndarray = np.full(shape=(phi_matrix.shape[1]),
    #                                        fill_value=np.inf)
    #     basis_norm: float = np.linalg.norm(curr_basis_vector, axis=0)
    #     alpha_i: float = basis_norm**2 / ((
    #         (curr_basis_vector.T @ target)**2 / basis_norm**2) -
    #                                       (1 / self._beta_))
    #     alpha_vector[curr_basis_idx] = alpha_i
    #     alpha_matrix: np.ndarray = np.diag(v=alpha_vector)
    #     return alpha_matrix

    @overrides
    def _compute_weight_posterior(
            self, target_hat: np.ndarray, alpha_matrix_active: np.ndarray,
            beta_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the "most probable" weight posterior statistics

        Args:
            target_hat (np.ndarray): (n_samples, ) The target hat vector.
            alpha_matrix_active (np.ndarray): (n_active_basis_vectors, n_active_basis_vectors) The current active alpha matrix.
            beta_matrix (np.ndarray): (n_samples, n_samples) The beta matrix

        Returns:
            weight_posterior_mean (np.ndarray): (n_active_basis_vectors, )The updated weight posterior mean
            weight_posterior_cov_matrix (np.ndarray): (n_active_basis_vectors, n_active_basis_vectors)
        """
        weight_posterior_cov_: np.ndarray = np.linalg.inv(
            alpha_matrix_active + self._beta_ *
            self._weight_posterior_cov_ @ self._weight_posterior_cov_.T)
        weight_posterior_mean: np.ndarray = self._beta_ * self._weight_posterior_cov_ @ self._phi_active_.T @ target_hat
        return weight_posterior_mean, weight_posterior_cov_

    @overrides
    def _compute_target_hat(self, X: np.ndarray,
                            target: np.ndarray) -> np.ndarray:
        """Compute target hat

        Args:
            X (np.ndarray): (n_samples, n_features) The input vector.
            y (np.ndarray): (n_samples, )The ground truth target.

        Returns:
            target_hat (np.ndarray): (n_samples, ) The predicted target.
        """
        return target

    @overrides
    def _update_beta_matrix(self, X: np.ndarray, target: np.ndarray,
                            alpha_matrix_active: np.ndarray,
                            beta_matrix: np.ndarray) -> np.ndarray:
        if self.update_y_var == False:
            return beta_matrix
        n_samples: int = target.shape[0]
        n_active_basis_vectors: int = alpha_matrix_active.shape[0]
        y: np.ndarray = self._apply_kernel_func(
            X=X, Y=self._phi_active_) @ self._weight_posterior_mean_
        loss_norm: float = np.linalg.norm(y - target)
        alpha_diag_vector: np.ndarray = np.diagonal(alpha_matrix_active)
        cov_diag_vector: np.ndarray = np.diagonal(self._weight_posterior_cov_)
        alpha_diag_vector_dot_cov_diag_vector: float = alpha_diag_vector @ cov_diag_vector
        self._beta_ = loss_norm**2 / (n_samples - n_active_basis_vectors +
                                      alpha_diag_vector_dot_cov_diag_vector)
        beta_matrix = np.diagonal(np.zeros_like(beta_matrix), self._beta_)
        return beta_matrix
