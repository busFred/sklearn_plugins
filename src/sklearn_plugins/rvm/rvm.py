from abc import ABC, abstractmethod
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import rbf_kernel


class BaseRVM(BaseEstimator, ABC):
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
                 verbose: bool = False) -> None:
        super().__init__()
        self._kernel_func = kernel_func
        self._include_bias = include_bias
        self._tol = tol
        self._max_iter = max_iter
        self._verbose = verbose

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
        n_active_basis_vectors: int = active_alpha_matrix.shape[1]
        self._init_weight_posterior(
            n_active_basis_vectors=n_active_basis_vectors)
        # step 3 - 8
        prev_alpha: np.ndarray = alpha_matrix.copy()
        curr_iter: int = 0
        while True if self._max_iter is None else curr_iter < self._max_iter:
            if self._verbose == True:
                print(str.format("iter {} ", curr_iter), end="")
            # step 3
            target_hat: np.ndarray = self._compute_target_hat(
                active_phi_matrix=active_phi_matrix,
                beta_matrix=beta_matrix,
                y=y)
            self._mu, self._sigma_matrix = self._update_weight_posterior(
                active_phi_matrix=active_phi_matrix,
                active_alpha_matrix=active_alpha_matrix,
                beta_matrix=beta_matrix,
                target_hat=target_hat)
            sparsity, quality = self._compute_sparsity_quality(
                active_basis_mask=active_basis_mask,
                phi_matrix=phi_matrix,
                beta_matrix=beta_matrix,
                target_hat=target_hat)
            # step 4 - 8
            alpha_matrix, active_basis_mask, active_alpha_matrix = self._update_alpha_matrix(
                alpha_matrix=alpha_matrix, sparsity=sparsity, quality=quality)
            active_phi_matrix: np.ndarray = self._get_active_phi_matrix(
                phi_matrix=phi_matrix, active_basis_mask=active_basis_mask)
            beta_matrix = self._update_beta_matrix(
                active_alpha_matrix=active_alpha_matrix,
                beta_matrix=beta_matrix,
                active_phi_matrix=active_phi_matrix,
                target=y)
            has_converged: bool = self._has_converged(
                curr_alpha_matrix=alpha_matrix, prev_alpha_matrix=prev_alpha)
            if has_converged:
                break
            prev_alpha: np.ndarray = alpha_matrix.copy()
            curr_iter = curr_iter + 1
        target_hat: np.ndarray = self._compute_target_hat(
            active_phi_matrix=active_phi_matrix, beta_matrix=beta_matrix, y=y)
        self._mu, self._sigma_matrix = self._update_weight_posterior(
            active_phi_matrix=active_phi_matrix,
            active_alpha_matrix=active_alpha_matrix,
            beta_matrix=beta_matrix,
            target_hat=target_hat)
        self._set_active_relevance_vectors(X=X,
                                           active_basis_mask=active_basis_mask)
        return self

    @abstractmethod
    def predict(self, X: np.ndarray):
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
            target (np.ndarray): (n_samples, ) The target vector.

        Returns:
            beta_matrix (np.ndarray): (n_samples, n_samples) The beta matrix.
            init_beta (float): The initial beta value for self._init_alpha_matrix  function call
        """
        pass

    @abstractmethod
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
        pass

    def _get_active_phi_matrix(self, phi_matrix: np.ndarray,
                               active_basis_mask: np.ndarray) -> np.ndarray:
        return phi_matrix[:, active_basis_mask]

    def _get_active_alpha_matrix(self, alpha_matrix: np.ndarray,
                                 active_basis_mask: np.ndarray) -> np.ndarray:
        return alpha_matrix[active_basis_mask, :][:, active_basis_mask]

    @abstractmethod
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
        pass

    def _init_weight_posterior(self, n_active_basis_vectors: int):
        self._weight_posterior_mean_ = np.zeros(shape=n_active_basis_vectors)
        self._weight_posterior_cov_ = np.ones(shape=(n_active_basis_vectors,
                                                     n_active_basis_vectors))

    # @abstractmethod
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
    #     pass

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
            mu (np.ndarray): (n_active_basis_vectors, )The updated weight posterior mean
            sigma_matrix (np.ndarray): (n_active_basis_vectors, n_active_basis_vectors)
        """
        # beta: float = beta_matrix[0, 0]
        sigma_matrix: np.ndarray = np.linalg.inv(
            active_alpha_matrix +
            active_phi_matrix.T @ beta_matrix @ active_phi_matrix)
        mu: np.ndarray = sigma_matrix @ active_phi_matrix.T @ beta_matrix @ target_hat
        return mu, sigma_matrix

    def _compute_sparsity_quality(
            self, active_basis_mask: np.ndarray, phi_matrix: np.ndarray,
            beta_matrix: np.ndarray,
            target_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sparsity_matrix and quality_matrix.

        Args:
            active_basis_mask (np.ndarray): (n_basis_vecor) or (M,) with all elements being boolean value
            phi_matrix (np.ndarray): (n_samples, n_basis_vectors) or (N, M) in Tipping 2003. The complete phi matrix.
            beta_matrix (np.ndarray): (n_samples, n_samples) or (N, N) the beta matrix with beta_i on the diagonal.
            target_hat (np.ndarray): (n_samples) regression should be target and classification should be current pred.

        Returns:
            sparsity (np.ndarray): (n_basis_vectors, ) The complete sparsity vector for all basis_vector.
            quality (np.ndarray): (n_basis_vectors, ) The complete quality vector for all basis_vector.
        """
        active_phi_matrix: np.ndarray = self._get_active_phi_matrix(
            phi_matrix=phi_matrix, active_basis_mask=active_basis_mask)
        phi_m = phi_matrix
        phi_m_tr_beta: np.ndarray = phi_m.T @ beta_matrix
        sigma_phi_tr: np.ndarray = self._sigma_matrix @ active_phi_matrix.T
        phi_m_tr_beta_phi_sigma_phi_tr_beta: np.ndarray = phi_m_tr_beta @ active_phi_matrix @ sigma_phi_tr @ beta_matrix
        sparsity_matrix: np.ndarray = phi_m_tr_beta @ phi_m - phi_m_tr_beta_phi_sigma_phi_tr_beta @ phi_m
        sparsity: np.ndarray = np.diagonal(sparsity_matrix)
        quality: np.ndarray = phi_m_tr_beta @ target_hat - phi_m_tr_beta_phi_sigma_phi_tr_beta @ target_hat
        return sparsity, quality

    def _update_alpha_matrix(
            self, alpha_matrix: np.ndarray, sparsity: np.ndarray,
            quality: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prune design matrix and update alpha matrix according to sparsity and quality analysis.

        Args:
            alpha_matrix (np.ndarray): (n_samples, n_basis_vectors) The complete alpha matrix.
            sparsity (np.ndarray): (n_basis_vectors, ) The sparsity of all basis vectors.
            quality (np.ndarray): (n_basis_vectors, ) The quality  of all basis vectors

        Returns:
            new_alpha_matrix (np.ndarray): (n_basis_vectors, n_basis_vectors) The updated complete alpha matrix.
            new_active_basis_mask (np.ndarray): (n_basis_vectors, ) The updated active basis mask.
            new_active_alpha_matrix (np.ndarray): (n_active_basis_vectors, n_active_basis_vectors) The updated new active alpha matrix.
        """
        theta: np.ndarray = quality**2 - sparsity
        alpha_diag: np.ndarray = np.diagonal(alpha_matrix).copy()
        re_estimate_mask: np.ndarray = (theta > 0) & (alpha_diag < np.inf)
        update_mask: np.ndarray = (theta > 0) & (alpha_diag == np.inf)
        delete_mask: np.ndarray = (theta <= 0) & (alpha_diag < np.inf)
        alpha_new_diag: np.ndarray = sparsity**2 / theta
        alpha_new_diag[delete_mask] = np.inf
        delta_log_weight_posterior: np.ndarray = np.zeros_like(alpha_diag)
        delta_log_weight_posterior[
            re_estimate_mask] = quality[re_estimate_mask]**2 / (
                sparsity[re_estimate_mask] + 1 /
                (1 / alpha_new_diag[re_estimate_mask] -
                 1 / alpha_diag[re_estimate_mask])) - np.log(
                     1 + sparsity[re_estimate_mask] *
                     (1 / alpha_new_diag[re_estimate_mask] -
                      1 / alpha_diag[re_estimate_mask]))
        delta_log_weight_posterior[update_mask] = (
            quality[update_mask]**2 -
            sparsity[update_mask]) / sparsity[update_mask] + np.log(
                sparsity[update_mask] / quality[update_mask]**2)
        delta_log_weight_posterior[delete_mask] = quality[delete_mask]**2 / (
            sparsity[delete_mask] - alpha_diag[delete_mask]) - np.log(
                1 - sparsity[delete_mask] / alpha_diag[delete_mask])
        selected_basis_idx: np.ndarray = np.argmax(delta_log_weight_posterior)
        alpha_diag[selected_basis_idx] = alpha_new_diag[selected_basis_idx]
        new_alpha_matrix: np.ndarray = np.diag(alpha_diag)
        new_active_basis_mask: np.ndarray = self._get_active_basis_mask(
            alpha_matrix=new_alpha_matrix)
        new_active_alpha_matrix: np.ndarray = self._get_active_alpha_matrix(
            alpha_matrix=new_alpha_matrix,
            active_basis_mask=new_active_basis_mask)
        return new_alpha_matrix, new_active_basis_mask, new_active_alpha_matrix

    def _has_converged(self, curr_alpha_matrix: np.ndarray,
                       prev_alpha_matrix: np.ndarray) -> bool:
        """Determine whether the algorithm has converged.

        Args:
            alpha_matrix (np.ndarray): (n_basis_vectors, n_basis_vectors) The current complete alpha matrix.
            prev_alpha_matrix (np.ndarray): (n_basis_vectors, n_basis_vectors) The previous alpha matrix.

        Returns:
            bool: True if has converged, False otherwise.
        """
        alpha_abs_diff: np.ndarray = np.abs(curr_alpha_matrix -
                                            prev_alpha_matrix)
        diff: float = np.nansum(alpha_abs_diff)
        if self._verbose == True:
            print(str.format("alpha_diff {}", diff))
        return True if diff <= self.tol else False

    def _set_active_relevance_vectors(self, X: np.ndarray,
                                      active_basis_mask: np.ndarray):
        if self._include_bias == True:
            self._include_bias = active_basis_mask[-1]
            active_basis_mask = active_basis_mask[:-1]
        self._relevance_vectors_ = X[active_basis_mask]

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

    # protected properties
    @property
    def _X_prime(self) -> np.ndarray:
        if self._relevance_vectors_ is not None:
            return self._relevance_vectors_
        else:
            raise ValueError("self._relevance_vectors_ is None")

    @property
    def _mu(self) -> np.ndarray:
        if self._weight_posterior_mean_ is not None:
            return self._weight_posterior_mean_
        else:
            raise ValueError("self._weight_posterior_mean_ is None")

    @_mu.setter
    def _mu(self, mu: np.ndarray):
        self._weight_posterior_mean_ = mu

    @property
    def _sigma_matrix(self):
        if self._weight_posterior_cov_ is not None:
            return self._weight_posterior_cov_
        else:
            raise ValueError("self._weight_posterior_cov_ is None")

    @_sigma_matrix.setter
    def _sigma_matrix(self, sigma: np.ndarray):
        self._weight_posterior_cov_ = sigma
