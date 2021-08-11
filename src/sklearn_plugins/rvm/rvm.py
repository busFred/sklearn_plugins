from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from numpy.random import RandomState
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import (linear_kernel, polynomial_kernel,
                                      rbf_kernel)


class BaseRVM(BaseEstimator, ABC):
    """BaseRVM

    Attributes:
        kernel (Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]], optional): The kernel function to be used. Defaults to 'rbf'.
        degree (int, optional): Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels. Defaults to 3.
        gamma (Union[str, float], optional): Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. Defaults to 'scale'.
        coef0 (float, optional): Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’. Defaults to 0.0.
        include_bias (bool, optional): Wheather or not to include bias in the design matrix. Defaults to True.
        tol (float, optional): tolerance for stopping criterion. Defaults to 1e-3.
        verbose (bool, optional): [description]. Defaults to False.
        max_iter (int, optional): [description]. Defaults to -1.
        random_state (Optional[Union[int, RandomState]], optional): [description]. Defaults to None.

        gamma_ (float): Actual Kernel coefficient used to compute ‘rbf’, ‘poly’ and ‘sigmoid’
        sigma_ (Union[np.ndarray, None]): The posterior 
    """
    kernel: Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]
    degree: int
    gamma: Union[str, float]
    coef0: float
    include_bias: bool
    tol: float
    verbose: bool
    max_iter: int
    random_state: Union[int, RandomState, None]

    gamma_: float
    weight_posterior_cov_: Union[np.ndarray, None]  # aka sigma
    weight_posterior_mean_: Union[np.ndarray, None]  # aka mu_mp
    design_matrix_: Union[np.ndarray, None]  # aka phi
    sparsity_: Union[np.ndarray, None]
    quality_: Union[np.ndarray, None]

    # public
    def __init__(
            self,
            kernel: Union[str, Callable[[np.ndarray, np.ndarray],
                                        np.ndarray]] = 'rbf',
            degree: int = 3,
            gamma: Union[str, float] = 'scale',
            coef0: float = 0.0,
            include_bias: bool = True,
            tol: float = 1e-3,
            verbose: bool = False,
            max_iter: int = -1,
            random_state: Optional[Union[int, RandomState]] = None) -> None:
        super().__init__()
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.include_bias = include_bias
        self.tol = tol
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state
        # inferred variables
        self.gamma_ = 1.0
        self.weight_posterior_cov_ = None
        self.weight_posterior_mean_ = None
        self.design_matrix_ = None
        self.sparsity_ = None
        self.quality_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseRVM":
        # step 0
        phi_matrix: np.ndarray = self._apply_kernel_func(
            X=X, Y=X)  # the design matrix
        # step 1
        beta_matrix: np.ndarray = self._init_beta_matrix(phi_matrix=phi_matrix,
                                                         target=y)
        # step 2
        alpha_matrix: np.ndarray = self.__init_alpha_matrix(
            phi_matrix=phi_matrix, target=y, beta_matrix=beta_matrix)
        # step 2.5: instantiate active alpha, and active phi (design matrix)
        active_basis_mask: np.ndarray = self.__get_active_basis_mask(
            alpha_matrix=alpha_matrix)
        alpha_matrix_active: np.ndarray = alpha_matrix[:, active_basis_mask][
            active_basis_mask, :]
        self.design_matrix_ = phi_matrix[:, active_basis_mask]
        # step 3: initialize posterior_weight_mean and posterior_weight_cov
        prev_alpha_matrix: np.ndarray = np.copy(a=alpha_matrix)
        target_hat: np.ndarray = self._compute_target_hat(X=X, target=y)
        self.weight_posterior_mean_, self.weight_posterior_cov_ = self._init_weight_posterior(
            n_active_basis_vectors=alpha_matrix_active.shape[0])
        sparsity, quality = self.__update_sparsity_quality(
            active_basis_mask=active_basis_mask,
            phi_matrix=phi_matrix,
            beta_matrix=beta_matrix,
            target_hat=target_hat)
        for curr_iter in range(self.max_iter):
            # step 4
            # TODO change to the implementation outlined in appendix
            curr_basis_idx: int = self._select_basis_vector(
                curr_iter=curr_iter,
                phi_matrix=phi_matrix,
                alpha_matrix=alpha_matrix,
                beta_matrix=beta_matrix)
            # step 5 - 8
            active_basis_mask, alpha_matrix_active = self.__prune(
                curr_basis_idx=curr_basis_idx,
                phi_matrix=phi_matrix,
                alpha_matrix=alpha_matrix,
                sparsity=sparsity,
                quality=quality)
            # step 9
            beta_matrix = self._update_beta_matrix(
                X=X,
                target=y,
                alpha_matrix_active=alpha_matrix_active,
                beta_matrix=beta_matrix)
            # step 10
            target_hat = self._compute_target_hat(X=X, target=y)
            self.weight_posterior_mean_, self.weight_posterior_cov_ = self._compute_weight_posterior(
                target_hat=target_hat,
                alpha_matrix_active=alpha_matrix_active,
                beta_matrix=beta_matrix)
            sparsity, quality = self.__update_sparsity_quality(
                active_basis_mask=active_basis_mask,
                phi_matrix=phi_matrix,
                beta_matrix=beta_matrix,
                target_hat=target_hat)
            # step 11
            has_converged: bool = self.__has_converged(
                curr_alpha_matrix=alpha_matrix,
                prev_alpha_matrix=prev_alpha_matrix)
            if has_converged == True:
                break
        return self

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def get_params(self, deep: bool = False):
        """
        Get parameters for this estimator.

        Args:
            deep (bool): If True, will return the parameters for this estimator and contained subobjects that are estimators. Default=True.

        Returns:
            params (dict): Parameter names mapped to their values.
        """

        params: Dict = super().get_params()
        return params

    def set_params(self, **parameters):
        """Set parameters using kwargs."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    # protected abstract
    @abstractmethod
    def _init_beta_matrix(self, phi_matrix: np.ndarray,
                          target: np.ndarray) -> np.ndarray:
        """Compute the beta matrix or B matrix in the paper.

        Args:
            phi_matrix (np.ndarray): (n_samples, n_basis_vectors) The complete phi matrix.
            target (np.ndarray): (n_samples, )the target vector of the problem.

        Returns:
            beta_matrix (np.ndarray): (n_samples, n_samples) The beta matrix B with beta_i on the diagonal.
        """
        pass

    @abstractmethod
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
        # force subclass to return so that the corresponding instance variables will definitely get updated.
        pass

    @abstractmethod
    def _compute_target_hat(self, X: np.ndarray,
                            target: np.ndarray) -> np.ndarray:
        """Compute target hat

        Args:
            X (np.ndarray): (n_samples, n_features) The input vector.
            y (np.ndarray): (n_samples, )The ground truth target.

        Returns:
            target_hat (np.ndarray): (n_samples, ) The predicted target.
        """
        pass

    # protected
    def _apply_kernel_func(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Apply kernel function.

        Args:
            X (np.ndarray): (n_samples_X, n_features)
            Y (np.ndarray): (n_samples_Y, n_features) or (N, M) in origianl literature where M is number of basis vectors.

        Raises:
            ValueError: "Custom kernel function did not return 2D matrix"
            ValueError: "Custom kernel function did not return matrix with rows equal to number of data points."
            ValueError: "Kernel selection is invalid."

        Returns:
            phi_matrix (np.ndarray): (n_samples_X, n_samples_Y) or (n_samples_X, M)
        """
        phi_matrix: np.ndarray
        if self.kernel == "linear":
            phi_matrix = linear_kernel(X, Y)
        elif self.kernel == "rbf":
            self.__compute_gamma(X=X)
            phi_matrix = rbf_kernel(X, Y, self.gamma_)
        elif self.kernel == "poly":
            self.__compute_gamma(X=X)
            phi_matrix = polynomial_kernel(X, Y, self.degree, self.gamma_,
                                           self.coef0)
        elif callable(self.kernel):
            phi_matrix = self.kernel(X, Y)
            if len(phi_matrix.shape) != 2:
                raise ValueError(
                    "Custom kernel function did not return 2D matrix")
            if phi_matrix.shape[0] != X.shape[0]:
                raise ValueError(
                    "Custom kernel function did not return matrix with rows equal to number of data points."
                )
        else:
            raise ValueError("Kernel selection is invalid.")
        if self.include_bias:
            phi_matrix = np.append(phi_matrix,
                                   np.ones((phi_matrix.shape[0], 1)),
                                   axis=1)
        return phi_matrix

    def _init_weight_posterior(
            self,
            n_active_basis_vectors: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the "most probable" weight posterior statistics

        Args:
            target_hat (np.ndarray): (n_samples, ) The target hat vector.
            alpha_matrix_active (np.ndarray): (n_active_basis_vectors, n_active_basis_vectors) The current active alpha matrix.
            beta_matrix (np.ndarray): (n_samples, n_samples) The beta matrix

        Returns:
            weight_posterior_mean (np.ndarray): (n_active_basis_vectors, )The updated weight posterior mean
            weight_posterior_cov_matrix (np.ndarray): (n_active_basis_vectors, n_active_basis_vectors)
        """
        weight_posterior_mean: np.ndarray = np.zeros(
            shape=n_active_basis_vectors)
        weight_posterior_cov: np.ndarray = np.zeros(
            shape=(n_active_basis_vectors, n_active_basis_vectors))
        return weight_posterior_mean, weight_posterior_cov

    def _select_basis_vector(self, curr_iter: int, phi_matrix: np.ndarray,
                             alpha_matrix: np.ndarray,
                             beta_matrix: np.ndarray) -> int:
        """Select the largest normalized projection onto the target vector.

        Args:
            curr_iter (int)
            phi_matrix (np.ndarray): (n_samples, n_basis_vectors) or (N, M) in Tipping 2003. The complete phi matrix.
            target (np.ndarray): (n_sampels, ) the target vector.

        Returns:
            idx (int): index of the selected vector
        """
        return curr_iter % phi_matrix.shape[1]

    def _update_beta_matrix(self, X: np.ndarray, target: np.ndarray,
                            alpha_matrix_active: np.ndarray,
                            beta_matrix: np.ndarray) -> np.ndarray:
        return beta_matrix

    @property
    def _phi_active_(self) -> np.ndarray:
        """self.__phi_active is self.design_matrix_

        Raises:
            ValueError: if self.design_matrix_ is None

        Returns:
            self.design_matrix_ np.ndarray: (n_samples, n_basis_vectors_active)
        """
        phi_active: np.ndarray
        if self.design_matrix_ is not None:
            phi_active = self.design_matrix_
        else:
            raise ValueError("self.design_matrix_ is None")
        return phi_active

    @property
    def _weight_posterior_mean_(self) -> np.ndarray:
        weight_posterior_mean: np.ndarray
        if self.weight_posterior_mean_ is not None:
            weight_posterior_mean = self.weight_posterior_mean_
        else:
            raise ValueError("self.weight_posterior_mean is None")
        return weight_posterior_mean

    @property
    def _weight_posterior_cov_(self) -> np.ndarray:
        weight_posterior_cov: np.ndarray
        if self.weight_posterior_cov_ is not None:
            weight_posterior_cov = self.weight_posterior_cov_
        else:
            raise ValueError("self.weight_posterior_cov is None")
        return weight_posterior_cov

    @property
    def _sparsity_(self) -> np.ndarray:
        sparsity: np.ndarray
        if self.sparsity_ is not None:
            sparsity = self.sparsity_
        else:
            raise ValueError("self.sparsity_ is None")
        return sparsity

    @property
    def _quality_(self) -> np.ndarray:
        quality: np.ndarray
        if self.quality_ is not None:
            quality = self.quality_
        else:
            raise ValueError("self.quality_ is None")
        return quality

    # private
    def __compute_gamma(self, X: np.ndarray):
        """Compute self._gamma

        Args:
            X (np.ndarray): (n_samples, n_features) input data

        Raises:
            ValueError: When 'gamma' is a string, it should be either "scale" or "auto".
            TypeError: Argument 'gamma' should only be str or float.
        """
        n_features: int = X.shape[1]
        if isinstance(self.gamma, str):
            if self.gamma == "scale":
                x_var: float = X.var()
                self.gamma_ = 1.0 / (n_features * x_var) if x_var != 0 else 1.0
            elif self.gamma == "auto":
                self.gamma_ = 1.0 / n_features
            else:
                raise ValueError(
                    "When 'gamma' is a string, it should be either 'scale' or 'auto'. Got '{}' instead."
                    .format(self.gamma))
        elif isinstance(self.gamma, float):
            self.gamma_ = self.gamma
        else:
            raise TypeError(
                str.format(
                    "Argument 'gamma' should be of type str or float, but 'gamma' is of type {} and has value {}",
                    type(self.gamma), self.gamma))

    def __init_select_basis_vector(
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

    def __init_alpha_matrix(self, phi_matrix: np.ndarray, target: np.ndarray,
                            beta_matrix: np.ndarray) -> np.ndarray:
        """Initialize alpha matrix

        Args:
            curr_basis_idx (int): current basis vector index number
            curr_basis_vector (np.ndarray): (n_samples, 1)current basis vector
            phi_matrix (np.ndarray): (n_samples, n_basis_vectors) or (N, M) in Tipping 2003. The complete phi matrix.
            target (np.ndarray): (n_samples) the target vector.
            beta_matrix (np.ndarray): (n_samples, n_samples) or (N, N) the beta matrix with beta_i on the diagonal.

        Returns:
            alpha_matrix (np.ndarray): (n_basis_vectors, n_basis_vectors) or (M, M) in Tipping 2003. The complete alpha matrix.
        """
        curr_basis_idx, curr_basis_vector = self.__init_select_basis_vector(
            phi_matrix=phi_matrix, target=target)
        # beta_i = 1 / sigma_i**2
        alpha_vector: np.ndarray = np.full(shape=(phi_matrix.shape[1]),
                                           fill_value=np.inf)
        basis_norm: float = np.linalg.norm(curr_basis_vector, axis=0)
        # TODO have index out of bound exception
        # modify beta as instance variable
        beta: float = beta_matrix[curr_basis_idx, curr_basis_idx]
        alpha_i: float = basis_norm**2 / ((
            (curr_basis_vector.T @ target)**2 / basis_norm**2) - (1 / beta))
        alpha_vector[curr_basis_idx] = alpha_i
        alpha_matrix: np.ndarray = np.diag(v=alpha_vector)
        return alpha_matrix

    def __get_active_basis_mask(self, alpha_matrix: np.ndarray) -> np.ndarray:
        """Get active basis mask

        Args:
            alpha_matrix (np.ndarray): (n_basis_vector, n_basis_vector) or (M, M). The complete alpha matrix.

        Returns:
            active_basis_mask (np.ndarray): (n_basis_vecor) or (M,) with all elements being boolean value
        """
        alpha_vector: np.ndarray = np.diagonal(a=alpha_matrix)
        active_basis_mask: np.ndarray = (alpha_vector != np.inf)
        return active_basis_mask

    def __update_sparsity_quality(
            self, active_basis_mask: np.ndarray, phi_matrix: np.ndarray,
            beta_matrix: np.ndarray,
            target_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute sparsity_matrix and quality_matrix.

        The method first compute the sparsity and quality for all basis vector, and then using the active_basis_mask to set the self.sparsity_ and self.quality_. Both self.sparsity_ and self.quality_ should have shape of (n_active_basis_vectors, ).

        Args:
            active_basis_mask (np.ndarray): (n_basis_vecor) or (M,) with all elements being boolean value
            phi_matrix (np.ndarray): (n_samples, n_basis_vectors) or (N, M) in Tipping 2003. The complete phi matrix.
            beta_matrix (np.ndarray): (n_samples, n_samples) or (N, N) the beta matrix with beta_i on the diagonal.
            target_hat (np.ndarray): (n_samples) regression should be target and classification should be current pred.

        Returns:
            sparsity (np.ndarray): (n_basis_vectors, ) The complete sparsity vector for all basis_vector.
            quality (np.ndarray): (n_basis_vectors, ) The complete quality vector for all basis_vector.
        """
        phi_active: np.ndarray = self._phi_active_
        phi_m_tr_beta: np.ndarray = phi_matrix.T @ beta_matrix
        sigma_phi_tr: np.ndarray = self._weight_posterior_cov_ @ phi_active.T
        phi_m_tr_beta_phi_sigma_phi_tr_beta: np.ndarray = phi_m_tr_beta @ phi_active @ sigma_phi_tr @ beta_matrix
        sparsity_matrix = phi_m_tr_beta @ phi_matrix - phi_m_tr_beta_phi_sigma_phi_tr_beta @ phi_matrix
        quality_matrix = phi_m_tr_beta @ target_hat - phi_m_tr_beta_phi_sigma_phi_tr_beta @ target_hat
        sparsity = np.diagonal(sparsity_matrix)
        quality = np.diagonal(quality_matrix)
        self.sparsity_ = sparsity[active_basis_mask]
        self.quality_ = quality[active_basis_mask]
        return sparsity, quality

    # def __compute_single_basis_sparsity_quality(self, basis_vector: np.ndarray,
    #                                             beta_matrix: np.ndarray,
    #                                             target_hat: np.ndarray):
    #     """Compute sparsity_matrix and quality_matrix

    #     Args:
    #         basis_vector (np.ndarray): (n_samples, 1) The single selected basis vector to be evaluated.
    #         beta_matrix (np.ndarray): (n_samples, n_samples) or (N, N) the beta matrix with beta_i on the diagonal.
    #         target_hat (np.ndarray): (n_samples) regression should be target and classification should be current pred.

    #     Returns:
    #         sparsity (float): The sparsity metric of the passed in basis_vector.
    #         quality (float): The quality metric of the passed in basis_vector.
    #     """
    #     sigma: np.ndarray = self._weight_posterior_cov_
    #     phi: np.ndarray = self._phi_active_
    #     phi_m: np.ndarray = basis_vector
    #     phi_m_tr: np.ndarray = phi_m.T
    #     sigma_phi_tr: np.ndarray = sigma @ phi.T
    #     phi_m_tr_beta: np.ndarray = phi_m_tr @ beta_matrix
    #     phi_m_tr_beta_phi_sigma_phi_tr_beta: np.ndarray = phi_m_tr_beta @ phi @ sigma_phi_tr @ beta_matrix
    #     sparsity: float = phi_m_tr_beta @ phi_m - phi_m_tr_beta_phi_sigma_phi_tr_beta @ phi_m
    #     quality: float = phi_m_tr_beta @ target_hat - phi_m_tr_beta_phi_sigma_phi_tr_beta @ target_hat
    #     return sparsity, quality

    def __prune(self, curr_basis_idx: int, phi_matrix: np.ndarray,
                alpha_matrix: np.ndarray, sparsity: np.ndarray,
                quality: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prune design matrix and update alpha matrix according to sparsity and quality analysis.

        The method updates the self.__design_matrix, modifies alpha_matrix in place, and then returns the new active_basis_mask.

        Args:
            curr_basis_idx (int): The index of the currently selected basis vector.
            phi_matrix (np.ndarray): (n_samples, n_basis_vectors) The complete phi matrix.
            alpha_matrix (np.ndarray): (n_basis_vectors, n_basis_vectors) The complete alpha_matrix.
            sparsity (np.ndarray): (n_basis_vectors, ) The sparsity of all basis vectors.
            quality (np.ndarray): (n_basis_vectors, ) The quality  of all basis vectors

        Returns:
            active_basis_mask (np.ndarray): (n_basis_vectors, ) The updated active basis mask.
            alpha_matrix_active (np.ndarray): (n_active_basis_vectors, ) The updated active alpha matrix.
        """
        curr_basis_sparsity: float = sparsity[curr_basis_idx]
        curr_basis_quality: float = quality[curr_basis_idx]
        curr_theta: float = curr_basis_sparsity**2 - curr_basis_quality
        if curr_theta > 0:
            curr_alpha: float = curr_basis_sparsity**2 / (
                curr_basis_quality**2 - curr_basis_sparsity)
            alpha_matrix[curr_basis_idx, curr_basis_idx] = curr_alpha
        else:
            alpha_matrix[curr_basis_idx, curr_basis_idx] = np.inf
        active_basis_mask: np.ndarray = self.__get_active_basis_mask(
            alpha_matrix=alpha_matrix)
        alpha_matrix_active: np.ndarray = alpha_matrix[:, active_basis_mask][
            active_basis_mask, :]
        self.design_matrix_ = phi_matrix[:, active_basis_mask]
        return active_basis_mask, alpha_matrix_active

    def __has_converged(self, curr_alpha_matrix: np.ndarray,
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
        return True if diff <= self.tol else False
