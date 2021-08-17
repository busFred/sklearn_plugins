from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from overrides.overrides import overrides
from scipy.special import expit as sigmoid
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import rbf_kernel

from .rvm import BaseRVM


class RVC(ClassifierMixin, BaseEstimator):
    class __BinaryRVC(BaseRVM):
        def __init__(self, kernel_func: Callable[[np.ndarray, np.ndarray],
                                                 np.ndarray],
                     include_bias: bool, tol: float, max_iter: Optional[int],
                     verbose: bool) -> None:
            super().__init__(kernel_func=kernel_func,
                             include_bias=include_bias,
                             tol=tol,
                             max_iter=max_iter,
                             verbose=verbose)

        @overrides
        def predict(self, X: np.ndarray) -> np.ndarray:
            phi_matrix: np.ndarray = self._compute_phi_matrix(
                X=X, X_prime=self._X_prime)
            pred: np.ndarray = self._predict_phi_matrix(
                active_phi_matrix=phi_matrix)
            return pred

        def predict_y(self, X: np.ndarray) -> np.ndarray:
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
            prob: np.ndarray = self._predict_phi_matrix(
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
            y = self._predict_phi_matrix(active_phi_matrix=active_phi_matrix)
            beta_inv: np.ndarray = np.linalg.inv(beta_matrix)
            target_hat: np.ndarray = active_phi_matrix @ self._mu + beta_inv @ (
                t - y)
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

        def _predict_phi_matrix(self,
                                active_phi_matrix: np.ndarray) -> np.ndarray:
            """Predict training dataset with current active_phi_matrix.

            Args:
                active_phi_matrix (np.ndarray): (n_samples, n_active_basis) The current active phi matrix.

            Returns:
                pred (np.ndarray): (n_samples, ) The prediction of training dataset given current active phi matrix.
            """
            y: np.ndarray = active_phi_matrix @ self._mu
            pred: np.ndarray = sigmoid(y)
            return pred

    _kernel_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    _include_bias: bool
    _tol: float
    _max_iter: Union[int, None]
    _verbose: bool

    _binary_rvc_list_: Union[List[__BinaryRVC], None]
    _n_classes_: Union[int, None]

    # _relevance_vectors_: Union[np.ndarray, None]  # aka self._X_prime
    # _weight_posterior_mean_: Union[np.ndarray, None]  # aka self._mu
    # _weight_posterior_cov_: Union[np.ndarray, None]  # aka self._sigma

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

        self._binary_rvc_list_ = None
        self._n_classes_ = None
        # self._relevance_vectors_ = None
        # self._weight_posterior_mean_ = None
        # self._weight_posterior_cov_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RVC":
        if np.any(np.bincount(y) == 0):
            raise ValueError("y contains class that has no sample")
        target: np.ndarray = y.astype(int)
        unique_classes: np.ndarray = np.unique(y).astype(int)
        self._n_classes_ = len(unique_classes)
        self._binary_rvc_list_ = list()
        for curr_class in unique_classes:
            curr_target: np.ndarray = np.where(target == curr_class, 1, 0)
            curr_rvc: _BinaryRVC = _BinaryRVC(kernel_func=self._kernel_func,
                                              include_bias=self._include_bias,
                                              tol=self._tol,
                                              max_iter=self._max_iter,
                                              verbose=self._verbose)
            curr_rvc.fit(X=X, y=curr_target)
            self._binary_rvc_list_.append(curr_rvc)
            if self._n_classes_ == 2:
                break
        return self

    @overrides
    def predict(self, X: np.ndarray) -> np.ndarray:
        prob: np.ndarray = self.predict_proba(X)
        pred: np.ndarray = np.argmax(prob, axis=1)
        return pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        n_samples: int = X.shape[0]
        if self._n_classes_ == 2:
            pos_prob: np.ndarray = self.binary_rvc_list_[0].predict(X)
            prob: np.ndarray = np.zeros(shape=(n_samples, 2))
            prob[:, 0] = pos_prob
            prob[:, 1] = 1 - pos_prob
            return prob
        y: np.ndarray = np.zeros(shape=(n_samples, self.n_classes_))
        for curr_class in range(self.n_classes_):
            curr_rvc: _BinaryRVC = self.binary_rvc_list_[curr_class]
            y[:, curr_class] = curr_rvc.predict_y(X)
        y = np.exp(y)
        denominator: np.ndarray = np.sum(y, axis=1, keepdims=True)
        prob: np.ndarray = y / denominator
        return prob

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
    def binary_rvc_list_(self) -> List[__BinaryRVC]:
        if self._binary_rvc_list_ is not None:
            return self._binary_rvc_list_
        raise ValueError("self._binary_rv_list_ is None")

    @property
    def n_classes_(self) -> int:
        if self._n_classes_ is not None:
            return self._n_classes_
        raise ValueError("self._n_classes_ is None")

    # @property
    # def relevance_vectors_(self) -> np.ndarray:
    #     if self._relevance_vectors_ is not None:
    #         return self._relevance_vectors_.copy()
    #     else:
    #         raise ValueError("self._relevance_vectors_ is None")

    # @property
    # def weight_posterior_mean_(self) -> np.ndarray:
    #     if self._weight_posterior_mean_ is not None:
    #         return self._weight_posterior_mean_.copy()
    #     else:
    #         raise ValueError("self._weight_posterior_mean_ is None")

    # @property
    # def weight_posterior_cov_(self):
    #     if self._weight_posterior_cov_ is not None:
    #         return self._weight_posterior_cov_.copy()
    #     else:
    #         raise ValueError("self._weight_posterior_cov_ is None")
