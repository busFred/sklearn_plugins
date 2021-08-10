from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
from numpy.core.fromnumeric import shape
from numpy.random import RandomState
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import (linear_kernel, rbf_kernel,
                                      polynomial_kernel)


class BaseRVM(BaseEstimator, ABC):
    """BaseRVM

    Args:
        kernel (Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]], optional): The kernel function to be used. Defaults to 'rbf'.
        degree (int, optional): Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels. Defaults to 3.
        gamma (Union[str, float], optional): Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. Defaults to 'scale'.
        coef0 (float, optional): Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid’. Defaults to 0.0.
        include_bias (bool, optional): include bias in the polynomial. Defaults to True.
        tol (float, optional): tolerance for stopping criterion. Defaults to 1e-3.
        verbose (bool, optional): [description]. Defaults to False.
        max_iter (int, optional): [description]. Defaults to -1.
        random_state (Optional[Union[int, RandomState]], optional): [description]. Defaults to None.
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

    @abstractmethod
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> "BaseRVM":
        phi: np.ndarray = self._apply_kernel_func(X=X,
                                                  Y=X)  # the design matrix
        beta: np.ndarray = self._compute_beta_matrix(phi=phi, target=y)
        alpha_matrix: np.ndarray = self._init_alpha_matrix(phi=phi,
                                                           target=y,
                                                           beta=beta)
        return self

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        # from setp 2 to step 11
        # regression override predict method and initialize target precision sigma_squared (precision of y)
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

    # protected
    def _compute_gamma(self, X: np.ndarray):
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
            phi (np.ndarray): (n_samples_X, n_samples_Y) or (n_samples_X, M)
        """
        phi: np.ndarray
        if self.kernel == "linear":
            phi = linear_kernel(X, Y)
        elif self.kernel == "rbf":
            self._compute_gamma(X=X)
            phi = rbf_kernel(X, Y, self.gamma_)
        elif self.kernel == "poly":
            self._compute_gamma(X=X)
            phi = polynomial_kernel(X, Y, self.degree, self.gamma_, self.coef0)
        elif callable(self.kernel):
            phi = self.kernel(X, Y)
            if len(phi.shape) != 2:
                raise ValueError(
                    "Custom kernel function did not return 2D matrix")
            if phi.shape[0] != X.shape[0]:
                raise ValueError(
                    "Custom kernel function did not return matrix with rows equal to number of data points."
                )
        else:
            raise ValueError("Kernel selection is invalid.")
        if self.include_bias:
            phi = np.append(phi, np.ones((phi.shape[0], 1)), axis=1)
        return phi

    def _select_basis_vector(self, phi: np.ndarray,
                             target: np.ndarray) -> Tuple[int, np.ndarray]:
        """Select the largest normalized projection onto the target vector.

        Args:
            phi (np.ndarray): (n_samples, n_basis_vectors) or (N, M) in Tipping 2003.
            target (np.ndarray): (n_sampels, ) the target vector.

        Returns:
            idx (int): index of the selected vector
            phi[:, idx] (np.ndarray): (n_samples, 1) the selected basis vector
        """
        proj: np.ndarray = np.matmul(phi.T, target)
        phi_norm: np.ndarray = np.linalg.norm(phi, axis=1)
        proj_norm: np.ndarray = np.divide(proj, phi_norm)
        idx: int = np.argmax(proj_norm)
        return idx, phi[:, idx]

    @abstractmethod
    def _compute_beta_matrix(self, phi: np.ndarray,
                             target: np.ndarray) -> np.ndarray:
        pass

    def _init_alpha_matrix(self, phi: np.ndarray, target: np.ndarray,
                           beta_matrix: np.ndarray) -> np.ndarray:
        """Initialize alpha matrix

        Args:
            phi (np.ndarray): (n_samples, n_basis_vectors) or (N, M) in Tipping 2003.
            target (np.ndarray): (n_samples) the target vector.
            beta_matrix (np.ndarray): (n_samples, n_samples) or (N, N) the beta matrix with beta_i on the diagonal.

        Returns:
            alpha_matrix (np.ndarray): (n_basis_vectors, n_basis_vectors) or (M, M) in Tipping 2003. the alpha matrix
        """
        # beta_i = 1 / sigma_i**2
        idx, basis_vector = self._select_basis_vector(phi=phi, target=target)
        alpha_vector: np.ndarray = np.full(shape=(phi.shape[1]),
                                           fill_value=np.inf)
        basis_norm: float = np.linalg.norm(basis_vector, axis=0)
        beta: float = beta_matrix[idx, idx]
        alpha_i: float = basis_norm**2 / (
            (np.matmul(x1=basis_vector.T, x2=target)**2 / basis_norm**2) -
            (1 / beta))
        alpha_vector[idx] = alpha_i
        alpha_matrix: np.ndarray = np.diag(v=alpha_vector)
        return alpha_matrix
