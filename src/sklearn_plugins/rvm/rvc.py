import gc
from typing import Callable, List, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from ..kernels.kernel_base import KernelBase
from ..kernels.rbf import RBFKernel
from ._binary_rvc import _BinaryRVC


class RVC(ClassifierMixin, BaseEstimator):

    _kernel_func: Callable[[np.ndarray, np.ndarray], np.ndarray]
    _include_bias: bool
    _tol: float
    _max_iter: Union[int, None]
    _verbose: bool

    _binary_rvc_list_: Union[List[_BinaryRVC], None]
    _n_classes_: Union[int, None]

    def __init__(self,
                 kernel_func: Union[KernelBase,
                                    Callable[[np.ndarray, np.ndarray],
                                             np.ndarray]] = RBFKernel(),
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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RVC":
        """Fit the model with the given input data and y

        Args:
            X (np.ndarray): (n_sampels, n_features) training data.
            y (np.ndarray): (n_samples, ) target values.

        Raises:
            ValueError: "y contains classes that has no samples"

        Returns:
            RVC: this instance.
        """
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
            gc.collect()
            self._binary_rvc_list_.append(curr_rvc)
            if self._n_classes_ == 2:
                break
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for the input X.

        Args:
            X (np.ndarray): (n_samples, n_features) Samples to be predicted.

        Returns:
            pred (np.ndarray): (n_samples) The class labels of each samples.
        """
        prob: np.ndarray = self.predict_proba(X)
        pred: np.ndarray = np.argmax(prob, axis=1)
        return pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Estimate probability for all samples in input X.

        Args:
            X (np.ndarray): (n_samples, n_features) Samples to be predicted.

        Returns:
            prob (np.ndarray): (n_samples, n_classes) The probabily of the samples for each class.
        """
        n_samples: int = X.shape[0]
        if self._n_classes_ == 2:
            pos_prob: np.ndarray = self.binary_rvc_list_[0].predict_proba(X)
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
    def binary_rvc_list_(self) -> List[_BinaryRVC]:
        if self._binary_rvc_list_ is not None:
            return self._binary_rvc_list_
        raise ValueError("self._binary_rv_list_ is None")

    @property
    def n_classes_(self) -> int:
        if self._n_classes_ is not None:
            return self._n_classes_
        raise ValueError("self._n_classes_ is None")


from skl2onnx import update_registered_converter

from ._onnx_transfrom import rvc_converter, rvc_shape_calculator

update_registered_converter(RVC,
                            "SklearnPluginsRVC",
                            shape_fct=rvc_shape_calculator,
                            convert_fct=rvc_converter)
