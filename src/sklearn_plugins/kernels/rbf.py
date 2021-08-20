from typing import Optional, Type, Union

import numpy as np
from overrides.overrides import overrides
from skl2onnx.algebra.onnx_operator import OnnxOperator
from skl2onnx.algebra.onnx_ops import OnnxExp, OnnxMul
from skl2onnx.common._topology import Variable
from skl2onnx.common.data_types import guess_numpy_type
from sklearn.metrics.pairwise import rbf_kernel

from .kernel_base import KernelBase
from .utils import compute_dist_onnx


class RBFKernel(KernelBase):
    __gamma: Union[float, None]
    gamma_: Union[float, None]

    def __init__(self, gamma: Optional[float] = None) -> None:
        self.__gamma = gamma
        self.gamma_ = gamma

    @overrides
    def __call__(self,
                 X: np.ndarray,
                 Y: np.ndarray,
                 fit: bool = False) -> np.ndarray:
        if fit == True and self.__gamma is None:
            self.gamma_ = 1.0 / X.shape[1]
        return rbf_kernel(X=X, Y=Y, gamma=self.gamma_)

    @overrides
    def convert_onnx(self, X: Variable, X_prime: np.ndarray,
                     op_version: Union[int, None]) -> OnnxOperator:
        if self.gamma_ is None:
            raise ValueError("RBF gamma parameter is not fitted.")
        NumPyType: Type = guess_numpy_type(X.type)
        dist: OnnxOperator = compute_dist_onnx(X, X_prime, op_version)
        neg_gamma_dist: OnnxOperator = OnnxMul(np.array(-1.0 * self.gamma_,
                                                        dtype=NumPyType),
                                               dist,
                                               op_version=op_version)
        rbf: OnnxOperator = OnnxExp(neg_gamma_dist, op_version=op_version)
        return rbf
