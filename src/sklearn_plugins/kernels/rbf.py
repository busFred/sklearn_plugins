from typing import Optional, Type, Union

import numpy as np
from overrides.overrides import overrides
from skl2onnx.algebra.onnx_operator import OnnxOperator
from skl2onnx.algebra.onnx_ops import OnnxExp, OnnxMul
from skl2onnx.common._topology import Variable
from skl2onnx.common.data_types import guess_numpy_type
from sklearn.metrics.pairwise import rbf_kernel

from .kernel_base import KernelFunction
from .utils import compute_dist_onnx


class RBFKernel(KernelFunction):
    __gamma: Union[float, None]

    def __init__(self, gamma: Optional[float] = None) -> None:
        self.__gamma = gamma

    @overrides
    def __call__(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return rbf_kernel(X=X, Y=Y, gamma=self.__gamma)

    @overrides
    def convert_onnx(self, X: Variable, X_prime: np.ndarray,
                     op_version: Union[int, None]) -> OnnxOperator:
        gamma: float
        if self.__gamma is None:
            gamma = 1.0 / X.type.shape[1]
        else:
            gamma = self.__gamma
        NumPyType: Type = guess_numpy_type(X.type)
        dist: OnnxOperator = compute_dist_onnx(X, X_prime, op_version)
        neg_gamma_dist: OnnxOperator = OnnxMul(np.array(-1.0 * gamma,
                                                        dtype=NumPyType),
                                               dist,
                                               op_version=op_version)
        rbf: OnnxOperator = OnnxExp(neg_gamma_dist, op_version=op_version)
        return rbf
