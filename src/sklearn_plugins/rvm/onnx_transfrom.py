"""Convert to onnx.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Type, Union

import numpy as np
from onnx.helper import make_tensor
from onnx.onnx_ml_pb2 import TensorProto
from onnxconverter_common.data_types import (DataType, DoubleTensorType,
                                             FloatTensorType, TensorType)
from overrides.overrides import overrides
from skl2onnx.algebra.onnx_operator import OnnxOperator
from skl2onnx.algebra.onnx_ops import (OnnxAdd, OnnxConstantOfShape, OnnxExp,
                                       OnnxMatMul, OnnxMul, OnnxPad,
                                       OnnxReshape, OnnxSum)
from skl2onnx.common._container import ModelComponentContainer
from skl2onnx.common._topology import Operator, Scope, Variable
from skl2onnx.common.data_types import guess_numpy_type
from skl2onnx.common.utils import check_input_and_output_types

from ._binary_rvc import _BinaryRVC
from .rvc import RVC
from .rvr import RVR

__author__ = "Hung-Tien Huang"
__copyright__ = "Copyright 2021, Hung-Tien Huang"


def rvr_shape_calculator(operator: Operator):
    check_input_and_output_types(
        operator,
        good_input_types=[FloatTensorType, DoubleTensorType],
        good_output_types=[FloatTensorType, DoubleTensorType])
    op_inputs: List[Variable] = operator.inputs
    if len(op_inputs) != 1:
        raise RuntimeError("Only one input matrix is allowed for RVR.")
    op_outputs: List[Variable] = operator.outputs
    if len(op_outputs) != 1:
        raise RuntimeError("Only one output is allowed for RVR.")
    # if len(op_outputs) != 2:
    #     raise RuntimeError("Only two outputs are allowed for RVR.")
    # retrieve rvr inputs dtype
    input_var_type: DataType = op_inputs[0].type
    # confirm rvr input and output shape
    n_samples: int = input_var_type.shape[0]
    # outputs[0], outputs[1] = rvr.predict_var(X)
    op_outputs[0].type.shape = [n_samples]
    op_outputs[0].onnx_name = "y"
    # op_outputs[1].type.shape = [n_samples]
    # op_outputs[1].onnx_name = "y_var"


class KernelFunction(ABC):
    # __TensorType_: Union[Type[TensorType], None]
    # __NumPyType_: Union[Type, None]

    def __init__(self) -> None:
        super().__init__()
        # self.__TensorType_ = None
        # self.__NumPyType_ = None

    @abstractmethod
    def __call__(self, X: Variable, X_prime: np.ndarray,
                 op_version: Union[int, None]) -> OnnxOperator:
        pass

    # @property
    # def TensorType_(self):
    #     if self.__TensorType_ is not None:
    #         return self.__TensorType_
    #     raise ValueError("self.__TensorType is None")

    # @TensorType_.setter
    # def TensorType_(self, type: Type[TensorType]):
    #     self.__TensorType_ = type

    # @property
    # def NumpyType_(self):
    #     if self.__NumPyType_ is not None:
    #         return self.__NumPyType_
    #     raise ValueError("self.__NumPyType_ is None")

    # @NumpyType_.setter
    # def NumpyType_(self, type: Type):
    #     self.__NumPyType_ = type


class RBFKernelFunction(KernelFunction):
    __gamma: Union[float, None]

    def __init__(self, gamma: Optional[float] = None) -> None:
        self.__gamma = gamma

    # TODO error
    def __compute_dist(self, X: Variable, X_prime: np.ndarray,
                       op_version: Union[int, None]) -> OnnxOperator:
        """
            For efficiency reasons, the euclidean distance between a pair of row
            vector x and y is computed as::

                dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
        """
        input_type: Union[FloatTensorType, DoubleTensorType] = X.type
        n_features: int = input_type.shape[1]
        NumPyType: Type = guess_numpy_type(data_type=input_type)
        x_squared: OnnxOperator = OnnxMul(X, X, op_version=op_version)
        ones_value: TensorProto = make_tensor(
            "const_of_shape_value",
            data_type=input_type._get_element_onnx_type(),
            dims=[1],
            vals=[1.0])
        ones: OnnxOperator = OnnxConstantOfShape(np.array([n_features],
                                                          dtype=int),
                                                 value=ones_value,
                                                 op_version=op_version)
        x_squared = OnnxMatMul(x_squared, ones, op_version=op_version)
        x_squared = OnnxReshape(x_squared,
                                np.array([-1, 1], dtype=np.int64),
                                op_version=op_version)
        y_squared: np.ndarray = X_prime**2
        y_squared = np.sum(y_squared, axis=1, dtype=NumPyType)
        y_squared = np.reshape(y_squared, newshape=(1, -1))
        neg_two_x_dot_y: OnnxOperator = OnnxMatMul(
            X, (-2 * X_prime.T).astype(NumPyType), op_version=op_version)
        dist: OnnxOperator = OnnxAdd(x_squared,
                                     y_squared,
                                     op_version=op_version)
        dist = OnnxSum(dist, neg_two_x_dot_y, op_version=op_version)
        return dist

    @overrides
    def __call__(self, X: Variable, X_prime: np.ndarray,
                 op_version: Union[int, None]) -> OnnxOperator:

        gamma: float
        if self.__gamma is None:
            gamma = 1.0 / X.type.shape[1]
        else:
            gamma = self.__gamma
        NumPyType: Type = guess_numpy_type(X.type)
        dist: OnnxOperator = self.__compute_dist(X, X_prime, op_version)
        neg_gamma_dist: OnnxOperator = OnnxMul(np.array(-1.0 * gamma,
                                                        dtype=NumPyType),
                                               dist,
                                               op_version=op_version)
        rbf: OnnxOperator = OnnxExp(neg_gamma_dist, op_version=op_version)
        return rbf


def rvr_converter(scope: Scope, operator: Operator,
                  container: ModelComponentContainer,
                  kernel_func: KernelFunction):
    rvr: RVR = operator.raw_operator
    input: Variable = operator.inputs[0]
    NumPyType: Type = guess_numpy_type(input.type)
    op_version: Union[int, None] = container.target_opset
    op_outputs: List[Variable] = operator.outputs
    phi_matrix: OnnxOperator = kernel_func(
        input, rvr.relevance_vectors_.astype(NumPyType), op_version)
    if rvr.include_bias == True:
        phi_matrix = OnnxPad(phi_matrix,
                             np.array([0, 0, 0, 1]),
                             np.array(1.0).astype(NumPyType),
                             op_version=op_version)
    y: OnnxOperator = OnnxMatMul(phi_matrix,
                                 rvr.weight_posterior_mean_.astype(NumPyType),
                                 op_version=op_version,
                                 output_names=[op_outputs[0]])
    y.add_to(scope=scope, container=container)
