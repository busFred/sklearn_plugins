"""Convert to onnx.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Type, Union

import numpy as np
from onnxconverter_common.data_types import (DataType, DoubleTensorType,
                                             FloatTensorType, Int64TensorType)
from overrides.overrides import overrides
from skl2onnx.algebra.onnx_operator import OnnxOperator, OnnxSubEstimator
from skl2onnx.algebra.onnx_ops import (OnnxArgMax, OnnxExp, OnnxMatMul,
                                       OnnxMul, OnnxSum, OnnxTranspose)
from skl2onnx.common._container import ModelComponentContainer
from skl2onnx.common._topology import Operator, Scope, Variable
from skl2onnx.common.data_types import guess_numpy_type
from skl2onnx.common.utils import check_input_and_output_types
from sklearn.preprocessing import Normalizer

from ._binary_rvc import _BinaryRVC
from .rvc import RVC
from .rvr import RVR

__author__ = "Hung-Tien Huang"
__copyright__ = "Copyright 2021, Hung-Tien Huang"


class KernelFunction(ABC):
    @abstractmethod
    def __call__(self, X, X_prime):
        pass


class RBFKernelFunction(KernelFunction):
    __gamma: Union[float, None]

    def __init__(self, gamma: Optional[float]) -> None:
        self.__gamma = gamma

    def __compute_dist(self, X: Variable, X_prime: np.ndarray) -> OnnxOperator:
        """
            For efficiency reasons, the euclidean distance between a pair of row
            vector x and y is computed as::

                dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))
        """
        Y: np.ndarray = X_prime
        X_tr: OnnxOperator = OnnxTranspose(X)
        Y_tr: OnnxOperator = OnnxTranspose(Y)
        X_tr_X: OnnxOperator = OnnxMatMul(X_tr, X)
        Y_tr_Y: OnnxOperator = OnnxMatMul(Y_tr, Y)
        X_tr_Y: OnnxOperator = OnnxMatMul(X_tr, Y)
        two_X_tr_Y: OnnxOperator = OnnxMul(2.0, X_tr_Y)
        dist: OnnxOperator = OnnxSum([X_tr_X, two_X_tr_Y, Y_tr_Y])
        return dist

    @overrides
    def __call__(self, X: Variable, X_prime: np.ndarray):
        gamma: float
        if self.__gamma is None:
            gamma = 1.0 / X.type.shape[1]
        else:
            gamma = self.__gamma
        dist: OnnxOperator = self.__compute_dist(X, X_prime)
        neg_gamma_dist: OnnxOperator = OnnxMul(-1.0 * gamma, dist)
        rbf: OnnxOperator = OnnxExp(neg_gamma_dist)
        return rbf


def rvr_shape_calculator(operator: Operator):
    check_input_and_output_types(
        operator,
        good_input_types=[FloatTensorType, DoubleTensorType],
        good_output_types=[FloatTensorType, DoubleTensorType])
    op_inputs: List[Variable] = operator.inputs
    if len(op_inputs) != 1:
        raise RuntimeError("Only one input matrix is allowed for RVR.")
    op_outputs: List[Variable] = operator.outputs
    if len(op_outputs) != 2:
        raise RuntimeError("Only two outputs are allowed for RVR.")
    # retrieve rvr inputs dtype
    input_var_type: DataType = op_inputs[0].type
    # confirm rvr input and output shape
    n_samples: int = input_var_type.shape[0]
    # outputs[0], outputs[1] = rvr.predict_var(X)
    op_outputs[0].type.shape = [n_samples]
    op_outputs[0].onnx_name = "y"
    op_outputs[1].type.shape = [n_samples]
    op_outputs[1].onnx_name = "y_var"


def rvr_converter(scope: Scope, operator: Operator,
                  container: ModelComponentContainer):

    pass


# def spherical_kmeans_converter(scope: Scope, operator: Operator,
#                                container: ModelComponentContainer):
#     """The ONNX converter for sklearn_plugins.cluster.SphericalKMeans.

#     Original SphericalKMeans Implementation:
#     ```
#     # preprocess input
#     if skm.normalize:
#         X = normalize(X, norm="l2", axis=1, copy=skm.copy)
#     # each features in the dataset has zero mean and unit variance; dataset level
#     if skm.standarize:
#         X = skm.__std_scalar_.transform(X, copy=copy)
#     # PCA whiten
#     X = skm.__pca_.transform(X)
#     # calculate projection
#     proj: np.ndarray = np.matmul(X, centroids)
#     # calculate lables
#     labels: np.ndarray = np.argmax(S_proj, axis=1)
#     ```

#     Args:
#         scope (Scope): [description]
#         operator (Operator): [description]
#         container (ModelComponentContainer): [description]
#     """
#     skm: SphericalKMeans = operator.raw_operator
#     op_version: Union[int, None] = container.target_opset
#     op_outputs: List[Variable] = operator.outputs
#     # retreive input
#     input: Variable = operator.inputs[0]
#     np_dtype = guess_numpy_type(input.type)
#     # normalize input
#     normalize_op: Union[OnnxOperator, Variable] = input
#     if skm.normalize == True:
#         normalize_op = OnnxSubEstimator(Normalizer(norm="l2"),
#                                         normalize_op,
#                                         op_versionrow_norms=op_version)
#     # standardize input
#     std_scalar_op: Union[OnnxOperator, Variable] = normalize_op
#     if skm.standardize == True:
#         std_scalar_op = OnnxSubEstimator(skm.std_scalar_,
#                                          normalize_op,
#                                          op_versionrow_norms=op_version)
#     # pca whitening
#     pca_op: OnnxOperator = OnnxSubEstimator(skm.pca_,
#                                             std_scalar_op,
#                                             op_version=op_version)
#     # calculate projection
#     proj_op: OnnxOperator = OnnxMatMul(pca_op,
#                                        skm.centroids_.astype(np_dtype),
#                                        op_version=op_version,
#                                        output_names=[op_outputs[1]])
#     proj_op.add_to(scope=scope, container=container)
#     labels_op: OnnxOperator = OnnxArgMax(proj_op,
#                                          op_version=op_version,
#                                          output_names=[op_outputs[0]],
#                                          axis=1,
#                                          keepdims=0)
#     labels_op.add_to(scope=scope, container=container)

# TODO implement parser to export intermediate steps.
# def _spherical_kmeans_parser(scope: Scope,
#                              model: SphericalKMeans,
#                              inputs: List[Variable],
#                              custom_parsers=None):
#     alias = get_model_alias(type(model))
#     this_op = scope.declare_local_operator(alias, model)

#     # inputs
#     this_op.inputs.append(inputs[0])

#     # outputs
#     cls_type = inputs[0].type.__class__
#     val_y1 = scope.declare_local_variable('nogemm', cls_type())
#     val_y2 = scope.declare_local_variable('gemm', cls_type())
#     this_op.outputs.append(val_y1)
#     this_op.outputs.append(val_y2)

#     # ends
#     return this_op.outputs
