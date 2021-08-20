from typing import Type, Union

import numpy as np
from onnx.helper import make_tensor
from onnx.onnx_ml_pb2 import TensorProto
from onnxconverter_common.data_types import DoubleTensorType, FloatTensorType
from skl2onnx.algebra.onnx_operator import OnnxOperator
from skl2onnx.algebra.onnx_ops import (OnnxAdd, OnnxConstantOfShape,
                                       OnnxMatMul, OnnxMul, OnnxReshape,
                                       OnnxSum)
from skl2onnx.common._topology import Variable
from skl2onnx.common.data_types import guess_numpy_type


def compute_dist_onnx(X: Variable, Y: np.ndarray,
                      op_version: Union[int, None]) -> OnnxOperator:
    """Compute distance between an ONNX variabel and an np.ndarray.

    For efficiency reasons, the euclidean distance between a pair of row
    vector x and y is computed as::

        dist(x, y) = sqrt(dot(x, x) - 2 * dot(x, y) + dot(y, y))

    Args:
        X (Variable): (n_samples_X, n_features) The first matrix to compute.
        Y (np.ndarray): (n_samples_Y, n_features) The second matrix to compute.

    Return:
        OnnxOperator: (n_samples_X, n_samples_Y) The operator that computes the distance between X and Y matrix.
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
    ones: OnnxOperator = OnnxConstantOfShape(np.array([n_features], dtype=int),
                                             value=ones_value,
                                             op_version=op_version)
    x_squared = OnnxMatMul(x_squared, ones, op_version=op_version)
    x_squared = OnnxReshape(x_squared,
                            np.array([-1, 1], dtype=np.int64),
                            op_version=op_version)
    y_squared: np.ndarray = Y**2
    y_squared = np.sum(y_squared, axis=1, dtype=NumPyType)
    y_squared = np.reshape(y_squared, newshape=(1, -1))
    neg_two_x_dot_y: OnnxOperator = OnnxMatMul(X, (-2 * Y.T).astype(NumPyType),
                                               op_version=op_version)
    dist: OnnxOperator = OnnxAdd(x_squared, y_squared, op_version=op_version)
    dist = OnnxSum(dist, neg_two_x_dot_y, op_version=op_version)
    return dist
