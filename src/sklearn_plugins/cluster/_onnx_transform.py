from typing import List

from onnxconverter_common.data_types import (DataType, DoubleTensorType,
                                             FloatTensorType, Int64TensorType)
from skl2onnx import get_model_alias
from skl2onnx.algebra.onnx_operator import OnnxSubOperator
from skl2onnx.algebra.onnx_ops import OnnxIdentity
from skl2onnx.common._container import ModelComponentContainer
from skl2onnx.common._topology import Operator, Scope, Variable
from skl2onnx.common.utils import check_input_and_output_types
from sklearn_plugins.cluster import SphericalKMeans


def _spherical_kmeans_shape_calculator(operator: Operator):
    """Calculate the input and output shape for SphericalKMeans.

    Args:
        operator (Operator): An Operator container.
    """
    check_input_and_output_types(
        operator,
        good_input_types=[Int64TensorType, FloatTensorType, DoubleTensorType],
        good_output_types=[Int64TensorType, FloatTensorType, DoubleTensorType])
    op_inputs: List[Variable] = operator.inputs
    if len(op_inputs) != 1:
        raise RuntimeError(
            "Only one input matrix is allowed for SphericalKMeans.")
    op_outputs: List[Variable] = operator.outputs
    if len(op_outputs) != 2:
        raise RuntimeError("Two outputs are expected for SphericalKMeans.")
    skm_op: SphericalKMeans = operator.raw_operator
    # retrieve skm inputs dtype
    input_var_type: DataType = op_inputs[0].type
    # retrieve skm inputs outputs shape
    n_samples: int = input_var_type.shape[0]
    n_clusters: int = skm_op.n_clusters
    # type alias
    InputVarDtype = input_var_type.__class__
    # output[0] = skm_op.fit_predict(X)
    op_outputs[0].type = InputVarDtype(shape=[n_samples])
    # output[1] = skm_op.fit_transform(X)
    op_outputs[1].type = InputVarDtype(shape=[n_samples, n_clusters])
    # output[2] = skm_op.score(X)
    op_outputs[2].type = InputVarDtype(shape=[n_samples])


def _spherical_kmeans_converter(scope: Scope, operator: Operator,
                                container: ModelComponentContainer):
    skm_op: SphericalKMeans = operator.raw_operator
    op_version = container.target_opset
    out = operator.outputs

    # We retrieve the unique input.
    X: Variable = operator.inputs[0]

    # We tell in ONNX language how to compute the unique output.
    # op_version=opv tells which opset is requested
    subop = OnnxSubOperator(skm_op.pca_, X, op_version=op_version)
    Y = OnnxIdentity(subop, op_version=op_version, output_names=out[:1])
    Y.add_to(scope, container)
    pass


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
