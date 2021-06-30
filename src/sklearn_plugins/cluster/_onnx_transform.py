from typing import List, Union

from onnxconverter_common.data_types import (DataType, DoubleTensorType,
                                             FloatTensorType, Int64TensorType)
from skl2onnx import get_model_alias
from skl2onnx.algebra.onnx_operator import OnnxOperator, OnnxSubOperator
from skl2onnx.algebra.onnx_ops import (OnnxArgMax, OnnxIdentity, OnnxMatMul,
                                       OnnxNormalizer)
from skl2onnx.common._container import ModelComponentContainer
from skl2onnx.common._topology import Operator, Scope, Variable
from skl2onnx.common.data_types import guess_numpy_type
from skl2onnx.common.utils import check_input_and_output_types
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans


def spherical_kmeans_shape_calculator(operator: Operator):
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
    # TODO move to optioanl output according to custom_parsers
    # # output[2] = skm_op.score(X)
    # op_outputs[2].type = InputVarDtype(shape=[n_samples])


def spherical_kmeans_converter(scope: Scope, operator: Operator,
                               container: ModelComponentContainer):
    """The ONNX converter for sklearn_plugins.cluster.SphericalKMeans.

    Original SphericalKMeans Implementation:
    ```
    # preprocess input
    if skm.normalize:
        X = normalize(X, norm="l2", axis=1, copy=skm.copy)
    # each features in the dataset has zero mean and unit variance; dataset level
    if skm.standarize:
        X = skm.__std_scalar_.transform(X, copy=copy)
    # PCA whiten
    X = skm.__pca_.transform(X)
    # calculate projection
    proj: np.ndarray = np.matmul(X, centroids)
    # calculate lables
    labels: np.ndarray = np.argmax(S_proj, axis=1)
    ```

    Args:
        scope (Scope): [description]
        operator (Operator): [description]
        container (ModelComponentContainer): [description]
    """
    skm: SphericalKMeans = operator.raw_operator
    # The targeted ONNX operator set (referred to as opset) that matches the ONNX version.
    op_version: Union[int, None] = container.target_opset
    op_outputs: List[Variable] = operator.outputs
    # retreive input
    X: Variable = operator.inputs[0]
    input_op: OnnxOperator = OnnxIdentity(X, op_version=op_version)
    input_op.add_to(scope=scope, container=container)
    # module computation
    # normalize input
    normalize_op: OnnxOperator = input_op
    if skm.normalize == True:
        normalize_op = OnnxNormalizer(input_op,
                                      norm="L2",
                                      op_version=op_version)
        normalize_op.add_to(scope=scope, container=container)
    # standardize input
    std_scalar_op: OnnxOperator = normalize_op
    if skm.standardize == True:
        std_scalar_sub_op: OnnxSubOperator = OnnxSubOperator(
            op=skm.std_scalar_,
            inputs=normalize_op.outputs,
            op_version=op_version)
        std_scalar_op = OnnxIdentity(std_scalar_sub_op, op_version=op_version)
        std_scalar_op.add_to(scope=scope, container=container)
    # pca whitening
    pca_sub_op: OnnxSubOperator = OnnxSubOperator(op=skm.pca_,
                                                  inputs=std_scalar_op.outputs,
                                                  op_version=op_version)
    pca_op: OnnxOperator = OnnxIdentity(pca_sub_op, op_version=op_version)
    # calculate projection
    proj_op: OnnxOperator = OnnxMatMul(pca_op,
                                       skm.centroids_,
                                       op_version=op_version,
                                       output_names=[op_outputs[1]])
    labels_op: OnnxOperator = OnnxArgMax(proj_op,
                                         op_version=op_version,
                                         output_names=[op_outputs[0]])
    proj_op.add_to(scope=scope, container=container)
    labels_op.add_to(scope=scope, container=container)


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
