"""Convert to onnx.
"""
from typing import List, Type, Union

from onnxconverter_common.data_types import (DataType, DoubleTensorType,
                                             FloatTensorType, Int64TensorType)
from skl2onnx.algebra.onnx_operator import OnnxOperator, OnnxSubEstimator
from skl2onnx.algebra.onnx_ops import OnnxArgMax, OnnxMatMul, OnnxNormalizer
from skl2onnx.common._container import ModelComponentContainer
from skl2onnx.common._topology import Operator, Scope, Variable
from skl2onnx.common.data_types import guess_numpy_type
from skl2onnx.common.utils import check_input_and_output_types
from sklearn.preprocessing import Normalizer
from sklearn_plugins.cluster.spherical_kmeans import SphericalKMeans

__author__ = "Hung-Tien Huang"
__copyright__ = "Copyright 2021, Hung-Tien Huang"


def spherical_kmeans_shape_calculator(operator: Operator):
    """Calculate the input and output shape for SphericalKMeans.

    Args:
        operator (Operator): An Operator container.
    """
    check_input_and_output_types(
        operator,
        good_input_types=[FloatTensorType, DoubleTensorType],
        good_output_types=[Int64TensorType, FloatTensorType, DoubleTensorType])
    op_inputs: List[Variable] = operator.inputs
    if len(op_inputs) != 1:
        raise RuntimeError(
            "Only one input matrix is allowed for SphericalKMeans.")
    op_outputs: List[Variable] = operator.outputs
    if len(op_outputs) != 2:
        raise RuntimeError("Two outputs are expected for SphericalKMeans.")
    skm: SphericalKMeans = operator.raw_operator
    # retrieve skm inputs dtype
    input_var_type: DataType = op_inputs[0].type
    # retrieve skm inputs outputs shape
    n_samples: int = input_var_type.shape[0]
    n_clusters: int = skm.n_clusters
    # type alias
    InputVarDtype: Type[DataType] = input_var_type.__class__
    # output[0] = skm_op.fit_predict(X)
    op_outputs[0].type.shape = [n_samples]
    # output[1] = skm_op.fit_transform(X)
    op_outputs[1].type.shape = [n_samples, n_clusters]
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
    op_version: Union[int, None] = container.target_opset
    op_outputs: List[Variable] = operator.outputs
    # retreive input
    input: Variable = operator.inputs[0]
    np_dtype = guess_numpy_type(input.type)
    # normalize input
    normalize_op: Union[OnnxOperator, Variable] = input
    if skm.normalize == True:
        # TODO temporary fix as input can only be np.float32 for SubEstimator approach; whereas OnnxNormalizer has inputs shape RuntimeError.
        normalize_op = OnnxSubEstimator(Normalizer(norm="l2"),
                                        normalize_op,
                                        op_versionrow_norms=op_version)
        # normalize_op = OnnxNormalizer(input, norm="L2", op_version=op_version)
    # standardize input
    std_scalar_op: Union[OnnxOperator, Variable] = normalize_op
    if skm.standardize == True:
        std_scalar_op = OnnxSubEstimator(skm.std_scalar_,
                                         normalize_op,
                                         op_versionrow_norms=op_version)
    # pca whitening
    pca_op: OnnxOperator = OnnxSubEstimator(skm.pca_,
                                            std_scalar_op,
                                            op_version=op_version)
    # calculate projection
    proj_op: OnnxOperator = OnnxMatMul(pca_op,
                                       skm.centroids_.astype(np_dtype),
                                       op_version=op_version,
                                       output_names=[op_outputs[1]])
    proj_op.add_to(scope=scope, container=container)
    labels_op: OnnxOperator = OnnxArgMax(proj_op,
                                         op_version=op_version,
                                         output_names=[op_outputs[0]],
                                         axis=1,
                                         keepdims=0)
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
