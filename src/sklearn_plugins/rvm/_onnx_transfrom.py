"""Convert to onnx.
"""
from typing import List, Type, Union

import numpy as np
from onnxconverter_common.data_types import (DataType, DoubleTensorType,
                                             FloatTensorType)
from skl2onnx.algebra.onnx_operator import OnnxOperator
from skl2onnx.algebra.onnx_ops import OnnxMatMul, OnnxPad
from skl2onnx.common._container import ModelComponentContainer
from skl2onnx.common._topology import Operator, Scope, Variable
from skl2onnx.common.data_types import guess_numpy_type
from skl2onnx.common.utils import check_input_and_output_types

from ..kernels.kernel_base import KernelBase
from .rvm import BaseRVM
from .rvr import RVR

__author__ = "Hung-Tien Huang"
__copyright__ = "Copyright 2021, Hung-Tien Huang"


# TODO support y_var output
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


def rvr_converter(scope: Scope, operator: Operator,
                  container: ModelComponentContainer):
    rvr: RVR = operator.raw_operator
    input: Variable = operator.inputs[0]
    NumPyType: Type = guess_numpy_type(input.type)
    op_version: Union[int, None] = container.target_opset
    op_outputs: List[Variable] = operator.outputs
    kernel_func: KernelBase = _get_kernel_function(rvm=rvr)
    phi_matrix: OnnxOperator = kernel_func.convert_onnx(
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


def _get_kernel_function(rvm: BaseRVM) -> KernelBase:
    if isinstance(rvm.kernel_func, KernelBase):
        return rvm.kernel_func
    raise ValueError(
        "rvm.kernel_func is not an instance of BaseRVM, and therefore ineligible for conversion"
    )
