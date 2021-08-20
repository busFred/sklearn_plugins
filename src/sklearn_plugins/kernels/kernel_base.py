from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from skl2onnx.algebra.onnx_operator import OnnxOperator
from skl2onnx.common._topology import Variable


class KernelBase(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self,
                 X: np.ndarray,
                 X_prime: np.ndarray,
                 fit: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def convert_onnx(self, X: Variable, X_prime: np.ndarray,
                     op_version: Union[int, None]) -> OnnxOperator:
        pass
