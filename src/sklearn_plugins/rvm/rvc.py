from functools import partial
from typing import Callable, Optional

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from .rvm import BaseRVM


class RVC(BaseRVM):
    def __init__(self,
                 kernel_func: Callable[[np.ndarray, np.ndarray],
                                       np.ndarray] = partial(rbf_kernel,
                                                             gamma=None),
                 include_bias: bool = True,
                 tol: float = 1e-3,
                 max_iter: Optional[int] = None,
                 verbose: bool = True) -> None:
        super().__init__(kernel_func=kernel_func,
                         include_bias=include_bias,
                         tol=tol,
                         max_iter=max_iter,
                         verbose=verbose)

    pass
