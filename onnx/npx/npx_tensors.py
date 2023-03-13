# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unnecessary-pass

from typing import Any, Callable, List, Tuple

import numpy as np

from onnx import ModelProto
from onnx.npx.npx_array_api import ArrayApi
from onnx.npx.npx_types import TensorType
from onnx.reference import ReferenceEvaluator


class JitTensor:
    """
    Defines a value for a specific jit mode
    """

    pass


class EagerTensor(ArrayApi):
    """
    Defines a value for a specific eager mode.
    An eager tensor must overwrite every call to a method listed in class
    :class:`ArrayApi`.
    """

    @staticmethod
    def _op_impl(*inputs, method_name=None):
        # avoids circular imports.
        from onnx.npx.npx_var import Var

        for i, x in enumerate(inputs):
            if not isinstance(x, Var):
                raise TypeError(f"Input {i} must be a Var not {type(x)}.")
        meth = getattr(Var, method_name)
        return meth(*inputs)

    def generic_method(self, method_name, *args: Any, **kwargs: Any) -> Any:
        """
        The method converts the method into an ONNX graph build by the
        corresponding method in class Var.
        """
        # avoids circular imports.
        from onnx.npx.npx_var import Var
        from onnx.npx.npx_jit_eager import eager_onnx

        if not hasattr(Var, method_name):
            raise AttributeError(
                f"Class Var does not implement method {method_name!r}. "
                f"This method cannot be converted into an ONNX graph."
            )
        if method_name.startswith("__") and method_name.endswith("__"):
            # An operator.
            if len(args) not in (0, 1):
                raise ValueError(
                    f"An operator must have zero or one argument not {len(args)}."
                )
            if len(kwargs) not in (0, 1):
                raise ValueError(f"Operators do not support parameters {len(kwargs)}.")

            eag = eager_onnx(EagerTensor._op_impl, self.__class__, bypass_eager=True)
            res = eag(self, *args, method_name=method_name, already_eager=True)
            if isinstance(res, tuple) and len(res) == 1:
                return res[0]
            return res

        return ArrayApi.generic_method(self, method_name, *args, **kwargs)
