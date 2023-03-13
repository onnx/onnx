# SPDX-License-Identifier: Apache-2.0
# pylint: disable=unnecessary-pass,import-outside-toplevel

from typing import Any

from onnx.npx.npx_array_api import ArrayApi


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

    @staticmethod
    def _reduce_impl(x, axes, keepdims=0, method_name=None):
        # avoids circular imports.
        from onnx.npx.npx_var import Var

        if not isinstance(x, Var):
            raise TypeError(f"Input 0 must be a Var not {type(x)}.")
        meth = getattr(Var, method_name)
        return meth(x, axes, keepdims=keepdims)

    @staticmethod
    def _reduce_impl_noaxes(x, keepdims=0, method_name=None):
        # avoids circular imports.
        from onnx.npx.npx_var import Var

        if not isinstance(x, Var):
            raise TypeError(f"Input 0 must be a Var not {type(x)}.")
        meth = getattr(Var, method_name)
        return meth(x, keepdims=keepdims)

    @staticmethod
    def _getitem_impl_var(obj, index, method_name=None):
        # avoids circular imports.
        from onnx.npx.npx_var import Var

        if not isinstance(obj, Var):
            raise TypeError(f"obj must be a Var not {type(obj)}.")
        meth = getattr(Var, method_name)
        return meth(obj, index)

    @staticmethod
    def _getitem_impl_tuple(obj, index=None, method_name=None):
        # avoids circular imports.
        from onnx.npx.npx_var import Var

        if not isinstance(obj, Var):
            raise TypeError(f"obj must be a Var not {type(obj)}.")
        meth = getattr(Var, method_name)
        return meth(obj, index)

    def generic_method(  # pylint: disable=too-many-branches
        self, method_name, *args: Any, **kwargs: Any
    ) -> Any:
        """
        The method converts the method into an ONNX graph build by the
        corresponding method in class Var.
        """
        # avoids circular imports.
        from onnx.npx.npx_jit_eager import eager_onnx
        from onnx.npx.npx_var import Var

        if not hasattr(Var, method_name):
            raise AttributeError(
                f"Class Var does not implement method {method_name!r}. "
                f"This method cannot be converted into an ONNX graph."
            )
        if method_name == "__getitem__":
            if len(args) != 1:
                raise ValueError(
                    f"Unexpected number of argument {len(args)}, it should be one."
                )
            if isinstance(args[0], tuple):
                eag = eager_onnx(
                    EagerTensor._getitem_impl_tuple, self.__class__, bypass_eager=True
                )
                res = eag(
                    self, index=args[0], method_name=method_name, already_eager=True
                )
            else:
                eag = eager_onnx(
                    EagerTensor._getitem_impl_var, self.__class__, bypass_eager=True
                )
                res = eag(self, args[0], method_name=method_name, already_eager=True)
            if isinstance(res, tuple) and len(res) == 1:
                return res[0]
            return res

        if method_name == "__setitem__":
            return ArrayApi.generic_method(self, method_name, *args, **kwargs)

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

        if method_name in {"mean", "sum", "min", "max", "prod"}:
            # ReduceFunction
            if len(args) not in (0, 1):
                raise ValueError(
                    f"An operator must have zero or one argument not {len(args)}."
                )

            if "axis" in kwargs:
                axes = kwargs["axis"]
                del kwargs["axis"]
            else:
                axes = None
            if axes is None:
                eag = eager_onnx(
                    EagerTensor._reduce_impl_noaxes, self.__class__, bypass_eager=True
                )
                res = eag(self, method_name=method_name, already_eager=True, **kwargs)
            else:
                eag = eager_onnx(
                    EagerTensor._reduce_impl, self.__class__, bypass_eager=True
                )
                res = eag(
                    self, axes, method_name=method_name, already_eager=True, **kwargs
                )
            if isinstance(res, tuple) and len(res) == 1:
                return res[0]
            return res

        return ArrayApi.generic_method(self, method_name, *args, **kwargs)
