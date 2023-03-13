# SPDX-License-Identifier: Apache-2.0
# pylint: disable=too-many-statements,too-many-branches

from inspect import _empty, signature
from typing import Any, Callable, Dict, Sequence, Union

import numpy as np

from onnx import FunctionProto, ModelProto, NodeProto
from onnx.npx.npx_tensors import EagerTensor
from onnx.npx.npx_types import (
    ElemType,
    OptParType,
    ParType,
    TupleType,
)
from onnx.npx.npx_var import Cst, Input, ManyIdentity, Par, Var


def cst(*args, **kwargs):
    """
    Wraps a call to the building of class :class:`Cst`.
    """
    return Cst(*args, **kwargs)


def tuple_var(*args: Sequence[Var]) -> Var:
    """
    Tie many results all together before being returned by a function.
    """
    return ManyIdentity(*args)


def make_tuple(
    n_elements_or_first_variable: Union[int, Var],
    *args: Sequence[Var],
    **kwargs: Dict[str, Any],
) -> Var:
    """
    Wraps a call to the building of class :class:`Tuple`.
    *n_elements_or_first_variable*
    is the number of elements in the tuple or the number of
    detected arguments if not specified.
    """
    if isinstance(n_elements_or_first_variable, int):
        n_elements = n_elements_or_first_variable
        return Var(*args, n_var_outputs=n_elements, **kwargs)
    args = [n_elements_or_first_variable, *args]
    return tuple_var(*args, **kwargs)


def var(*args: Sequence[Var], **kwargs: Dict[str, Any]) -> Var:
    """
    Wraps a call to the building of class :class:`Var`.
    """
    return Var(*args, **kwargs)


def _process_parameter(fn, sig, k, v, new_pars, inline):
    annotation = sig.parameters[k].annotation if k in sig.parameters else None
    if v is None and len(new_pars) == 0 and annotation is None:
        # It could be an optional input or a parameter.
        raise NotImplementedError(
            f"Unable to decide between an optional input or a "
            f"parameter for name={k!r}."
        )
    if isinstance(v, Par):
        if inline:
            new_pars[k] = v.value
        else:
            new_pars[k] = v
        return
    if isinstance(v, type) and k == "dtype":
        vto = ElemType.numpy_map[v]
        if inline:
            new_pars[k] = vto
        else:
            new_pars[k] = Par(
                k,
                dtype=ParType[int],
                value=vto,
                parent_op=(fn.__module__, fn.__name__, 0),
            )
        return
    if isinstance(v, (int, float, str, tuple)):
        if inline:
            new_pars[k] = v
        else:
            new_pars[k] = Par(
                k,
                dtype=ParType[type(v)],
                value=v,
                parent_op=(fn.__module__, fn.__name__, 0),
            )
        return
    if isinstance(v, (Cst, Var)):
        raise TypeError(
            f"Parameter {k!r} is a tensor ({type(v)}), it is not "
            f"supported for a named parameter."
        )

    if isinstance(v, (FunctionProto, NodeProto, ModelProto)):
        new_pars[k] = v
        return

    if v is None and issubclass(annotation, OptParType):
        return
    raise TypeError(
        f"Unexpected type for parameter {k!r}, type={type(v)}, "
        f"annotation={annotation}."
    )


def _xapi(fn: Callable, inline: bool):
    """
    Decorator to use before any function using part of the numpy API.
    The function inspects the input and decides which version of the function
    to call.

    :param fn: function
    :param inline: inline the function instead of creating
        a function
    """
    sig = signature(fn)
    eager_onnx_tensor_classes = {}

    # It has the same signature
    def wrapper(*inputs, **kwargs):
        if any(map(lambda x: isinstance(x, EagerTensor), inputs)):
            tensor_class = None
            for x in inputs:
                if isinstance(x, EagerTensor):
                    tensor_class = x.__class__
            if tensor_class is None:
                raise RuntimeError(
                    f"Unable to find an EagerTensor in types {[type(x) for x in inputs]}."
                )

            if tensor_class not in eager_onnx_tensor_classes:
                from onnx.npx.npx_jit_eager import eager_onnx

                eager_onnx_tensor_classes[tensor_class] = eager_onnx(fn, tensor_class)
            res = eager_onnx_tensor_classes[tensor_class](
                *inputs, already_eager=True, **kwargs
            )
            if not isinstance(res, tuple):
                raise TypeError(f"Return of the eager must be a tuple not {type(res)}.")
            return res if len(res) > 1 else res[0]

        # conversion to onnx
        new_inputs = []
        new_pars = {}
        parnames = {}
        pos = 0
        for name, par in sig.parameters.items():
            if par.kind == par.VAR_POSITIONAL:
                break
            if par.kind in (par.POSITIONAL_ONLY, par.POSITIONAL_OR_KEYWORD):
                parnames[pos] = name
                pos += 1
                continue
        last_input = -1
        for ind, i in enumerate(inputs):
            annotation = (
                sig.parameters[parnames[ind]].annotation if ind in parnames else None
            )
            if (
                annotation is not None
                and isinstance(annotation, type)
                and issubclass(annotation, ParType)
            ):
                # no more inputs
                break
            last_input = ind
            if isinstance(i, (Var, np.ndarray)):
                new_inputs.append(i)
            elif isinstance(i, (int, float)):
                new_inputs.append(
                    np.array([i], dtype=np.int64 if isinstance(i, int) else np.float32)
                )
            elif isinstance(i, str):
                new_inputs.append(Input(i))
            elif i is None:
                # optional input
                new_inputs.append(None)
            else:
                raise TypeError(
                    f"Unexpected type for input {ind}, type={type(i)}. "
                    f"Did you forget to wrap the constant with 'cst(.)'?"
                )
        for ind in range(last_input + 1, len(inputs)):
            k = parnames[ind]
            if k in kwargs:
                break
            _process_parameter(fn, sig, k, inputs[ind], new_pars, inline)
        for k, v in kwargs.items():
            _process_parameter(fn, sig, k, v, new_pars, inline)

        if issubclass(sig.return_annotation, TupleType):
            n_var_outputs = sig.return_annotation.len()
            return Var(
                *new_inputs,
                op=fn,
                inline=inline,
                n_var_outputs=n_var_outputs,
                **new_pars,
            )
        return Var(*new_inputs, op=fn, inline=inline, **new_pars)

    rows = ["", "", "Signature:", "", "::", "", "    ("]
    for p in sig.parameters.values():
        if p.annotation == _empty:
            rows.append(f"        {p.name},")
        else:
            if hasattr(p.annotation, "__args__"):
                args = p.annotation.__args__
                if (
                    isinstance(args, tuple)
                    and len(args) == 2
                    and isinstance(None, args[1])
                ):  # args[1] == type(None)
                    # optional
                    annot = args[0]
                else:
                    raise TypeError(
                        f"Unable to interpret annotation for parameter "
                        f"{p.name!r} with {p.annotation} and args={args}."
                    )
            else:
                annot = p.annotation
            try:
                a_name = annot.type_name()
            except AttributeError as e:
                raise AttributeError(
                    f"Unexpected annotation type {p.annotation!r}."
                ) from e
            rows.append(f"        {p.name}: {a_name},")
    if sig.return_annotation == _empty:
        rows.append("    ):")
    else:
        rows.append(f"    ) -> {sig.return_annotation.type_name()}:")
    wrapper.__doc__ = (fn.__doc__ or "") + "\n" + "\n".join(rows)
    return wrapper


def npxapi_function(fn):
    """
    Decorator to use before any function using part of the numpy API.
    The function inspects the input and decides which version of the function
    to call.
    """
    return _xapi(fn, inline=False)


def npxapi_inline(fn):
    """
    Decorator to use before any function using part of the numpy API.
    The function inspects the input and decides which version of the function
    to call.
    """
    return _xapi(fn, inline=True)
