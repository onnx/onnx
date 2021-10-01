# SPDX-License-Identifier: Apache-2.0

"""onnx checker

This implements graphalities that allows us to check whether a serialized
proto is legal.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import functools

from onnx import (ValueInfoProto,
                  AttributeProto,
                  TensorProto,
                  SparseTensorProto,
                  NodeProto,
                  ModelProto,
                  GraphProto,
                  IR_VERSION)
import onnx.onnx_cpp2py_export.checker as C
import onnx.defs
from google.protobuf.message import Message
from typing import TypeVar, Callable, Any, Type, cast, Union, Text
from six import string_types, binary_type
import onnx.shape_inference
import sys


# Limitation of single protobuf file is 2GB
MAXIMUM_PROTOBUF = 2000000000

# TODO: This thing where we reserialize the protobuf back into the
# string, only to deserialize it at the call site, is really goofy.
# Stop doing that.


# NB: Please don't edit this context!
DEFAULT_CONTEXT = C.CheckerContext()
DEFAULT_CONTEXT.ir_version = IR_VERSION
# TODO: Maybe ONNX-ML should also be defaulted?
DEFAULT_CONTEXT.opset_imports = {'': onnx.defs.onnx_opset_version()}


FuncType = TypeVar('FuncType', bound=Callable[..., Any])


# TODO: This really doesn't seem worth the metaprogramming...
def _create_checker(proto_type):  # type: (Type[Message]) -> Callable[[FuncType], FuncType]
    def decorator(py_func):  # type: (FuncType) -> FuncType
        @functools.wraps(py_func)
        def checker(proto, ctx=DEFAULT_CONTEXT):  # type: (Message, C.CheckerContext) -> Any
            if not isinstance(proto, proto_type):
                raise RuntimeError(
                    'You cannot pass an object that is not of type {}'.format(
                        proto_type.__name__))
            return getattr(C, py_func.__name__)(
                proto.SerializeToString(), ctx)
        return cast(FuncType, checker)
    return decorator


@_create_checker(ValueInfoProto)
def check_value_info(value_info, ctx=DEFAULT_CONTEXT):  # type: (ValueInfoProto, C.CheckerContext) -> None
    pass


@_create_checker(TensorProto)
def check_tensor(tensor, ctx=DEFAULT_CONTEXT):  # type: (TensorProto, C.CheckerContext) -> None
    pass


@_create_checker(AttributeProto)
def check_attribute(attr, ctx=DEFAULT_CONTEXT):  # type: (AttributeProto, C.CheckerContext) -> None
    pass


@_create_checker(NodeProto)
def check_node(node, ctx=DEFAULT_CONTEXT):  # type: (NodeProto, C.CheckerContext) -> None
    pass


@_create_checker(GraphProto)
def check_graph(graph, ctx=DEFAULT_CONTEXT):  # type: (GraphProto, C.CheckerContext) -> None
    pass


def check_sparse_tensor(sparse, ctx=DEFAULT_CONTEXT):  # type: (SparseTensorProto, C.CheckerContext) -> None
    C.check_sparse_tensor(sparse.SerializeToString(), ctx)


def check_model(model, full_check=False):  # type: (Union[ModelProto, Text, bytes], bool) -> None
    # If model is a path instead of ModelProto
    if isinstance(model, string_types):
        C.check_model_path(model)
        if full_check:
            onnx.shape_inference.infer_shapes_path(model, check_type=True, strict_mode=True)
    else:
        protobuf_string = model if isinstance(model, binary_type) else model.SerializeToString()
        # If the protobuf is larger than 2GB,
        # remind users should use the model path to check
        if sys.getsizeof(protobuf_string) > MAXIMUM_PROTOBUF:
            raise ValueError('This protobuf of onnx model is too large (>2GB). Call check_model with model path instead.')
        C.check_model(protobuf_string)
        if full_check:
            onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True)


ValidationError = C.ValidationError
