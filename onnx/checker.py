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
                  NodeProto,
                  GraphProto,
                  IR_VERSION)
import onnx.onnx_cpp2py_export.checker as C
import onnx.defs


# TODO: This thing where we reserialize the protobuf back into the
# string, only to deserialize it at the call site, is really goofy.
# Stop doing that.


# NB: Please don't edit this context!
DEFAULT_CONTEXT = C.CheckerContext()
DEFAULT_CONTEXT.ir_version = IR_VERSION
# TODO: Maybe ONNX-ML should also be defaulted?
DEFAULT_CONTEXT.opset_imports = {'': onnx.defs.onnx_opset_version()}


# TODO: This really doesn't seem worth the metaprogramming...
def _create_checker(proto_type):
    def decorator(py_func):
        @functools.wraps(py_func)
        def checker(proto, ctx=DEFAULT_CONTEXT):
            if not isinstance(proto, proto_type):
                raise RuntimeError(
                    'You cannot pass an object that is not of type {}'.format(
                        proto_type.__name__))
            return getattr(C, py_func.__name__)(
                proto.SerializeToString(), ctx)
        return checker
    return decorator


@_create_checker(ValueInfoProto)
def check_value_info(value_info, ctx=DEFAULT_CONTEXT):
    pass


@_create_checker(TensorProto)
def check_tensor(tensor, ctx=DEFAULT_CONTEXT):
    pass


@_create_checker(AttributeProto)
def check_attribute(attr, ctx=DEFAULT_CONTEXT):
    pass


@_create_checker(NodeProto)
def check_node(node, ctx=DEFAULT_CONTEXT):
    pass


@_create_checker(GraphProto)
def check_graph(graph, ctx=DEFAULT_CONTEXT):
    pass


def check_model(model):
    C.check_model(model.SerializeToString())


ValidationError = C.ValidationError
