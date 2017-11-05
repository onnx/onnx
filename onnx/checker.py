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
                  ModelProto,
                  IR_VERSION)
import onnx.onnx_cpp2py_export.checker as C


def _create_checker(proto_type):
    def decorator(py_func):
        @functools.wraps(py_func)
        def checker(proto, ir_version=IR_VERSION):
            if not isinstance(proto, proto_type):
                raise RuntimeError(
                    'You cannot pass an object that is not of type {}'.format(
                        proto_type.__name__))
            return getattr(C, py_func.__name__)(
                proto.SerializeToString(), ir_version)
        return checker
    return decorator


@_create_checker(ValueInfoProto)
def check_value_info(value_info, ir_version=IR_VERSION):
    pass


@_create_checker(TensorProto)
def check_tensor(tensor, ir_version=IR_VERSION):
    pass


@_create_checker(AttributeProto)
def check_attribute(attr, ir_version=IR_VERSION):
    pass


@_create_checker(NodeProto)
def check_node(node, ir_version=IR_VERSION):
    pass


@_create_checker(GraphProto)
def check_graph(graph, ir_version=IR_VERSION):
    pass


@_create_checker(ModelProto)
def check_model(model, ir_version=IR_VERSION):
    pass

ValidationError = C.ValidationError
