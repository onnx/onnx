"""onnx checker

This implements graphalities that allows us to check whether a serialized
proto is legal.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import logging

from onnx.onnx_pb2 import *
from onnx import defs


def check_node(node):
    """Checks if a node is legal.

    Inputs:
        node: a NodeProto object.
    Returns:
        None
    An exception is thrown if it does not pass the test.
    """
    # General checks.
    if not isinstance(node, NodeProto):
        raise RuntimeError('You cannot pass an object that is not NodeProto.')
    if not node.op_type:
        raise NameError('NodeProto does not have a proper op_type set.')
    if not node.input and not node.output:
        raise ValueError('NodeProto has zero input and zero output.')
    if not defs.has(node.op_type):
        raise NameError(
            'Node op_type {} not recognized by onnx.'.format(node.op_type))
    if not defs.get_schema(node.op_type).verify(node.SerializeToString()):
        raise ValueError(
            'NodeProto of type {} did not pass defs schema check.'.format(str(node.op_type)))


def check_tensor_value_info(value_info,
                            type_required=True,
                            shape_required=True):
    if not isinstance(value_info, ValueInfoProto):
        raise RuntimeError('You cannot pass an object that is not ValueInfoProto.')
    if not value_info.name:
        raise NameError('ValueInfoProto must have its name set.')

    if not type_required and not shape_required:
        return
    if not value_info.HasField('type'):
        raise ValueError('type field of ValueInfoProto is missing')
    value = value_info.type.WhichOneof('value')

    if value == 'tensor_type':
        if type_required and not value_info.type.tensor_type.HasField('elem_type'):
            raise ValueError('elem_type field of TensorTypeProto is missing')
        if shape_required and not value_info.type.tensor_type.HasField('shape'):
            raise ValueError('shape field of TensorTypeProto is missing')
    elif value == 'sparse_tensor_type':
        if type_required and not value_info.type.sparse_tensor_type.HasField('elem_type'):
            raise ValueError('elem_type field of SparseTensorTypeProto is missing')
        if shape_required and not value_info.type.sparse_tensor_type.HasField('shape'):
            raise ValueError('shape field of SparseTensorTypeProto is missing')
    else:
        raise ValueError(
            'TypeProto.value should be either tensor_type or sparse_tensor_type')


def check_graph(graph):
    """Checks if a GraphProto is legal.

    Inputs:
        graph: a GraphProto object.
    Returns:
        None
    An exception is thrown if it does not pass the test.
    """
    if not isinstance(graph, GraphProto):
        raise RuntimeError('You cannot pass an object that is not GraphProto.')
    if not graph.name:
        raise NameError(
            'The graph does not have a proper name set.')
    for value_info in itertools.chain(graph.input, graph.output):
        check_tensor_value_info(value_info)
    for value_info in graph.value_info:
        check_tensor_value_info(value_info)
    for node in graph.node:
        check_node(node)


def check_model(model):
    """Checks if a ModelProto is legal.

    Inputs:
        model: a ModelProto object.
    Returns:
        None
    An exception is thrown if it does not pass the test.
    """
    if not isinstance(model, ModelProto):
        raise RuntimeError('You cannot pass an object that is not ModelProto.')
    if not model.HasField('ir_version'):
        raise ValueError('The model does not have an ir_version set properly.')
    if model.ir_version > IR_VERSION:
        logging.warning(
            'Your model ir_version is higher than the checker\'s, so it might '
            'not interpret the higher version correctly.')
    check_graph(model.model.graph)
