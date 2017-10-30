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

from onnx.onnx_ml_pb2 import TensorProto, ValueInfoProto, NodeProto, AttributeProto,  \
    GraphProto, ModelProto, IR_VERSION
from onnx import defs, mapping




def check_attr(attr, ir_version=IR_VERSION):
    """Checks if a node attribute is legal.

    Inputs:
        node: an AttributeProto object.
        ir_version: the IR_VERSION to validate against.
    Returns:
        None
    An exception is thrown if it does not pass the test.
    """

    require_type_field = ir_version >= IR_VERSION
    has_attr_type = attr.HasField('type') and attr.type != AttributeProto.UNDEFINED

    if require_type_field and not has_attr_type:
        raise RuntimeError('AttributeProto missing type field where IR version requires it.')

    def check_mismatch(field_name, type_value, type_value_name):
        if attr.HasField(field_name) and has_attr_type and attr.type != type_value:
            raise RuntimeError('AttributeProto.type is wrong value (expected '
                               + type_value_name + ').')


    check_mismatch('s', AttributeProto.STRING, 'STRING')
    check_mismatch('f', AttributeProto.FLOAT, 'FLOAT')
    check_mismatch('i', AttributeProto.INT, 'INT')
    check_mismatch('t', AttributeProto.TENSOR, 'TENSOR')
    check_mismatch('g', AttributeProto.GRAPH, 'GRAPH')

    check_mismatch('strings', AttributeProto.STRINGS, 'STRINGS')
    check_mismatch('floats', AttributeProto.FLOATS, 'FLOATS')
    check_mismatch('ints', AttributeProto.INTS, 'INTS')
    check_mismatch('tensors', AttributeProto.TENSORS, 'TENSORS')
    check_mismatch('graphs', AttributeProto.GRAPHS, 'GRAPHS')

    if attr.HasField('t'):
        check_tensor(attr.t)
    if attr.HasField('tensors'):
        for tensor in attr.tensors:
            check_tensor(tensor)


def check_node(node, ir_version=IR_VERSION):
    """Checks if a node is legal.

    Inputs:
        node: a NodeProto object.
        ir_version: the IR_VERSION to check against.
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
    for attr in node.attribute:
        check_attr(attr, ir_version)


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
        if type_required and \
           not value_info.type.sparse_tensor_type.HasField('elem_type'):
            raise ValueError('elem_type field of SparseTensorTypeProto is missing')
        if shape_required and not value_info.type.sparse_tensor_type.HasField('shape'):
            raise ValueError('shape field of SparseTensorTypeProto is missing')
    else:
        raise ValueError(
            'TypeProto.value should be either tensor_type or sparse_tensor_type')


def check_tensor(tensor):
    if not isinstance(tensor, TensorProto):
        raise RuntimeError('You cannot pass an object that is not TensorProto.')

    fields = [field
              for field in set(mapping.STORAGE_TENSOR_TYPE_TO_FIELD.values())
              if getattr(tensor, field)]
    has_raw_field = tensor.HasField('raw_data')
    if has_raw_field:
        fields.append('raw_data')

    if len(fields) != 1:
        raise ValueError(
            'There should be exactly one data field set: {}'.format(
                ', '.join(fields)))

    if has_raw_field:
        if tensor.data_type == TensorProto.STRING:
            raise ValueError(
                'STRING data should not be stored in "raw_data" field')
    else:
        field = fields[0]
        if field != mapping.STORAGE_TENSOR_TYPE_TO_FIELD[
                mapping.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE[tensor.data_type]]:
            raise ValueError(
                'Mismatched data type ({}) and field ({})'.format(
                    tensor.data_type, field
                ))


def check_graph(graph, ir_version=IR_VERSION):
    """Checks if a GraphProto is legal.

    Inputs:
        graph: a GraphProto object.
        ir_version: the IR_VERSION to check against.
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
        check_node(node, ir_version)

    input_names = {value_info.name for value_info in graph.input}
    for init in graph.initializer:
        if init.name not in input_names:
            raise ValueError(
                '{} in initializer but not in graph input'.format(init.name))
        check_tensor(init)


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
    check_graph(model.graph, model.ir_version)
