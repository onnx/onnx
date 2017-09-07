from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numbers
import sys

from onnx.onnx_pb2 import \
    AttributeProto, TensorProto, NodeProto, GraphProto, IR_VERSION
import onnx.onnx_cpp2py_export as C

def make_node(
        op_type, inputs, outputs,
        name=None, **kwargs):
    node = NodeProto()
    node.op_type = op_type
    node.input.extend(inputs)
    node.output.extend(outputs)
    if name:
        node.name = name
    if kwargs:
        node.attribute.extend(
            make_attribute(key, value)
            for key, value in kwargs.items())
    return node


def make_graph(nodes, name, inputs, outputs, initializer=[]):
    graph = GraphProto()
    # Touch graph.ir_version so it is stored as the version from which it is
    # generated.
    graph.ir_version = IR_VERSION
    graph.node.extend(nodes)
    graph.name = name
    graph.input.extend(inputs)
    graph.output.extend(outputs)
    graph.initializer.extend(initializer)
    return graph

def make_tensor(name, data_type, dims, vals):
    tensor = TensorProto()
    tensor.data_type = data_type
    tensor.name = name
    if data_type == TensorProto.FLOAT:
        tensor.float_data.extend(vals)
    elif data_type in [TensorProto.UINT8,
                       TensorProto.INT8,
                       TensorProto.UINT16,
                       TensorProto.INT16,
                       TensorProto.INT32,
                       TensorProto.FLOAT16,
                       TensorProto.BOOL]:
        tensor.int32_data.extend(vals)
    elif data_type == TensorProto.INT64:
        tensor.int64_data.extend(vals)
    elif data_type == TensorProto.STRING:
        tensor.string_data.extend(vals)
    tensor.dims.extend(dims)
    return tensor

def _to_bytes_or_false(val):
    """An internal graph to convert the input to a bytes or to False.

    The criteria for conversion is as follows and should be python 2 and 3
    compatible:
    - If val is py2 str or py3 bytes: return bytes
    - If val is py2 unicode or py3 str: return val.decode('ascii')
    - Otherwise, return False
    """
    if isinstance(val, bytes):
        return val
    else:
        try:
            return val.encode('ascii')
        except AttributeError:
            return False


def make_attribute(key, value):
    """Makes an AttributeProto based on the value type."""
    attr = AttributeProto()
    attr.name = key

    is_iterable = isinstance(value, collections.Iterable)
    bytes_or_false = _to_bytes_or_false(value)
    # First, singular cases
    # float
    if isinstance(value, float):
        attr.f = value
    # integer
    elif isinstance(value, numbers.Integral):
        attr.i = value
    # string
    elif bytes_or_false:
        attr.s = bytes_or_false
    elif isinstance(value, GraphProto):
        attr.graph.CopyFrom(value)
    # third, iterable cases
    elif is_iterable:
        byte_array = [_to_bytes_or_false(v) for v in value]
        if all(isinstance(v, float) for v in value):
            attr.floats.extend(value)
        elif all(isinstance(v, numbers.Integral) for v in value):
            attr.ints.extend(value)
        elif all(byte_array):
            attr.strings.extend(byte_array)
        elif all(isinstance(v, TensorProto) for v in value):
            attr.tensors.extend(value)
        elif all(isinstance(v, GraphProto) for v in value):
            attr.graphs.extend(value)
        else:
            raise ValueError(
                "You passed in an iterable attribute but I cannot figure out "
                "its applicable type.")
    else:
        raise ValueError(
            "Your attribute {}:{} is not float, integer or string type, and is not"
            "iterable.".format(key, str(value)))
    return attr


def is_attribute_legal(attr):
    """Checks if an AttributeProto is legal.

    Inputs:
        arg: an AttributeProto object.
    Returns:
        bool.
    """
    if not isinstance(attr, AttributeProto):
        raise RuntimeError("You cannot pass an object that is not AttributeProto.")
    return C.is_attribute_legal(attr.SerializeToString())


def _sanitize_str(s):
    if isinstance(s, text_type):
        sanitized = s
    elif isinstance(s, binary_type):
        sanitized = s.decode('ascii', errors='ignore')
    else:
        sanitized = str(s)
    if len(sanitized) < 64:
        return sanitized
    else:
        return sanitized[:64] + '...<+len=%d>' % (len(sanitized) - 64)


def printable_attribute(attr):
    content = []
    content.append(attr.name)
    content.append("=")
    if attr.HasField("f"):
        content.append(str(attr.f))
    elif attr.HasField("i"):
        content.append('{}'.format(attr.i))
    elif attr.HasField("s"):
        content.append(_sanitize_str(attr.s))
    elif attr.HasField("t"):
        content.append("<Tensor>")
    elif attr.floats:
        content.append(str(list(attr.floats)))
    elif attr.ints:
        content.append('[' + ', '.join(map(lambda i: '{}'.format(i), attr.ints)) + ']')
    elif attr.strings:
        content.append(str(list(map(_sanitize_str, attr.strings))))
    elif attr.tensors:
        content.append("[<Tensor>, ...]")
    else:
        content.append("<Unknown>")
    return ' '.join(content)


def printable_node(node, prefix=''):
    content = []
    if len(node.output):
        content.append(
            ', '.join(['%{}'.format(name) for name in node.output]))
        content.append('=')
    printed_attributes = ', '.join(map(printable_attribute, node.attribute))
    printed_inputs = ', '.join(['%{}'.format(name) for name in node.input])
    if node.attribute:
        content.append("{}[{}]({})".format(node.op_type, printed_attributes, printed_inputs))
    else:
        content.append("{}({})".format(node.op_type, printed_inputs))
    # TODO: subgr
    return prefix + ' '.join(content)

def printable_graph(graph, prefix=''):
    content = []
    indent = prefix + '  '
    # header
    header = ['graph', graph.name]
    if len(graph.input):
        header.append(
            "(" + ', '.join(['%{}'.format(name) for name in graph.input]) + ")")
    header.append('{')
    content.append(prefix + ' '.join(header))
    # body
    for node in graph.node:
        content.append(printable_node(node, indent))
    # tail
    tail = ['return']
    if len(graph.output):
        tail.append(
            ', '.join(['%{}'.format(name) for name in graph.output]))
    content.append(indent + ' '.join(tail))
    # closing bracket
    content.append(prefix + '}')
    return '\n'.join(content)
