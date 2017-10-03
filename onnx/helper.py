from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numbers
import sys

from six import text_type, integer_types

from onnx.onnx_pb2 import *
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
    graph.node.extend(nodes)
    graph.name = name
    graph.input.extend(inputs)
    graph.output.extend(outputs)
    graph.initializer.extend(initializer)
    return graph


def make_model(graph, **kwargs):
    model = ModelProto()
    # Touch model.ir_version so it is stored as the version from which it is
    # generated.
    model.ir_version = IR_VERSION
    model.graph.CopyFrom(graph)

    for k, v in kwargs.items():
        setattr(model, k, v)
    return model


def make_tensor(name, data_type, dims, vals, raw=False):
    '''
    Make a TensorProto with specified arguments.  If raw is False, this
    function will choose the corresponding proto field to store the
    values based on data_type. If raw is True, use "raw_data" proto
    field to store the values, and values should be of type bytes in
    this case.
    '''
    tensor = TensorProto()
    tensor.data_type = data_type
    tensor.name = name

    if data_type == TensorProto.STRING:
        assert not raw, "Can not use raw_data to store string type"
        tensor.string_data.extend(vals)
    elif data_type in [TensorProto.UINT8,
                       TensorProto.INT8,
                       TensorProto.UINT16,
                       TensorProto.INT16,
                       TensorProto.INT32,
                       TensorProto.FLOAT16,
                       TensorProto.BOOL,
                       TensorProto.FLOAT]:
        if raw:
            tensor.raw_data = vals
        else:
            if data_type == TensorProto.FLOAT:
                tensor.float_data.extend(vals)
            elif data_type == TensorProto.INT64:
                tensor.int64_data.extend(vals)
            else:
                tensor.int32_data.extend(vals)
    else:
        raise RuntimeError('Unrecognized data_type: {}'.format(data_type))
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
    elif isinstance(value, TensorProto):
        attr.t.CopyFrom(value)
    elif isinstance(value, GraphProto):
        attr.g.CopyFrom(value)
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
            'Value "{}" is not valid attribute data type.'.format(value))
    return attr


def make_tensor_value_info(name, elem_type, shape):
    """Makes a TypeProto based on the data type and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name

    tensor_type_proto = value_info_proto.type.tensor_type
    tensor_type_proto.elem_type = elem_type

    tensor_shape_proto = tensor_type_proto.shape.dim
    for d in shape:
        dim = tensor_shape_proto.add()
        if isinstance(d, integer_types):
            dim.dim_value = d
        elif isinstance(d, text_type):
            dim.dim_param = d
        else:
            raise ValueError(
                'Invalid item in shape: {}. '
                'Needs to of integer_types or text_type.'.format(d))

    return value_info_proto


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
    def str_float(f):
        # NB: Different Python versions print different numbers of trailing
        # decimals, specifying this explicitly keeps it consistent for all
        # versions
        return '{:.15g}'.format(f)
    def str_int(i):
        # NB: In Python 2, longs will repr() as '2L', which is ugly and
        # unnecessary.  Explicitly format it to keep it consistent.
        return '{:d}'.format(i)
    def str_str(s):
        return repr(s)
    def str_list(str_elem, xs):
        return '[' + ', '.join(map(str_elem, xs)) + ']'
    if attr.HasField("f"):
        content.append(str_float(attr.f))
    elif attr.HasField("i"):
        content.append(str_int(attr.i))
    elif attr.HasField("s"):
        # TODO: Bit nervous about Python 2 / Python 3 determinism implications
        content.append(repr(_sanitize_str(attr.s)))
    elif attr.HasField("t"):
        content.append("<Tensor>")
    elif attr.floats:
        content.append(str_list(str_float, attr.floats))
    elif attr.ints:
        content.append(str_list(str_int, attr.ints))
    elif attr.strings:
        # TODO: Bit nervous about Python 2 / Python 3 determinism implications
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
