from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import numbers
import numpy as np

from six import text_type, integer_types, binary_type

from onnx.onnx_pb2 import TensorProto, AttributeProto, ValueInfoProto, \
    NodeProto, ModelProto, GraphProto, IR_VERSION
import onnx.defs as defs
from onnx import mapping

def make_node(
        op_type, inputs, outputs,
        name=None, doc_string=None, **kwargs):
    node = NodeProto()
    node.op_type = op_type
    node.input.extend(inputs)
    node.output.extend(outputs)
    if name:
        node.name = name
    if doc_string:
        node.doc_string = doc_string
    if kwargs:
        node.attribute.extend(
            make_attribute(key, value)
            for key, value in kwargs.items())
    return node


def make_graph(nodes, name, inputs, outputs, initializer=None, doc_string=None):
    if initializer is None:
        initializer = []
    graph = GraphProto()
    graph.node.extend(nodes)
    graph.name = name
    graph.input.extend(inputs)
    graph.output.extend(outputs)
    graph.initializer.extend(initializer)
    if doc_string:
        graph.doc_string = doc_string
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


def split_complex_to_pairs(ca):
    return [(ca[i // 2].real if (i % 2 == 0) else ca[i // 2].imag)
            for i in range(len(ca) * 2)]


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

    if (data_type == TensorProto.COMPLEX64 or
            data_type == TensorProto.COMPLEX128):
        vals = split_complex_to_pairs(vals)
    if raw:
        tensor.raw_data = vals
    else:
        field = mapping.STORAGE_TENSOR_TYPE_TO_FIELD[
            mapping.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE[data_type]]
        getattr(tensor, field).extend(vals)

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


def make_attribute(key, value, doc_string=None):
    """Makes an AttributeProto based on the value type."""
    attr = AttributeProto()
    attr.name = key
    if doc_string:
        attr.doc_string = doc_string

    is_iterable = isinstance(value, collections.Iterable)
    bytes_or_false = _to_bytes_or_false(value)
    # First, singular cases
    # float
    if isinstance(value, float):
        attr.f = value
        attr.type = AttributeProto.FLOAT
    # integer
    elif isinstance(value, numbers.Integral):
        attr.i = value
        attr.type = AttributeProto.INT
    # string
    elif bytes_or_false:
        attr.s = bytes_or_false
        attr.type = AttributeProto.STRING
    elif isinstance(value, TensorProto):
        attr.t.CopyFrom(value)
        attr.type = AttributeProto.TENSOR
    elif isinstance(value, GraphProto):
        attr.g.CopyFrom(value)
        attr.type = AttributeProto.GRAPH
    # third, iterable cases
    elif is_iterable:
        byte_array = [_to_bytes_or_false(v) for v in value]
        if all(isinstance(v, float) for v in value):
            attr.floats.extend(value)
            attr.type = AttributeProto.FLOATS
        elif all(isinstance(v, numbers.Integral) for v in value):
            # Turn np.int32/64 into Python built-in int.
            attr.ints.extend(int(v) for v in value)
            attr.type = AttributeProto.INTS
        elif all(byte_array):
            attr.strings.extend(byte_array)
            attr.type = AttributeProto.STRINGS
        elif all(isinstance(v, TensorProto) for v in value):
            attr.tensors.extend(value)
            attr.type = AttributeProto.TENSORS
        elif all(isinstance(v, GraphProto) for v in value):
            attr.graphs.extend(value)
            attr.type = AttributeProto.GRAPHS
        else:
            raise ValueError(
                "You passed in an iterable attribute but I cannot figure out "
                "its applicable type.")
    else:
        raise ValueError(
            'Value "{}" is not valid attribute data type.'.format(value))
    return attr

def get_attribute_value(attr):
    if attr.HasField('f'):
        return attr.f
    elif attr.HasField('i'):
        return attr.i
    elif attr.HasField('s'):
        return attr.s
    elif attr.HasField('t'):
        return attr.t
    elif attr.HasField('g'):
        return onnn_attr.g
    elif len(attr.floats):
        return list(attr.floats)
    elif len(attr.ints):
        return list(attr.ints)
    elif len(attr.strings):
        return list(attr.strings)
    elif len(attr.tensors):
        return list(attr.tensors)
    elif len(attr.graphs):
        return list(attr.graphs)
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(onnx_arg))


def make_tensor_value_info(name, elem_type, shape, doc_string=""):
    """Makes a TypeProto based on the data type and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

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

    # for now, this logic should continue to work as long as we are running on a proto3
    # implementation. If/when we switch to proto3, we will need to use attr.type  
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
