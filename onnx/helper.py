# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections.abc  # type: ignore
import numbers

import google.protobuf.message
from onnx import TensorProto, SparseTensorProto, AttributeProto, ValueInfoProto, \
    TensorShapeProto, NodeProto, ModelProto, GraphProto, OperatorSetIdProto, \
    TypeProto, SequenceProto, MapProto, IR_VERSION, TrainingInfoProto, OptionalProto, \
    FunctionProto
from onnx import defs
from onnx import mapping
from onnx.mapping import STORAGE_TENSOR_TYPE_TO_FIELD
from typing import Text, Sequence, Any, Optional, Dict, Union, TypeVar, Callable, Tuple, List, cast
import numpy as np  # type: ignore
import warnings

VersionRowType = Union[Tuple[Text, int, int, int], Tuple[Text, int, int, int, int]]
VersionTableType = List[VersionRowType]
AssignmentBindingType = List[Tuple[Text, Text]]

# This is a copy of the documented version in https://github.com/onnx/onnx/blob/main/docs/Versioning.md#released-versions
# Both must be updated whenever a new version of ONNX is released.
VERSION_TABLE: VersionTableType = [
    # Release-version, IR version, ai.onnx version, ai.onnx.ml version, (optional) ai.onnx.training version
    ('1.0', 3, 1, 1),
    ('1.1', 3, 5, 1),
    ('1.1.2', 3, 6, 1),
    ('1.2', 3, 7, 1),
    ('1.3', 3, 8, 1),
    ('1.4.1', 4, 9, 1),
    ('1.5.0', 5, 10, 1),
    ('1.6.0', 6, 11, 2),
    ('1.7.0', 7, 12, 2, 1),
    ('1.8.0', 7, 13, 2, 1),
    ('1.8.1', 7, 13, 2, 1),
    ('1.9.0', 7, 14, 2, 1),
    ('1.10.0', 8, 15, 2, 1),
    ('1.10.1', 8, 15, 2, 1),
    ('1.10.2', 8, 15, 2, 1),
    ('1.11.0', 8, 16, 3, 1)
]

VersionMapType = Dict[Tuple[Text, int], int]


# create a map from (opset-domain, opset-version) to ir-version from above table
def create_op_set_id_version_map(table: VersionTableType) -> VersionMapType:
    result: VersionMapType = dict()

    def process(release_version: Text, ir_version: int, *args: Any) -> None:
        for pair in zip(['ai.onnx', 'ai.onnx.ml', 'ai.onnx.training'], args):
            if (pair not in result):
                result[pair] = ir_version
    for row in table:
        process(*row)
    return result


OP_SET_ID_VERSION_MAP = create_op_set_id_version_map(VERSION_TABLE)


# Given list of opset ids, determine minimum IR version required
def find_min_ir_version_for(opsetidlist: List[OperatorSetIdProto]) -> int:
    default_min_version = 3

    def find_min(domain: Union[Text, None], version: int) -> int:
        key = (domain if domain else 'ai.onnx', version)
        if (key in OP_SET_ID_VERSION_MAP):
            return OP_SET_ID_VERSION_MAP[key]
        else:
            raise ValueError("Unsupported opset-version.")
    if (opsetidlist):
        return max([find_min(x.domain, x.version) for x in opsetidlist])
    return default_min_version  # if no opsets specified


def make_node(
        op_type: Text,
        inputs: Sequence[Text],
        outputs: Sequence[Text],
        name: Optional[Text] = None,
        doc_string: Optional[Text] = None,
        domain: Optional[Text] = None,
        **kwargs: Any
) -> NodeProto:
    """Construct a NodeProto.

    Arguments:
        op_type (string): The name of the operator to construct
        inputs (list of string): list of input names
        outputs (list of string): list of output names
        name (string, default None): optional unique identifier for NodeProto
        doc_string (string, default None): optional documentation string for NodeProto
        domain (string, default None): optional domain for NodeProto.
            If it's None, we will just use default domain (which is empty)
        **kwargs (dict): the attributes of the node.  The acceptable values
            are documented in :func:`make_attribute`.
    """

    node = NodeProto()
    node.op_type = op_type
    node.input.extend(inputs)
    node.output.extend(outputs)
    if name:
        node.name = name
    if doc_string:
        node.doc_string = doc_string
    if domain is not None:
        node.domain = domain
    if kwargs:
        node.attribute.extend(
            make_attribute(key, value)
            for key, value in sorted(kwargs.items())
            if value is not None)
    return node


def make_operatorsetid(
        domain: Text,
        version: int,
) -> OperatorSetIdProto:
    """Construct an OperatorSetIdProto.

    Arguments:
        domain (string): The domain of the operator set id
        version (integer): Version of operator set id
    """
    operatorsetid = OperatorSetIdProto()
    operatorsetid.domain = domain
    operatorsetid.version = version
    return operatorsetid


def make_graph(
    nodes: Sequence[NodeProto],
    name: Text,
    inputs: Sequence[ValueInfoProto],
    outputs: Sequence[ValueInfoProto],
    initializer: Optional[Sequence[TensorProto]] = None,
    doc_string: Optional[Text] = None,
    value_info: Sequence[ValueInfoProto] = [],
    sparse_initializer: Optional[Sequence[SparseTensorProto]] = None,
) -> GraphProto:
    if initializer is None:
        initializer = []
    if sparse_initializer is None:
        sparse_initializer = []
    if value_info is None:
        value_info = []
    graph = GraphProto()
    graph.node.extend(nodes)
    graph.name = name
    graph.input.extend(inputs)
    graph.output.extend(outputs)
    graph.initializer.extend(initializer)
    graph.sparse_initializer.extend(sparse_initializer)
    graph.value_info.extend(value_info)
    if doc_string:
        graph.doc_string = doc_string
    return graph


def make_opsetid(domain: Text, version: int) -> OperatorSetIdProto:
    opsetid = OperatorSetIdProto()
    opsetid.domain = domain
    opsetid.version = version
    return opsetid


def make_function(
    domain: Text,
    fname: Text,
    inputs: Sequence[Text],
    outputs: Sequence[Text],
    nodes: Sequence[NodeProto],
    opset_imports: Sequence[OperatorSetIdProto],
    attributes: Optional[Sequence[Text]] = [],
    doc_string: Optional[Text] = None
) -> FunctionProto:
    f = FunctionProto()
    f.domain = domain
    f.name = fname
    f.input.extend(inputs)
    f.output.extend(outputs)
    f.node.extend(nodes)
    f.opset_import.extend(opset_imports)
    f.attribute.extend(attributes)
    if doc_string:
        f.doc_string = doc_string
    return f


def make_model(graph: GraphProto, **kwargs: Any) -> ModelProto:
    model = ModelProto()
    # Touch model.ir_version so it is stored as the version from which it is
    # generated.
    model.ir_version = IR_VERSION
    model.graph.CopyFrom(graph)

    opset_imports: Optional[Sequence[OperatorSetIdProto]] = None
    opset_imports = kwargs.pop('opset_imports', None)  # type: ignore
    if opset_imports is not None:
        model.opset_import.extend(opset_imports)
    else:
        # Default import
        imp = model.opset_import.add()
        imp.version = defs.onnx_opset_version()

    functions: Optional[Sequence[FunctionProto]] = None
    functions = kwargs.pop('functions', None)  # type: ignore
    if functions is not None:
        model.functions.extend(functions)

    for k, v in kwargs.items():
        # TODO: Does this work with repeated fields?
        setattr(model, k, v)
    return model


# An extension of make_model that infers an IR_VERSION for the model,
# if not specified, using a best-effort-basis.
def make_model_gen_version(graph: GraphProto, **kwargs: Any) -> ModelProto:
    ir_version_field = str('ir_version')
    if (ir_version_field not in kwargs):
        opset_imports_field = str('opset_imports')
        imports = (kwargs[opset_imports_field] if opset_imports_field in kwargs else [])
        kwargs[ir_version_field] = find_min_ir_version_for(imports)
    return make_model(graph, **kwargs)


def set_model_props(model: ModelProto, dict_value: Dict[Text, Text]) -> None:
    del model.metadata_props[:]
    for (k, v) in dict_value.items():
        entry = model.metadata_props.add()
        entry.key = k
        entry.value = v
        # model.metadata_properties.append(entry)


def split_complex_to_pairs(ca: Sequence[np.complex64]) -> Sequence[int]:
    return [(ca[i // 2].real if (i % 2 == 0) else ca[i // 2].imag)
            for i in range(len(ca) * 2)]


def make_tensor(
        name: Text,
        data_type: int,
        dims: Sequence[int],
        vals: Any,
        raw: bool = False
) -> TensorProto:
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

    # Check number of vals specified equals tensor size
    expected_size = 1 if (not raw) else (mapping.TENSOR_TYPE_TO_NP_TYPE[data_type].itemsize)
    # Flatten a numpy array if its rank > 1
    if type(vals) is np.ndarray and len(vals.shape) > 1:
        vals = vals.flatten()
    for d in dims:
        expected_size = expected_size * d

    if len(vals) != expected_size:
        raise ValueError("Number of values does not match tensor's size. Expected {}, but it is {}. "
            .format(expected_size, len(vals)))

    if raw:
        tensor.raw_data = vals
    else:
        if (data_type == TensorProto.COMPLEX64
                or data_type == TensorProto.COMPLEX128):
            vals = split_complex_to_pairs(vals)
        # floa16/bfloat16 are stored as uint16
        elif (data_type == TensorProto.FLOAT16
                or data_type == TensorProto.BFLOAT16):
            vals = np.array(vals).astype(np.float16).view(dtype=np.uint16).flatten().tolist()
        field = mapping.STORAGE_TENSOR_TYPE_TO_FIELD[
            mapping.TENSOR_TYPE_TO_STORAGE_TENSOR_TYPE[data_type]]
        getattr(tensor, field).extend(vals)
    tensor.dims.extend(dims)
    return tensor


def make_sparse_tensor(
    values: TensorProto,
    indices: TensorProto,
    dims: Sequence[int]
) -> SparseTensorProto:
    sparse = SparseTensorProto()
    sparse.values.CopyFrom(values)
    sparse.indices.CopyFrom(indices)
    sparse.dims.extend(dims)
    return sparse


def make_sequence(
        name: Text,
        elem_type: SequenceProto.DataType,
        values: Sequence[Any],
) -> SequenceProto:
    '''
    Make a Sequence with specified value arguments.
    '''
    sequence = SequenceProto()
    sequence.name = name
    sequence.elem_type = elem_type
    values_field = mapping.STORAGE_ELEMENT_TYPE_TO_FIELD[elem_type]
    getattr(sequence, values_field).extend(values)
    return sequence


def make_map(
        name: Text,
        key_type: int,
        keys: List[Any],
        values: SequenceProto
) -> MapProto:
    '''
    Make a Map with specified key-value pair arguments.

    Criteria for conversion:
    - Keys and Values must have the same number of elements
    - Every key in keys must be of the same type
    - Every value in values must be of the same type
    '''
    map = MapProto()
    valid_key_int_types = [TensorProto.INT8, TensorProto.INT16, TensorProto.INT32,
                           TensorProto.INT64, TensorProto.UINT8, TensorProto.UINT16,
                           TensorProto.UINT32, TensorProto.UINT64]
    map.name = name
    map.key_type = key_type
    if key_type == TensorProto.STRING:
        map.string_keys.extend(keys)
    elif key_type in valid_key_int_types:
        map.keys.extend(keys)
    map.values.CopyFrom(values)
    return map


def make_optional(
        name: Text,
        elem_type: OptionalProto.DataType,
        value: Optional[Any],
) -> OptionalProto:
    '''
    Make an Optional with specified value arguments.
    '''
    optional = OptionalProto()
    optional.name = name
    optional.elem_type = elem_type
    if elem_type != 0:
        values_field = mapping.OPTIONAL_ELEMENT_TYPE_TO_FIELD[elem_type]
        getattr(optional, values_field).CopyFrom(value)
    return optional


def _to_bytes_or_false(val: Union[Text, bytes]) -> Union[bytes, bool]:
    """An internal graph to convert the input to a bytes or to False.

    The criteria for conversion is as follows and should be python 2 and 3
    compatible:
    - If val is py2 str or py3 bytes: return bytes
    - If val is py2 unicode or py3 str: return val.decode('utf-8')
    - Otherwise, return False
    """
    if isinstance(val, bytes):
        return val
    try:
        return val.encode('utf-8')
    except AttributeError:
        return False


def make_attribute(
        key: Text,
        value: Any,
        doc_string: Optional[Text] = None
) -> AttributeProto:
    """Makes an AttributeProto based on the value type."""
    attr = AttributeProto()
    attr.name = key
    if doc_string:
        attr.doc_string = doc_string

    is_iterable = isinstance(value, collections.abc.Iterable)
    bytes_or_false = _to_bytes_or_false(value)
    # First, singular cases
    # float
    if isinstance(value, float):
        attr.f = value
        attr.type = AttributeProto.FLOAT
    # integer
    elif isinstance(value, numbers.Integral):
        attr.i = cast(int, value)
        attr.type = AttributeProto.INT
    # string
    elif bytes_or_false is not False:
        assert isinstance(bytes_or_false, bytes)
        attr.s = bytes_or_false
        attr.type = AttributeProto.STRING
    elif isinstance(value, TensorProto):
        attr.t.CopyFrom(value)
        attr.type = AttributeProto.TENSOR
    elif isinstance(value, SparseTensorProto):
        attr.sparse_tensor.CopyFrom(value)
        attr.type = AttributeProto.SPARSE_TENSOR
    elif isinstance(value, GraphProto):
        attr.g.CopyFrom(value)
        attr.type = AttributeProto.GRAPH
    elif isinstance(value, TypeProto):
        attr.tp.CopyFrom(value)
        attr.type = AttributeProto.TYPE_PROTO
    # third, iterable cases
    elif is_iterable:
        byte_array = [_to_bytes_or_false(v) for v in value]
        if all(isinstance(v, numbers.Integral) for v in value):
            # Turn np.int32/64 into Python built-in int.
            attr.ints.extend(int(v) for v in value)
            attr.type = AttributeProto.INTS
        elif all(isinstance(v, numbers.Real) for v in value):
            # Since ints and floats are members of Real, this allows a mix of ints and floats
            # (and converts the ints to floats).
            attr.floats.extend(float(v) for v in value)
            attr.type = AttributeProto.FLOATS
        elif all(map(lambda bytes_or_false: bytes_or_false is not False, byte_array)):
            attr.strings.extend(cast(List[bytes], byte_array))
            attr.type = AttributeProto.STRINGS
        elif all(isinstance(v, TensorProto) for v in value):
            attr.tensors.extend(value)
            attr.type = AttributeProto.TENSORS
        elif all(isinstance(v, SparseTensorProto) for v in value):
            attr.sparse_tensors.extend(value)
            attr.type = AttributeProto.SPARSE_TENSORS
        elif all(isinstance(v, GraphProto) for v in value):
            attr.graphs.extend(value)
            attr.type = AttributeProto.GRAPHS
        elif all(isinstance(tp, TypeProto) for tp in value):
            attr.type_protos.extend(value)
            attr.type = AttributeProto.TYPE_PROTOS
        else:
            raise ValueError(
                "You passed in an iterable attribute but I cannot figure out "
                "its applicable type.")
    else:
        raise TypeError(
            'value "{}" is not valid attribute data type.'.format(value))
    return attr


def get_attribute_value(attr: AttributeProto) -> Any:
    if attr.type == AttributeProto.FLOAT:
        return attr.f
    if attr.type == AttributeProto.INT:
        return attr.i
    if attr.type == AttributeProto.STRING:
        return attr.s
    if attr.type == AttributeProto.TENSOR:
        return attr.t
    if attr.type == AttributeProto.SPARSE_TENSOR:
        return attr.sparse_tensor
    if attr.type == AttributeProto.GRAPH:
        return attr.g
    if attr.type == AttributeProto.TYPE_PROTO:
        return attr.tp
    if attr.type == AttributeProto.FLOATS:
        return list(attr.floats)
    if attr.type == AttributeProto.INTS:
        return list(attr.ints)
    if attr.type == AttributeProto.STRINGS:
        return list(attr.strings)
    if attr.type == AttributeProto.TENSORS:
        return list(attr.tensors)
    if attr.type == AttributeProto.SPARSE_TENSORS:
        return list(attr.sparse_tensors)
    if attr.type == AttributeProto.GRAPHS:
        return list(attr.graphs)
    if attr.type == AttributeProto.TYPE_PROTOS:
        return list(attr.type_protos)
    raise ValueError("Unsupported ONNX attribute: {}".format(attr))


def make_empty_tensor_value_info(name: Text) -> ValueInfoProto:
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    return value_info_proto


def make_tensor_type_proto(
        elem_type: int,
        shape: Optional[Sequence[Union[Text, int, None]]],
        shape_denotation: Optional[List[Text]] = None,
) -> TypeProto:
    """Makes a Tensor TypeProto based on the data type and shape."""

    type_proto = TypeProto()
    tensor_type_proto = type_proto.tensor_type
    tensor_type_proto.elem_type = elem_type
    tensor_shape_proto = tensor_type_proto.shape

    if shape is not None:
        # You might think this is a no-op (extending a normal Python
        # list by [] certainly is), but protobuf lists work a little
        # differently; if a field is never set, it is omitted from the
        # resulting protobuf; a list that is explicitly set to be
        # empty will get an (empty) entry in the protobuf. This
        # difference is visible to our consumers, so make sure we emit
        # an empty shape!
        tensor_shape_proto.dim.extend([])

        if shape_denotation:
            if len(shape_denotation) != len(shape):
                raise ValueError(
                    'Invalid shape_denotation. '
                    'Must be of the same length as shape.')

        for i, d in enumerate(shape):
            dim = tensor_shape_proto.dim.add()
            if d is None:
                pass
            elif isinstance(d, int):
                dim.dim_value = d
            elif isinstance(d, str):
                dim.dim_param = d
            else:
                raise ValueError(
                    'Invalid item in shape: {}. '
                    'Needs to be of int or str.'.format(d))

            if shape_denotation:
                dim.denotation = shape_denotation[i]

    return type_proto


def make_tensor_value_info(
        name: Text,
        elem_type: int,
        shape: Optional[Sequence[Union[Text, int, None]]],
        doc_string: Text = "",
        shape_denotation: Optional[List[Text]] = None,
) -> ValueInfoProto:
    """Makes a ValueInfoProto based on the data type and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    tensor_type_proto = make_tensor_type_proto(elem_type, shape, shape_denotation)
    value_info_proto.type.CopyFrom(tensor_type_proto)
    return value_info_proto


def make_sparse_tensor_type_proto(
        elem_type: int,
        shape: Optional[Sequence[Union[Text, int, None]]],
        shape_denotation: Optional[List[Text]] = None,
) -> TypeProto:
    """Makes a SparseTensor TypeProto based on the data type and shape."""

    type_proto = TypeProto()
    sparse_tensor_type_proto = type_proto.sparse_tensor_type
    sparse_tensor_type_proto.elem_type = elem_type
    sparse_tensor_shape_proto = sparse_tensor_type_proto.shape

    if shape is not None:
        # You might think this is a no-op (extending a normal Python
        # list by [] certainly is), but protobuf lists work a little
        # differently; if a field is never set, it is omitted from the
        # resulting protobuf; a list that is explicitly set to be
        # empty will get an (empty) entry in the protobuf. This
        # difference is visible to our consumers, so make sure we emit
        # an empty shape!
        sparse_tensor_shape_proto.dim.extend([])

        if shape_denotation:
            if len(shape_denotation) != len(shape):
                raise ValueError(
                    'Invalid shape_denotation. '
                    'Must be of the same length as shape.')

        for i, d in enumerate(shape):
            dim = sparse_tensor_shape_proto.dim.add()
            if d is None:
                pass
            elif isinstance(d, int):
                dim.dim_value = d
            elif isinstance(d, str):
                dim.dim_param = d
            else:
                raise ValueError(
                    'Invalid item in shape: {}. '
                    'Needs to be of int or text.'.format(d))

            if shape_denotation:
                dim.denotation = shape_denotation[i]

    return type_proto


def make_sparse_tensor_value_info(
        name: Text,
        elem_type: int,
        shape: Optional[Sequence[Union[Text, int, None]]],
        doc_string: Text = "",
        shape_denotation: Optional[List[Text]] = None,
) -> ValueInfoProto:
    """Makes a SparseTensor ValueInfoProto based on the data type and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    sparse_tensor_type_proto = make_sparse_tensor_type_proto(elem_type, shape, shape_denotation)
    value_info_proto.type.sparse_tensor_type.CopyFrom(sparse_tensor_type_proto.sparse_tensor_type)
    return value_info_proto


def make_sequence_type_proto(
        inner_type_proto: TypeProto,
) -> TypeProto:
    """Makes a sequence TypeProto."""
    type_proto = TypeProto()
    type_proto.sequence_type.elem_type.CopyFrom(inner_type_proto)
    return type_proto


def make_optional_type_proto(
        inner_type_proto: TypeProto,
) -> TypeProto:
    """Makes an optional TypeProto."""
    type_proto = TypeProto()
    type_proto.optional_type.elem_type.CopyFrom(inner_type_proto)
    return type_proto


def make_value_info(
        name: Text,
        type_proto: TypeProto,
        doc_string: Text = "",
) -> ValueInfoProto:
    """Makes a ValueInfoProto with the given type_proto."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    value_info_proto.type.CopyFrom(type_proto)
    return value_info_proto


def _sanitize_str(s: Union[Text, bytes]) -> Text:
    if isinstance(s, str):
        sanitized = s
    elif isinstance(s, bytes):
        sanitized = s.decode('utf-8', errors='ignore')
    else:
        sanitized = str(s)
    if len(sanitized) < 64:
        return sanitized
    return sanitized[:64] + '...<+len=%d>' % (len(sanitized) - 64)


def make_tensor_sequence_value_info(
        name: Text,
        elem_type: int,
        shape: Optional[Sequence[Union[Text, int, None]]],
        doc_string: Text = "",
        elem_shape_denotation: Optional[List[Text]] = None,
) -> ValueInfoProto:
    """Makes a Sequence[Tensors] ValueInfoProto based on the data type and shape."""
    value_info_proto = ValueInfoProto()
    value_info_proto.name = name
    if doc_string:
        value_info_proto.doc_string = doc_string

    tensor_type_proto = make_tensor_type_proto(elem_type, shape, elem_shape_denotation)
    sequence_type_proto = make_sequence_type_proto(tensor_type_proto)
    value_info_proto.type.sequence_type.CopyFrom(sequence_type_proto.sequence_type)

    return value_info_proto


def printable_attribute(attr: AttributeProto, subgraphs: bool = False) -> Union[Text, Tuple[Text, List[GraphProto]]]:
    content = []
    content.append(attr.name)
    content.append("=")

    def str_float(f: float) -> Text:
        # NB: Different Python versions print different numbers of trailing
        # decimals, specifying this explicitly keeps it consistent for all
        # versions
        return '{:.15g}'.format(f)

    def str_int(i: int) -> Text:
        # NB: In Python 2, longs will repr() as '2L', which is ugly and
        # unnecessary.  Explicitly format it to keep it consistent.
        return '{:d}'.format(i)

    def str_str(s: Text) -> Text:
        return repr(s)

    _T = TypeVar('_T')  # noqa

    def str_list(str_elem: Callable[[_T], Text], xs: Sequence[_T]) -> Text:
        return '[' + ', '.join(map(str_elem, xs)) + ']'

    # for now, this logic should continue to work as long as we are running on a proto3
    # implementation. If/when we switch to proto3, we will need to use attr.type

    # To support printing subgraphs, if we find a graph attribute, print out
    # its name here and pass the graph itself up to the caller for later
    # printing.
    graphs = []
    if attr.HasField("f"):
        content.append(str_float(attr.f))
    elif attr.HasField("i"):
        content.append(str_int(attr.i))
    elif attr.HasField("s"):
        # TODO: Bit nervous about Python 2 / Python 3 determinism implications
        content.append(repr(_sanitize_str(attr.s)))
    elif attr.HasField("t"):
        if len(attr.t.dims) > 0:
            content.append("<Tensor>")
        else:
            # special case to print scalars
            field = STORAGE_TENSOR_TYPE_TO_FIELD[attr.t.data_type]
            content.append('<Scalar Tensor {}>'.format(str(getattr(attr.t, field))))
    elif attr.HasField("g"):
        content.append("<graph {}>".format(attr.g.name))
        graphs.append(attr.g)
    elif attr.HasField("tp"):
        content.append("<Type Proto {}>".format(attr.tp))
    elif attr.floats:
        content.append(str_list(str_float, attr.floats))
    elif attr.ints:
        content.append(str_list(str_int, attr.ints))
    elif attr.strings:
        # TODO: Bit nervous about Python 2 / Python 3 determinism implications
        content.append(str(list(map(_sanitize_str, attr.strings))))
    elif attr.tensors:
        content.append("[<Tensor>, ...]")
    elif attr.type_protos:
        content.append('[')
        for i, tp in enumerate(attr.type_protos):
            comma = ',' if i != len(attr.type_protos) - 1 else ''
            content.append('<Type Proto {}>{}'.format(tp, comma))
        content.append(']')
    elif attr.graphs:
        content.append('[')
        for i, g in enumerate(attr.graphs):
            comma = ',' if i != len(attr.graphs) - 1 else ''
            content.append('<graph {}>{}'.format(g.name, comma))
        content.append(']')
        graphs.extend(attr.graphs)
    else:
        content.append("<Unknown>")
    if subgraphs:
        return ' '.join(content), graphs
    else:
        return ' '.join(content)


def printable_dim(dim: TensorShapeProto.Dimension) -> Text:
    which = dim.WhichOneof('value')
    assert which is not None
    return str(getattr(dim, which))


def printable_type(t: TypeProto) -> Text:
    if t.WhichOneof('value') == "tensor_type":
        s = TensorProto.DataType.Name(t.tensor_type.elem_type)
        if t.tensor_type.HasField('shape'):
            if len(t.tensor_type.shape.dim):
                s += str(', ' + 'x'.join(map(printable_dim, t.tensor_type.shape.dim)))
            else:
                s += str(', scalar')
        return s
    if t.WhichOneof('value') is None:
        return ""
    return 'Unknown type {}'.format(t.WhichOneof('value'))


def printable_value_info(v: ValueInfoProto) -> Text:
    s = '%{}'.format(v.name)
    if v.type:
        s = '{}[{}]'.format(s, printable_type(v.type))
    return s


def printable_tensor_proto(t: TensorProto) -> Text:
    s = '%{}['.format(t.name)
    s += TensorProto.DataType.Name(t.data_type)
    if t.dims is not None:
        if len(t.dims):
            s += str(', ' + 'x'.join(map(str, t.dims)))
        else:
            s += str(', scalar')
    s += ']'
    return s


def printable_node(node: NodeProto, prefix: Text = '', subgraphs: bool = False) -> Union[Text, Tuple[Text, List[GraphProto]]]:
    content = []
    if len(node.output):
        content.append(
            ', '.join(['%{}'.format(name) for name in node.output]))
        content.append('=')
    # To deal with nested graphs
    graphs: List[GraphProto] = []
    printed_attrs = []
    for attr in node.attribute:
        if subgraphs:
            printed_attr_subgraphs = printable_attribute(attr, subgraphs)
            assert isinstance(printed_attr_subgraphs[1], list)
            graphs.extend(printed_attr_subgraphs[1])
            printed_attrs.append(printed_attr_subgraphs[0])
        else:
            printed = printable_attribute(attr)
            assert isinstance(printed, Text)
            printed_attrs.append(printed)
    printed_attributes = ', '.join(sorted(printed_attrs))
    printed_inputs = ', '.join(['%{}'.format(name) for name in node.input])
    if node.attribute:
        content.append("{}[{}]({})".format(node.op_type, printed_attributes, printed_inputs))
    else:
        content.append("{}({})".format(node.op_type, printed_inputs))
    if subgraphs:
        return prefix + ' '.join(content), graphs
    else:
        return prefix + ' '.join(content)


def printable_graph(graph: GraphProto, prefix: Text = '') -> Text:
    content = []
    indent = prefix + '  '
    # header
    header = ['graph', graph.name]
    initializers = {t.name for t in graph.initializer}
    if len(graph.input):
        header.append("(")
        in_strs = []  # required inputs
        in_with_init_strs = []  # optional inputs with initializer providing default value
        for inp in graph.input:
            if inp.name not in initializers:
                in_strs.append(printable_value_info(inp))
            else:
                in_with_init_strs.append(printable_value_info(inp))
        if in_strs:
            content.append(prefix + ' '.join(header))
            header = []
            for line in in_strs:
                content.append(prefix + '  ' + line)
        header.append(")")

        if in_with_init_strs:
            header.append("optional inputs with matching initializers (")
            content.append(prefix + ' '.join(header))
            header = []
            for line in in_with_init_strs:
                content.append(prefix + '  ' + line)
            header.append(")")

        # from IR 4 onwards an initializer is not required to have a matching graph input
        # so output the name, type and shape of those as well
        if len(in_with_init_strs) < len(initializers):
            graph_inputs = {i.name for i in graph.input}
            init_strs = [printable_tensor_proto(i) for i in graph.initializer
                         if i.name not in graph_inputs]
            header.append("initializers (")
            content.append(prefix + ' '.join(header))
            header = []
            for line in init_strs:
                content.append(prefix + '  ' + line)
            header.append(")")

    header.append('{')
    content.append(prefix + ' '.join(header))
    graphs: List[GraphProto] = []
    # body
    for node in graph.node:
        contents_subgraphs = printable_node(node, indent, subgraphs=True)
        assert isinstance(contents_subgraphs[1], list)
        content.append(contents_subgraphs[0])
        graphs.extend(contents_subgraphs[1])
    # tail
    tail = ['return']
    if len(graph.output):
        tail.append(
            ', '.join(['%{}'.format(out.name) for out in graph.output]))
    content.append(indent + ' '.join(tail))
    # closing bracket
    content.append(prefix + '}')
    for g in graphs:
        content.append('\n' + printable_graph(g))
    return '\n'.join(content)


def strip_doc_string(proto: google.protobuf.message.Message) -> None:
    """
    Empties `doc_string` field on any nested protobuf messages
    """
    assert isinstance(proto, google.protobuf.message.Message)
    for descriptor in proto.DESCRIPTOR.fields:
        if descriptor.name == 'doc_string':
            proto.ClearField(descriptor.name)
        elif descriptor.type == descriptor.TYPE_MESSAGE:
            if descriptor.label == descriptor.LABEL_REPEATED:
                for x in getattr(proto, descriptor.name):
                    strip_doc_string(x)
            elif proto.HasField(descriptor.name):
                strip_doc_string(getattr(proto, descriptor.name))


def make_training_info(algorithm: GraphProto, algorithm_bindings: AssignmentBindingType, initialization: Optional[GraphProto], initialization_bindings: Optional[AssignmentBindingType]) -> TrainingInfoProto:
    training_info = TrainingInfoProto()
    training_info.algorithm.CopyFrom(algorithm)
    for k, v in algorithm_bindings:
        binding = training_info.update_binding.add()
        binding.key = k
        binding.value = v

    if initialization:
        training_info.initialization.CopyFrom(initialization)
    if initialization_bindings:
        for k, v in initialization_bindings:
            binding = training_info.initialization_binding.add()
            binding.key = k
            binding.value = v

    return training_info


# For backwards compatibility
def make_sequence_value_info(
        name: Text,
        elem_type: int,
        shape: Optional[Sequence[Union[Text, int, None]]],
        doc_string: Text = "",
        elem_shape_denotation: Optional[List[Text]] = None,
) -> ValueInfoProto:
    """Makes a Sequence[Tensors] ValueInfoProto based on the data type and shape."""
    warnings.warn(str("`onnx.helper.make_sequence_value_info` is a deprecated alias for `onnx.helper.make_tensor_sequence_value_info`. To silence this warning, please use `make_tensor_sequence_value_info` for `TensorProto` sequences. Deprecated in ONNX v1.10.0, `onnx.helper.make_sequence_value_info alias` will be removed in an upcoming release."), DeprecationWarning, stacklevel=2)
    return make_tensor_sequence_value_info(name, elem_type, shape, doc_string, elem_shape_denotation)
