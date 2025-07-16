from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Version(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    _START_VERSION: _ClassVar[Version]
    IR_VERSION_2017_10_10: _ClassVar[Version]
    IR_VERSION_2017_10_30: _ClassVar[Version]
    IR_VERSION_2017_11_3: _ClassVar[Version]
    IR_VERSION_2019_1_22: _ClassVar[Version]
    IR_VERSION_2019_3_18: _ClassVar[Version]
    IR_VERSION_2019_9_19: _ClassVar[Version]
    IR_VERSION_2020_5_8: _ClassVar[Version]
    IR_VERSION_2021_7_30: _ClassVar[Version]
    IR_VERSION_2023_5_5: _ClassVar[Version]
    IR_VERSION_2024_3_25: _ClassVar[Version]
    IR_VERSION_2025_05_12: _ClassVar[Version]
    IR_VERSION: _ClassVar[Version]

class OperatorStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXPERIMENTAL: _ClassVar[OperatorStatus]
    STABLE: _ClassVar[OperatorStatus]
_START_VERSION: Version
IR_VERSION_2017_10_10: Version
IR_VERSION_2017_10_30: Version
IR_VERSION_2017_11_3: Version
IR_VERSION_2019_1_22: Version
IR_VERSION_2019_3_18: Version
IR_VERSION_2019_9_19: Version
IR_VERSION_2020_5_8: Version
IR_VERSION_2021_7_30: Version
IR_VERSION_2023_5_5: Version
IR_VERSION_2024_3_25: Version
IR_VERSION_2025_05_12: Version
IR_VERSION: Version
EXPERIMENTAL: OperatorStatus
STABLE: OperatorStatus

class AttributeProto(_message.Message):
    __slots__ = ("name", "ref_attr_name", "doc_string", "type", "f", "i", "s", "t", "g", "sparse_tensor", "tp", "floats", "ints", "strings", "tensors", "graphs", "sparse_tensors", "type_protos")
    class AttributeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNDEFINED: _ClassVar[AttributeProto.AttributeType]
        FLOAT: _ClassVar[AttributeProto.AttributeType]
        INT: _ClassVar[AttributeProto.AttributeType]
        STRING: _ClassVar[AttributeProto.AttributeType]
        TENSOR: _ClassVar[AttributeProto.AttributeType]
        GRAPH: _ClassVar[AttributeProto.AttributeType]
        SPARSE_TENSOR: _ClassVar[AttributeProto.AttributeType]
        TYPE_PROTO: _ClassVar[AttributeProto.AttributeType]
        FLOATS: _ClassVar[AttributeProto.AttributeType]
        INTS: _ClassVar[AttributeProto.AttributeType]
        STRINGS: _ClassVar[AttributeProto.AttributeType]
        TENSORS: _ClassVar[AttributeProto.AttributeType]
        GRAPHS: _ClassVar[AttributeProto.AttributeType]
        SPARSE_TENSORS: _ClassVar[AttributeProto.AttributeType]
        TYPE_PROTOS: _ClassVar[AttributeProto.AttributeType]
    UNDEFINED: AttributeProto.AttributeType
    FLOAT: AttributeProto.AttributeType
    INT: AttributeProto.AttributeType
    STRING: AttributeProto.AttributeType
    TENSOR: AttributeProto.AttributeType
    GRAPH: AttributeProto.AttributeType
    SPARSE_TENSOR: AttributeProto.AttributeType
    TYPE_PROTO: AttributeProto.AttributeType
    FLOATS: AttributeProto.AttributeType
    INTS: AttributeProto.AttributeType
    STRINGS: AttributeProto.AttributeType
    TENSORS: AttributeProto.AttributeType
    GRAPHS: AttributeProto.AttributeType
    SPARSE_TENSORS: AttributeProto.AttributeType
    TYPE_PROTOS: AttributeProto.AttributeType
    NAME_FIELD_NUMBER: _ClassVar[int]
    REF_ATTR_NAME_FIELD_NUMBER: _ClassVar[int]
    DOC_STRING_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    F_FIELD_NUMBER: _ClassVar[int]
    I_FIELD_NUMBER: _ClassVar[int]
    S_FIELD_NUMBER: _ClassVar[int]
    T_FIELD_NUMBER: _ClassVar[int]
    G_FIELD_NUMBER: _ClassVar[int]
    SPARSE_TENSOR_FIELD_NUMBER: _ClassVar[int]
    TP_FIELD_NUMBER: _ClassVar[int]
    FLOATS_FIELD_NUMBER: _ClassVar[int]
    INTS_FIELD_NUMBER: _ClassVar[int]
    STRINGS_FIELD_NUMBER: _ClassVar[int]
    TENSORS_FIELD_NUMBER: _ClassVar[int]
    GRAPHS_FIELD_NUMBER: _ClassVar[int]
    SPARSE_TENSORS_FIELD_NUMBER: _ClassVar[int]
    TYPE_PROTOS_FIELD_NUMBER: _ClassVar[int]
    name: str
    ref_attr_name: str
    doc_string: str
    type: AttributeProto.AttributeType
    f: float
    i: int
    s: bytes
    t: TensorProto
    g: GraphProto
    sparse_tensor: SparseTensorProto
    tp: TypeProto
    floats: _containers.RepeatedScalarFieldContainer[float]
    ints: _containers.RepeatedScalarFieldContainer[int]
    strings: _containers.RepeatedScalarFieldContainer[bytes]
    tensors: _containers.RepeatedCompositeFieldContainer[TensorProto]
    graphs: _containers.RepeatedCompositeFieldContainer[GraphProto]
    sparse_tensors: _containers.RepeatedCompositeFieldContainer[SparseTensorProto]
    type_protos: _containers.RepeatedCompositeFieldContainer[TypeProto]
    def __init__(self, name: _Optional[str] = ..., ref_attr_name: _Optional[str] = ..., doc_string: _Optional[str] = ..., type: _Optional[_Union[AttributeProto.AttributeType, str]] = ..., f: _Optional[float] = ..., i: _Optional[int] = ..., s: _Optional[bytes] = ..., t: _Optional[_Union[TensorProto, _Mapping]] = ..., g: _Optional[_Union[GraphProto, _Mapping]] = ..., sparse_tensor: _Optional[_Union[SparseTensorProto, _Mapping]] = ..., tp: _Optional[_Union[TypeProto, _Mapping]] = ..., floats: _Optional[_Iterable[float]] = ..., ints: _Optional[_Iterable[int]] = ..., strings: _Optional[_Iterable[bytes]] = ..., tensors: _Optional[_Iterable[_Union[TensorProto, _Mapping]]] = ..., graphs: _Optional[_Iterable[_Union[GraphProto, _Mapping]]] = ..., sparse_tensors: _Optional[_Iterable[_Union[SparseTensorProto, _Mapping]]] = ..., type_protos: _Optional[_Iterable[_Union[TypeProto, _Mapping]]] = ...) -> None: ...

class ValueInfoProto(_message.Message):
    __slots__ = ("name", "type", "doc_string", "metadata_props")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    DOC_STRING_FIELD_NUMBER: _ClassVar[int]
    METADATA_PROPS_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: TypeProto
    doc_string: str
    metadata_props: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]
    def __init__(self, name: _Optional[str] = ..., type: _Optional[_Union[TypeProto, _Mapping]] = ..., doc_string: _Optional[str] = ..., metadata_props: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]] = ...) -> None: ...

class NodeProto(_message.Message):
    __slots__ = ("input", "output", "name", "op_type", "domain", "overload", "attribute", "doc_string", "metadata_props", "device_configurations")
    INPUT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    OP_TYPE_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    OVERLOAD_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTE_FIELD_NUMBER: _ClassVar[int]
    DOC_STRING_FIELD_NUMBER: _ClassVar[int]
    METADATA_PROPS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_CONFIGURATIONS_FIELD_NUMBER: _ClassVar[int]
    input: _containers.RepeatedScalarFieldContainer[str]
    output: _containers.RepeatedScalarFieldContainer[str]
    name: str
    op_type: str
    domain: str
    overload: str
    attribute: _containers.RepeatedCompositeFieldContainer[AttributeProto]
    doc_string: str
    metadata_props: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]
    device_configurations: _containers.RepeatedCompositeFieldContainer[NodeDeviceConfigurationProto]
    def __init__(self, input: _Optional[_Iterable[str]] = ..., output: _Optional[_Iterable[str]] = ..., name: _Optional[str] = ..., op_type: _Optional[str] = ..., domain: _Optional[str] = ..., overload: _Optional[str] = ..., attribute: _Optional[_Iterable[_Union[AttributeProto, _Mapping]]] = ..., doc_string: _Optional[str] = ..., metadata_props: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]] = ..., device_configurations: _Optional[_Iterable[_Union[NodeDeviceConfigurationProto, _Mapping]]] = ...) -> None: ...

class IntIntListEntryProto(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: int
    value: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, key: _Optional[int] = ..., value: _Optional[_Iterable[int]] = ...) -> None: ...

class NodeDeviceConfigurationProto(_message.Message):
    __slots__ = ("configuration_id", "sharding_spec", "pipeline_stage")
    CONFIGURATION_ID_FIELD_NUMBER: _ClassVar[int]
    SHARDING_SPEC_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_STAGE_FIELD_NUMBER: _ClassVar[int]
    configuration_id: str
    sharding_spec: _containers.RepeatedCompositeFieldContainer[ShardingSpecProto]
    pipeline_stage: int
    def __init__(self, configuration_id: _Optional[str] = ..., sharding_spec: _Optional[_Iterable[_Union[ShardingSpecProto, _Mapping]]] = ..., pipeline_stage: _Optional[int] = ...) -> None: ...

class ShardingSpecProto(_message.Message):
    __slots__ = ("tensor_name", "device", "index_to_device_group_map", "sharded_dim")
    TENSOR_NAME_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    INDEX_TO_DEVICE_GROUP_MAP_FIELD_NUMBER: _ClassVar[int]
    SHARDED_DIM_FIELD_NUMBER: _ClassVar[int]
    tensor_name: str
    device: _containers.RepeatedScalarFieldContainer[int]
    index_to_device_group_map: _containers.RepeatedCompositeFieldContainer[IntIntListEntryProto]
    sharded_dim: _containers.RepeatedCompositeFieldContainer[ShardedDimProto]
    def __init__(self, tensor_name: _Optional[str] = ..., device: _Optional[_Iterable[int]] = ..., index_to_device_group_map: _Optional[_Iterable[_Union[IntIntListEntryProto, _Mapping]]] = ..., sharded_dim: _Optional[_Iterable[_Union[ShardedDimProto, _Mapping]]] = ...) -> None: ...

class ShardedDimProto(_message.Message):
    __slots__ = ("axis", "simple_sharding")
    AXIS_FIELD_NUMBER: _ClassVar[int]
    SIMPLE_SHARDING_FIELD_NUMBER: _ClassVar[int]
    axis: int
    simple_sharding: _containers.RepeatedCompositeFieldContainer[SimpleShardedDimProto]
    def __init__(self, axis: _Optional[int] = ..., simple_sharding: _Optional[_Iterable[_Union[SimpleShardedDimProto, _Mapping]]] = ...) -> None: ...

class SimpleShardedDimProto(_message.Message):
    __slots__ = ("dim_value", "dim_param", "num_shards")
    DIM_VALUE_FIELD_NUMBER: _ClassVar[int]
    DIM_PARAM_FIELD_NUMBER: _ClassVar[int]
    NUM_SHARDS_FIELD_NUMBER: _ClassVar[int]
    dim_value: int
    dim_param: str
    num_shards: int
    def __init__(self, dim_value: _Optional[int] = ..., dim_param: _Optional[str] = ..., num_shards: _Optional[int] = ...) -> None: ...

class TrainingInfoProto(_message.Message):
    __slots__ = ("initialization", "algorithm", "initialization_binding", "update_binding")
    INITIALIZATION_FIELD_NUMBER: _ClassVar[int]
    ALGORITHM_FIELD_NUMBER: _ClassVar[int]
    INITIALIZATION_BINDING_FIELD_NUMBER: _ClassVar[int]
    UPDATE_BINDING_FIELD_NUMBER: _ClassVar[int]
    initialization: GraphProto
    algorithm: GraphProto
    initialization_binding: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]
    update_binding: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]
    def __init__(self, initialization: _Optional[_Union[GraphProto, _Mapping]] = ..., algorithm: _Optional[_Union[GraphProto, _Mapping]] = ..., initialization_binding: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]] = ..., update_binding: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]] = ...) -> None: ...

class ModelProto(_message.Message):
    __slots__ = ("ir_version", "opset_import", "producer_name", "producer_version", "domain", "model_version", "doc_string", "graph", "metadata_props", "training_info", "functions", "configuration")
    IR_VERSION_FIELD_NUMBER: _ClassVar[int]
    OPSET_IMPORT_FIELD_NUMBER: _ClassVar[int]
    PRODUCER_NAME_FIELD_NUMBER: _ClassVar[int]
    PRODUCER_VERSION_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    MODEL_VERSION_FIELD_NUMBER: _ClassVar[int]
    DOC_STRING_FIELD_NUMBER: _ClassVar[int]
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    METADATA_PROPS_FIELD_NUMBER: _ClassVar[int]
    TRAINING_INFO_FIELD_NUMBER: _ClassVar[int]
    FUNCTIONS_FIELD_NUMBER: _ClassVar[int]
    CONFIGURATION_FIELD_NUMBER: _ClassVar[int]
    ir_version: int
    opset_import: _containers.RepeatedCompositeFieldContainer[OperatorSetIdProto]
    producer_name: str
    producer_version: str
    domain: str
    model_version: int
    doc_string: str
    graph: GraphProto
    metadata_props: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]
    training_info: _containers.RepeatedCompositeFieldContainer[TrainingInfoProto]
    functions: _containers.RepeatedCompositeFieldContainer[FunctionProto]
    configuration: _containers.RepeatedCompositeFieldContainer[DeviceConfigurationProto]
    def __init__(self, ir_version: _Optional[int] = ..., opset_import: _Optional[_Iterable[_Union[OperatorSetIdProto, _Mapping]]] = ..., producer_name: _Optional[str] = ..., producer_version: _Optional[str] = ..., domain: _Optional[str] = ..., model_version: _Optional[int] = ..., doc_string: _Optional[str] = ..., graph: _Optional[_Union[GraphProto, _Mapping]] = ..., metadata_props: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]] = ..., training_info: _Optional[_Iterable[_Union[TrainingInfoProto, _Mapping]]] = ..., functions: _Optional[_Iterable[_Union[FunctionProto, _Mapping]]] = ..., configuration: _Optional[_Iterable[_Union[DeviceConfigurationProto, _Mapping]]] = ...) -> None: ...

class DeviceConfigurationProto(_message.Message):
    __slots__ = ("name", "num_devices", "device")
    NAME_FIELD_NUMBER: _ClassVar[int]
    NUM_DEVICES_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    name: str
    num_devices: int
    device: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., num_devices: _Optional[int] = ..., device: _Optional[_Iterable[str]] = ...) -> None: ...

class StringStringEntryProto(_message.Message):
    __slots__ = ("key", "value")
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...

class TensorAnnotation(_message.Message):
    __slots__ = ("tensor_name", "quant_parameter_tensor_names")
    TENSOR_NAME_FIELD_NUMBER: _ClassVar[int]
    QUANT_PARAMETER_TENSOR_NAMES_FIELD_NUMBER: _ClassVar[int]
    tensor_name: str
    quant_parameter_tensor_names: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]
    def __init__(self, tensor_name: _Optional[str] = ..., quant_parameter_tensor_names: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]] = ...) -> None: ...

class GraphProto(_message.Message):
    __slots__ = ("node", "name", "initializer", "sparse_initializer", "doc_string", "input", "output", "value_info", "quantization_annotation", "metadata_props")
    NODE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    INITIALIZER_FIELD_NUMBER: _ClassVar[int]
    SPARSE_INITIALIZER_FIELD_NUMBER: _ClassVar[int]
    DOC_STRING_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    VALUE_INFO_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_ANNOTATION_FIELD_NUMBER: _ClassVar[int]
    METADATA_PROPS_FIELD_NUMBER: _ClassVar[int]
    node: _containers.RepeatedCompositeFieldContainer[NodeProto]
    name: str
    initializer: _containers.RepeatedCompositeFieldContainer[TensorProto]
    sparse_initializer: _containers.RepeatedCompositeFieldContainer[SparseTensorProto]
    doc_string: str
    input: _containers.RepeatedCompositeFieldContainer[ValueInfoProto]
    output: _containers.RepeatedCompositeFieldContainer[ValueInfoProto]
    value_info: _containers.RepeatedCompositeFieldContainer[ValueInfoProto]
    quantization_annotation: _containers.RepeatedCompositeFieldContainer[TensorAnnotation]
    metadata_props: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]
    def __init__(self, node: _Optional[_Iterable[_Union[NodeProto, _Mapping]]] = ..., name: _Optional[str] = ..., initializer: _Optional[_Iterable[_Union[TensorProto, _Mapping]]] = ..., sparse_initializer: _Optional[_Iterable[_Union[SparseTensorProto, _Mapping]]] = ..., doc_string: _Optional[str] = ..., input: _Optional[_Iterable[_Union[ValueInfoProto, _Mapping]]] = ..., output: _Optional[_Iterable[_Union[ValueInfoProto, _Mapping]]] = ..., value_info: _Optional[_Iterable[_Union[ValueInfoProto, _Mapping]]] = ..., quantization_annotation: _Optional[_Iterable[_Union[TensorAnnotation, _Mapping]]] = ..., metadata_props: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]] = ...) -> None: ...

class TensorProto(_message.Message):
    __slots__ = ("dims", "data_type", "segment", "float_data", "int32_data", "string_data", "int64_data", "name", "doc_string", "raw_data", "external_data", "data_location", "double_data", "uint64_data", "metadata_props")
    class DataType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        UNDEFINED: _ClassVar[TensorProto.DataType]
        FLOAT: _ClassVar[TensorProto.DataType]
        UINT8: _ClassVar[TensorProto.DataType]
        INT8: _ClassVar[TensorProto.DataType]
        UINT16: _ClassVar[TensorProto.DataType]
        INT16: _ClassVar[TensorProto.DataType]
        INT32: _ClassVar[TensorProto.DataType]
        INT64: _ClassVar[TensorProto.DataType]
        STRING: _ClassVar[TensorProto.DataType]
        BOOL: _ClassVar[TensorProto.DataType]
        FLOAT16: _ClassVar[TensorProto.DataType]
        DOUBLE: _ClassVar[TensorProto.DataType]
        UINT32: _ClassVar[TensorProto.DataType]
        UINT64: _ClassVar[TensorProto.DataType]
        COMPLEX64: _ClassVar[TensorProto.DataType]
        COMPLEX128: _ClassVar[TensorProto.DataType]
        BFLOAT16: _ClassVar[TensorProto.DataType]
        FLOAT8E4M3FN: _ClassVar[TensorProto.DataType]
        FLOAT8E4M3FNUZ: _ClassVar[TensorProto.DataType]
        FLOAT8E5M2: _ClassVar[TensorProto.DataType]
        FLOAT8E5M2FNUZ: _ClassVar[TensorProto.DataType]
        UINT4: _ClassVar[TensorProto.DataType]
        INT4: _ClassVar[TensorProto.DataType]
        FLOAT4E2M1: _ClassVar[TensorProto.DataType]
        FLOAT8E8M0: _ClassVar[TensorProto.DataType]
    UNDEFINED: TensorProto.DataType
    FLOAT: TensorProto.DataType
    UINT8: TensorProto.DataType
    INT8: TensorProto.DataType
    UINT16: TensorProto.DataType
    INT16: TensorProto.DataType
    INT32: TensorProto.DataType
    INT64: TensorProto.DataType
    STRING: TensorProto.DataType
    BOOL: TensorProto.DataType
    FLOAT16: TensorProto.DataType
    DOUBLE: TensorProto.DataType
    UINT32: TensorProto.DataType
    UINT64: TensorProto.DataType
    COMPLEX64: TensorProto.DataType
    COMPLEX128: TensorProto.DataType
    BFLOAT16: TensorProto.DataType
    FLOAT8E4M3FN: TensorProto.DataType
    FLOAT8E4M3FNUZ: TensorProto.DataType
    FLOAT8E5M2: TensorProto.DataType
    FLOAT8E5M2FNUZ: TensorProto.DataType
    UINT4: TensorProto.DataType
    INT4: TensorProto.DataType
    FLOAT4E2M1: TensorProto.DataType
    FLOAT8E8M0: TensorProto.DataType
    class DataLocation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        DEFAULT: _ClassVar[TensorProto.DataLocation]
        EXTERNAL: _ClassVar[TensorProto.DataLocation]
    DEFAULT: TensorProto.DataLocation
    EXTERNAL: TensorProto.DataLocation
    class Segment(_message.Message):
        __slots__ = ("begin", "end")
        BEGIN_FIELD_NUMBER: _ClassVar[int]
        END_FIELD_NUMBER: _ClassVar[int]
        begin: int
        end: int
        def __init__(self, begin: _Optional[int] = ..., end: _Optional[int] = ...) -> None: ...
    DIMS_FIELD_NUMBER: _ClassVar[int]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_FIELD_NUMBER: _ClassVar[int]
    FLOAT_DATA_FIELD_NUMBER: _ClassVar[int]
    INT32_DATA_FIELD_NUMBER: _ClassVar[int]
    STRING_DATA_FIELD_NUMBER: _ClassVar[int]
    INT64_DATA_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DOC_STRING_FIELD_NUMBER: _ClassVar[int]
    RAW_DATA_FIELD_NUMBER: _ClassVar[int]
    EXTERNAL_DATA_FIELD_NUMBER: _ClassVar[int]
    DATA_LOCATION_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_DATA_FIELD_NUMBER: _ClassVar[int]
    UINT64_DATA_FIELD_NUMBER: _ClassVar[int]
    METADATA_PROPS_FIELD_NUMBER: _ClassVar[int]
    dims: _containers.RepeatedScalarFieldContainer[int]
    data_type: int
    segment: TensorProto.Segment
    float_data: _containers.RepeatedScalarFieldContainer[float]
    int32_data: _containers.RepeatedScalarFieldContainer[int]
    string_data: _containers.RepeatedScalarFieldContainer[bytes]
    int64_data: _containers.RepeatedScalarFieldContainer[int]
    name: str
    doc_string: str
    raw_data: bytes
    external_data: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]
    data_location: TensorProto.DataLocation
    double_data: _containers.RepeatedScalarFieldContainer[float]
    uint64_data: _containers.RepeatedScalarFieldContainer[int]
    metadata_props: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]
    def __init__(self, dims: _Optional[_Iterable[int]] = ..., data_type: _Optional[int] = ..., segment: _Optional[_Union[TensorProto.Segment, _Mapping]] = ..., float_data: _Optional[_Iterable[float]] = ..., int32_data: _Optional[_Iterable[int]] = ..., string_data: _Optional[_Iterable[bytes]] = ..., int64_data: _Optional[_Iterable[int]] = ..., name: _Optional[str] = ..., doc_string: _Optional[str] = ..., raw_data: _Optional[bytes] = ..., external_data: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]] = ..., data_location: _Optional[_Union[TensorProto.DataLocation, str]] = ..., double_data: _Optional[_Iterable[float]] = ..., uint64_data: _Optional[_Iterable[int]] = ..., metadata_props: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]] = ...) -> None: ...

class SparseTensorProto(_message.Message):
    __slots__ = ("values", "indices", "dims")
    VALUES_FIELD_NUMBER: _ClassVar[int]
    INDICES_FIELD_NUMBER: _ClassVar[int]
    DIMS_FIELD_NUMBER: _ClassVar[int]
    values: TensorProto
    indices: TensorProto
    dims: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, values: _Optional[_Union[TensorProto, _Mapping]] = ..., indices: _Optional[_Union[TensorProto, _Mapping]] = ..., dims: _Optional[_Iterable[int]] = ...) -> None: ...

class TensorShapeProto(_message.Message):
    __slots__ = ("dim",)
    class Dimension(_message.Message):
        __slots__ = ("dim_value", "dim_param", "denotation")
        DIM_VALUE_FIELD_NUMBER: _ClassVar[int]
        DIM_PARAM_FIELD_NUMBER: _ClassVar[int]
        DENOTATION_FIELD_NUMBER: _ClassVar[int]
        dim_value: int
        dim_param: str
        denotation: str
        def __init__(self, dim_value: _Optional[int] = ..., dim_param: _Optional[str] = ..., denotation: _Optional[str] = ...) -> None: ...
    DIM_FIELD_NUMBER: _ClassVar[int]
    dim: _containers.RepeatedCompositeFieldContainer[TensorShapeProto.Dimension]
    def __init__(self, dim: _Optional[_Iterable[_Union[TensorShapeProto.Dimension, _Mapping]]] = ...) -> None: ...

class TypeProto(_message.Message):
    __slots__ = ("tensor_type", "sequence_type", "map_type", "optional_type", "sparse_tensor_type", "opaque_type", "denotation")
    class Tensor(_message.Message):
        __slots__ = ("elem_type", "shape")
        ELEM_TYPE_FIELD_NUMBER: _ClassVar[int]
        SHAPE_FIELD_NUMBER: _ClassVar[int]
        elem_type: int
        shape: TensorShapeProto
        def __init__(self, elem_type: _Optional[int] = ..., shape: _Optional[_Union[TensorShapeProto, _Mapping]] = ...) -> None: ...
    class Sequence(_message.Message):
        __slots__ = ("elem_type",)
        ELEM_TYPE_FIELD_NUMBER: _ClassVar[int]
        elem_type: TypeProto
        def __init__(self, elem_type: _Optional[_Union[TypeProto, _Mapping]] = ...) -> None: ...
    class Map(_message.Message):
        __slots__ = ("key_type", "value_type")
        KEY_TYPE_FIELD_NUMBER: _ClassVar[int]
        VALUE_TYPE_FIELD_NUMBER: _ClassVar[int]
        key_type: int
        value_type: TypeProto
        def __init__(self, key_type: _Optional[int] = ..., value_type: _Optional[_Union[TypeProto, _Mapping]] = ...) -> None: ...
    class Optional(_message.Message):
        __slots__ = ("elem_type",)
        ELEM_TYPE_FIELD_NUMBER: _ClassVar[int]
        elem_type: TypeProto
        def __init__(self, elem_type: _Optional[_Union[TypeProto, _Mapping]] = ...) -> None: ...
    class SparseTensor(_message.Message):
        __slots__ = ("elem_type", "shape")
        ELEM_TYPE_FIELD_NUMBER: _ClassVar[int]
        SHAPE_FIELD_NUMBER: _ClassVar[int]
        elem_type: int
        shape: TensorShapeProto
        def __init__(self, elem_type: _Optional[int] = ..., shape: _Optional[_Union[TensorShapeProto, _Mapping]] = ...) -> None: ...
    class Opaque(_message.Message):
        __slots__ = ("domain", "name")
        DOMAIN_FIELD_NUMBER: _ClassVar[int]
        NAME_FIELD_NUMBER: _ClassVar[int]
        domain: str
        name: str
        def __init__(self, domain: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...
    TENSOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    SEQUENCE_TYPE_FIELD_NUMBER: _ClassVar[int]
    MAP_TYPE_FIELD_NUMBER: _ClassVar[int]
    OPTIONAL_TYPE_FIELD_NUMBER: _ClassVar[int]
    SPARSE_TENSOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    OPAQUE_TYPE_FIELD_NUMBER: _ClassVar[int]
    DENOTATION_FIELD_NUMBER: _ClassVar[int]
    tensor_type: TypeProto.Tensor
    sequence_type: TypeProto.Sequence
    map_type: TypeProto.Map
    optional_type: TypeProto.Optional
    sparse_tensor_type: TypeProto.SparseTensor
    opaque_type: TypeProto.Opaque
    denotation: str
    def __init__(self, tensor_type: _Optional[_Union[TypeProto.Tensor, _Mapping]] = ..., sequence_type: _Optional[_Union[TypeProto.Sequence, _Mapping]] = ..., map_type: _Optional[_Union[TypeProto.Map, _Mapping]] = ..., optional_type: _Optional[_Union[TypeProto.Optional, _Mapping]] = ..., sparse_tensor_type: _Optional[_Union[TypeProto.SparseTensor, _Mapping]] = ..., opaque_type: _Optional[_Union[TypeProto.Opaque, _Mapping]] = ..., denotation: _Optional[str] = ...) -> None: ...

class OperatorSetIdProto(_message.Message):
    __slots__ = ("domain", "version")
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    domain: str
    version: int
    def __init__(self, domain: _Optional[str] = ..., version: _Optional[int] = ...) -> None: ...

class FunctionProto(_message.Message):
    __slots__ = ("name", "input", "output", "attribute", "attribute_proto", "node", "doc_string", "opset_import", "domain", "overload", "value_info", "metadata_props")
    NAME_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTE_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTE_PROTO_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    DOC_STRING_FIELD_NUMBER: _ClassVar[int]
    OPSET_IMPORT_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    OVERLOAD_FIELD_NUMBER: _ClassVar[int]
    VALUE_INFO_FIELD_NUMBER: _ClassVar[int]
    METADATA_PROPS_FIELD_NUMBER: _ClassVar[int]
    name: str
    input: _containers.RepeatedScalarFieldContainer[str]
    output: _containers.RepeatedScalarFieldContainer[str]
    attribute: _containers.RepeatedScalarFieldContainer[str]
    attribute_proto: _containers.RepeatedCompositeFieldContainer[AttributeProto]
    node: _containers.RepeatedCompositeFieldContainer[NodeProto]
    doc_string: str
    opset_import: _containers.RepeatedCompositeFieldContainer[OperatorSetIdProto]
    domain: str
    overload: str
    value_info: _containers.RepeatedCompositeFieldContainer[ValueInfoProto]
    metadata_props: _containers.RepeatedCompositeFieldContainer[StringStringEntryProto]
    def __init__(self, name: _Optional[str] = ..., input: _Optional[_Iterable[str]] = ..., output: _Optional[_Iterable[str]] = ..., attribute: _Optional[_Iterable[str]] = ..., attribute_proto: _Optional[_Iterable[_Union[AttributeProto, _Mapping]]] = ..., node: _Optional[_Iterable[_Union[NodeProto, _Mapping]]] = ..., doc_string: _Optional[str] = ..., opset_import: _Optional[_Iterable[_Union[OperatorSetIdProto, _Mapping]]] = ..., domain: _Optional[str] = ..., overload: _Optional[str] = ..., value_info: _Optional[_Iterable[_Union[ValueInfoProto, _Mapping]]] = ..., metadata_props: _Optional[_Iterable[_Union[StringStringEntryProto, _Mapping]]] = ...) -> None: ...
