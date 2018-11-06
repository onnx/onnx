from google.protobuf.any_pb2 import (
    Any,
)
from google.protobuf.duration_pb2 import (
    Duration,
)
from google.protobuf.field_mask_pb2 import (
    FieldMask,
)
from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer,
)
from google.protobuf.message import (
    Message,
)
from google.protobuf.struct_pb2 import (
    Struct,
    Value,
)
from google.protobuf.timestamp_pb2 import (
    Timestamp,
)
from google.protobuf.wrappers_pb2 import (
    BoolValue,
    BytesValue,
    DoubleValue,
    FloatValue,
    Int32Value,
    Int64Value,
    StringValue,
    UInt32Value,
    UInt64Value,
)
from typing import (
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Text,
    Tuple,
    cast,
)


class ForeignEnum(int):

    @classmethod
    def Name(cls, number: int) -> str: ...

    @classmethod
    def Value(cls, name: str) -> ForeignEnum: ...

    @classmethod
    def keys(cls) -> List[str]: ...

    @classmethod
    def values(cls) -> List[ForeignEnum]: ...

    @classmethod
    def items(cls) -> List[Tuple[str, ForeignEnum]]: ...
FOREIGN_FOO: ForeignEnum
FOREIGN_BAR: ForeignEnum
FOREIGN_BAZ: ForeignEnum


class TestAllTypesProto3(Message):

    class NestedEnum(int):

        @classmethod
        def Name(cls, number: int) -> str: ...

        @classmethod
        def Value(cls, name: str) -> TestAllTypesProto3.NestedEnum: ...

        @classmethod
        def keys(cls) -> List[str]: ...

        @classmethod
        def values(cls) -> List[TestAllTypesProto3.NestedEnum]: ...

        @classmethod
        def items(cls) -> List[Tuple[str, TestAllTypesProto3.NestedEnum]]: ...
    FOO: NestedEnum
    BAR: NestedEnum
    BAZ: NestedEnum
    NEG: NestedEnum

    class NestedMessage(Message):
        a = ...  # type: int

        @property
        def corecursive(self) -> TestAllTypesProto3: ...

        def __init__(self,
                     a: Optional[int] = ...,
                     corecursive: Optional[TestAllTypesProto3] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.NestedMessage: ...

    class MapInt32Int32Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapInt32Int32Entry: ...

    class MapInt64Int64Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapInt64Int64Entry: ...

    class MapUint32Uint32Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapUint32Uint32Entry: ...

    class MapUint64Uint64Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapUint64Uint64Entry: ...

    class MapSint32Sint32Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapSint32Sint32Entry: ...

    class MapSint64Sint64Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapSint64Sint64Entry: ...

    class MapFixed32Fixed32Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapFixed32Fixed32Entry: ...

    class MapFixed64Fixed64Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapFixed64Fixed64Entry: ...

    class MapSfixed32Sfixed32Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapSfixed32Sfixed32Entry: ...

    class MapSfixed64Sfixed64Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapSfixed64Sfixed64Entry: ...

    class MapInt32FloatEntry(Message):
        key = ...  # type: int
        value = ...  # type: float

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[float] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapInt32FloatEntry: ...

    class MapInt32DoubleEntry(Message):
        key = ...  # type: int
        value = ...  # type: float

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[float] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapInt32DoubleEntry: ...

    class MapBoolBoolEntry(Message):
        key = ...  # type: bool
        value = ...  # type: bool

        def __init__(self,
                     key: Optional[bool] = ...,
                     value: Optional[bool] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapBoolBoolEntry: ...

    class MapStringStringEntry(Message):
        key = ...  # type: Text
        value = ...  # type: Text

        def __init__(self,
                     key: Optional[Text] = ...,
                     value: Optional[Text] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapStringStringEntry: ...

    class MapStringBytesEntry(Message):
        key = ...  # type: Text
        value = ...  # type: str

        def __init__(self,
                     key: Optional[Text] = ...,
                     value: Optional[str] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapStringBytesEntry: ...

    class MapStringNestedMessageEntry(Message):
        key = ...  # type: Text

        @property
        def value(self) -> TestAllTypesProto3.NestedMessage: ...

        def __init__(self,
                     key: Optional[Text] = ...,
                     value: Optional[TestAllTypesProto3.NestedMessage] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapStringNestedMessageEntry: ...

    class MapStringForeignMessageEntry(Message):
        key = ...  # type: Text

        @property
        def value(self) -> ForeignMessage: ...

        def __init__(self,
                     key: Optional[Text] = ...,
                     value: Optional[ForeignMessage] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapStringForeignMessageEntry: ...

    class MapStringNestedEnumEntry(Message):
        key = ...  # type: Text
        value = ...  # type: TestAllTypesProto3.NestedEnum

        def __init__(self,
                     key: Optional[Text] = ...,
                     value: Optional[TestAllTypesProto3.NestedEnum] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapStringNestedEnumEntry: ...

    class MapStringForeignEnumEntry(Message):
        key = ...  # type: Text
        value = ...  # type: ForeignEnum

        def __init__(self,
                     key: Optional[Text] = ...,
                     value: Optional[ForeignEnum] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto3.MapStringForeignEnumEntry: ...
    optional_int32 = ...  # type: int
    optional_int64 = ...  # type: int
    optional_uint32 = ...  # type: int
    optional_uint64 = ...  # type: int
    optional_sint32 = ...  # type: int
    optional_sint64 = ...  # type: int
    optional_fixed32 = ...  # type: int
    optional_fixed64 = ...  # type: int
    optional_sfixed32 = ...  # type: int
    optional_sfixed64 = ...  # type: int
    optional_float = ...  # type: float
    optional_double = ...  # type: float
    optional_bool = ...  # type: bool
    optional_string = ...  # type: Text
    optional_bytes = ...  # type: str
    optional_nested_enum = ...  # type: TestAllTypesProto3.NestedEnum
    optional_foreign_enum = ...  # type: ForeignEnum
    optional_string_piece = ...  # type: Text
    optional_cord = ...  # type: Text
    repeated_int32 = ...  # type: RepeatedScalarFieldContainer[int]
    repeated_int64 = ...  # type: RepeatedScalarFieldContainer[int]
    repeated_uint32 = ...  # type: RepeatedScalarFieldContainer[int]
    repeated_uint64 = ...  # type: RepeatedScalarFieldContainer[int]
    repeated_sint32 = ...  # type: RepeatedScalarFieldContainer[int]
    repeated_sint64 = ...  # type: RepeatedScalarFieldContainer[int]
    repeated_fixed32 = ...  # type: RepeatedScalarFieldContainer[int]
    repeated_fixed64 = ...  # type: RepeatedScalarFieldContainer[int]
    repeated_sfixed32 = ...  # type: RepeatedScalarFieldContainer[int]
    repeated_sfixed64 = ...  # type: RepeatedScalarFieldContainer[int]
    repeated_float = ...  # type: RepeatedScalarFieldContainer[float]
    repeated_double = ...  # type: RepeatedScalarFieldContainer[float]
    repeated_bool = ...  # type: RepeatedScalarFieldContainer[bool]
    repeated_string = ...  # type: RepeatedScalarFieldContainer[Text]
    repeated_bytes = ...  # type: RepeatedScalarFieldContainer[str]
    repeated_nested_enum = ...  # type: RepeatedScalarFieldContainer[TestAllTypesProto3.NestedEnum]
    repeated_foreign_enum = ...  # type: RepeatedScalarFieldContainer[ForeignEnum]
    repeated_string_piece = ...  # type: RepeatedScalarFieldContainer[Text]
    repeated_cord = ...  # type: RepeatedScalarFieldContainer[Text]
    oneof_uint32 = ...  # type: int
    oneof_string = ...  # type: Text
    oneof_bytes = ...  # type: str
    oneof_bool = ...  # type: bool
    oneof_uint64 = ...  # type: int
    oneof_float = ...  # type: float
    oneof_double = ...  # type: float
    oneof_enum = ...  # type: TestAllTypesProto3.NestedEnum
    fieldname1 = ...  # type: int
    field_name2 = ...  # type: int
    _field_name3 = ...  # type: int
    field__name4_ = ...  # type: int
    field0name5 = ...  # type: int
    field_0_name6 = ...  # type: int
    fieldName7 = ...  # type: int
    FieldName8 = ...  # type: int
    field_Name9 = ...  # type: int
    Field_Name10 = ...  # type: int
    FIELD_NAME11 = ...  # type: int
    FIELD_name12 = ...  # type: int
    __field_name13 = ...  # type: int
    __Field_name14 = ...  # type: int
    field__name15 = ...  # type: int
    field__Name16 = ...  # type: int
    field_name17__ = ...  # type: int
    Field_name18__ = ...  # type: int

    @property
    def optional_nested_message(self) -> TestAllTypesProto3.NestedMessage: ...

    @property
    def optional_foreign_message(self) -> ForeignMessage: ...

    @property
    def recursive_message(self) -> TestAllTypesProto3: ...

    @property
    def repeated_nested_message(self) -> RepeatedCompositeFieldContainer[TestAllTypesProto3.NestedMessage]: ...

    @property
    def repeated_foreign_message(self) -> RepeatedCompositeFieldContainer[ForeignMessage]: ...

    @property
    def map_int32_int32(self) -> MutableMapping[int, int]: ...

    @property
    def map_int64_int64(self) -> MutableMapping[int, int]: ...

    @property
    def map_uint32_uint32(self) -> MutableMapping[int, int]: ...

    @property
    def map_uint64_uint64(self) -> MutableMapping[int, int]: ...

    @property
    def map_sint32_sint32(self) -> MutableMapping[int, int]: ...

    @property
    def map_sint64_sint64(self) -> MutableMapping[int, int]: ...

    @property
    def map_fixed32_fixed32(self) -> MutableMapping[int, int]: ...

    @property
    def map_fixed64_fixed64(self) -> MutableMapping[int, int]: ...

    @property
    def map_sfixed32_sfixed32(self) -> MutableMapping[int, int]: ...

    @property
    def map_sfixed64_sfixed64(self) -> MutableMapping[int, int]: ...

    @property
    def map_int32_float(self) -> MutableMapping[int, float]: ...

    @property
    def map_int32_double(self) -> MutableMapping[int, float]: ...

    @property
    def map_bool_bool(self) -> MutableMapping[bool, bool]: ...

    @property
    def map_string_string(self) -> MutableMapping[Text, Text]: ...

    @property
    def map_string_bytes(self) -> MutableMapping[Text, str]: ...

    @property
    def map_string_nested_message(self) -> MutableMapping[Text, TestAllTypesProto3.NestedMessage]: ...

    @property
    def map_string_foreign_message(self) -> MutableMapping[Text, ForeignMessage]: ...

    @property
    def map_string_nested_enum(self) -> MutableMapping[Text, TestAllTypesProto3.NestedEnum]: ...

    @property
    def map_string_foreign_enum(self) -> MutableMapping[Text, ForeignEnum]: ...

    @property
    def oneof_nested_message(self) -> TestAllTypesProto3.NestedMessage: ...

    @property
    def optional_bool_wrapper(self) -> BoolValue: ...

    @property
    def optional_int32_wrapper(self) -> Int32Value: ...

    @property
    def optional_int64_wrapper(self) -> Int64Value: ...

    @property
    def optional_uint32_wrapper(self) -> UInt32Value: ...

    @property
    def optional_uint64_wrapper(self) -> UInt64Value: ...

    @property
    def optional_float_wrapper(self) -> FloatValue: ...

    @property
    def optional_double_wrapper(self) -> DoubleValue: ...

    @property
    def optional_string_wrapper(self) -> StringValue: ...

    @property
    def optional_bytes_wrapper(self) -> BytesValue: ...

    @property
    def repeated_bool_wrapper(self) -> RepeatedCompositeFieldContainer[BoolValue]: ...

    @property
    def repeated_int32_wrapper(self) -> RepeatedCompositeFieldContainer[Int32Value]: ...

    @property
    def repeated_int64_wrapper(self) -> RepeatedCompositeFieldContainer[Int64Value]: ...

    @property
    def repeated_uint32_wrapper(self) -> RepeatedCompositeFieldContainer[UInt32Value]: ...

    @property
    def repeated_uint64_wrapper(self) -> RepeatedCompositeFieldContainer[UInt64Value]: ...

    @property
    def repeated_float_wrapper(self) -> RepeatedCompositeFieldContainer[FloatValue]: ...

    @property
    def repeated_double_wrapper(self) -> RepeatedCompositeFieldContainer[DoubleValue]: ...

    @property
    def repeated_string_wrapper(self) -> RepeatedCompositeFieldContainer[StringValue]: ...

    @property
    def repeated_bytes_wrapper(self) -> RepeatedCompositeFieldContainer[BytesValue]: ...

    @property
    def optional_duration(self) -> Duration: ...

    @property
    def optional_timestamp(self) -> Timestamp: ...

    @property
    def optional_field_mask(self) -> FieldMask: ...

    @property
    def optional_struct(self) -> Struct: ...

    @property
    def optional_any(self) -> Any: ...

    @property
    def optional_value(self) -> Value: ...

    @property
    def repeated_duration(self) -> RepeatedCompositeFieldContainer[Duration]: ...

    @property
    def repeated_timestamp(self) -> RepeatedCompositeFieldContainer[Timestamp]: ...

    @property
    def repeated_fieldmask(self) -> RepeatedCompositeFieldContainer[FieldMask]: ...

    @property
    def repeated_struct(self) -> RepeatedCompositeFieldContainer[Struct]: ...

    @property
    def repeated_any(self) -> RepeatedCompositeFieldContainer[Any]: ...

    @property
    def repeated_value(self) -> RepeatedCompositeFieldContainer[Value]: ...

    def __init__(self,
                 optional_int32: Optional[int] = ...,
                 optional_int64: Optional[int] = ...,
                 optional_uint32: Optional[int] = ...,
                 optional_uint64: Optional[int] = ...,
                 optional_sint32: Optional[int] = ...,
                 optional_sint64: Optional[int] = ...,
                 optional_fixed32: Optional[int] = ...,
                 optional_fixed64: Optional[int] = ...,
                 optional_sfixed32: Optional[int] = ...,
                 optional_sfixed64: Optional[int] = ...,
                 optional_float: Optional[float] = ...,
                 optional_double: Optional[float] = ...,
                 optional_bool: Optional[bool] = ...,
                 optional_string: Optional[Text] = ...,
                 optional_bytes: Optional[str] = ...,
                 optional_nested_message: Optional[TestAllTypesProto3.NestedMessage] = ...,
                 optional_foreign_message: Optional[ForeignMessage] = ...,
                 optional_nested_enum: Optional[TestAllTypesProto3.NestedEnum] = ...,
                 optional_foreign_enum: Optional[ForeignEnum] = ...,
                 optional_string_piece: Optional[Text] = ...,
                 optional_cord: Optional[Text] = ...,
                 recursive_message: Optional[TestAllTypesProto3] = ...,
                 repeated_int32: Optional[Iterable[int]] = ...,
                 repeated_int64: Optional[Iterable[int]] = ...,
                 repeated_uint32: Optional[Iterable[int]] = ...,
                 repeated_uint64: Optional[Iterable[int]] = ...,
                 repeated_sint32: Optional[Iterable[int]] = ...,
                 repeated_sint64: Optional[Iterable[int]] = ...,
                 repeated_fixed32: Optional[Iterable[int]] = ...,
                 repeated_fixed64: Optional[Iterable[int]] = ...,
                 repeated_sfixed32: Optional[Iterable[int]] = ...,
                 repeated_sfixed64: Optional[Iterable[int]] = ...,
                 repeated_float: Optional[Iterable[float]] = ...,
                 repeated_double: Optional[Iterable[float]] = ...,
                 repeated_bool: Optional[Iterable[bool]] = ...,
                 repeated_string: Optional[Iterable[Text]] = ...,
                 repeated_bytes: Optional[Iterable[str]] = ...,
                 repeated_nested_message: Optional[Iterable[TestAllTypesProto3.NestedMessage]] = ...,
                 repeated_foreign_message: Optional[Iterable[ForeignMessage]] = ...,
                 repeated_nested_enum: Optional[Iterable[TestAllTypesProto3.NestedEnum]] = ...,
                 repeated_foreign_enum: Optional[Iterable[ForeignEnum]] = ...,
                 repeated_string_piece: Optional[Iterable[Text]] = ...,
                 repeated_cord: Optional[Iterable[Text]] = ...,
                 map_int32_int32: Optional[Mapping[int, int]]=...,
                 map_int64_int64: Optional[Mapping[int, int]]=...,
                 map_uint32_uint32: Optional[Mapping[int, int]]=...,
                 map_uint64_uint64: Optional[Mapping[int, int]]=...,
                 map_sint32_sint32: Optional[Mapping[int, int]]=...,
                 map_sint64_sint64: Optional[Mapping[int, int]]=...,
                 map_fixed32_fixed32: Optional[Mapping[int, int]]=...,
                 map_fixed64_fixed64: Optional[Mapping[int, int]]=...,
                 map_sfixed32_sfixed32: Optional[Mapping[int, int]]=...,
                 map_sfixed64_sfixed64: Optional[Mapping[int, int]]=...,
                 map_int32_float: Optional[Mapping[int, float]]=...,
                 map_int32_double: Optional[Mapping[int, float]]=...,
                 map_bool_bool: Optional[Mapping[bool, bool]]=...,
                 map_string_string: Optional[Mapping[Text, Text]]=...,
                 map_string_bytes: Optional[Mapping[Text, str]]=...,
                 map_string_nested_message: Optional[Mapping[Text, TestAllTypesProto3.NestedMessage]]=...,
                 map_string_foreign_message: Optional[Mapping[Text, ForeignMessage]]=...,
                 map_string_nested_enum: Optional[Mapping[Text, TestAllTypesProto3.NestedEnum]]=...,
                 map_string_foreign_enum: Optional[Mapping[Text, ForeignEnum]]=...,
                 oneof_uint32: Optional[int] = ...,
                 oneof_nested_message: Optional[TestAllTypesProto3.NestedMessage] = ...,
                 oneof_string: Optional[Text] = ...,
                 oneof_bytes: Optional[str] = ...,
                 oneof_bool: Optional[bool] = ...,
                 oneof_uint64: Optional[int] = ...,
                 oneof_float: Optional[float] = ...,
                 oneof_double: Optional[float] = ...,
                 oneof_enum: Optional[TestAllTypesProto3.NestedEnum] = ...,
                 optional_bool_wrapper: Optional[BoolValue] = ...,
                 optional_int32_wrapper: Optional[Int32Value] = ...,
                 optional_int64_wrapper: Optional[Int64Value] = ...,
                 optional_uint32_wrapper: Optional[UInt32Value] = ...,
                 optional_uint64_wrapper: Optional[UInt64Value] = ...,
                 optional_float_wrapper: Optional[FloatValue] = ...,
                 optional_double_wrapper: Optional[DoubleValue] = ...,
                 optional_string_wrapper: Optional[StringValue] = ...,
                 optional_bytes_wrapper: Optional[BytesValue] = ...,
                 repeated_bool_wrapper: Optional[Iterable[BoolValue]] = ...,
                 repeated_int32_wrapper: Optional[Iterable[Int32Value]] = ...,
                 repeated_int64_wrapper: Optional[Iterable[Int64Value]] = ...,
                 repeated_uint32_wrapper: Optional[Iterable[UInt32Value]] = ...,
                 repeated_uint64_wrapper: Optional[Iterable[UInt64Value]] = ...,
                 repeated_float_wrapper: Optional[Iterable[FloatValue]] = ...,
                 repeated_double_wrapper: Optional[Iterable[DoubleValue]] = ...,
                 repeated_string_wrapper: Optional[Iterable[StringValue]] = ...,
                 repeated_bytes_wrapper: Optional[Iterable[BytesValue]] = ...,
                 optional_duration: Optional[Duration] = ...,
                 optional_timestamp: Optional[Timestamp] = ...,
                 optional_field_mask: Optional[FieldMask] = ...,
                 optional_struct: Optional[Struct] = ...,
                 optional_any: Optional[Any] = ...,
                 optional_value: Optional[Value] = ...,
                 repeated_duration: Optional[Iterable[Duration]] = ...,
                 repeated_timestamp: Optional[Iterable[Timestamp]] = ...,
                 repeated_fieldmask: Optional[Iterable[FieldMask]] = ...,
                 repeated_struct: Optional[Iterable[Struct]] = ...,
                 repeated_any: Optional[Iterable[Any]] = ...,
                 repeated_value: Optional[Iterable[Value]] = ...,
                 fieldname1: Optional[int] = ...,
                 field_name2: Optional[int] = ...,
                 _field_name3: Optional[int] = ...,
                 field__name4_: Optional[int] = ...,
                 field0name5: Optional[int] = ...,
                 field_0_name6: Optional[int] = ...,
                 fieldName7: Optional[int] = ...,
                 FieldName8: Optional[int] = ...,
                 field_Name9: Optional[int] = ...,
                 Field_Name10: Optional[int] = ...,
                 FIELD_NAME11: Optional[int] = ...,
                 FIELD_name12: Optional[int] = ...,
                 __field_name13: Optional[int] = ...,
                 __Field_name14: Optional[int] = ...,
                 field__name15: Optional[int] = ...,
                 field__Name16: Optional[int] = ...,
                 field_name17__: Optional[int] = ...,
                 Field_name18__: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestAllTypesProto3: ...


class ForeignMessage(Message):
    c = ...  # type: int

    def __init__(self,
                 c: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> ForeignMessage: ...
