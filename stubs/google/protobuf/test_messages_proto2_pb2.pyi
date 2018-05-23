from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer,
)
from google.protobuf.message import (
    Message,
)
import builtins
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


class ForeignEnumProto2(int):

    @classmethod
    def Name(cls, number: int) -> str: ...

    @classmethod
    def Value(cls, name: str) -> ForeignEnumProto2: ...

    @classmethod
    def keys(cls) -> List[str]: ...

    @classmethod
    def values(cls) -> List[ForeignEnumProto2]: ...

    @classmethod
    def items(cls) -> List[Tuple[str, ForeignEnumProto2]]: ...


FOREIGN_FOO: ForeignEnumProto2
FOREIGN_BAR: ForeignEnumProto2
FOREIGN_BAZ: ForeignEnumProto2


class TestAllTypesProto2(Message):

    class NestedEnum(int):

        @classmethod
        def Name(cls, number: int) -> str: ...

        @classmethod
        def Value(cls, name: str) -> TestAllTypesProto2.NestedEnum: ...

        @classmethod
        def keys(cls) -> List[str]: ...

        @classmethod
        def values(cls) -> List[TestAllTypesProto2.NestedEnum]: ...

        @classmethod
        def items(cls) -> List[Tuple[str, TestAllTypesProto2.NestedEnum]]: ...
    FOO: NestedEnum
    BAR: NestedEnum
    BAZ: NestedEnum
    NEG: NestedEnum

    class NestedMessage(Message):
        a = ...  # type: int

        @property
        def corecursive(self) -> TestAllTypesProto2: ...

        def __init__(self,
                     a: Optional[int] = ...,
                     corecursive: Optional[TestAllTypesProto2] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto2.NestedMessage: ...

    class MapInt32Int32Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapInt32Int32Entry: ...

    class MapInt64Int64Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapInt64Int64Entry: ...

    class MapUint32Uint32Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapUint32Uint32Entry: ...

    class MapUint64Uint64Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapUint64Uint64Entry: ...

    class MapSint32Sint32Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapSint32Sint32Entry: ...

    class MapSint64Sint64Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapSint64Sint64Entry: ...

    class MapFixed32Fixed32Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapFixed32Fixed32Entry: ...

    class MapFixed64Fixed64Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapFixed64Fixed64Entry: ...

    class MapSfixed32Sfixed32Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapSfixed32Sfixed32Entry: ...

    class MapSfixed64Sfixed64Entry(Message):
        key = ...  # type: int
        value = ...  # type: int

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapSfixed64Sfixed64Entry: ...

    class MapInt32FloatEntry(Message):
        key = ...  # type: int
        value = ...  # type: float

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[float] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapInt32FloatEntry: ...

    class MapInt32DoubleEntry(Message):
        key = ...  # type: int
        value = ...  # type: float

        def __init__(self,
                     key: Optional[int] = ...,
                     value: Optional[float] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapInt32DoubleEntry: ...

    class MapBoolBoolEntry(Message):
        key = ...  # type: bool
        value = ...  # type: bool

        def __init__(self,
                     key: Optional[bool] = ...,
                     value: Optional[bool] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto2.MapBoolBoolEntry: ...

    class MapStringStringEntry(Message):
        key = ...  # type: Text
        value = ...  # type: Text

        def __init__(self,
                     key: Optional[Text] = ...,
                     value: Optional[Text] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapStringStringEntry: ...

    class MapStringBytesEntry(Message):
        key = ...  # type: Text
        value = ...  # type: str

        def __init__(self,
                     key: Optional[Text] = ...,
                     value: Optional[str] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapStringBytesEntry: ...

    class MapStringNestedMessageEntry(Message):
        key = ...  # type: Text

        @property
        def value(self) -> TestAllTypesProto2.NestedMessage: ...

        def __init__(self,
                     key: Optional[Text] = ...,
                     value: Optional[TestAllTypesProto2.NestedMessage] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapStringNestedMessageEntry: ...

    class MapStringForeignMessageEntry(Message):
        key = ...  # type: Text

        @property
        def value(self) -> ForeignMessageProto2: ...

        def __init__(self,
                     key: Optional[Text] = ...,
                     value: Optional[ForeignMessageProto2] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapStringForeignMessageEntry: ...

    class MapStringNestedEnumEntry(Message):
        key = ...  # type: Text
        value = ...  # type: TestAllTypesProto2.NestedEnum

        def __init__(self,
                     key: Optional[Text] = ...,
                     value: Optional[TestAllTypesProto2.NestedEnum] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapStringNestedEnumEntry: ...

    class MapStringForeignEnumEntry(Message):
        key = ...  # type: Text
        value = ...  # type: ForeignEnumProto2

        def __init__(self,
                     key: Optional[Text] = ...,
                     value: Optional[ForeignEnumProto2] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MapStringForeignEnumEntry: ...

    class Data(Message):
        group_int32 = ...  # type: int
        group_uint32 = ...  # type: int

        def __init__(self,
                     group_int32: Optional[int] = ...,
                     group_uint32: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypesProto2.Data: ...

    class MessageSetCorrect(Message):

        def __init__(self,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MessageSetCorrect: ...

    class MessageSetCorrectExtension1(Message):
        str = ...  # type: Text

        def __init__(self,
                     str: Optional[Text] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: builtins.str) -> TestAllTypesProto2.MessageSetCorrectExtension1: ...

    class MessageSetCorrectExtension2(Message):
        i = ...  # type: int

        def __init__(self,
                     i: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> TestAllTypesProto2.MessageSetCorrectExtension2: ...
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
    optional_nested_enum = ...  # type: TestAllTypesProto2.NestedEnum
    optional_foreign_enum = ...  # type: ForeignEnumProto2
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
    repeated_nested_enum = ...  # type: RepeatedScalarFieldContainer[TestAllTypesProto2.NestedEnum]
    repeated_foreign_enum = ...  # type: RepeatedScalarFieldContainer[ForeignEnumProto2]
    repeated_string_piece = ...  # type: RepeatedScalarFieldContainer[Text]
    repeated_cord = ...  # type: RepeatedScalarFieldContainer[Text]
    oneof_uint32 = ...  # type: int
    oneof_string = ...  # type: Text
    oneof_bytes = ...  # type: str
    oneof_bool = ...  # type: bool
    oneof_uint64 = ...  # type: int
    oneof_float = ...  # type: float
    oneof_double = ...  # type: float
    oneof_enum = ...  # type: TestAllTypesProto2.NestedEnum
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
    def optional_nested_message(self) -> TestAllTypesProto2.NestedMessage: ...

    @property
    def optional_foreign_message(self) -> ForeignMessageProto2: ...

    @property
    def recursive_message(self) -> TestAllTypesProto2: ...

    @property
    def repeated_nested_message(
        self) -> RepeatedCompositeFieldContainer[TestAllTypesProto2.NestedMessage]: ...

    @property
    def repeated_foreign_message(
        self) -> RepeatedCompositeFieldContainer[ForeignMessageProto2]: ...

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
    def map_string_nested_message(
        self) -> MutableMapping[Text, TestAllTypesProto2.NestedMessage]: ...

    @property
    def map_string_foreign_message(
        self) -> MutableMapping[Text, ForeignMessageProto2]: ...

    @property
    def map_string_nested_enum(
        self) -> MutableMapping[Text, TestAllTypesProto2.NestedEnum]: ...

    @property
    def map_string_foreign_enum(
        self) -> MutableMapping[Text, ForeignEnumProto2]: ...

    @property
    def oneof_nested_message(self) -> TestAllTypesProto2.NestedMessage: ...

    @property
    def data(self) -> TestAllTypesProto2.Data: ...

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
                 optional_nested_message: Optional[TestAllTypesProto2.NestedMessage] = ...,
                 optional_foreign_message: Optional[ForeignMessageProto2] = ...,
                 optional_nested_enum: Optional[TestAllTypesProto2.NestedEnum] = ...,
                 optional_foreign_enum: Optional[ForeignEnumProto2] = ...,
                 optional_string_piece: Optional[Text] = ...,
                 optional_cord: Optional[Text] = ...,
                 recursive_message: Optional[TestAllTypesProto2] = ...,
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
                 repeated_nested_message: Optional[Iterable[TestAllTypesProto2.NestedMessage]] = ...,
                 repeated_foreign_message: Optional[Iterable[ForeignMessageProto2]] = ...,
                 repeated_nested_enum: Optional[Iterable[TestAllTypesProto2.NestedEnum]] = ...,
                 repeated_foreign_enum: Optional[Iterable[ForeignEnumProto2]] = ...,
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
                 map_string_nested_message: Optional[Mapping[Text, TestAllTypesProto2.NestedMessage]]=...,
                 map_string_foreign_message: Optional[Mapping[Text, ForeignMessageProto2]]=...,
                 map_string_nested_enum: Optional[Mapping[Text, TestAllTypesProto2.NestedEnum]]=...,
                 map_string_foreign_enum: Optional[Mapping[Text, ForeignEnumProto2]]=...,
                 oneof_uint32: Optional[int] = ...,
                 oneof_nested_message: Optional[TestAllTypesProto2.NestedMessage] = ...,
                 oneof_string: Optional[Text] = ...,
                 oneof_bytes: Optional[str] = ...,
                 oneof_bool: Optional[bool] = ...,
                 oneof_uint64: Optional[int] = ...,
                 oneof_float: Optional[float] = ...,
                 oneof_double: Optional[float] = ...,
                 oneof_enum: Optional[TestAllTypesProto2.NestedEnum] = ...,
                 data: Optional[TestAllTypesProto2.Data] = ...,
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
    def FromString(cls, s: str) -> TestAllTypesProto2: ...


class ForeignMessageProto2(Message):
    c = ...  # type: int

    def __init__(self,
                 c: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> ForeignMessageProto2: ...
