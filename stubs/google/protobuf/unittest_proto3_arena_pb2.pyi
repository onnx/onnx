from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer,
)
from google.protobuf.message import (
    Message,
)
from google.protobuf.unittest_import_pb2 import (
    ImportMessage,
)
from google.protobuf.unittest_import_public_pb2 import (
    PublicImportMessage,
)
from typing import (
    Iterable,
    List,
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


FOREIGN_ZERO: ForeignEnum
FOREIGN_FOO: ForeignEnum
FOREIGN_BAR: ForeignEnum
FOREIGN_BAZ: ForeignEnum


class TestAllTypes(Message):

    class NestedEnum(int):

        @classmethod
        def Name(cls, number: int) -> str: ...

        @classmethod
        def Value(cls, name: str) -> TestAllTypes.NestedEnum: ...

        @classmethod
        def keys(cls) -> List[str]: ...

        @classmethod
        def values(cls) -> List[TestAllTypes.NestedEnum]: ...

        @classmethod
        def items(cls) -> List[Tuple[str, TestAllTypes.NestedEnum]]: ...
    ZERO: NestedEnum
    FOO: NestedEnum
    BAR: NestedEnum
    BAZ: NestedEnum
    NEG: NestedEnum

    class NestedMessage(Message):
        bb = ...  # type: int

        def __init__(self,
                     bb: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> TestAllTypes.NestedMessage: ...
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
    optional_nested_enum = ...  # type: TestAllTypes.NestedEnum
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
    repeated_nested_enum = ...  # type: RepeatedScalarFieldContainer[TestAllTypes.NestedEnum]
    repeated_foreign_enum = ...  # type: RepeatedScalarFieldContainer[ForeignEnum]
    repeated_string_piece = ...  # type: RepeatedScalarFieldContainer[Text]
    repeated_cord = ...  # type: RepeatedScalarFieldContainer[Text]
    oneof_uint32 = ...  # type: int
    oneof_string = ...  # type: Text
    oneof_bytes = ...  # type: str

    @property
    def optional_nested_message(self) -> TestAllTypes.NestedMessage: ...

    @property
    def optional_foreign_message(self) -> ForeignMessage: ...

    @property
    def optional_import_message(self) -> ImportMessage: ...

    @property
    def optional_public_import_message(self) -> PublicImportMessage: ...

    @property
    def optional_lazy_message(self) -> TestAllTypes.NestedMessage: ...

    @property
    def optional_lazy_import_message(self) -> ImportMessage: ...

    @property
    def repeated_nested_message(
        self) -> RepeatedCompositeFieldContainer[TestAllTypes.NestedMessage]: ...

    @property
    def repeated_foreign_message(
        self) -> RepeatedCompositeFieldContainer[ForeignMessage]: ...

    @property
    def repeated_import_message(
        self) -> RepeatedCompositeFieldContainer[ImportMessage]: ...

    @property
    def repeated_lazy_message(
        self) -> RepeatedCompositeFieldContainer[TestAllTypes.NestedMessage]: ...

    @property
    def oneof_nested_message(self) -> TestAllTypes.NestedMessage: ...

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
                 optional_nested_message: Optional[TestAllTypes.NestedMessage] = ...,
                 optional_foreign_message: Optional[ForeignMessage] = ...,
                 optional_import_message: Optional[ImportMessage] = ...,
                 optional_nested_enum: Optional[TestAllTypes.NestedEnum] = ...,
                 optional_foreign_enum: Optional[ForeignEnum] = ...,
                 optional_string_piece: Optional[Text] = ...,
                 optional_cord: Optional[Text] = ...,
                 optional_public_import_message: Optional[PublicImportMessage] = ...,
                 optional_lazy_message: Optional[TestAllTypes.NestedMessage] = ...,
                 optional_lazy_import_message: Optional[ImportMessage] = ...,
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
                 repeated_nested_message: Optional[Iterable[TestAllTypes.NestedMessage]] = ...,
                 repeated_foreign_message: Optional[Iterable[ForeignMessage]] = ...,
                 repeated_import_message: Optional[Iterable[ImportMessage]] = ...,
                 repeated_nested_enum: Optional[Iterable[TestAllTypes.NestedEnum]] = ...,
                 repeated_foreign_enum: Optional[Iterable[ForeignEnum]] = ...,
                 repeated_string_piece: Optional[Iterable[Text]] = ...,
                 repeated_cord: Optional[Iterable[Text]] = ...,
                 repeated_lazy_message: Optional[Iterable[TestAllTypes.NestedMessage]] = ...,
                 oneof_uint32: Optional[int] = ...,
                 oneof_nested_message: Optional[TestAllTypes.NestedMessage] = ...,
                 oneof_string: Optional[Text] = ...,
                 oneof_bytes: Optional[str] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestAllTypes: ...


class TestPackedTypes(Message):
    packed_int32 = ...  # type: RepeatedScalarFieldContainer[int]
    packed_int64 = ...  # type: RepeatedScalarFieldContainer[int]
    packed_uint32 = ...  # type: RepeatedScalarFieldContainer[int]
    packed_uint64 = ...  # type: RepeatedScalarFieldContainer[int]
    packed_sint32 = ...  # type: RepeatedScalarFieldContainer[int]
    packed_sint64 = ...  # type: RepeatedScalarFieldContainer[int]
    packed_fixed32 = ...  # type: RepeatedScalarFieldContainer[int]
    packed_fixed64 = ...  # type: RepeatedScalarFieldContainer[int]
    packed_sfixed32 = ...  # type: RepeatedScalarFieldContainer[int]
    packed_sfixed64 = ...  # type: RepeatedScalarFieldContainer[int]
    packed_float = ...  # type: RepeatedScalarFieldContainer[float]
    packed_double = ...  # type: RepeatedScalarFieldContainer[float]
    packed_bool = ...  # type: RepeatedScalarFieldContainer[bool]
    packed_enum = ...  # type: RepeatedScalarFieldContainer[ForeignEnum]

    def __init__(self,
                 packed_int32: Optional[Iterable[int]] = ...,
                 packed_int64: Optional[Iterable[int]] = ...,
                 packed_uint32: Optional[Iterable[int]] = ...,
                 packed_uint64: Optional[Iterable[int]] = ...,
                 packed_sint32: Optional[Iterable[int]] = ...,
                 packed_sint64: Optional[Iterable[int]] = ...,
                 packed_fixed32: Optional[Iterable[int]] = ...,
                 packed_fixed64: Optional[Iterable[int]] = ...,
                 packed_sfixed32: Optional[Iterable[int]] = ...,
                 packed_sfixed64: Optional[Iterable[int]] = ...,
                 packed_float: Optional[Iterable[float]] = ...,
                 packed_double: Optional[Iterable[float]] = ...,
                 packed_bool: Optional[Iterable[bool]] = ...,
                 packed_enum: Optional[Iterable[ForeignEnum]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestPackedTypes: ...


class TestUnpackedTypes(Message):
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
    repeated_nested_enum = ...  # type: RepeatedScalarFieldContainer[TestAllTypes.NestedEnum]

    def __init__(self,
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
                 repeated_nested_enum: Optional[Iterable[TestAllTypes.NestedEnum]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestUnpackedTypes: ...


class NestedTestAllTypes(Message):

    @property
    def child(self) -> NestedTestAllTypes: ...

    @property
    def payload(self) -> TestAllTypes: ...

    @property
    def repeated_child(
        self) -> RepeatedCompositeFieldContainer[NestedTestAllTypes]: ...

    def __init__(self,
                 child: Optional[NestedTestAllTypes] = ...,
                 payload: Optional[TestAllTypes] = ...,
                 repeated_child: Optional[Iterable[NestedTestAllTypes]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> NestedTestAllTypes: ...


class ForeignMessage(Message):
    c = ...  # type: int

    def __init__(self,
                 c: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> ForeignMessage: ...


class TestEmptyMessage(Message):

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestEmptyMessage: ...
