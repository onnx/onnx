from google.protobuf.descriptor_pb2 import (
    FileOptions,
)
from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer,
    RepeatedScalarFieldContainer,
)
from google.protobuf.message import (
    Message,
)
from typing import (
    Iterable,
    List,
    Optional,
    Text,
    Tuple,
    cast,
)


class MethodOpt1(int):

    @classmethod
    def Name(cls, number: int) -> str: ...

    @classmethod
    def Value(cls, name: str) -> MethodOpt1: ...

    @classmethod
    def keys(cls) -> List[str]: ...

    @classmethod
    def values(cls) -> List[MethodOpt1]: ...

    @classmethod
    def items(cls) -> List[Tuple[str, MethodOpt1]]: ...


METHODOPT1_VAL1: MethodOpt1
METHODOPT1_VAL2: MethodOpt1


class AggregateEnum(int):

    @classmethod
    def Name(cls, number: int) -> str: ...

    @classmethod
    def Value(cls, name: str) -> AggregateEnum: ...

    @classmethod
    def keys(cls) -> List[str]: ...

    @classmethod
    def values(cls) -> List[AggregateEnum]: ...

    @classmethod
    def items(cls) -> List[Tuple[str, AggregateEnum]]: ...


VALUE: AggregateEnum


class TestMessageWithCustomOptions(Message):

    class AnEnum(int):

        @classmethod
        def Name(cls, number: int) -> str: ...

        @classmethod
        def Value(cls, name: str) -> TestMessageWithCustomOptions.AnEnum: ...

        @classmethod
        def keys(cls) -> List[str]: ...

        @classmethod
        def values(cls) -> List[TestMessageWithCustomOptions.AnEnum]: ...

        @classmethod
        def items(cls) -> List[Tuple[str,
                                     TestMessageWithCustomOptions.AnEnum]]: ...
    ANENUM_VAL1: AnEnum
    ANENUM_VAL2: AnEnum
    field1 = ...  # type: Text
    oneof_field = ...  # type: int

    def __init__(self,
                 field1: Optional[Text] = ...,
                 oneof_field: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestMessageWithCustomOptions: ...


class CustomOptionFooRequest(Message):

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> CustomOptionFooRequest: ...


class CustomOptionFooResponse(Message):

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> CustomOptionFooResponse: ...


class CustomOptionFooClientMessage(Message):

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> CustomOptionFooClientMessage: ...


class CustomOptionFooServerMessage(Message):

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> CustomOptionFooServerMessage: ...


class DummyMessageContainingEnum(Message):

    class TestEnumType(int):

        @classmethod
        def Name(cls, number: int) -> str: ...

        @classmethod
        def Value(cls, name: str) -> DummyMessageContainingEnum.TestEnumType: ...

        @classmethod
        def keys(cls) -> List[str]: ...

        @classmethod
        def values(cls) -> List[DummyMessageContainingEnum.TestEnumType]: ...

        @classmethod
        def items(cls) -> List[Tuple[str,
                                     DummyMessageContainingEnum.TestEnumType]]: ...
    TEST_OPTION_ENUM_TYPE1: TestEnumType
    TEST_OPTION_ENUM_TYPE2: TestEnumType

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> DummyMessageContainingEnum: ...


class DummyMessageInvalidAsOptionType(Message):

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> DummyMessageInvalidAsOptionType: ...


class CustomOptionMinIntegerValues(Message):

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> CustomOptionMinIntegerValues: ...


class CustomOptionMaxIntegerValues(Message):

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> CustomOptionMaxIntegerValues: ...


class CustomOptionOtherValues(Message):

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> CustomOptionOtherValues: ...


class SettingRealsFromPositiveInts(Message):

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> SettingRealsFromPositiveInts: ...


class SettingRealsFromNegativeInts(Message):

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> SettingRealsFromNegativeInts: ...


class ComplexOptionType1(Message):
    foo = ...  # type: int
    foo2 = ...  # type: int
    foo3 = ...  # type: int
    foo4 = ...  # type: RepeatedScalarFieldContainer[int]

    def __init__(self,
                 foo: Optional[int] = ...,
                 foo2: Optional[int] = ...,
                 foo3: Optional[int] = ...,
                 foo4: Optional[Iterable[int]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> ComplexOptionType1: ...


class ComplexOptionType2(Message):

    class ComplexOptionType4(Message):
        waldo = ...  # type: int

        def __init__(self,
                     waldo: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> ComplexOptionType2.ComplexOptionType4: ...
    baz = ...  # type: int

    @property
    def bar(self) -> ComplexOptionType1: ...

    @property
    def fred(self) -> ComplexOptionType2.ComplexOptionType4: ...

    @property
    def barney(
        self) -> RepeatedCompositeFieldContainer[ComplexOptionType2.ComplexOptionType4]: ...

    def __init__(self,
                 bar: Optional[ComplexOptionType1] = ...,
                 baz: Optional[int] = ...,
                 fred: Optional[ComplexOptionType2.ComplexOptionType4] = ...,
                 barney: Optional[Iterable[ComplexOptionType2.ComplexOptionType4]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> ComplexOptionType2: ...


class ComplexOptionType3(Message):

    class ComplexOptionType5(Message):
        plugh = ...  # type: int

        def __init__(self,
                     plugh: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(
            cls, s: str) -> ComplexOptionType3.ComplexOptionType5: ...
    qux = ...  # type: int

    @property
    def complexoptiontype5(self) -> ComplexOptionType3.ComplexOptionType5: ...

    def __init__(self,
                 qux: Optional[int] = ...,
                 complexoptiontype5: Optional[ComplexOptionType3.ComplexOptionType5] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> ComplexOptionType3: ...


class ComplexOpt6(Message):
    xyzzy = ...  # type: int

    def __init__(self,
                 xyzzy: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> ComplexOpt6: ...


class VariousComplexOptions(Message):

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> VariousComplexOptions: ...


class AggregateMessageSet(Message):

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> AggregateMessageSet: ...


class AggregateMessageSetElement(Message):
    s = ...  # type: Text

    def __init__(self,
                 s: Optional[Text] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> AggregateMessageSetElement: ...


class Aggregate(Message):
    i = ...  # type: int
    s = ...  # type: Text

    @property
    def sub(self) -> Aggregate: ...

    @property
    def file(self) -> FileOptions: ...

    @property
    def mset(self) -> AggregateMessageSet: ...

    def __init__(self,
                 i: Optional[int] = ...,
                 s: Optional[Text] = ...,
                 sub: Optional[Aggregate] = ...,
                 file: Optional[FileOptions] = ...,
                 mset: Optional[AggregateMessageSet] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> Aggregate: ...


class AggregateMessage(Message):
    fieldname = ...  # type: int

    def __init__(self,
                 fieldname: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> AggregateMessage: ...


class NestedOptionType(Message):

    class NestedEnum(int):

        @classmethod
        def Name(cls, number: int) -> str: ...

        @classmethod
        def Value(cls, name: str) -> NestedOptionType.NestedEnum: ...

        @classmethod
        def keys(cls) -> List[str]: ...

        @classmethod
        def values(cls) -> List[NestedOptionType.NestedEnum]: ...

        @classmethod
        def items(cls) -> List[Tuple[str, NestedOptionType.NestedEnum]]: ...
    NESTED_ENUM_VALUE: NestedEnum

    class NestedMessage(Message):
        nested_field = ...  # type: int

        def __init__(self,
                     nested_field: Optional[int] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> NestedOptionType.NestedMessage: ...

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> NestedOptionType: ...


class OldOptionType(Message):

    class TestEnum(int):

        @classmethod
        def Name(cls, number: int) -> str: ...

        @classmethod
        def Value(cls, name: str) -> OldOptionType.TestEnum: ...

        @classmethod
        def keys(cls) -> List[str]: ...

        @classmethod
        def values(cls) -> List[OldOptionType.TestEnum]: ...

        @classmethod
        def items(cls) -> List[Tuple[str, OldOptionType.TestEnum]]: ...
    OLD_VALUE: TestEnum
    value = ...  # type: OldOptionType.TestEnum

    def __init__(self,
                 value: OldOptionType.TestEnum,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> OldOptionType: ...


class NewOptionType(Message):

    class TestEnum(int):

        @classmethod
        def Name(cls, number: int) -> str: ...

        @classmethod
        def Value(cls, name: str) -> NewOptionType.TestEnum: ...

        @classmethod
        def keys(cls) -> List[str]: ...

        @classmethod
        def values(cls) -> List[NewOptionType.TestEnum]: ...

        @classmethod
        def items(cls) -> List[Tuple[str, NewOptionType.TestEnum]]: ...
    OLD_VALUE: TestEnum
    NEW_VALUE: TestEnum
    value = ...  # type: NewOptionType.TestEnum

    def __init__(self,
                 value: NewOptionType.TestEnum,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> NewOptionType: ...


class TestMessageWithRequiredEnumOption(Message):

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestMessageWithRequiredEnumOption: ...
