from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer,
)
from google.protobuf.message import (
    Message,
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


class NullValue(int):
    @classmethod
    def Name(cls, number: int) -> str: ...

    @classmethod
    def Value(cls, name: str) -> NullValue: ...

    @classmethod
    def keys(cls) -> List[str]: ...

    @classmethod
    def values(cls) -> List[NullValue]: ...

    @classmethod
    def items(cls) -> List[Tuple[str, NullValue]]: ...


NULL_VALUE: NullValue


class Struct(Message):
    class FieldsEntry(Message):
        key = ...  # type: Text

        @property
        def value(self) -> Value: ...

        def __init__(self,
                     key: Optional[Text] = ...,
                     value: Optional[Value] = ...,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> Struct.FieldsEntry: ...

    @property
    def fields(self) -> MutableMapping[Text, Value]: ...

    def __init__(self,
                 fields: Optional[Mapping[Text, Value]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> Struct: ...


class _Value(Message):
    null_value = ...  # type: NullValue
    number_value = ...  # type: float
    string_value = ...  # type: Text
    bool_value = ...  # type: bool

    @property
    def struct_value(self) -> Struct: ...

    @property
    def list_value(self) -> ListValue: ...

    def __init__(self,
                 null_value: Optional[NullValue] = ...,
                 number_value: Optional[float] = ...,
                 string_value: Optional[Text] = ...,
                 bool_value: Optional[bool] = ...,
                 struct_value: Optional[Struct] = ...,
                 list_value: Optional[ListValue] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> _Value: ...


Value = _Value


class ListValue(Message):

    @property
    def values(self) -> RepeatedCompositeFieldContainer[Value]: ...

    def __init__(self,
                 values: Optional[Iterable[Value]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> ListValue: ...
