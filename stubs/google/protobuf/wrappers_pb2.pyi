from google.protobuf.message import (
    Message,
)
from typing import (
    Optional,
    Text,
)


class DoubleValue(Message):
    value = ...  # type: float

    def __init__(self,
                 value: Optional[float] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> DoubleValue: ...


class FloatValue(Message):
    value = ...  # type: float

    def __init__(self,
                 value: Optional[float] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> FloatValue: ...


class Int64Value(Message):
    value = ...  # type: int

    def __init__(self,
                 value: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> Int64Value: ...


class UInt64Value(Message):
    value = ...  # type: int

    def __init__(self,
                 value: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> UInt64Value: ...


class Int32Value(Message):
    value = ...  # type: int

    def __init__(self,
                 value: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> Int32Value: ...


class UInt32Value(Message):
    value = ...  # type: int

    def __init__(self,
                 value: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> UInt32Value: ...


class BoolValue(Message):
    value = ...  # type: bool

    def __init__(self,
                 value: Optional[bool] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> BoolValue: ...


class StringValue(Message):
    value = ...  # type: Text

    def __init__(self,
                 value: Optional[Text] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> StringValue: ...


class BytesValue(Message):
    value = ...  # type: str

    def __init__(self,
                 value: Optional[str] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> BytesValue: ...
