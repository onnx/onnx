from google.protobuf.message import (
    Message,
)
from typing import (
    List,
    Optional,
    Tuple,
    cast,
)


class TestEnum(int):
    @classmethod
    def Name(cls, number: int) -> str: ...

    @classmethod
    def Value(cls, name: str) -> TestEnum: ...

    @classmethod
    def keys(cls) -> List[str]: ...

    @classmethod
    def values(cls) -> List[TestEnum]: ...

    @classmethod
    def items(cls) -> List[Tuple[str, TestEnum]]: ...


FOO: TestEnum


class TestMessage(Message):
    a = ...  # type: int

    def __init__(self,
                 a: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestMessage: ...
