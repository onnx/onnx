from google.protobuf.internal.containers import (
    RepeatedCompositeFieldContainer,
)
from google.protobuf.message import (
    Message,
)
from google.protobuf.unittest_mset_wire_format_pb2 import (
    TestMessageSet,
)
import builtins
from typing import (
    Iterable,
    Optional,
    Text,
)


class TestMessageSetContainer(Message):

    @property
    def message_set(self) -> TestMessageSet: ...

    def __init__(self,
                 message_set: Optional[TestMessageSet] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestMessageSetContainer: ...


class TestMessageSetExtension1(Message):
    i = ...  # type: int

    def __init__(self,
                 i: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> TestMessageSetExtension1: ...


class TestMessageSetExtension2(Message):
    str = ...  # type: Text

    def __init__(self,
                 str: Optional[Text] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: builtins.str) -> TestMessageSetExtension2: ...


class RawMessageSet(Message):

    class Item(Message):
        type_id = ...  # type: int
        message = ...  # type: str

        def __init__(self,
                     type_id: int,
                     message: str,
                     ) -> None: ...

        @classmethod
        def FromString(cls, s: str) -> RawMessageSet.Item: ...

    @property
    def item(self) -> RepeatedCompositeFieldContainer[RawMessageSet.Item]: ...

    def __init__(self,
                 item: Optional[Iterable[RawMessageSet.Item]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> RawMessageSet: ...
