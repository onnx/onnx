from google.protobuf.internal.containers import (
    RepeatedScalarFieldContainer,
)
from google.protobuf.message import (
    Message,
)
from typing import (
    Iterable,
    Optional,
    Text,
)


class FieldMask(Message):
    paths = ...  # type: RepeatedScalarFieldContainer[Text]

    def __init__(self,
                 paths: Optional[Iterable[Text]] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> FieldMask: ...
