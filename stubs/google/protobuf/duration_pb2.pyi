from google.protobuf.message import (
    Message,
)
from typing import (
    Optional,
)


class Duration(Message):
    seconds = ...  # type: int
    nanos = ...  # type: int

    def __init__(self,
                 seconds: Optional[int] = ...,
                 nanos: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> Duration: ...
