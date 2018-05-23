from google.protobuf.message import (
    Message,
)
from typing import (
    Optional,
)


class PublicImportMessage(Message):
    e = ...  # type: int

    def __init__(self,
                 e: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> PublicImportMessage: ...
