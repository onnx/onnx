from google.protobuf.message import (
    Message,
)


class Empty(Message):

    def __init__(self,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> Empty: ...
