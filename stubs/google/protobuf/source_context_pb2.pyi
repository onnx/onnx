from google.protobuf.message import (
    Message,
)
from typing import (
    Optional,
    Text,
)


class SourceContext(Message):
    file_name = ...  # type: Text

    def __init__(self,
                 file_name: Optional[Text] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> SourceContext: ...
