from google.protobuf.message import (
    Message,
)
from typing import (
    List,
    Optional,
    Tuple,
    cast,
)


class ImportEnum(int):

    @classmethod
    def Name(cls, number: int) -> str: ...

    @classmethod
    def Value(cls, name: str) -> ImportEnum: ...

    @classmethod
    def keys(cls) -> List[str]: ...

    @classmethod
    def values(cls) -> List[ImportEnum]: ...

    @classmethod
    def items(cls) -> List[Tuple[str, ImportEnum]]: ...


IMPORT_FOO: ImportEnum
IMPORT_BAR: ImportEnum
IMPORT_BAZ: ImportEnum


class ImportEnumForMap(int):

    @classmethod
    def Name(cls, number: int) -> str: ...

    @classmethod
    def Value(cls, name: str) -> ImportEnumForMap: ...

    @classmethod
    def keys(cls) -> List[str]: ...

    @classmethod
    def values(cls) -> List[ImportEnumForMap]: ...

    @classmethod
    def items(cls) -> List[Tuple[str, ImportEnumForMap]]: ...


UNKNOWN: ImportEnumForMap
FOO: ImportEnumForMap
BAR: ImportEnumForMap


class ImportMessage(Message):
    d = ...  # type: int

    def __init__(self,
                 d: Optional[int] = ...,
                 ) -> None: ...

    @classmethod
    def FromString(cls, s: str) -> ImportMessage: ...
