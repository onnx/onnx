from typing import Sequence

class ConvertError(Exception):
    ...

# Where the first bytes are a serialized ModelProto
def convert_version(bytes: bytes, target: int) -> bytes: ...
