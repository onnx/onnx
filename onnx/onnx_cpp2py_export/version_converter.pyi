from typing import Sequence, Text


# Where the first bytes are a serialized ModelProto
def convert_version(bytes: bytes, target: int) -> bytes: ...
