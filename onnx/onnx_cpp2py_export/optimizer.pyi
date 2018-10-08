from typing import Sequence, Text


def optimize(
    bytes: bytes, names: Sequence[Text]) -> bytes: ...

def optimize_fixedpoint(
    bytes: bytes, names: Sequence[Text]) -> bytes: ...


def get_available_passes() -> Sequence[Text]: ...
