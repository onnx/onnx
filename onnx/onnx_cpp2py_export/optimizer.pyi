from typing import Sequence, Text


def optimize(
    bytes: bytes, names: Sequence[Text], fixed_point: bool = False) -> bytes: ...


def get_available_passes() -> Sequence[Text]: ...
