from typing import List

class InferenceError(Exception): ...

def infer_shapes(
    b: bytes, check_type: bool, strict_mode: bool, data_prop: bool
) -> bytes: ...

def infer_shapes_path(
    model_path: str,
    output_path: str,
    check_type: bool,
    strict_mode: bool,
    data_prop: bool,
) -> None: ...

def infer_function_output_types(bytes: bytes, input_types: List[bytes], attributes: List[bytes]) -> List[bytes]: ...
