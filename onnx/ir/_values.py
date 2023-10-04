from typing import Protocol, Any, Sequence


class Tensor(Protocol):
    # TODO: We need to support symbolic shapes
    # This is a concrete tensor value. The dims are for interpreting the data
    dims: Sequence[int]
    data_type: int
    name: str
    doc_string: str
    data: Any


# TODO: For symbolic shapes, the value may be an object dependent
# on other shapes. Potentially allow replacing the dimension implementation?
class Dimension(Protocol):
    value: int | str
    denotation: str

class Shape(Protocol):
    # TODO: Integrate Dimension Denotation
    # https://github.com/onnx/onnx/blob/main/docs/DimensionDenotation.md#denotation-definition

    dims: Sequence[Dimension]


class TensorType(Protocol):
    # NOTE: This is TypeProto.Tensor
    ...
    # TODO: It can be nested

# TODO(figure out how to represent types)
