from typing import Text, Union
from onnx import GraphProto, FunctionProto


def to_text(proto : Union[GraphProto, FunctionProto]) -> Text:
    ...
