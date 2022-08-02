import itertools
from typing import Iterable, Dict

import onnx
from onnx.defs import get_all_schemas_with_history

SCHEMAS = get_all_schemas_with_history()

SCHEMAS_BY_NAME = {
    name: sorted(group, key=lambda s: s.since_version)
    for name, group in itertools.groupby(
        sorted(SCHEMAS, key=lambda s: s.name), lambda s: s.name
    )
}

DOMAINS = {s.domain for s in SCHEMAS}


def infer_types(op_name: str, inputs: Iterable[onnx.TypeProto], num_outputs: int) -> Iterable[onnx.TypeProto]:
    inputs = list(inputs)

    node = onnx.helper.make_node(
        op_name,
        [f'in{i}' for i in range(len(inputs))],
        [f'out{i}' for i in range(num_outputs)]
    )

    outputs = SCHEMAS_BY_NAME[op_name][-1].infer_types(
        node.SerializeToString(),
        {f'in{i}': in_type.SerializeToString() for i, in_type in enumerate(inputs)}, {}, {}
    )

    for output in outputs:
        yield onnx.TypeProto.FromString(output)


print(list(infer_types("Add", [
    onnx.helper.make_tensor_type_proto(1, (None, 2)),
    onnx.helper.make_tensor_type_proto(1, (1, 2))
], 1)))