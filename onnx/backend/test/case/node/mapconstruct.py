# SPDX-License-Identifier: Apache-2.0

from typing import List, Dict, Any

import numpy as np

import onnx

from ..base import Base
from . import expect


def map_construct_reference_implementation(
    keys: np.ndarray = None,
    values: List[Any] = None,
) -> Dict[Any, Any]: 
    if keys is not None and values is not None:
        return dict(zip(keys.tolist(), values))
    

class MapConstruct(Base):
    @staticmethod
    def export_map_construct_with_keys_values() -> None:
        keys = np.array([100, 200]).astype(np.int32)
        values = [
            np.array([1, 2, 3, 4]).astype(np.int32),
            np.array([5, 6, 7, 8]).astype(np.int32)
        ]

        tensor_type_proto = onnx.helper.make_tensor_type_proto(
            elem_type=onnx.TensorProto.INT32,
            shape=[4],
        )
        seq_type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
        map_proto = onnx.helper.make_map_type_proto(onnx.TensorProto.INT32, seq_type_proto)

        node = onnx.helper.make_node(
            "MapConstruct",
            inputs=["keys", "values"],
            outputs=["map"],
        )

        map = map_construct_reference_implementation(keys=keys, values=values)
        
        expect(
            node, 
            inputs=[keys, values],
            outputs=[map],
            output_type_protos=[map_proto],
            name="test_map_construct_with_inputs",
        )
