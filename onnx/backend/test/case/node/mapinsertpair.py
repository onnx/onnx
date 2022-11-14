# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any, Union

import numpy as np

import onnx

from ..base import Base
from . import expect


def map_insert_pair_reference_implementation(
    mapping: Dict[Any, Any],
    key: Union[int, str] = None,
    value: Any = None,
) -> Dict[Any, Any]: 
    mapping[key] = value
    return mapping
    

class MapInsertPair(Base):
    @staticmethod
    def export_map_insert_pair() -> None:
        mapping = {
            100 : np.array([1, 2, 3, 4]).astype(np.int32),
            200 : np.array([5, 6, 7, 8]).astype(np.int32)
        }

        key = 300
        value = np.array([7, 8, 9, 0]).astype(np.int32)

        key_type_proto = onnx.helper.make_tensor_type_proto(
            elem_type=onnx.TensorProto.INT32,
            shape=[1],
        )

        tensor_type_proto = onnx.helper.make_tensor_type_proto(
            elem_type=onnx.TensorProto.INT32,
            shape=[4],
        )
        seq_type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
        map_proto = onnx.helper.make_map_type_proto(onnx.TensorProto.INT32, seq_type_proto)

        node = onnx.helper.make_node(
            "MapInsertPair",
            inputs=["map", "key", "value"],
            outputs=["output_map"],
        )

        map_out = map_insert_pair_reference_implementation(mapping, key, value)
        key = np.array([key]).astype(np.int32)
        
        expect(
            node, 
            inputs=[mapping, key, value],
            outputs=[map_out],
            input_type_protos=[map_proto, key_type_proto, seq_type_proto],
            output_type_protos=[map_proto],
            name="test_map_insert_pair",
        )
