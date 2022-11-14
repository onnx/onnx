# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Any

import numpy as np

import onnx

from ..base import Base
from . import expect


def map_keys_reference_implementation(
    mapping: Dict[Any, Any],
) -> np.ndarray: 
    keys = list(mapping.keys())
    return np.array(keys).astype(np.int32)


class MapKeys(Base):
    @staticmethod
    def export_map_keys() -> None:
        mapping = {
            100 : np.array([1, 2, 3, 4]).astype(np.int32),
            200 : np.array([5, 6, 7, 8]).astype(np.int32)
        }

        tensor_type_proto = onnx.helper.make_tensor_type_proto(
            elem_type=onnx.TensorProto.INT32,
            shape=[4],
        )
        seq_type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
        map_proto = onnx.helper.make_map_type_proto(onnx.TensorProto.INT32, seq_type_proto)

        keys_proto = onnx.helper.make_tensor_type_proto(
            elem_type=onnx.TensorProto.INT32,
            shape=[2],
        )

        node = onnx.helper.make_node(
            "MapKeys",
            inputs=["map"],
            outputs=["keys"],
        )

        keys = map_keys_reference_implementation(mapping)
        
        expect(
            node, 
            inputs=[mapping],
            outputs=[keys],
            input_type_protos=[map_proto],
            output_type_protos=[keys_proto],
            name="test_map_keys",
        )
