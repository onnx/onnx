# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from typing import List, Optional, Union, Dict, Any

import numpy as np

import onnx
from onnx import TensorProto, TypeProto

from ..base import Base
from . import expect

from .sequence import SequenceConstructImpl


def MapConstructImpl(
    keys: np.ndarray = None,
    values: List[np.ndarray] = None,
    key_type: Union[int, str] = None,
    value_type: TypeProto = None,
) -> Dict[Union[int, str], np.ndarray]:
    if keys is not None and values is not None:
        return dict(zip(keys.tolist(), values))


def MapKeysImpl(mapping: Dict[Union[int, str], np.ndarray]) -> np.ndarray:
    return np.array(list(mapping.keys())).astype(mapping.keys()[0])


def MapValuesImpl(mapping: Dict[Union[int, str], List[np.ndarray]]) -> List[np.ndarray]:
    return list(mapping.values())


def MapInsertPairImpl(
    mapping: Dict[Union[int, str], np.ndarray],
    key: Union[int, str],
    value: np.ndarray,
) -> Dict[Union[int, str], np.ndarray]: 
    mapping[key] = value
    return mapping


def MapDeletePairImpl(
    mapping: Dict[Union[int, str], np.ndarray],
    key: Union[int, str],
) -> Dict[Union[int, str], np.ndarray]: 
    mapping.pop(key)
    return mapping


def MapHasKeyImpl(
    mapping: Dict[Union[int, str], np.ndarray],
    key: Union[int, str],
) -> np.ndarray: 
    return np.array(key in list(mapping.keys()))

def MapGetValueImpl(
    mapping: Dict[Union[int, str], np.ndarray],
    key: Union[int, str],
) -> np.ndarray: 
    return mapping[key]

class Map(Base):
    @staticmethod
    def export() -> None:
        def make_graph(
            nodes: List[onnx.helper.NodeProto],
            input_shapes: List[Optional[typing.Sequence[Union[str, int]]]],
            output_shapes: List[Optional[typing.Sequence[Union[str, int]]]],
            input_names: List[str],
            output_names: List[str],
            input_types: List[TensorProto.DataType],
            output_types: List[TensorProto.DataType],
            initializers: Optional[List[TensorProto]] = None,
        ) -> onnx.helper.GraphProto:
            graph = onnx.helper.make_graph(
                nodes=nodes,
                name="Map",
                inputs=[
                    onnx.helper.make_tensor_value_info(name, input_type, input_shape)
                    for name, input_type, input_shape in zip(
                        input_names, input_types, input_shapes
                    )
                ],
                outputs=[
                    onnx.helper.make_tensor_value_info(name, output_type, output_shape)
                    for name, output_type, output_shape in zip(
                        output_names, output_types, output_shapes
                    )
                ],
                initializer=initializers,
            )
            return graph

        # 1st testcase - create map, insert pair and delete pair
        # After deletion, check if that key exists in the map
        # 1. MapConstruct:                -> {k1: v1, k2: v2}
        # 2. MapInsertPair((k3, v3)):     -> {k1: v1, k2: v2, k3: v3} 
        # 3. MapDeletePair(k2):           -> {k1: v3, k2: v3}
        # 4. MapHasKey(k2)                -> False
        seq_construct_node = onnx.helper.make_node(
            "SequenceConstruct", ["V1", "V2"], ["Values"]
        )
        map_construct_node = onnx.helper.make_node(
            "MapConstruct", ["Keys", "Values"], ["Map_1"]
        )
        map_insert_pair_node = onnx.helper.make_node(
            "MapInsertPair", ["Map_1", "K3", "V3"], ["Map_2"]
        )
        map_delete_pair_node = onnx.helper.make_node(
            "MapDeletePair", ["Map_2", "K2"], ["Map_3"]
        )
        map_has_key_node = onnx.helper.make_node(
            "MapHasKey", ["Map_3", "K2"], ["out"]
        )

        x_shape = [2, 3, 4]
        y_shape = [1, 3, 4]
        z_shape = [3, 3, 4]
        out_shape = [1,]

        x = np.ones(x_shape, dtype=np.float32)
        y = np.zeros(y_shape, dtype=np.float32)
        z = np.ones(z_shape, dtype=np.float32) * 2
        keys = np.array([10, 20]).astype(np.int32)
        key_insert = 30
        key_delete = 20

        out = SequenceConstructImpl(x, y)
        out = MapConstructImpl(keys=keys, values=out)
        out = MapInsertPairImpl(out, key_insert, z)
        out = MapDeletePairImpl(out, key_delete)
        out = MapHasKeyImpl(out, key_delete)
        assert np.array_equal(out, np.array(False))

        key_insert = onnx.helper.make_tensor("K3", TensorProto.INT32, (), (key_insert,))
        key_delete = onnx.helper.make_tensor("K2", TensorProto.INT32, (), (key_delete,))

        graph = make_graph(
            [
                seq_construct_node,
                map_construct_node,
                map_insert_pair_node,
                map_delete_pair_node,
                map_has_key_node,
            ],
            [x_shape, y_shape, [2,], [], z_shape, []],  # type: ignore
            [out_shape],  # type: ignore
            ["V1", "V2", "Keys", "K3", "V3", "K2"],
            ["out"],
            [onnx.TensorProto.FLOAT] * 2 + [onnx.TensorProto.INT32] * 2 + [onnx.TensorProto.INT64, onnx.TensorProto.INT64],  # type: ignore
            [onnx.TensorProto.BOOL],
            [key_insert, key_delete],
        )
        model = onnx.helper.make_model_gen_version(
            graph,
            producer_name="backend-test",
        )
        expect(model, inputs=[x, y, z], outputs=[out], name="test_map_model1")
