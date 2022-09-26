# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np  # type: ignore

import onnx
import onnx.parser

from ..base import Base
from . import expect


class AttributeHasValue(Base):
    @staticmethod
    def export() -> None:
        def test_one_attribute(name: str, **kwargs: Any) -> None:
            node = onnx.helper.make_node(
                "AttributeHasValue",
                inputs=[],
                outputs=["output"],
            )

            output = np.array(False)
            expect(
                node,
                inputs=[],
                outputs=[output],
                name=f"test_attribute_has_{name}_false",
            )

            node = onnx.helper.make_node(
                "AttributeHasValue",
                inputs=[],
                outputs=["output"],
                **kwargs,
            )

            output = np.array(True)
            expect(
                node,
                inputs=[],
                outputs=[output],
                name=f"test_attribute_has_{name}_true",
            )

        value_float = 0.1
        test_one_attribute("value_float", value_float=value_float)

        value_int = 1
        test_one_attribute("value_int", value_int=value_int)

        value_string = "test"
        test_one_attribute("value_string", value_string=value_string)

        tensor_values = np.random.randn(5, 5).astype(np.float32)
        value_tensor = onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.FLOAT,
            dims=tensor_values.shape,
            vals=tensor_values.flatten().astype(float),
        )
        test_one_attribute("value_tensor", value_tensor=value_tensor)

        value_graph = onnx.parser.parse_graph("agraph (X) => (Y) {Y = Identity(X)}")
        test_one_attribute("value_graph", value_graph=value_graph)

        value_sparse_tensor = onnx.helper.make_sparse_tensor(
            onnx.helper.make_tensor(
                name="",
                data_type=onnx.TensorProto.FLOAT,
                dims=(5,),
                vals=[1.1, 2.2, 3.3, 4.4, 5.5],
            ),
            onnx.helper.make_tensor(
                name="",
                data_type=onnx.TensorProto.INT64,
                dims=(5,),
                vals=[1, 3, 5, 7, 9],
            ),
            [10],
        )

        test_one_attribute(
            "value_sparse_tensor", value_sparse_tensor=value_sparse_tensor
        )

        value_type_proto = onnx.helper.make_tensor_type_proto(
            onnx.TensorProto.FLOAT, shape=[5]
        )
        test_one_attribute("value_type_proto", value_type_proto=value_type_proto)

        value_floats = [0.0, 1.1]
        test_one_attribute("value_floats", value_floats=value_floats)

        value_ints = [0, 1]
        test_one_attribute("value_ints", value_ints=value_ints)

        value_strings = ["test strings"]
        test_one_attribute("value_strings", value_strings=value_strings)

        value_tensors = [value_tensor, value_tensor]
        test_one_attribute("value_tensors", value_tensors=value_tensors)

        value_graphs = [value_graph, value_graph]
        test_one_attribute("value_graphs", value_graphs=value_graphs)

        value_sparse_tensors = [value_sparse_tensor, value_sparse_tensor]
        test_one_attribute(
            "value_sparse_tensors", value_sparse_tensors=value_sparse_tensors
        )

        value_type_protos = [value_type_proto, value_type_proto]
        test_one_attribute("value_type_protos", value_type_protos=value_type_protos)
