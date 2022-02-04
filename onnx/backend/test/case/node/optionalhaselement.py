# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore
from typing import Optional

import onnx
from ..base import Base
from . import expect


def optional_has_element_reference_implementation(optional: Optional[np.ndarray]) -> np.ndarray:
    if optional is None:
        return np.array(False)
    else:
        return np.array(True)


class OptionalHasElement(Base):

    @staticmethod
    def export() -> None:
        optional = np.array([1, 2, 3, 4]).astype(np.float32)
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=onnx.TensorProto.FLOAT, shape=[4, ])
        input_type_proto = onnx.helper.make_optional_type_proto(tensor_type_proto)
        node = onnx.helper.make_node(
            'OptionalHasElement',
            inputs=['optional_input'],
            outputs=['output']
        )
        output = optional_has_element_reference_implementation(optional)
        expect(node, inputs=[optional], outputs=[output],
               input_type_protos=[input_type_proto],
               name='test_optional_has_element')

    @staticmethod
    def export_empty() -> None:
        optional = None
        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=onnx.TensorProto.INT32, shape=[])
        input_type_proto = onnx.helper.make_optional_type_proto(tensor_type_proto)
        node = onnx.helper.make_node(
            'OptionalHasElement',
            inputs=['optional_input'],
            outputs=['output']
        )
        output = optional_has_element_reference_implementation(optional)
        expect(node, inputs=[optional], outputs=[output],
               input_type_protos=[input_type_proto],
               name='test_optional_has_element_empty')
