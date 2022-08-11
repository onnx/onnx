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
        optional_type_proto = onnx.helper.make_optional_type_proto(tensor_type_proto)

        # OptionalHasElement takes a tensor or optional as input
        for input_type_protos in [tensor_type_proto, optional_type_proto]:
            node = onnx.helper.make_node(
                'OptionalHasElement',
                inputs=['optional_input'],
                outputs=['output']
            )
            output = optional_has_element_reference_implementation(optional)
            test_name = 'test_optional_has_element_' +\
                ('optional_input' if input_type_protos == optional_type_proto else 'tensor_input')
            expect(node, inputs=[optional], outputs=[output],
                input_type_protos=[optional_type_proto],
                name=test_name)

    @staticmethod
    def export_empty() -> None:
        optional = None

        tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=onnx.TensorProto.INT32, shape=[])
        optional_type_proto = onnx.helper.make_optional_type_proto(tensor_type_proto)
        
        # OptionalHasElement takes a tensor or optional as input
        for input_type_proto in [tensor_type_proto, optional_type_proto]:
            input_name_options = {
                'empty': 'optional_input',
                'empty_no_input_name': '',
                'empty_no_input': None,
                }
            for test_name_surfix, input_name in input_name_options.items():
                node = onnx.helper.make_node(
                    'OptionalHasElement',
                    inputs=[] if input_name is None else [input_name],
                    outputs=['output']
                )
                output = optional_has_element_reference_implementation(optional)
                test_name = name='test_optional_has_element_' + test_name_surfix\
                    + ('_optional_input' if input_type_proto == optional_type_proto else '_tensor_input')
                expect(node, inputs=[optional] if input_name else [], outputs=[output],
                    input_type_protos=[input_type_proto] if input_name else [],
                    name=test_name)
