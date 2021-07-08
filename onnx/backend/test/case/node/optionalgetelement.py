# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
from typing import Optional, Any

import onnx
from ..base import Base
from . import expect


def optional_get_element_reference_implementation(optional):
    # type: (Optional[Any]) -> Any
    assert optional is not None
    return optional


class OptionalHasElement(Base):

    @staticmethod
    def export():  # type: () -> None
        optional = np.array([1, 2, 3, 4])

        node = onnx.helper.make_node(
            'OptionalGetElement',
            inputs=['optional_input'],
            outputs=['output']
        )
        output = optional_get_element_reference_implementation(optional)
        expect(node, inputs=[optional], outputs=[output],
               name='test_optional_get_element')

    @staticmethod
    def export_empty():  # type: () -> None
        optional = [np.array([1, 2, 3, 4])]

        node = onnx.helper.make_node(
            'OptionalGetElement',
            inputs=['optional_input'],
            outputs=['output']
        )
        output = optional_get_element_reference_implementation(optional)
        expect(node, inputs=[optional], outputs=[output],
               name='test_optional_get_element_sequence')
