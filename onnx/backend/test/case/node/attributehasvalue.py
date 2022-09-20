# SPDX-License-Identifier: Apache-2.0

import numpy as np  # type: ignore

import onnx

from ..base import Base
from . import expect


class AttributeHasValue(Base):
    @staticmethod
    def export() -> None:
        def test_one_attribute(name, **kwargs):
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
                name="test_attribute_has_value_{name}_false".format(name=name),
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
                name="test_attribute_has_value_{name}_true".format(name=name),
            )

        ints = [0, 1]
        test_one_attribute("ints", ints=ints)

        int = 1
        test_one_attribute("int", int=int)
