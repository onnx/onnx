# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import unittest

import onnx
from onnx import checker, helper, inliner, shape_inference, version_converter


class TestOpsetVersionRange(unittest.TestCase):
    def make_model(self) -> onnx.ModelProto:
        return helper.make_model(
            helper.make_graph(
                [helper.make_node("Identity", ["X"], ["Y"])],
                "identity",
                [helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [2])],
                [helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [2])],
            ),
            opset_imports=[helper.make_opsetid("", 18)],
        )

    def test_converter_accepts_int64_at_binding_boundary(self) -> None:
        model = self.make_model()
        self.assertEqual(
            version_converter.convert_version(model, 19).opset_import[0].version, 19
        )
        for version in (-(1 << 63), -(1 << 31) - 1, 1 << 31, (1 << 63) - 1):
            # This must reach C++ validation, rather than fail int argument conversion.
            with (
                self.subTest(version=version),
                self.assertRaisesRegex(checker.ValidationError, str(version)),
            ):
                version_converter.convert_version(model, version)

    def test_unused_imports_are_validated_before_processing(self) -> None:
        for local in (False, True):
            with self.subTest(local=local):
                model = self.make_model()
                imports = model.opset_import
                if local:
                    function = model.functions.add()
                    function.name = "Unused"
                    function.domain = "local"
                    imports = function.opset_import
                imports.append(helper.make_opsetid("com.vendor", (1 << 63) - 1))
                for operation in (
                    shape_inference.infer_shapes,
                    inliner.inline_local_functions,
                    lambda m: version_converter.convert_version(m, 19),
                ):
                    with self.assertRaisesRegex(checker.ValidationError, "com.vendor"):
                        operation(model)


if __name__ == "__main__":
    unittest.main()
