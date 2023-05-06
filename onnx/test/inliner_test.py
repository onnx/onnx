# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
import unittest

from onnx import inliner, parser


class InlinerTest(unittest.TestCase):
    def test_basic(self):
        model = parser.parse_model(
            """
            <ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
            agraph (float[N] X) => (float[N] Y)
            {
                Y = local.foo (X)
            }

            <opset_import: [ "" : 17, "local" : 1 ], domain: "local">
            foo (x) => (y) {
                temp = Add(x, x)
                y = local.bar(temp)
            }

            <opset_import: [ "" : 17 ], domain: "local">
            bar (x) => (y) {
                y = Mul (x, x)
            }
        """
        )
        inlined = inliner.inline_local_functions(model)
        inlined_nodes = inlined.graph.node
        # function-call should be replaced by Add, followed by Mul
        self.assertEqual(len(inlined_nodes), 2)
        self.assertEqual(inlined_nodes[0].op_type, "Add")
        self.assertEqual(inlined_nodes[1].op_type, "Mul")


if __name__ == "__main__":
    unittest.main()
