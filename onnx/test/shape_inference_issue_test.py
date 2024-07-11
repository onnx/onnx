# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

import onnx
import onnx.helper as oh
import onnx.numpy_helper as onh
import parameterized


class TestShapeInferenceIssue(unittest.TestCase):
    @parameterized.parameterized.expand([False, True])
    def test_issue_6182(self, convert_attribute):
        # import torch
        # import onnx
        # class Linear(torch.nn.Module):
        #     def __init__(self, input_shape: int = 784):
        #         super(Linear, self).__init__()
        #         self.input_shape = input_shape
        #         self.fc = torch.nn.Linear(input_shape, 10)
        #     def forward(self, x):
        #         x = torch.concat((x, torch.randn(10, 784)), 0)
        #         x = x.view(
        #            x.shape[0], self.input_shape
        #         )  # num samples is first dim. Then flatten the rest
        #         x = self.fc(x)
        #         return x
        # Simple model representation
        # model_path = "/tmp/model.onnx"
        # torch.onnx.export(Linear(), args=(torch.randn(10, 784)), f=model_path)
        # onnx.checker.check_model(model_path, full_check=True)
        # Ensure that the weights aren't inlined.
        # model = onnx.load(model_path)

        opset_imports = [oh.make_opsetid("", 17)]
        initializers = [
            onh.from_array(
                np.random.randn(10, 784).astype(np.float32), name="fc.weight"
            ),
            onh.from_array(np.random.randn(10).astype(np.float32), name="fc.bias"),
        ]
        inputs = [
            oh.make_tensor_value_info(
                "onnx::Concat_0", onnx.TensorProto.FLOAT, shape=(10, 784)
            )
        ]
        nodes = [
            oh.make_node(
                "RandomNormal", [], ["/RandomNormal_output_0"], dtype=1, shape=[10, 784]
            ),
            oh.make_node(
                "Concat",
                ["onnx::Concat_0", "/RandomNormal_output_0"],
                ["/Concat_output_0"],
                axis=0,
            ),
            oh.make_node(
                "Constant",
                [],
                ["/Constant_output_0"],
                value=onh.from_array(np.array([20, 784], dtype=np.int64), name="value"),
            ),
            oh.make_node(
                "Reshape",
                ["/Concat_output_0", "/Constant_output_0"],
                ["/Reshape_output_0"],
                allowzero=0,
            ),
            oh.make_node(
                "Gemm",
                ["/Reshape_output_0", "fc.weight", "fc.bias"],
                ["13"],
                alpha=1.0,
                beta=1.0,
                transB=1,
            ),
        ]
        outputs = [
            oh.make_tensor_value_info("13", onnx.TensorProto.FLOAT, shape=(20, 10))
        ]
        graph = oh.make_graph(nodes, "main_graph", inputs, outputs, initializers)
        model = oh.make_model(graph, opset_imports=opset_imports)

        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, "model.onnx")

            onnx.save(
                model,
                model_path,
                save_as_external_data=True,
                all_tensors_to_one_file=True,
                size_threshold=0,
                convert_attribute=convert_attribute,
            )

            # Check the model
            onnx.checker.check_model(model_path, full_check=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
