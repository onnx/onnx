# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

import numpy as np

from onnx import ModelProto, TensorProto, checker, load_model
from onnx.external_data_helper import _get_all_tensors, uses_external_data
from onnx.helper import make_graph, make_model, make_node, make_tensor_value_info
from onnx.large_helper import (
    LargeModelContainer,
    LargeModelFileFormat,
    make_large_model,
    make_large_tensor_proto,
)
from onnx.numpy_helper import from_array


class TestLargeOnnx(unittest.TestCase):
    @staticmethod
    def _linear_regression():
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        graph = make_graph(
            [
                make_node("MatMul", ["X", "A"], ["XA"]),
                make_node("MatMul", ["XA", "B"], ["XB"]),
                make_node("MatMul", ["XB", "C"], ["Y"]),
            ],
            "mm",
            [X],
            [Y],
            [
                from_array(np.arange(9).astype(np.float32).reshape((-1, 3)), name="A"),
                from_array(
                    (np.arange(9) * 10).astype(np.float32).reshape((-1, 3)),
                    name="B",
                ),
                from_array(
                    (np.arange(9) * 10).astype(np.float32).reshape((-1, 3)),
                    name="C",
                ),
            ],
        )
        onnx_model = make_model(graph)
        checker.check_model(onnx_model)
        return onnx_model

    def test_large_onnx_no_large_initializer(self):
        model_proto = self._linear_regression()
        assert isinstance(model_proto, ModelProto)
        large_model = make_large_model(model_proto.graph)
        assert isinstance(large_model, LargeModelContainer)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnx")
            large_model.save(filename)
            copy = LargeModelContainer()
            with self.assertRaises(RuntimeError):
                assert copy.model_proto
            copy.load(filename)
            assert copy.model_proto is not None
            checker.check_model(copy.model_proto)

    @staticmethod
    def _large_linear_regression():
        X = make_tensor_value_info("X", TensorProto.FLOAT, [None, None])
        Y = make_tensor_value_info("Y", TensorProto.FLOAT, [None])
        graph = make_graph(
            [
                make_node("MatMul", ["X", "A"], ["XA"]),
                make_node("MatMul", ["XA", "B"], ["XB"]),
                make_node("MatMul", ["XB", "C"], ["Y"]),
            ],
            "mm",
            [X],
            [Y],
            [
                make_large_tensor_proto("#loc0", "A", TensorProto.FLOAT, (3, 3)),
                from_array(np.arange(9).astype(np.float32).reshape((-1, 3)), name="B"),
                make_large_tensor_proto("#loc1", "C", TensorProto.FLOAT, (3, 3)),
            ],
        )
        onnx_model = make_model(graph)
        large_model = make_large_model(
            onnx_model.graph,
            {
                "#loc0": (np.arange(9) * 100).astype(np.float32).reshape((-1, 3)),
                "#loc1": (np.arange(9) + 10).astype(np.float32).reshape((-1, 3)),
            },
        )
        large_model.check_model()
        return large_model

    def test_large_one_weight_file(self):
        large_model = self._large_linear_regression()
        assert isinstance(large_model, LargeModelContainer)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnx")
            saved_proto = large_model.save(
                filename, LargeModelFileFormat.SINGLE_TENSOR_FILE
            )
            assert isinstance(saved_proto, ModelProto)
            copy = LargeModelContainer()
            copy.load(filename)
            copy.check_model()
            copy2 = load_model(filename, load_external_data=True)
            checker.check_model(copy2)

    def test_large_multi_files(self):
        large_model = self._large_linear_regression()
        assert isinstance(large_model, LargeModelContainer)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnx")
            saved_proto = large_model.save(
                filename, LargeModelFileFormat.ONE_TENSOR_PER_FILE
            )
            assert isinstance(saved_proto, ModelProto)
            copy = load_model(filename)
            checker.check_model(copy)
            for tensor in _get_all_tensors(copy):
                if uses_external_data(tensor):
                    for ext in tensor.external_data:
                        if ext.key == "location":  # type: ignore[attr-defined]
                            assert os.path.exists(ext.value)
            copy2 = load_model(filename, load_external_data=True)
            checker.check_model(copy2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
