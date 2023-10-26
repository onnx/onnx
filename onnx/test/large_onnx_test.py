# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import unittest

import numpy as np

import onnx
import onnx.external_data_helper as ext_data
import onnx.helper
import onnx.large_helper
import onnx.numpy_helper


def _linear_regression():
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, None])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [None])
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["X", "A"], ["XA"]),
            onnx.helper.make_node("MatMul", ["XA", "B"], ["XB"]),
            onnx.helper.make_node("MatMul", ["XB", "C"], ["Y"]),
        ],
        "mm",
        [X],
        [Y],
        [
            onnx.numpy_helper.from_array(
                np.arange(9).astype(np.float32).reshape((-1, 3)), name="A"
            ),
            onnx.numpy_helper.from_array(
                (np.arange(9) * 10).astype(np.float32).reshape((-1, 3)),
                name="B",
            ),
            onnx.numpy_helper.from_array(
                (np.arange(9) * 10).astype(np.float32).reshape((-1, 3)),
                name="C",
            ),
        ],
    )
    onnx_model = onnx.helper.make_model(graph)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def _large_linear_regression():
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [None, None])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [None])
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("MatMul", ["X", "A"], ["XA"]),
            onnx.helper.make_node("MatMul", ["XA", "B"], ["XB"]),
            onnx.helper.make_node("MatMul", ["XB", "C"], ["Y"]),
        ],
        "mm",
        [X],
        [Y],
        [
            onnx.large_helper.make_large_tensor_proto(
                "#loc0", "A", onnx.TensorProto.FLOAT, (3, 3)
            ),
            onnx.numpy_helper.from_array(
                np.arange(9).astype(np.float32).reshape((-1, 3)), name="B"
            ),
            onnx.large_helper.make_large_tensor_proto(
                "#loc1", "C", onnx.TensorProto.FLOAT, (3, 3)
            ),
        ],
    )
    onnx_model = onnx.helper.make_model(graph)
    large_model = onnx.large_helper.make_large_model(
        onnx_model.graph,
        {
            "#loc0": (np.arange(9) * 100).astype(np.float32).reshape((-1, 3)),
            "#loc1": (np.arange(9) + 10).astype(np.float32).reshape((-1, 3)),
        },
    )
    large_model.check_model()
    return large_model


class TestLargeOnnx(unittest.TestCase):
    def test_large_onnx_no_large_initializer(self):
        model_proto = _linear_regression()
        assert isinstance(model_proto, onnx.ModelProto)
        large_model = onnx.large_helper.make_large_model(model_proto.graph)
        assert isinstance(large_model, onnx.large_helper.ModelContainer)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnx")
            large_model.save(filename)
            copy = onnx.large_helper.ModelContainer()
            with self.assertRaises(RuntimeError):
                assert copy.model_proto
            copy.load(filename)
            assert copy.model_proto is not None
            onnx.checker.check_model(copy.model_proto)

    def test_large_one_weight_file(self):
        large_model = _large_linear_regression()
        assert isinstance(large_model, onnx.large_helper.ModelContainer)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnx")
            saved_proto = large_model.save(filename, True)
            assert isinstance(saved_proto, onnx.ModelProto)
            copy = onnx.large_helper.ModelContainer()
            copy.load(filename)
            copy.check_model()
            copy2 = onnx.load_model(filename, load_external_data=True)
            onnx.checker.check_model(copy2)

    def test_large_multi_files(self):
        large_model = _large_linear_regression()
        assert isinstance(large_model, onnx.large_helper.ModelContainer)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnx")
            saved_proto = large_model.save(filename, False)
            assert isinstance(saved_proto, onnx.ModelProto)
            copy = onnx.load_model(filename)
            onnx.checker.check_model(copy)
            for tensor in ext_data._get_all_tensors(copy):
                if ext_data.uses_external_data(tensor):
                    for ext in tensor.external_data:
                        if ext.key == "location":  # type: ignore[attr-defined]
                            assert os.path.exists(ext.value)
            copy2 = onnx.load_model(filename, load_external_data=True)
            onnx.checker.check_model(copy2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
