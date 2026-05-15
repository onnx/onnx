# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import os
import shutil
import tarfile
import tempfile
import unittest

import onnx
from onnx import TensorProto, helper, utils


class TestUtilityFunctions(unittest.TestCase):
    def test_extract_model(self) -> None:
        def create_tensor(name):
            return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2])

        A0 = create_tensor("A0")
        A1 = create_tensor("A1")
        B0 = create_tensor("B0")
        B1 = create_tensor("B1")
        B2 = create_tensor("B2")
        C0 = create_tensor("C0")
        C1 = create_tensor("C1")
        D0 = create_tensor("D0")
        L0_0 = helper.make_node("Add", ["A0", "A1"], ["B0"])
        L0_1 = helper.make_node("Sub", ["A0", "A1"], ["B1"])
        L0_2 = helper.make_node("Mul", ["A0", "A1"], ["B2"])
        L1_0 = helper.make_node("Add", ["B0", "B1"], ["C0"])
        L1_1 = helper.make_node("Sub", ["B1", "B2"], ["C1"])
        L2_0 = helper.make_node("Mul", ["C0", "C1"], ["D0"])

        g0 = helper.make_graph(
            [L0_0, L0_1, L0_2, L1_0, L1_1, L2_0], "test", [A0, A1], [D0]
        )
        m0 = helper.make_model(g0, producer_name="test")
        tdir = tempfile.mkdtemp()
        p0 = os.path.join(tdir, "original.onnx")
        onnx.save(m0, p0)

        p1 = os.path.join(tdir, "extracted.onnx")
        input_names = ["B0", "B1", "B2"]
        output_names = ["C0", "C1"]
        onnx.utils.extract_model(p0, p1, input_names, output_names)

        m1 = onnx.load(p1)
        self.assertEqual(m1.producer_name, "onnx.utils.extract_model")
        self.assertEqual(m1.ir_version, m0.ir_version)
        self.assertEqual(m1.opset_import, m0.opset_import)
        self.assertEqual(len(m1.graph.node), 2)
        self.assertEqual(len(m1.graph.input), 3)
        self.assertEqual(len(m1.graph.output), 2)
        self.assertEqual(m1.graph.input[0], B0)
        self.assertEqual(m1.graph.input[1], B1)
        self.assertEqual(m1.graph.input[2], B2)
        self.assertEqual(m1.graph.output[0], C0)
        self.assertEqual(m1.graph.output[1], C1)
        shutil.rmtree(tdir, ignore_errors=True)

    def test_tar_members_filter_rejects_sibling_prefix_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tdir:
            base = os.path.join(tdir, "model")
            os.mkdir(base)
            tar_path = os.path.join(tdir, "payload.tar")

            with tarfile.open(tar_path, "w") as tar:
                payload = b"outside extraction root"
                info = tarfile.TarInfo("../model_evil/pwned.txt")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))

            with (
                tarfile.open(tar_path) as tar,
                self.assertRaisesRegex(RuntimeError, "directory traversal"),
            ):
                onnx.utils._tar_members_filter(tar, base)


def test_get_model_initializers_size(self) -> None:
    # Empty
    graph_empty = helper.make_graph([], "test_empty", [], [])
    model_empty = helper.make_model(graph_empty)
    self.assertEqual(utils.get_model_initializers_size(model_empty), 0)

    # Standard (FLOAT=4 bytes, INT64=8 bytes)
    t1 = helper.make_tensor(
        "t1", TensorProto.FLOAT, [2, 2], [1.0, 2.0, 3.0, 4.0]
    )  # 16 bytes
    t2 = helper.make_tensor("t2", TensorProto.INT64, [3], [1, 2, 3])  # 24 bytes
    graph_std = helper.make_graph([], "test_std", [], [], initializer=[t1, t2])
    model_std = helper.make_model(graph_std)
    self.assertEqual(utils.get_model_initializers_size(model_std), 40)

    # Raw Data
    t3 = helper.make_tensor("t3", TensorProto.FLOAT, [2], b"12345678", raw=True)
    graph_raw = helper.make_graph([], "test_raw", [], [], initializer=[t3])
    model_raw = helper.make_model(graph_raw)
    self.assertEqual(utils.get_model_initializers_size(model_raw), 8)

    # INT4 Packed Tensor
    t4 = helper.make_tensor(
        "int4_tensor",
        TensorProto.INT4,
        [5],
        [1, 2, 3, 4, 5],
    )
    graph_int4 = helper.make_graph([], "test_int4", [], [], initializer=[t4])
    model_int4 = helper.make_model(graph_int4)

    # ceil(5 * 0.5) = 3 bytes
    self.assertEqual(utils.get_model_initializers_size(model_int4), 3)

    # UINT2 Packed Tensor
    t5 = helper.make_tensor(
        "uint2_tensor",
        TensorProto.UINT2,
        [7],
        [0, 1, 2, 3, 0, 1, 2],
    )
    graph_uint2 = helper.make_graph([], "test_uint2", [], [], initializer=[t5])
    model_uint2 = helper.make_model(graph_uint2)

    # ceil(7 * 0.25) = 2 bytes
    self.assertEqual(utils.get_model_initializers_size(model_uint2), 2)

    # STRING Tensor
    t6 = helper.make_tensor(
        "str_tensor",
        TensorProto.STRING,
        [3],
        [b"abc", b"de", b"f"],
    )
    graph_string = helper.make_graph([], "test_string", [], [], initializer=[t6])
    model_string = helper.make_model(graph_string)

    # 3 + 2 + 1 = 6 bytes
    self.assertEqual(utils.get_model_initializers_size(model_string), 6)


if __name__ == "__main__":
    unittest.main()
