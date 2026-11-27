# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import os
import shutil
import tarfile
import tempfile

import numpy as np
import pytest

import onnx
from onnx import TensorProto, helper


class TestUtilityFunctions:
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
        assert m1.producer_name == "onnx.utils.extract_model"
        assert m1.ir_version == m0.ir_version
        assert m1.opset_import == m0.opset_import
        assert len(m1.graph.node) == 2
        assert len(m1.graph.input) == 3
        assert len(m1.graph.output) == 2
        assert m1.graph.input[0] == B0
        assert m1.graph.input[1] == B1
        assert m1.graph.input[2] == B2
        assert m1.graph.output[0] == C0
        assert m1.graph.output[1] == C1
        shutil.rmtree(tdir, ignore_errors=True)

    def test_extract_large_model_loads_external_data_from_input_directory(
        self, tmp_path, monkeypatch
    ) -> None:
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        input_path = input_dir / "model.onnx"
        output_path = output_dir / "model.onnx"
        expected = np.array([2.0, 3.0], dtype=np.float32)
        weight = onnx.numpy_helper.from_array(expected, "W")
        graph = helper.make_graph(
            [helper.make_node("Add", ["X", "W"], ["Y"])],
            "test",
            [helper.make_tensor_value_info("X", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2])],
            [weight],
        )
        onnx.save_model(
            helper.make_model(graph),
            input_path,
            save_as_external_data=True,
            location="weights.bin",
            size_threshold=0,
        )
        (output_dir / "weights.bin").write_bytes(
            np.array([99.0, 99.0], dtype=np.float32).tobytes()
        )

        original_getsize = os.path.getsize

        def oversized_input(path):
            if os.fspath(path) == os.fspath(input_path):
                return onnx.checker.MAXIMUM_PROTOBUF + 1
            return original_getsize(path)

        monkeypatch.setattr(onnx.utils.os.path, "getsize", oversized_input)
        onnx.utils.extract_model(
            input_path, output_path, ["X"], ["Y"], check_model=False
        )

        extracted = onnx.load(output_path)
        actual = onnx.numpy_helper.to_array(extracted.graph.initializer[0])
        np.testing.assert_array_equal(actual, expected)

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

            with tarfile.open(tar_path) as tar:  # noqa: SIM117
                with pytest.raises(RuntimeError, match="directory traversal"):
                    onnx.utils._tar_members_filter(tar, base)
