# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import tempfile
from typing import IO

import numpy as np
import pytest

import onnx
import onnx.external_data_helper as ext_data
import onnx.helper
import onnx.model_container
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
            onnx.model_container.make_large_tensor_proto(
                "#loc0", "A", onnx.TensorProto.FLOAT, (3, 3)
            ),
            onnx.numpy_helper.from_array(
                np.arange(9).astype(np.float32).reshape((-1, 3)), name="B"
            ),
            onnx.model_container.make_large_tensor_proto(
                "#loc1", "C", onnx.TensorProto.FLOAT, (3, 3)
            ),
        ],
    )
    onnx_model = onnx.helper.make_model(graph)
    large_model = onnx.model_container.make_large_model(
        onnx_model.graph,
        {
            "#loc0": (np.arange(9) * 100).astype(np.float32).reshape((-1, 3)),
            "#loc1": (np.arange(9) + 10).astype(np.float32).reshape((-1, 3)),
        },
    )
    large_model.check_model()
    return large_model


class TestLargeOnnx:
    def test_large_onnx_no_large_initializer(self):
        model_proto = _linear_regression()
        assert isinstance(model_proto, onnx.ModelProto)
        large_model = onnx.model_container.make_large_model(model_proto.graph)
        assert isinstance(large_model, onnx.model_container.ModelContainer)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnx")
            large_model.save(filename)
            copy = onnx.model_container.ModelContainer()
            with pytest.raises(RuntimeError):
                assert copy.model_proto
            copy.load(filename)
            assert copy.model_proto is not None
            onnx.checker.check_model(copy.model_proto)

    def test_large_one_weight_file(self):
        large_model = _large_linear_regression()
        assert isinstance(large_model, onnx.model_container.ModelContainer)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnx")
            saved_proto = large_model.save(filename, True)
            assert isinstance(saved_proto, onnx.ModelProto)
            copy = onnx.model_container.ModelContainer()
            copy.load(filename)
            copy.check_model()
            loaded_model = onnx.load_model(filename, load_external_data=True)
            onnx.checker.check_model(loaded_model)

    @pytest.mark.parametrize("all_tensors_to_one_file", [True, False])
    def test_large_model_values_roundtrip(self, all_tensors_to_one_file):
        # Regression test: writes now go through the same hardened
        # save_external_data primitive as write_external_data_tensors,
        # rather than the container computing/writing offsets itself.
        # Verify the actual tensor values still round-trip correctly,
        # including with multiple large tensors sharing one file.
        large_model = _large_linear_regression()
        expected = {
            "A": large_model["#loc0"],
            "C": large_model["#loc1"],
        }
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnx")
            large_model.save(filename, all_tensors_to_one_file)

            copy = onnx.model_container.ModelContainer()
            copy.load(filename)
            copy.check_model()

            for tensor in copy.model_proto.graph.initializer:
                if tensor.name in expected:
                    for ext in tensor.external_data:
                        if ext.key == "location":
                            np.testing.assert_array_equal(
                                copy[ext.value], expected[tensor.name]
                            )

    def test_large_model_save_does_not_stage_data_in_proto(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        large_model = _large_linear_regression()
        write_external_data = ext_data._write_external_data_to_file
        tensors_with_raw_data: list[bool] = []

        def check_no_raw_data(
            tensor: onnx.TensorProto, data_file: IO[bytes], data: bytes
        ) -> None:
            tensors_with_raw_data.append(tensor.HasField("raw_data"))
            write_external_data(tensor, data_file, data)

        monkeypatch.setattr(ext_data, "_write_external_data_to_file", check_no_raw_data)
        with tempfile.TemporaryDirectory() as temp:
            large_model.save(os.path.join(temp, "model.onnx"), True)

        assert tensors_with_raw_data
        assert not any(tensors_with_raw_data)

    @pytest.mark.parametrize("all_tensors_to_one_file", [True, False])
    def test_save_rejects_hardlinked_weight_without_modifying_target(
        self, all_tensors_to_one_file: bool
    ) -> None:
        large_model = _large_linear_regression()
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnx")
            weight = (
                f"{filename}.weight"
                if all_tensors_to_one_file
                else os.path.join(temp, "model-#loc0.weight")
            )
            victim = os.path.join(temp, "victim.bin")
            expected = b"do not overwrite"
            with open(victim, "wb") as f:
                f.write(expected)
            os.link(victim, weight)

            with pytest.raises(
                onnx.checker.ValidationError, match="multiple hard links"
            ):
                large_model.save(filename, all_tensors_to_one_file)

            with open(victim, "rb") as f:
                assert f.read() == expected

    @pytest.mark.parametrize("all_tensors_to_one_file", [True, False])
    def test_save_rejects_symlinked_weight_without_modifying_target(
        self, all_tensors_to_one_file: bool
    ) -> None:
        large_model = _large_linear_regression()
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnx")
            weight = (
                f"{filename}.weight"
                if all_tensors_to_one_file
                else os.path.join(temp, "model-#loc0.weight")
            )
            victim = os.path.join(temp, "victim.bin")
            expected = b"do not overwrite"
            with open(victim, "wb") as f:
                f.write(expected)
            try:
                os.symlink(victim, weight)
            except OSError as error:
                pytest.skip(f"Cannot create symlinks on this platform: {error}")

            with pytest.raises(onnx.checker.ValidationError):
                large_model.save(filename, all_tensors_to_one_file)

            with open(victim, "rb") as f:
                assert f.read() == expected

    @pytest.mark.parametrize("all_tensors_to_one_file", [True, False])
    def test_save_truncates_existing_regular_weight(
        self, all_tensors_to_one_file: bool
    ) -> None:
        large_model = _large_linear_regression()
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnx")
            weight = (
                f"{filename}.weight"
                if all_tensors_to_one_file
                else os.path.join(temp, "model-#loc0.weight")
            )
            with open(weight, "wb") as f:
                f.write(b"stale" * 100)

            large_model.save(filename, all_tensors_to_one_file)

            expected_size = (
                sum(array.nbytes for array in large_model.large_initializers.values())
                if all_tensors_to_one_file
                else large_model["#loc0"].nbytes
            )
            assert os.path.getsize(weight) == expected_size

    def test_large_multi_files(self):
        large_model = _large_linear_regression()
        assert isinstance(large_model, onnx.model_container.ModelContainer)
        with tempfile.TemporaryDirectory() as temp:
            filename = os.path.join(temp, "model.onnx")
            saved_proto = large_model.save(filename, False)
            assert isinstance(saved_proto, onnx.ModelProto)
            copy = onnx.load_model(filename)
            onnx.checker.check_model(copy)
            for tensor in ext_data._get_all_tensors(copy):
                if ext_data.uses_external_data(tensor):
                    tested = 0
                    for ext in tensor.external_data:
                        if ext.key == "location":
                            assert os.path.exists(ext.value)
                            tested += 1
                    assert tested == 1
            loaded_model = onnx.load_model(filename, load_external_data=True)
            onnx.checker.check_model(loaded_model)
