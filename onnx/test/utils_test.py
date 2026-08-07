# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io
import os
import shutil
import tarfile
import tempfile

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

    def test_tar_members_filter_rejects_special_files(self) -> None:
        with tempfile.TemporaryDirectory() as tdir:
            base = os.path.join(tdir, "model")
            os.mkdir(base)
            tar_path = os.path.join(tdir, "payload.tar")

            with tarfile.open(tar_path, "w") as tar:
                info = tarfile.TarInfo("named-pipe")
                info.type = tarfile.FIFOTYPE
                tar.addfile(info)

            with tarfile.open(tar_path) as tar:
                with pytest.raises(RuntimeError, match="regular file or directory"):
                    onnx.utils._tar_members_filter(tar, base)

    def test_tar_members_filter_enforces_limits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with tempfile.TemporaryDirectory() as tdir:
            base = os.path.join(tdir, "model")
            os.mkdir(base)
            tar_path = os.path.join(tdir, "payload.tar")
            info = tarfile.TarInfo("large.bin")
            info.size = 2
            with tarfile.open(tar_path, "w") as tar:
                tar.addfile(info, io.BytesIO(b"12"))

            monkeypatch.setattr(onnx.utils, "_MAX_TAR_MEMBER_SIZE", 1)
            with tarfile.open(tar_path) as tar:
                with pytest.raises(RuntimeError, match="too large"):
                    onnx.utils._tar_members_filter(tar, base)

    def test_tar_members_filter_enforces_member_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with tempfile.TemporaryDirectory() as tdir:
            base = os.path.join(tdir, "model")
            os.mkdir(base)
            tar_path = os.path.join(tdir, "payload.tar")
            with tarfile.open(tar_path, "w") as tar:
                tar.addfile(tarfile.TarInfo("file"))

            monkeypatch.setattr(onnx.utils, "_MAX_TAR_MEMBERS", 0)
            with tarfile.open(tar_path) as tar:
                with pytest.raises(RuntimeError, match="too many members"):
                    onnx.utils._tar_members_filter(tar, base)

    def test_tar_members_filter_rejects_duplicate_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tdir:
            base = os.path.join(tdir, "model")
            os.mkdir(base)
            tar_path = os.path.join(tdir, "payload.tar")
            with tarfile.open(tar_path, "w") as tar:
                for name in ("file", "./file"):
                    info = tarfile.TarInfo(name)
                    info.size = 1
                    tar.addfile(info, io.BytesIO(b"x"))

            with tarfile.open(tar_path) as tar:
                with pytest.raises(RuntimeError, match="duplicate member path"):
                    onnx.utils._tar_members_filter(tar, base)

    def test_extract_model_safe_does_not_partially_extract(self) -> None:
        with tempfile.TemporaryDirectory() as tdir:
            destination = os.path.join(tdir, "model")
            os.mkdir(destination)
            marker = os.path.join(destination, "keep.txt")
            with open(marker, "w", encoding="utf-8") as file:
                file.write("keep")
            tar_path = os.path.join(tdir, "payload.tar")

            with tarfile.open(tar_path, "w") as tar:
                payload = b"new"
                info = tarfile.TarInfo("new.txt")
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
                special = tarfile.TarInfo("named-pipe")
                special.type = tarfile.FIFOTYPE
                tar.addfile(special)

            with pytest.raises((RuntimeError, tarfile.SpecialFileError)):
                onnx.utils._extract_model_safe(tar_path, destination)
            assert os.path.exists(marker)
            assert not os.path.exists(os.path.join(destination, "new.txt"))

    def test_extract_model_safe_rejects_destination_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as tdir:
            destination = os.path.join(tdir, "model")
            os.mkdir(destination)
            outside = os.path.join(tdir, "outside")
            os.mkdir(outside)
            try:
                os.symlink(outside, os.path.join(destination, "link"))
            except (NotImplementedError, OSError):
                pytest.skip("symbolic links are unavailable")

            tar_path = os.path.join(tdir, "payload.tar")
            with tarfile.open(tar_path, "w") as tar:
                tar.addfile(tarfile.TarInfo("file"))

            with pytest.raises(RuntimeError, match="destination contains symbolic links"):
                onnx.utils._extract_model_safe(tar_path, destination)
