# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

import onnx
from onnx.onnx_cpp2py_export import checker

if TYPE_CHECKING:
    from pathlib import Path


def test_directory_context_survives_path_replacement(tmp_path: Path) -> None:
    base = tmp_path / "model"
    base.mkdir()
    (base / "weights.bin").write_bytes(b"original")
    context = checker._ExternalDataContext(str(base))
    original = tmp_path / "original"
    base.rename(original)
    base.mkdir()
    (base / "weights.bin").write_bytes(b"replacement")
    with os.fdopen(context.open("weights.bin", "t", True), "rb") as stream:
        assert stream.read() == b"original"
    with os.fdopen(context.open("created.bin", "t", False), "wb") as stream:
        stream.write(b"created")
    assert (original / "created.bin").read_bytes() == b"created"
    assert not (base / "created.bin").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFOs")
@pytest.mark.parametrize("read_only", [True, False])
def test_fifo_open_does_not_block(tmp_path: Path, read_only: bool) -> None:
    os.mkfifo(tmp_path / "weights.bin")
    # Keep the parent's working directory: sanitizer suppression paths may be
    # relative to it. Pass the temporary directory explicitly to the child.
    result = subprocess.run(  # noqa: S603 -- fixed test code, no shell
        [
            sys.executable,
            "-c",
            (
                "import sys; from onnx.onnx_cpp2py_export import checker; "
                "checker._ExternalDataContext(sys.argv[1]).open('weights.bin', 't', sys.argv[2] == 'True')"
            ),
            str(tmp_path),
            str(read_only),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode != 0
    assert "external data is not a regular file" in result.stderr


@pytest.mark.parametrize("location", ["weights.bin", "#weights.bin"])
@pytest.mark.parametrize("file_size", [None, 12, 24])
def test_path_checker_validates_hash_prefixed_sidecar(
    tmp_path: Path, location: str, file_size: int | None
) -> None:
    tensor = onnx.TensorProto(
        name="weight",
        data_type=onnx.TensorProto.FLOAT,
        dims=[6],
        data_location=onnx.TensorProto.EXTERNAL,
    )
    tensor.external_data.add(key="location", value=location)
    tensor.external_data.add(key="length", value="24")
    model = onnx.helper.make_model(onnx.helper.make_graph([], "test", [], [], [tensor]))
    path = tmp_path / "model.onnx"
    path.write_bytes(model.SerializeToString())
    if file_size is not None:
        (tmp_path / location).write_bytes(bytes(file_size))

    # Metadata-only checking deliberately does not inspect either sidecar.
    onnx.checker.check_model(path, check_external_data_files=False)
    expected_size = 6 * 4
    if file_size == expected_size:
        onnx.checker.check_model(path)
    else:
        with pytest.raises(onnx.checker.ValidationError) as error:
            onnx.checker.check_model(path)
        assert error.value.code == (
            "external-data-file-missing"
            if file_size is None
            else "external-data-out-of-range"
        )
