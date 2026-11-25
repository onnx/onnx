# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

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
