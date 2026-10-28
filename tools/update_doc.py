# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    # The Python build backend owns the CMake configuration and generated protos.
    # Respect the caller's ONNX_ML/CMAKE_ARGS instead of OS-specific overrides.
    commands = [
        [sys.executable, "-m", "pip", "install", "-e", ".", "--no-build-isolation"],
        [sys.executable, "onnx/backend/test/stat_coverage.py"],
        [sys.executable, "onnx/defs/gen_doc.py"],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True)  # noqa: S603


if __name__ == "__main__":
    main()
