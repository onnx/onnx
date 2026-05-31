# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""Wrapper that runs a command via pixi if available, otherwise directly.

Usage: python tools/pixi-shim.py <command> [args...]

This allows non-pixi users to run pre-commit hooks by having the
required tools (ruff, mypy, clang-format, shellcheck, reuse, typos) on their PATH.

The 'default' environment is used if Pixi is available.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import warnings

args = sys.argv[1:]

if shutil.which("pixi"):
    raise SystemExit(subprocess.call(["pixi", "run", "--", *args]))  # noqa: S603, S607

warnings.warn(
    "pixi not found. Running tools directly from PATH. "
    "Install pixi for a fully managed environment: https://pixi.sh",
    stacklevel=1,
)
raise SystemExit(subprocess.call(args))  # noqa: S603
