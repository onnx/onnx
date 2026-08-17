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

import os
import shutil
import subprocess
import sys
import warnings

args = sys.argv[1:]

if shutil.which("pixi"):
    # If this script itself is running under a Python that sets PYTHONHOME
    # (e.g. a uv-managed interpreter), that value leaks into the pixi-managed
    # environment's own Python and breaks its site-packages resolution.
    # PYTHONHOME/PYTHONPATH are specific to *this* interpreter, not the one
    # pixi is about to invoke, so don't propagate them.
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    raise SystemExit(subprocess.call(["pixi", "run", "--", *args], env=env))  # noqa: S603, S607

warnings.warn(
    "pixi not found. Running tools directly from PATH. "
    "Install pixi for a fully managed environment: https://pixi.sh",
    stacklevel=1,
)
raise SystemExit(subprocess.call(args))  # noqa: S603
