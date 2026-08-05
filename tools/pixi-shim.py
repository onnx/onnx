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
    # Some Python distributions (e.g. standalone builds managed by uv) set
    # PYTHONHOME/PYTHONPATH for themselves. Inherited by this subprocess, those
    # variables leak into whichever interpreter pixi activates for the target
    # environment and make it look for packages in the wrong place, so strip
    # them and let pixi fully manage the environment.
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    raise SystemExit(subprocess.call(["pixi", "run", "--", *args], env=env))  # noqa: S603, S607

warnings.warn(
    "pixi not found. Running tools directly from PATH. "
    "Install pixi for a fully managed environment: https://pixi.sh",
    stacklevel=1,
)
raise SystemExit(subprocess.call(args))  # noqa: S603
