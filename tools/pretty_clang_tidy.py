# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper around run-clang-tidy that filters noisy output."""

from __future__ import annotations

import re
import subprocess
import sys

NOISE = re.compile(
    r"^\[|^\d+ warnings? generated\.|^Suppressed \d|^Use -header-filter"
    r"|\[\d+/\d+\] \(\d+/\d+\) Processing file"
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def main() -> int:
    result = subprocess.run(sys.argv[1:], capture_output=True, text=True)  # noqa: S603,PLW1510
    prev_blank = False
    for line in (result.stdout + result.stderr).splitlines(keepends=True):
        if NOISE.match(ANSI_ESCAPE.sub("", line)):
            continue
        if line.strip():
            prev_blank = False
        else:
            if prev_blank:
                continue
            prev_blank = True
        sys.stdout.write(line)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
