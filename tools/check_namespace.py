# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Check that C++ files do not hardcode the onnx namespace.

Other libraries that statically link with onnx can hide onnx symbols
in a private namespace, so the namespace should not be hardcoded.
"""

from __future__ import annotations

import sys


def main() -> int:
    violations = []
    for path in sys.argv[1:]:
        with open(path) as f:
            for line_no, line in enumerate(f, 1):
                if "namespace onnx" in line or "onnx::" in line:
                    violations.append(f"{path}:{line_no}: {line.rstrip()}")
    if violations:
        print("Hardcoded onnx namespace found:")
        print("\n".join(violations))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
