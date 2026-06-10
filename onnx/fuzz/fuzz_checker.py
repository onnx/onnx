# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    from onnx import checker


def TestOneInput(data):
    try:
        checker.check_model(data, full_check=True)
    except Exception:
        return


def main():
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
