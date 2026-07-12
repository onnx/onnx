# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    from onnx import parser


def TestOneInput(data):
    try:
        text = data.decode("utf-8", "surrogatepass")
        parser.parse_model(text)
    except Exception:
        return


def main():
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
