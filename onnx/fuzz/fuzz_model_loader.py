# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    import onnx


def TestOneInput(data):
    try:
        model = onnx.load_model_from_string(data)
        _ = len(model.graph.node)
        _ = len(model.graph.input)
        _ = len(model.graph.output)
        onnx.checker.check_model(model)
    except Exception:
        return


def main():
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
