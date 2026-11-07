# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

import atheris
from google.protobuf.descriptor import FieldDescriptor

with atheris.instrument_imports():
    import onnx
    from onnx import checker


def _check_strings(message, path="model"):
    """Every string field of a valid model must hold valid UTF-8."""
    for field, value in message.ListFields():
        values = value if _is_repeated(field) else [value]
        if field.type == FieldDescriptor.TYPE_STRING:
            for item in values:
                assert isinstance(item, str), (
                    f"{path}.{field.name}",
                    item,
                    "Checker accepted a string field that is not valid UTF-8",
                )
        elif field.type == FieldDescriptor.TYPE_MESSAGE:
            for item in values:
                _check_strings(item, f"{path}.{field.name}")


def _is_repeated(field):
    is_repeated = getattr(field, "is_repeated", None)
    if is_repeated is not None:
        return is_repeated
    return field.label == FieldDescriptor.LABEL_REPEATED


def TestOneInput(data):
    try:
        checker.check_model(data, full_check=True)
        model = onnx.load_model_from_string(data)
    except Exception:
        return
    _check_strings(model)


def main():
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
