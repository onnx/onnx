# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    import onnx
    from onnx import version_converter


def _default_opset_version(model):
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version
    return None


def _candidate_target_versions(model):
    current = _default_opset_version(model)
    latest = onnx.defs.onnx_opset_version()
    if current is None:
        return [latest]

    targets = []
    if current > 1:
        targets.append(current - 1)
    if current < latest:
        targets.append(current + 1)
    if latest not in targets and current != latest:
        targets.append(latest)
    return targets


def _is_valid(model):
    try:
        onnx.checker.check_model(model, full_check=True)
    except Exception:
        return False
    return True


def TestOneInput(data):
    try:
        model = onnx.load_model_from_string(data)
    except Exception:
        return

    valid = _is_valid(model)
    for target_version in _candidate_target_versions(model):
        try:
            converted = version_converter.convert_version(model, target_version)
        except Exception:
            # Refusing a conversion is fine; producing an invalid model is not.
            continue
        if valid:
            try:
                onnx.checker.check_model(converted, full_check=True)
            except Exception as error:
                raise AssertionError(
                    f"Converting a valid model to opset {target_version} "
                    "produced an invalid model"
                ) from error


def main():
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
