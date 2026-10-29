# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Filesystem-aware Atheris harness for ONNX external data.

The input selects a read, write, model-load, or ModelContainer operation and
mutates external-data paths plus offset/length metadata. All paths resolve in a
disposable workspace. Traversal, absolute-path, symlink, and hardlink cases
point at a sentinel outside the model directory; opening or modifying it is a
finding.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
import tempfile
from pathlib import Path

import atheris

with atheris.instrument_imports():
    import numpy as np
    from google.protobuf.message import DecodeError

    import onnx
    from onnx import TensorProto, external_data_helper, helper
    from onnx.model_container import ModelContainer, make_large_tensor_proto

_TEMP_DIR = tempfile.TemporaryDirectory(prefix="onnx-external-data-fuzz-")
_ROOT = Path(_TEMP_DIR.name)
_BASE = _ROOT / "model"
_SENTINEL = _ROOT / "sentinel.bin"
_SENTINEL_CONTENT = b"ONNX external-data sentinel\n"

_INTEGER_VALUES: tuple[str | None, ...] = (
    None,
    "0",
    "1",
    "-1",
    "+1",
    " 1 ",
    "not-an-integer",
    str(2**31),
    str(2**63 - 1),
    str(2**63),
    str(2**64),
)

_EXPECTED_ERRORS = (
    DecodeError,
    OSError,
    OverflowError,
    RuntimeError,
    TypeError,
    UnicodeError,
    ValueError,
    onnx.checker.ValidationError,
)

_original_open_external_data_fd = external_data_helper._open_external_data_fd


def _checked_open_external_data_fd(
    base_dir: str, location: str, tensor_name: str, read_only: bool
) -> int:
    # Every external-data read and write, including ModelContainer.load, opens
    # through this helper. Checking the descriptor here reports an escape even
    # when a later bounds or reshape error aborts the operation.
    fd = _original_open_external_data_fd(base_dir, location, tensor_name, read_only)
    fd_stat = os.fstat(fd)
    sentinel_stat = _SENTINEL.stat()
    if (fd_stat.st_dev, fd_stat.st_ino) == (sentinel_stat.st_dev, sentinel_stat.st_ino):
        os.close(fd)
        raise AssertionError("external-data open resolved to the sentinel")
    return fd


external_data_helper._open_external_data_fd = _checked_open_external_data_fd


def _reset_workspace(payload: bytes) -> bytes:
    if _BASE.exists():
        shutil.rmtree(_BASE)
    _BASE.mkdir()
    (_BASE / "nested").mkdir()
    (_BASE / "sub").mkdir()

    file_data = (b"external tensor data:" + payload[:1024]) or b"\0"
    (_BASE / "data.bin").write_bytes(file_data)
    (_BASE / "nested" / "data.bin").write_bytes(file_data)

    _SENTINEL.write_bytes(_SENTINEL_CONTENT)
    os.link(_SENTINEL, _BASE / "hardlink.bin")
    # Symlink creation may require an extra privilege on Windows. The hardlink
    # and all non-link path cases remain active there.
    with contextlib.suppress(OSError):
        os.symlink(_SENTINEL, _BASE / "symlink.bin")
    return file_data


def _locations() -> tuple[str, ...]:
    return (
        "data.bin",
        "nested/data.bin",
        "missing.bin",
        "../sentinel.bin",
        "sub/../../sentinel.bin",
        str(_SENTINEL),
        "symlink.bin",
        "hardlink.bin",
        "..\\sentinel.bin",
        "\\\\server\\share\\data.bin",
        "C:data.bin",
        "C:\\data.bin",
        "%2e%2e/sentinel.bin",
        "./data.bin",
        "",
        "data.bin/child",
    )


def _add_metadata(tensor: TensorProto, key: str, value: str) -> None:
    entry = tensor.external_data.add()
    entry.key = key
    entry.value = value


def _make_external_tensor(
    data: bytes,
    location: str,
    alternate_location: str,
    tensor_name: str,
    tensor_size: int,
    raw_data: bytes | None = None,
) -> TensorProto:
    flags = data[2]
    tensor = TensorProto(
        name=tensor_name,
        data_type=TensorProto.UINT8,
        dims=[tensor_size],
        data_location=TensorProto.EXTERNAL,
    )
    if raw_data is not None:
        tensor.raw_data = raw_data

    _add_metadata(tensor, "location", location)
    if flags & 0x01:
        _add_metadata(tensor, "location", alternate_location)

    offset = _INTEGER_VALUES[data[3] % len(_INTEGER_VALUES)]
    if offset is not None:
        _add_metadata(tensor, "offset", offset)
    if flags & 0x02:
        duplicate = _INTEGER_VALUES[(data[3] + 1) % len(_INTEGER_VALUES)]
        if duplicate is not None:
            _add_metadata(tensor, "offset", duplicate)

    length = _INTEGER_VALUES[data[4] % len(_INTEGER_VALUES)]
    if length is not None:
        _add_metadata(tensor, "length", length)
    if flags & 0x04:
        duplicate = _INTEGER_VALUES[(data[4] + 1) % len(_INTEGER_VALUES)]
        if duplicate is not None:
            _add_metadata(tensor, "length", duplicate)

    if flags & 0x08:
        _add_metadata(tensor, "unknown", data[6:38].hex())
        _add_metadata(tensor, "another_unknown", "value")
    if flags & 0x10:
        _add_metadata(tensor, "checksum", data[6:22].hex())
    if flags & 0x20:
        _add_metadata(tensor, "basepath", alternate_location)
    return tensor


def _make_model(*tensors: TensorProto) -> onnx.ModelProto:
    outputs = [
        helper.make_tensor_value_info(tensor.name, TensorProto.UINT8, list(tensor.dims))
        for tensor in tensors
    ]
    graph = helper.make_graph(
        [], "external-data-fuzz", [], outputs, initializer=list(tensors)
    )
    return helper.make_model(graph)


def _exercise(
    data: bytes,
    location: str,
    alternate_location: str,
    payload: bytes,
    file_data: bytes,
) -> None:
    operation = data[0] % 8
    needs_raw_data = operation in (2, 4)
    tensor = _make_external_tensor(
        data,
        location,
        alternate_location,
        "weight",
        len(file_data),
        payload if needs_raw_data else None,
    )

    if operation == 0:
        external_data_helper.ExternalDataInfo(tensor)
    elif operation == 1:
        external_data_helper.load_external_data_for_tensor(tensor, str(_BASE))
    elif operation == 2:
        external_data_helper.save_external_data(tensor, str(_BASE))
    elif operation == 3:
        model_path = _BASE / "model.onnx"
        model_path.write_bytes(_make_model(tensor).SerializeToString())
        onnx.load_model(model_path, load_external_data=True)
    elif operation == 4:
        second_data = bytearray(data)
        second_data[3] = (second_data[3] + 1) % 256
        second = _make_external_tensor(
            bytes(second_data),
            location,
            alternate_location,
            "second_weight",
            len(payload),
            payload[::-1],
        )
        external_data_helper.write_external_data_tensors(
            _make_model(tensor, second), str(_BASE)
        )
    elif operation == 5:
        model_path = _BASE / "container-input.onnx"
        model_path.write_bytes(_make_model(tensor).SerializeToString())
        container = ModelContainer()
        container.load(str(model_path), load_large_initializers=True)
    elif operation == 6:
        embedded = TensorProto(
            name="weight",
            data_type=TensorProto.UINT8,
            dims=[len(payload)],
            raw_data=payload,
        )
        model_path = _BASE / "saved.onnx"
        onnx.save_model(
            _make_model(embedded),
            model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=location,
            size_threshold=0,
        )
        onnx.load_model(model_path, load_external_data=True)
    else:
        array = np.frombuffer(payload, dtype=np.uint8).copy()
        large_tensor = make_large_tensor_proto(
            "#weight", "weight", TensorProto.UINT8, array.shape
        )
        container = ModelContainer()
        container.model_proto = _make_model(large_tensor)
        container.set_large_initializers({"#weight": array})
        model_path = _BASE / "container-output.onnx"
        container.save(str(model_path), all_tensors_to_one_file=bool(data[2] & 1))
        loaded = ModelContainer()
        loaded.load(str(model_path), load_large_initializers=True)


def TestOneInput(data: bytes) -> None:
    if len(data) < 6:
        return
    locations = _locations()
    location_index = data[1] % len(locations)
    alternate_index = (location_index + data[5] + 1) % len(locations)
    payload = data[6:4102] or b"\0"
    # Fixture setup runs before the expected-error suppression so its failures
    # are not mistaken for expected target errors.
    file_data = _reset_workspace(payload)

    with contextlib.suppress(*_EXPECTED_ERRORS):
        _exercise(
            data,
            locations[location_index],
            locations[alternate_index],
            payload,
            file_data,
        )

    # Writes through plain open() (ModelContainer.save) bypass the open check.
    if not _SENTINEL.exists() or _SENTINEL.read_bytes() != _SENTINEL_CONTENT:
        raise AssertionError("external-data operation modified the sentinel")


def main() -> None:
    atheris.instrument_all()
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
