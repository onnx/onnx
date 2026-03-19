# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import pathlib
import re
import sys
import uuid
import warnings
from itertools import chain
from typing import IO, TYPE_CHECKING

import onnx.checker as onnx_checker
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx.onnx_pb import (
    AttributeProto,
    FunctionProto,
    GraphProto,
    ModelProto,
    TensorProto,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

# Security: 3-layer defense against malicious external_data entries (GHSA-538c-55jv-c5g9)
#
# Layer 1 (here) — Attribute whitelist: Only spec-defined keys are accepted.
#   Unknown keys are warned and ignored, preventing arbitrary attribute injection (CWE-915).
#
# Layer 2 (ExternalDataInfo.__init__) — Bounds validation: offset and length must be
#   non-negative integers. Catches invalid values at parse time (CWE-400).
#
# Layer 3 (load_external_data_for_tensor) — File-size validation: offset and length are
#   checked against actual file size before reading. This is the critical safety net that
#   prevents memory exhaustion regardless of how the model was constructed (CWE-400).
#
# 'basepath' is included because set_external_data() and model_container
# write it to protobuf entries; it must survive save/load round-trips.
_ALLOWED_EXTERNAL_DATA_KEYS = frozenset(
    {"location", "offset", "length", "checksum", "basepath"}
)
_SORTED_ALLOWED_KEYS = sorted(_ALLOWED_EXTERNAL_DATA_KEYS)
_MAX_UNKNOWN_KEYS_IN_WARNING = 10
_MAX_KEY_DISPLAY_LENGTH = 100


class ExternalDataInfo:
    def __init__(self, tensor: TensorProto) -> None:
        self.location = ""
        self.offset = None
        self.length = None
        self.checksum = None
        self.basepath = ""

        unknown_keys: set[str] = set()
        unknown_key_count = 0
        for entry in tensor.external_data:
            # Layer 1: reject unknown keys (CWE-915 defense-in-depth)
            if entry.key in _ALLOWED_EXTERNAL_DATA_KEYS:
                setattr(self, entry.key, entry.value)
            else:
                unknown_key_count += 1
                if len(unknown_keys) < _MAX_UNKNOWN_KEYS_IN_WARNING:
                    truncated = entry.key[:_MAX_KEY_DISPLAY_LENGTH]
                    if len(entry.key) > _MAX_KEY_DISPLAY_LENGTH:
                        truncated += "..."
                    unknown_keys.add(truncated)

        if unknown_keys:
            shown = sorted(unknown_keys)
            extra = unknown_key_count - len(shown)
            key_list = repr(shown)
            if extra > 0:
                key_list += f" and {extra} more"
            warnings.warn(
                f"Ignoring unknown external data key(s) {key_list} "
                f"for tensor {tensor.name!r}. "
                f"Allowed keys: {_SORTED_ALLOWED_KEYS}",
                stacklevel=2,
            )

        if self.offset is not None:
            self.offset = int(self.offset)
            if self.offset < 0:
                raise ValueError(
                    f"External data offset must be non-negative, got {self.offset} "
                    f"for tensor {tensor.name!r}"
                )

        if self.length is not None:
            self.length = int(self.length)
            if self.length < 0:
                raise ValueError(
                    f"External data length must be non-negative, got {self.length} "
                    f"for tensor {tensor.name!r}"
                )


def _validate_external_data_file_bounds(
    data_file: IO[bytes],
    info: ExternalDataInfo,
    tensor_name: str,
) -> bytes:
    """Validate offset/length against actual file size and read data.

    Layer 3 defense-in-depth (CWE-400): prevents memory exhaustion even if the
    model was crafted via direct protobuf APIs that bypass Python parsing.

    Returns the raw bytes read from the file.
    """
    file_size = os.fstat(data_file.fileno()).st_size

    if info.offset is not None:
        if info.offset > file_size:
            raise ValueError(
                f"External data offset ({info.offset}) exceeds file size "
                f"({file_size}) for tensor {tensor_name!r}"
            )
        data_file.seek(info.offset)

    if info.length is not None:
        read_start = info.offset if info.offset is not None else 0
        available = file_size - read_start
        if info.length > available:
            raise ValueError(
                f"External data length ({info.length}) exceeds available data "
                f"({available} bytes from offset {read_start}) "
                f"for tensor {tensor_name!r}"
            )
        return data_file.read(info.length)
    return data_file.read()


def _validate_external_data_path(
    base_dir: str,
    data_path: str,
    tensor_name: str,
    *,
    check_exists: bool = True,
) -> str:
    """Validate that an external data path is safe to open.

    Performs three security checks:
    1. Canonical path containment — resolved path must stay within base_dir.
    2. Symlink rejection — final-component symlinks are not allowed.
    3. Hardlink count — files with multiple hard links are rejected.

    Args:
        base_dir: The model base directory that data_path must be contained in.
        data_path: The external data file path to validate.
        tensor_name: Tensor name for error messages.
        check_exists: If True (default), check hardlink count. Set to False
            for save-side paths where the file may not exist yet.

    Returns:
        The validated data_path (unchanged).

    Raises:
        onnx.checker.ValidationError: If any security check fails.
    """
    real_base = os.path.realpath(base_dir)
    real_path = os.path.realpath(data_path)
    if not real_path.startswith(real_base + os.sep) and real_path != real_base:
        raise onnx_checker.ValidationError(
            f"Tensor {tensor_name!r} external data path resolves to "
            f"{real_path!r} which is outside the model directory {real_base!r}."
        )
    if os.path.islink(data_path):
        raise onnx_checker.ValidationError(
            f"Tensor {tensor_name!r} external data path {data_path!r} "
            f"is a symbolic link, which is not allowed for security reasons."
        )
    if check_exists and os.path.exists(data_path) and os.stat(data_path).st_nlink > 1:
        raise onnx_checker.ValidationError(
            f"Tensor {tensor_name!r} external data path {data_path!r} "
            f"has multiple hard links, which is not allowed for security reasons."
        )
    return data_path


def load_external_data_for_tensor(tensor: TensorProto, base_dir: str) -> None:
    """Loads data from an external file for tensor.
    Ideally TensorProto should not hold any raw data but if it does it will be ignored.

    Arguments:
        tensor: a TensorProto object.
        base_dir: directory that contains the external data.
    """
    info = ExternalDataInfo(tensor)
    external_data_file_path = c_checker._resolve_external_data_location(  # type: ignore[attr-defined]
        base_dir, info.location, tensor.name
    )
    # Security checks (symlink, containment, hardlink) already performed
    # by C++ _resolve_external_data_location() above.
    # Use O_NOFOLLOW where available as defense-in-depth for symlink protection
    open_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    fd = os.open(external_data_file_path, open_flags)
    with os.fdopen(fd, "rb") as data_file:
        tensor.raw_data = _validate_external_data_file_bounds(
            data_file, info, tensor.name
        )


def load_external_data_for_model(model: ModelProto, base_dir: str) -> None:
    """Loads external tensors into model

    Arguments:
        model: ModelProto to load external data to
        base_dir: directory that contains external data
    """
    for tensor in _get_all_tensors(model):
        if uses_external_data(tensor):
            load_external_data_for_tensor(tensor, base_dir)
            # After loading raw_data from external_data, change the state of tensors
            tensor.data_location = TensorProto.DEFAULT
            # and remove external data
            del tensor.external_data[:]


def set_external_data(
    tensor: TensorProto,
    location: str,
    offset: int | None = None,
    length: int | None = None,
    checksum: str | None = None,
    basepath: str | None = None,
) -> None:
    if not tensor.HasField("raw_data"):
        raise ValueError(
            "Tensor "
            + tensor.name
            + " does not have raw_data field. Cannot set external data for this tensor."
        )

    del tensor.external_data[:]
    tensor.data_location = TensorProto.EXTERNAL
    for k, v in {
        "location": location,
        "offset": int(offset) if offset is not None else None,
        "length": int(length) if length is not None else None,
        "checksum": checksum,
        "basepath": basepath,
    }.items():
        if v is not None:
            entry = tensor.external_data.add()
            entry.key = k
            entry.value = str(v)


def convert_model_to_external_data(
    model: ModelProto,
    all_tensors_to_one_file: bool = True,
    location: str | None = None,
    size_threshold: int = 1024,
    convert_attribute: bool = False,
) -> None:
    """Call to set all tensors with raw data as external data. This call should precede 'save_model'.
    'save_model' saves all the tensors data as external data after calling this function.

    Arguments:
        model (ModelProto): Model to be converted.
        all_tensors_to_one_file (bool): If true, save all tensors to one external file specified by location.
            If false, save each tensor to a file named with the tensor name.
        location: specify the external file relative to the model that all tensors to save to.
            Path is relative to the model path.
            If not specified, will use the model name.
        size_threshold: Threshold for size of data. Only when tensor's data is >= the size_threshold
            it will be converted to external data. To convert every tensor with raw data to external data set size_threshold=0.
        convert_attribute (bool): If true, convert all tensors to external data
                       If false, convert only non-attribute tensors to external data

    Raise:
        ValueError: If location is not a relative path.
        FileExistsError: If a file already exists in location.
    """
    tensors = _get_initializer_tensors(model)
    if convert_attribute:
        tensors = _get_all_tensors(model)

    if all_tensors_to_one_file:
        file_name = str(uuid.uuid1()) + ".data"
        if location:
            if os.path.isabs(location):
                raise ValueError(
                    "location must be a relative path that is relative to the model path."
                )
            if os.path.exists(location):
                raise FileExistsError(f"External data file exists in {location}.")
            file_name = location
        for tensor in tensors:
            if (
                tensor.HasField("raw_data")
                and sys.getsizeof(tensor.raw_data) >= size_threshold
            ):
                set_external_data(tensor, file_name)
    else:
        for tensor in tensors:
            if (
                tensor.HasField("raw_data")
                and sys.getsizeof(tensor.raw_data) >= size_threshold
            ):
                tensor_location = tensor.name
                if not _is_valid_filename(tensor_location):
                    tensor_location = str(uuid.uuid1())
                set_external_data(tensor, tensor_location)


def convert_model_from_external_data(model: ModelProto) -> None:
    """Call to set all tensors which use external data as embedded data.
    save_model saves all the tensors data as embedded data after
    calling this function.

    Arguments:
        model (ModelProto): Model to be converted.
    """
    for tensor in _get_all_tensors(model):
        if uses_external_data(tensor):
            if not tensor.HasField("raw_data"):
                raise ValueError("raw_data field doesn't exist.")
            del tensor.external_data[:]
            tensor.data_location = TensorProto.DEFAULT


def save_external_data(tensor: TensorProto, base_path: str) -> None:
    """Writes tensor data to an external file according to information in the `external_data` field.
    The function checks the external is a valid name and located in folder `base_path`.

    Arguments:
        tensor (TensorProto): Tensor object to be serialized
        base_path: System path of a folder where tensor data is to be stored

    Raises:
        ValueError: If the external file is invalid.
    """
    info = ExternalDataInfo(tensor)

    # Let's check the tensor location is valid.
    location_path = pathlib.Path(info.location)
    if location_path.is_absolute():
        raise onnx_checker.ValidationError(
            f"Tensor {tensor.name!r} is external and must not be defined "
            f"with an absolute path such as {info.location!r}, "
            f"base_path={base_path!r}"
        )
    if ".." in location_path.parts:
        raise onnx_checker.ValidationError(
            f"Tensor {tensor.name!r} is external and must be placed in folder "
            f"{base_path!r}, '..' is not needed in {info.location!r}."
        )
    if location_path.name in (".", ".."):
        raise onnx_checker.ValidationError(
            f"Tensor {tensor.name!r} is external and its name "
            f"{info.location!r} is invalid."
        )

    external_data_file_path = os.path.join(base_path, info.location)

    # C++ _resolve_external_data_location() cannot be used on save path
    # (file may not exist yet), so Python performs its own security validation.
    _validate_external_data_path(
        base_path, external_data_file_path, tensor.name, check_exists=True
    )

    # Retrieve the tensor's data from raw_data or load external file
    if not tensor.HasField("raw_data"):
        raise onnx_checker.ValidationError("raw_data field doesn't exist.")

    # Atomic file creation with symlink protection (O_NOFOLLOW where available)
    open_flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        open_flags |= os.O_NOFOLLOW
    # Use restrictive permissions: owner read/write only (0o600)
    fd = os.open(external_data_file_path, open_flags, 0o600)

    # Open file for reading and writing at random locations ('r+b')
    with os.fdopen(fd, "r+b") as data_file:
        data_file.seek(0, 2)
        if info.offset is not None:
            # Pad file to required offset if needed
            file_size = data_file.tell()
            if info.offset > file_size:
                data_file.write(b"\0" * (info.offset - file_size))

            data_file.seek(info.offset)
        offset = data_file.tell()
        data_file.write(tensor.raw_data)
        set_external_data(tensor, info.location, offset, data_file.tell() - offset)


def _get_all_tensors(onnx_model_proto: ModelProto) -> Iterable[TensorProto]:
    """Scan an ONNX model for all tensors and return as an iterator."""
    return chain(
        _get_initializer_tensors(onnx_model_proto),
        _get_attribute_tensors(onnx_model_proto),
    )


def _recursive_attribute_processor(
    attribute: AttributeProto, func: Callable[[GraphProto], Iterable[TensorProto]]
) -> Iterable[TensorProto]:
    """Create an iterator through processing ONNX model attributes with functor."""
    if attribute.type == AttributeProto.GRAPH:
        yield from func(attribute.g)
    if attribute.type == AttributeProto.GRAPHS:
        for graph in attribute.graphs:
            yield from func(graph)


def _get_initializer_tensors_from_graph(graph: GraphProto, /) -> Iterable[TensorProto]:
    """Create an iterator of initializer tensors from ONNX model graph."""
    yield from graph.initializer
    for node in graph.node:
        for attribute in node.attribute:
            yield from _recursive_attribute_processor(
                attribute, _get_initializer_tensors_from_graph
            )


def _get_initializer_tensors(onnx_model_proto: ModelProto) -> Iterable[TensorProto]:
    """Create an iterator of initializer tensors from ONNX model."""
    yield from _get_initializer_tensors_from_graph(onnx_model_proto.graph)


def _get_attribute_tensors_from_graph(
    graph_or_function: GraphProto | FunctionProto, /
) -> Iterable[TensorProto]:
    """Create an iterator of tensors from node attributes of an ONNX model graph/function."""
    for node in graph_or_function.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            yield from _recursive_attribute_processor(
                attribute, _get_attribute_tensors_from_graph
            )


def _get_attribute_tensors(onnx_model_proto: ModelProto) -> Iterable[TensorProto]:
    """Create an iterator of tensors from node attributes of an ONNX model."""
    yield from _get_attribute_tensors_from_graph(onnx_model_proto.graph)
    for function in onnx_model_proto.functions:
        yield from _get_attribute_tensors_from_graph(function)


def _is_valid_filename(filename: str) -> bool:
    """Utility to check whether the provided filename is valid."""
    exp = re.compile('^[^<>:;,?"*|/]+$')
    match = exp.match(filename)
    return bool(match)


def uses_external_data(tensor: TensorProto) -> bool:
    """Returns true if the tensor stores data in an external location."""
    return (
        tensor.HasField("data_location")
        and tensor.data_location == TensorProto.EXTERNAL
    )


def remove_external_data_field(tensor: TensorProto, field_key: str) -> None:
    """Removes a field from a Tensor's external_data key-value store.

    Modifies tensor object in place.

    Arguments:
        tensor (TensorProto): Tensor object from which value will be removed
        field_key (string): The key of the field to be removed
    """
    for i, field in enumerate(tensor.external_data):
        if field.key == field_key:
            del tensor.external_data[i]


def write_external_data_tensors(model: ModelProto, filepath: str) -> ModelProto:
    """Serializes data for all the tensors which have data location set to TensorProto.External.

    Note: This function also strips basepath information from all tensors' external_data fields.

    Arguments:
        model (ModelProto): Model object which is the source of tensors to serialize.
        filepath: System path to the directory which should be treated as base path for external data.

    Returns:
        ModelProto: The modified model object.
    """
    for tensor in _get_all_tensors(model):
        # Writing to external data happens in 2 passes:
        # 1. Tensors with raw data which pass the necessary conditions (size threshold etc) are marked for serialization
        # 2. The raw data in these tensors is serialized to a file
        # Thus serialize only if tensor has raw data and it was marked for serialization
        if uses_external_data(tensor) and tensor.HasField("raw_data"):
            save_external_data(tensor, filepath)
            tensor.ClearField("raw_data")

    return model
