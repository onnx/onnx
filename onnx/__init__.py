# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

__all__ = [
    # Constants
    "ONNX_ML",
    "IR_VERSION",
    "IR_VERSION_2017_10_10",
    "IR_VERSION_2017_10_30",
    "IR_VERSION_2017_11_3",
    "IR_VERSION_2019_1_22",
    "IR_VERSION_2019_3_18",
    "IR_VERSION_2019_9_19",
    "IR_VERSION_2020_5_8",
    "IR_VERSION_2021_7_30",
    "IR_VERSION_2023_5_5",
    "IR_VERSION_2024_3_25",
    "EXPERIMENTAL",
    "STABLE",
    # Modules
    "checker",
    "compose",
    "defs",
    "gen_proto",
    "helper",
    "numpy_helper",
    "parser",
    "printer",
    "shape_inference",
    "utils",
    "version_converter",
    # Proto classes
    "AttributeProto",
    "DeviceConfigurationProto",
    "FunctionProto",
    "GraphProto",
    "IntIntListEntryProto",
    "MapProto",
    "ModelProto",
    "NodeDeviceConfigurationProto",
    "NodeProto",
    "OperatorProto",
    "OperatorSetIdProto",
    "OperatorSetProto",
    "OperatorStatus",
    "OptionalProto",
    "SequenceProto",
    "SimpleShardedDimProto",
    "ShardedDimProto",
    "ShardingSpecProto",
    "SparseTensorProto",
    "StringStringEntryProto",
    "TensorAnnotation",
    "TensorProto",
    "TensorShapeProto",
    "TrainingInfoProto",
    "TypeProto",
    "ValueInfoProto",
    "Version",
    # Utility functions
    "convert_model_to_external_data",
    "load",
    "load_external_data_for_model",
    "load_model_from_string",
    "load_model",
    "load_tensor_from_string",
    "load_tensor",
    "save",
    "save_model",
    "save_tensor",
    "write_external_data_tensors",
]
# isort:skip_file

import os
import typing
from typing import IO, Literal


from onnx import serialization
from onnx.onnx_cpp2py_export import ONNX_ML
from onnx.external_data_helper import (
    _ExternalDataContext,
    _get_all_tensors,
    _open_model_file_for_write,
    load_external_data_for_model,
    uses_external_data,
    write_external_data_tensors,
    convert_model_to_external_data,
)
from onnx.onnx_pb import (
    AttributeProto,
    DeviceConfigurationProto,
    EXPERIMENTAL,
    FunctionProto,
    GraphProto,
    IntIntListEntryProto,
    IR_VERSION,
    IR_VERSION_2017_10_10,
    IR_VERSION_2017_10_30,
    IR_VERSION_2017_11_3,
    IR_VERSION_2019_1_22,
    IR_VERSION_2019_3_18,
    IR_VERSION_2019_9_19,
    IR_VERSION_2020_5_8,
    IR_VERSION_2021_7_30,
    IR_VERSION_2023_5_5,
    IR_VERSION_2024_3_25,
    ModelProto,
    NodeDeviceConfigurationProto,
    NodeProto,
    OperatorSetIdProto,
    OperatorStatus,
    STABLE,
    SimpleShardedDimProto,
    ShardedDimProto,
    ShardingSpecProto,
    SparseTensorProto,
    StringStringEntryProto,
    TensorAnnotation,
    TensorProto,
    TensorShapeProto,
    TrainingInfoProto,
    TypeProto,
    ValueInfoProto,
    Version,
)
from onnx.onnx_operators_pb import OperatorProto, OperatorSetProto
from onnx.onnx_data_pb import MapProto, OptionalProto, SequenceProto
import importlib.metadata

# Import common subpackages so they're available when you 'import onnx'
from onnx import (
    checker,
    compose,
    defs,
    gen_proto,
    helper,
    numpy_helper,
    parser,
    printer,
    shape_inference,
    utils,
    version_converter,
)

if typing.TYPE_CHECKING:
    from collections.abc import Sequence

try:
    __version__ = importlib.metadata.version("onnx")
except importlib.metadata.PackageNotFoundError:
    try:
        __version__ = importlib.metadata.version("onnx-weekly")
    except importlib.metadata.PackageNotFoundError:
        __version__ = "unknown"

# Supported model formats that can be loaded from and saved to
# The literals are formats with built-in support. But we also allow users to
# register their own formats. So we allow str as well.
_SupportedFormat = Literal["protobuf", "textproto", "onnxtxt", "json"] | str  # noqa: PYI051
# Default serialization format
_DEFAULT_FORMAT = "protobuf"


def _load_bytes(f: IO[bytes] | str | os.PathLike) -> bytes:
    if hasattr(f, "read") and callable(typing.cast("IO[bytes]", f).read):
        content = typing.cast("IO[bytes]", f).read()
    else:
        f = typing.cast("str | os.PathLike", f)
        with open(f, "rb") as readable:
            content = readable.read()
    return content


def _save_bytes(content: bytes, f: IO[bytes] | str | os.PathLike) -> None:
    if hasattr(f, "write") and callable(typing.cast("IO[bytes]", f).write):
        typing.cast("IO[bytes]", f).write(content)
    else:
        f = typing.cast("str | os.PathLike", f)
        with open(f, "wb") as writable:
            writable.write(content)


def _get_file_path(f: IO[bytes] | str | os.PathLike | None) -> str | None:
    if isinstance(f, (str, os.PathLike)):
        return os.path.abspath(f)
    if hasattr(f, "name"):
        assert f is not None
        return os.path.abspath(f.name)
    return None


def _context_for_named_file(
    f: IO[bytes] | str | os.PathLike,
    model_filepath: str,
    *,
    read_only: bool,
) -> _ExternalDataContext:
    """Pin a named file object's directory and verify both handles name one file."""
    try:
        file_object = typing.cast("IO[bytes]", f)
        file_object_stat = os.fstat(file_object.fileno())
        base_dir = os.path.dirname(model_filepath)
        context = _ExternalDataContext(base_dir)
        context_fd = context.open_model_file(
            os.path.basename(model_filepath), read_only
        )
        try:
            context_stat = os.fstat(context_fd)
        finally:
            os.close(context_fd)
    except Exception as error:
        raise ValueError(
            "Cannot bind the named model file object to its external-data directory."
        ) from error
    if not os.path.samestat(file_object_stat, context_stat):
        raise ValueError(
            "The named model file object does not belong to its current directory."
        )
    return context


def _get_serializer(
    fmt: _SupportedFormat | None, f: str | os.PathLike | IO[bytes] | None = None
) -> serialization.ProtoSerializer:
    """Get the serializer for the given path and format from the serialization registry."""
    # Use fmt if it is specified
    if fmt is not None:
        return serialization.registry.get(fmt)

    if (file_path := _get_file_path(f)) is not None:
        _, ext = os.path.splitext(file_path)
        fmt = serialization.registry.get_format_from_file_extension(ext)

    # Failed to resolve format if fmt is None. Use protobuf as default
    fmt = fmt or _DEFAULT_FORMAT
    assert fmt is not None

    return serialization.registry.get(fmt)


def load_model(
    f: IO[bytes] | str | os.PathLike,
    format: _SupportedFormat | None = None,  # noqa: A002
    load_external_data: bool = True,
    *,
    _external_data_context: _ExternalDataContext | None = None,
) -> ModelProto:
    """Loads a serialized ModelProto into memory.

    Args:
        f: can be a file-like object (has "read" function) or a string/PathLike containing a file name
        format: The serialization format. When it is not specified, it is inferred
            from the file extension when ``f`` is a path. If not specified _and_
            ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
            be "utf-8" when the format is a text format.
        load_external_data: Whether to load the external data.
            Set to True if the data is under the same directory of the model.
            If not, users need to call :func:`load_external_data_for_model`
            with directory to load external data from.

    Returns:
        Loaded in-memory ModelProto.
    """
    model_filepath = _get_file_path(f)
    external_data_context = _external_data_context
    context_error: ValueError | None = None
    if isinstance(f, (str, os.PathLike)):
        assert model_filepath is not None
        if not os.path.exists(model_filepath):
            raise FileNotFoundError(model_filepath)
        base_dir = os.path.dirname(model_filepath)
        if external_data_context is None:
            external_data_context = _ExternalDataContext(base_dir)
        fd = external_data_context.open_model_file(
            os.path.basename(model_filepath), True
        )
        with os.fdopen(fd, "rb") as readable:
            serialized = readable.read()
    else:
        if load_external_data and model_filepath and external_data_context is None:
            try:
                external_data_context = _context_for_named_file(
                    f, model_filepath, read_only=True
                )
            except ValueError as error:
                context_error = error
        serialized = _load_bytes(f)
    model = _get_serializer(format, f).deserialize_proto(serialized, ModelProto())

    if load_external_data:
        if model_filepath:
            if context_error is not None and any(
                uses_external_data(tensor) for tensor in _get_all_tensors(model)
            ):
                raise context_error
            base_dir = os.path.dirname(model_filepath)
            load_external_data_for_model(
                model, base_dir, _context=external_data_context
            )

    return model


def load_tensor(
    f: IO[bytes] | str | os.PathLike,
    format: _SupportedFormat | None = None,  # noqa: A002
) -> TensorProto:
    """Loads a serialized TensorProto into memory.

    Args:
        f: can be a file-like object (has "read" function) or a string/PathLike containing a file name
        format: The serialization format. When it is not specified, it is inferred
            from the file extension when ``f`` is a path. If not specified _and_
            ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
            be "utf-8" when the format is a text format.

    Returns:
        Loaded in-memory TensorProto.
    """
    return _get_serializer(format, f).deserialize_proto(_load_bytes(f), TensorProto())


def load_model_from_string(
    s: bytes | str,
    format: _SupportedFormat = _DEFAULT_FORMAT,  # noqa: A002
) -> ModelProto:
    """Loads a binary string (bytes) that contains serialized ModelProto.

    Args:
        s: a string, which contains serialized ModelProto
        format: The serialization format. When it is not specified, it is inferred
            from the file extension when ``f`` is a path. If not specified _and_
            ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
            be "utf-8" when the format is a text format.

    Returns:
        Loaded in-memory ModelProto.
    """
    return _get_serializer(format).deserialize_proto(s, ModelProto())


def load_tensor_from_string(
    s: bytes,
    format: _SupportedFormat = _DEFAULT_FORMAT,  # noqa: A002
) -> TensorProto:
    """Loads a binary string (bytes) that contains serialized TensorProto.

    Args:
        s: a string, which contains serialized TensorProto
        format: The serialization format. When it is not specified, it is inferred
            from the file extension when ``f`` is a path. If not specified _and_
            ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
            be "utf-8" when the format is a text format.

    Returns:
        Loaded in-memory TensorProto.
    """
    return _get_serializer(format).deserialize_proto(s, TensorProto())


def save_model(
    proto: ModelProto | bytes,
    f: IO[bytes] | str | os.PathLike,
    format: _SupportedFormat | None = None,  # noqa: A002
    *,
    save_as_external_data: bool = False,
    all_tensors_to_one_file: bool = True,
    location: str | None = None,
    size_threshold: int = 1024,
    convert_attribute: bool = False,
) -> None:
    """Saves the ModelProto to the specified path and optionally, serialize tensors with raw data as external data before saving.

    Args:
        proto: should be a in-memory ModelProto
        f: can be a file-like object (has "write" function) or a string containing
        a file name or a pathlike object
        format: The serialization format. When it is not specified, it is inferred
            from the file extension when ``f`` is a path. If not specified _and_
            ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
            be "utf-8" when the format is a text format.
        save_as_external_data: If true, save tensors to external file(s).
        all_tensors_to_one_file: Effective only if save_as_external_data is True.
            If true, save all tensors to one external file specified by location.
            If false, save each tensor to a file named with the tensor name.
        location: Effective only if save_as_external_data is true.
            Specify the external file that all tensors to save to.
            Path is relative to the model path.
            If not specified, will use the model name.
        size_threshold: Effective only if save_as_external_data is True.
            Threshold for size of data. Only when tensor's data is >= the size_threshold it will be converted
            to external data. To convert every tensor with raw data to external data set size_threshold=0.
        convert_attribute: Effective only if save_as_external_data is True.
            If true, convert all tensors to external data
            If false, convert only non-attribute tensors to external data
    """
    model_filepath = _get_file_path(f)
    external_data_context: _ExternalDataContext | None = None
    context_error: ValueError | None = None
    model_file: IO[bytes] | None = None
    if isinstance(f, (str, os.PathLike)):
        assert model_filepath is not None
        basepath = os.path.dirname(model_filepath)
        if not os.path.isdir(basepath):
            raise FileNotFoundError(basepath)
        external_data_context = _ExternalDataContext(basepath)
        try:
            model_file = _open_model_file_for_write(
                external_data_context,
                os.path.basename(model_filepath),
                truncate=False,
                existing_only=True,
            )
        except checker.ValidationError as error:
            if error.code != "external-data-file-missing":
                raise
    elif model_filepath is not None:
        try:
            external_data_context = _context_for_named_file(
                f, model_filepath, read_only=True
            )
        except ValueError as error:
            context_error = error

    try:
        if isinstance(proto, bytes):
            proto = _get_serializer(_DEFAULT_FORMAT).deserialize_proto(
                proto, ModelProto()
            )

        if save_as_external_data:
            convert_model_to_external_data(
                proto,
                all_tensors_to_one_file,
                location,
                size_threshold,
                convert_attribute,
            )

        if model_filepath is not None:
            if context_error is not None and any(
                uses_external_data(tensor) and tensor.HasField("raw_data")
                for tensor in _get_all_tensors(proto)
            ):
                raise context_error
            basepath = os.path.dirname(model_filepath)
            proto = write_external_data_tensors(
                proto, basepath, _context=external_data_context
            )

        serialized = _get_serializer(format, model_filepath).serialize_proto(proto)
        if (
            model_file is None
            and external_data_context is not None
            and isinstance(f, (str, os.PathLike))
        ):
            assert model_filepath is not None
            model_file = _open_model_file_for_write(
                external_data_context,
                os.path.basename(model_filepath),
                truncate=False,
                create_new=True,
            )
        if model_file is not None:
            model_file.seek(0)
            model_file.truncate()
            model_file.write(serialized)
        else:
            _save_bytes(serialized, f)
    finally:
        if model_file is not None:
            model_file.close()


def save_tensor(
    proto: TensorProto,
    f: IO[bytes] | str | os.PathLike,
    format: _SupportedFormat | None = None,  # noqa: A002
) -> None:
    """Saves the TensorProto to the specified path.

    Args:
        proto: should be a in-memory TensorProto
        f: can be a file-like object (has "write" function) or a string
        containing a file name or a pathlike object.
        format: The serialization format. When it is not specified, it is inferred
            from the file extension when ``f`` is a path. If not specified _and_
            ``f`` is not a path, 'protobuf' is used. The encoding is assumed to
            be "utf-8" when the format is a text format.
    """
    serialized = _get_serializer(format, f).serialize_proto(proto)
    _save_bytes(serialized, f)


# For backward compatibility
load = load_model
load_from_string = load_model_from_string
save = save_model


def _model_proto_repr(self: ModelProto) -> str:
    if self.domain:
        domain = f", domain='{self.domain}'"
    else:
        domain = ""
    if self.producer_name:
        producer_name = f", producer_name='{self.producer_name}'"
    else:
        producer_name = ""
    if self.producer_version:
        producer_version = f", producer_version='{self.producer_version}'"
    else:
        producer_version = ""
    if self.graph:
        graph = f", graph={self.graph!r}"
    else:
        graph = ""
    if self.functions:
        functions = f", functions=<{len(self.functions)} functions>"
    else:
        functions = ""
    if self.opset_import:
        opset_import = f", opset_import={_operator_set_protos_repr(self.opset_import)}"
    else:
        opset_import = ""
    return f"ModelProto(ir_version={self.ir_version}{opset_import}{domain}{producer_name}{producer_version}{graph}{functions})"


def _graph_proto_repr(self: GraphProto) -> str:
    if self.initializer:
        initializer = f", initializer=<{len(self.initializer)} initializers>"
    else:
        initializer = ""
    if self.node:
        node = f", node=<{len(self.node)} nodes>"
    else:
        node = ""
    if self.value_info:
        value_info = f", value_info=<{len(self.value_info)} value_info>"
    else:
        value_info = ""
    if self.input:
        input = f", input=<{len(self.input)} inputs>"
    else:
        input = ""
    if self.output:
        output = f", output=<{len(self.output)} outputs>"
    else:
        output = ""
    return f"GraphProto('{self.name}'{input}{output}{initializer}{node}{value_info})"


def _function_proto_repr(self: FunctionProto) -> str:
    if self.domain:
        domain = f", domain='{self.domain}'"
    else:
        domain = ""
    if self.overload:
        overload = f", overload='{self.overload}'"
    else:
        overload = ""
    if self.node:
        node = f", node=<{len(self.node)} nodes>"
    else:
        node = ""
    if self.attribute:
        attribute = f", attribute={self.attribute}"
    else:
        attribute = ""
    if self.opset_import:
        opset_import = f", opset_import={_operator_set_protos_repr(self.opset_import)}"
    else:
        opset_import = ""
    if self.input:
        input = f", input=<{len(self.input)} inputs>"
    else:
        input = ""
    if self.output:
        output = f", output=<{len(self.output)} outputs>"
    else:
        output = ""
    return f"FunctionProto('{self.name}'{domain}{overload}{opset_import}{input}{output}{attribute}{node})"


def _operator_set_protos_repr(protos: Sequence[OperatorSetIdProto]) -> str:
    opset_imports = {proto.domain: proto.version for proto in protos}
    return repr(opset_imports)


# Override __repr__ for some proto classes to make it more efficient
ModelProto.__repr__ = _model_proto_repr  # type: ignore[method-assign,assignment]
GraphProto.__repr__ = _graph_proto_repr  # type: ignore[method-assign,assignment]
FunctionProto.__repr__ = _function_proto_repr  # type: ignore[method-assign,assignment]
