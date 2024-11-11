#!/usr/bin/env python

# Copyright (c) ONNX Project Contributors

# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

# NOTE: This is deprecated in favor of protogen's own .pyi generation method.
#       See: https://github.com/onnx/onnx/pull/6096

# Taken from https://github.com/dropbox/mypy-protobuf/blob/d984389124eae6dbbb517f766b9266bb32171510/python/protoc-gen-mypy
# (Apache 2.0 License)
# with own fixes to
# - appease flake8
# - exit without error when protobuf isn't installed
# - fix recognition of whether an identifier is defined locally
#   (unfortunately, we use a python package name ONNX_NAMESPACE_FOO_BAR_FOR_CI
#    on CI, which by the original protoc-gen-mypy script was recognized to be
#    camel case and therefore handled as an entry in the local package)


"""Protoc Plugin to generate mypy stubs. Loosely based on @zbarsky's go implementation"""
from __future__ import annotations

import sys
from collections import defaultdict
from contextlib import contextmanager
from typing import Any, Callable, Generator, cast

try:
    import google.protobuf.descriptor_pb2 as d_typed
    from google.protobuf.compiler import plugin_pb2 as plugin
except ImportError as e:
    raise RuntimeError("Failed to generate mypy stubs") from e


# Hax to get around fact that google protobuf libraries aren't in typeshed yet
d: Any = d_typed

# Split the string so phabricator doesn't think this file is generated
GENERATED = "@ge" + "nerated"
HEADER = (
    f"# {GENERATED} by protoc-gen-mypy.py.  Do not edit!\n"
    "# mypy: disable-error-code=override\n"
)


class Descriptors:
    def __init__(self, request: plugin.CodeGeneratorRequest) -> None:
        files = {f.name: f for f in request.proto_file}
        to_generate = {n: files[n] for n in request.file_to_generate}
        self.files: dict[str, d.FileDescriptorProto] = files
        self.to_generate: dict[str, d.FileDescriptorProto] = to_generate
        self.messages: dict[str, d.DescriptorProto] = {}
        self.message_to_fd: dict[str, d.FileDescriptorProto] = {}

        def _add_enums(
            enums: d.EnumDescriptorProto, prefix: str, fd: d.FileDescriptorProto
        ) -> None:
            for enum in enums:
                self.message_to_fd[prefix + enum.name] = fd

        def _add_messages(
            messages: d.DescriptorProto, prefix: str, fd: d.FileDescriptorProto
        ) -> None:
            for message in messages:
                self.messages[prefix + message.name] = message
                self.message_to_fd[prefix + message.name] = fd
                sub_prefix = prefix + message.name + "."
                _add_messages(message.nested_type, sub_prefix, fd)
                _add_enums(message.enum_type, sub_prefix, fd)

        for fd in request.proto_file:
            start_prefix = "." + fd.package + "."
            _add_messages(fd.message_type, start_prefix, fd)
            _add_enums(fd.enum_type, start_prefix, fd)


class PkgWriter:
    """Writes a single pyi file"""

    def __init__(self, fd: d.FileDescriptorProto, descriptors: Descriptors) -> None:
        self.fd = fd
        self.descriptors = descriptors
        self.lines: list[str] = []
        self.indent = ""

        # dictionary of x->y for `from {x} import {y}`
        self.imports: dict[str, set[str]] = defaultdict(set)
        self.locals: set[str] = set()

    def _import(self, path: str, name: str, import_as: str | None = None) -> str:
        """Imports a stdlib path and returns a handle to it
        eg. self._import("typing", "Optional") -> "Optional"
        """
        imp = path.replace("/", ".")
        if import_as is not None:
            self.imports[imp].add(f"{name} as {import_as}")
            return import_as
        else:
            self.imports[imp].add(name)
            return name

    def _import_message(self, type_name: d.FieldDescriptorProto) -> str:
        """Import a referenced message and return a handle"""
        name = cast(str, type_name)

        if name[0] == "." and name[1].isupper() and name[2].islower():
            # Message defined in this file
            return name[1:]

        message_fd = self.descriptors.message_to_fd[name]
        if message_fd.name == self.fd.name:
            # message defined in this package
            split = name.split(".")
            for i, segment in enumerate(split):
                if segment and segment[0].isupper() and segment[1].islower():
                    return ".".join(split[i:])

        # Not in package. Must import
        split = name.split(".")
        for i, segment in enumerate(split):
            if segment and segment[0].isupper() and segment[1].islower():
                assert message_fd.name.endswith(".proto")
                import_name = self._import(
                    message_fd.name[:-6].replace("-", "_") + "_pb2", segment
                )
                remains = ".".join(split[i + 1 :])
                if not remains:
                    return import_name
                raise AssertionError("Don't support nested imports yet")
                # return new_nested_import(import_name, remains)

        raise AssertionError("Could not parse local name " + name)

    @contextmanager  # type: ignore
    def _indent(self) -> Generator[None, None, None]:
        self.indent = self.indent + "    "
        yield
        self.indent = self.indent[:-4]

    def _write_line(self, line: str, *args: str) -> None:
        self.lines.append(self.indent + line.format(*args))

    def write_enums(self, enums: list[d.EnumDescriptorProto]) -> None:
        line = self._write_line
        for enum in enums:
            line("class {}(int):", enum.name)
            with self._indent():
                line("@classmethod")
                line("def Name(cls, number: int) -> str: ...")
                line("@classmethod")
                line("def Value(cls, name: str) -> int: ...")
                line("@classmethod")
                line("def keys(cls) -> {}[str]: ...", self._import("typing", "List"))
                line("@classmethod")
                line("def values(cls) -> {}[int]: ...", self._import("typing", "List"))
                line("@classmethod")
                line(
                    "def items(cls) -> {}[{}[str, int]]: ...",
                    self._import("typing", "List"),
                    self._import("typing", "Tuple"),
                )

            for val in enum.value:
                line(
                    "{} = {}({}, {})",
                    val.name,
                    self._import("typing", "cast"),
                    enum.name,
                    val.number,
                )
            line("")

    def write_messages(self, messages: list[d.DescriptorProto], prefix: str) -> None:
        line = self._write_line
        message_class = self._import("google.protobuf.message", "Message")

        for desc in messages:
            self.locals.add(desc.name)
            qualified_name = prefix + desc.name
            line("class {}({}):", desc.name, message_class)
            with self._indent():
                # Nested enums/messages
                self.write_enums(desc.enum_type)
                self.write_messages(desc.nested_type, qualified_name + ".")

                # Scalar fields
                for field in [f for f in desc.field if is_scalar(f)]:
                    if field.label == d.FieldDescriptorProto.LABEL_REPEATED:
                        container = self._import(
                            "google.protobuf.internal.containers",
                            "RepeatedScalarFieldContainer",
                        )
                        line(
                            "{} = ... # type: {}[{}]",
                            field.name,
                            container,
                            self.python_type(field),
                        )
                    else:
                        line("{} = ... # type: {}", field.name, self.python_type(field))
                line("")

                # Getters for non-scalar fields
                for field in [f for f in desc.field if not is_scalar(f)]:
                    line("@property")
                    if field.label == d.FieldDescriptorProto.LABEL_REPEATED:
                        msg = self.descriptors.messages[field.type_name]
                        if msg.options.map_entry:
                            # map generates a special Entry wrapper message
                            container = self._import("typing", "MutableMapping")
                            line(
                                "def {}(self) -> {}[{}, {}]: ...",
                                field.name,
                                container,
                                self.python_type(msg.field[0]),
                                self.python_type(msg.field[1]),
                            )
                        else:
                            container = self._import(
                                "google.protobuf.internal.containers",
                                "RepeatedCompositeFieldContainer",
                            )
                            line(
                                "def {}(self) -> {}[{}]: ...",
                                field.name,
                                container,
                                self.python_type(field),
                            )
                    else:
                        line(
                            "def {}(self) -> {}: ...",
                            field.name,
                            self.python_type(field),
                        )
                    line("")

                # Constructor
                line("def __init__(self,")
                with self._indent():
                    # Required args
                    for field in [
                        f
                        for f in desc.field
                        if f.label == d.FieldDescriptorProto.LABEL_REQUIRED
                    ]:
                        line("{} : {},", field.name, self.python_type(field))
                    for field in [
                        f
                        for f in desc.field
                        if f.label != d.FieldDescriptorProto.LABEL_REQUIRED
                    ]:
                        if field.label == d.FieldDescriptorProto.LABEL_REPEATED:
                            if (
                                field.type_name != ""
                                and self.descriptors.messages[
                                    field.type_name
                                ].options.map_entry
                            ):
                                msg = self.descriptors.messages[field.type_name]
                                line(
                                    "{} : {}[{}[{}, {}]] = None,",
                                    field.name,
                                    self._import("typing", "Optional", "OptionalType"),
                                    self._import("typing", "Mapping"),
                                    self.python_type(msg.field[0]),
                                    self.python_type(msg.field[1]),
                                )
                            else:
                                line(
                                    "{} : {}[{}[{}]] = None,",
                                    field.name,
                                    self._import("typing", "Optional", "OptionalType"),
                                    self._import("typing", "Iterable"),
                                    self.python_type(field),
                                )
                        else:
                            line(
                                "{} : {}[{}] = None,",
                                field.name,
                                self._import("typing", "Optional", "OptionalType"),
                                self.python_type(field),
                            )
                    line(") -> None: ...")

                # Standard message methods
                line("@classmethod")
                line("def FromString(cls, s: bytes) -> {}: ...", qualified_name)
                line("def MergeFrom(self, other_msg: {}) -> None: ...", message_class)
                line("def CopyFrom(self, other_msg: {}) -> None: ...", message_class)
            line("")

    def write_services(self, services: d.ServiceDescriptorProto) -> None:
        line = self._write_line

        for service in services:
            # The service definition interface
            line(
                "class {}({}, metaclass={}):",
                service.name,
                self._import("google.protobuf.service", "Service"),
                self._import("abc", "ABCMeta"),
            )
            with self._indent():
                for method in service.method:
                    line("@{}", self._import("abc", "abstractmethod"))
                    line("def {}(self,", method.name)
                    with self._indent():
                        line(
                            "rpc_controller: {},",
                            self._import("google.protobuf.service", "RpcController"),
                        )
                        line("request: {},", self._import_message(method.input_type))
                        line(
                            "done: {}[{}[[{}], None]],",
                            self._import("typing", "Optional"),
                            self._import("typing", "Callable"),
                            self._import_message(method.output_type),
                        )
                    line(
                        ") -> {}[{}]: ...",
                        self._import("concurrent.futures", "Future"),
                        self._import_message(method.output_type),
                    )

            # The stub client
            line("class {}({}):", service.name + "_Stub", service.name)
            with self._indent():
                line(
                    "def __init__(self, rpc_channel: {}) -> None: ...",
                    self._import("google.protobuf.service", "RpcChannel"),
                )

    def python_type(self, field: d.FieldDescriptorProto) -> str:
        mapping: dict[int, Callable[[], str]] = {
            d.FieldDescriptorProto.TYPE_DOUBLE: lambda: "float",
            d.FieldDescriptorProto.TYPE_FLOAT: lambda: "float",
            d.FieldDescriptorProto.TYPE_INT64: lambda: "int",
            d.FieldDescriptorProto.TYPE_UINT64: lambda: "int",
            d.FieldDescriptorProto.TYPE_FIXED64: lambda: "int",
            d.FieldDescriptorProto.TYPE_SFIXED64: lambda: "int",
            d.FieldDescriptorProto.TYPE_SINT64: lambda: "int",
            d.FieldDescriptorProto.TYPE_INT32: lambda: "int",
            d.FieldDescriptorProto.TYPE_UINT32: lambda: "int",
            d.FieldDescriptorProto.TYPE_FIXED32: lambda: "int",
            d.FieldDescriptorProto.TYPE_SFIXED32: lambda: "int",
            d.FieldDescriptorProto.TYPE_SINT32: lambda: "int",
            d.FieldDescriptorProto.TYPE_BOOL: lambda: "bool",
            d.FieldDescriptorProto.TYPE_STRING: lambda: "str",
            d.FieldDescriptorProto.TYPE_BYTES: lambda: "bytes",
            d.FieldDescriptorProto.TYPE_ENUM: lambda: self._import_message(
                field.type_name
            ),
            d.FieldDescriptorProto.TYPE_MESSAGE: lambda: self._import_message(
                field.type_name
            ),
            d.FieldDescriptorProto.TYPE_GROUP: lambda: self._import_message(
                field.type_name
            ),
        }

        assert field.type in mapping, "Unrecognized type: " + field.type
        return mapping[field.type]()

    def write(self) -> str:
        imports = []
        for pkg, items in self.imports.items():
            if pkg.startswith("google."):
                imports.append(f"from {pkg} import (  # type: ignore")
            else:
                imports.append(f"from {pkg} import (")
            for item in sorted(items):
                imports.append(f"    {item},")  # noqa: PERF401
            imports.append(")\n")

        return "\n".join(imports + self.lines)


def is_scalar(fd: d.FileDescriptorProto) -> bool:
    return not (
        fd.type == d.FieldDescriptorProto.TYPE_MESSAGE  # noqa: PLR1714
        or fd.type == d.FieldDescriptorProto.TYPE_GROUP
    )


def generate_mypy_stubs(
    descriptors: Descriptors, response: plugin.CodeGeneratorResponse
) -> None:
    for name, fd in descriptors.to_generate.items():
        pkg_writer = PkgWriter(fd, descriptors)
        pkg_writer.write_enums(fd.enum_type)
        pkg_writer.write_messages(fd.message_type, "")
        pkg_writer.write_services(fd.service)

        assert name == fd.name
        assert fd.name.endswith(".proto")
        output = response.file.add()
        output.name = fd.name[:-6].replace("-", "_") + "_pb2.pyi"
        output.content = HEADER + pkg_writer.write()
        print("Writing mypy to", output.name, file=sys.stderr)


def main() -> None:
    # Read request message from stdin
    data = sys.stdin.buffer.read()

    # Parse request
    request = plugin.CodeGeneratorRequest()
    request.ParseFromString(data)

    # Create response
    response = plugin.CodeGeneratorResponse()

    # Generate mypy
    generate_mypy_stubs(Descriptors(request), response)

    # Serialise response message
    output = response.SerializeToString()

    # Write to stdout
    sys.stdout.buffer.write(output)


if __name__ == "__main__":
    main()
