# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models of the operator spec files and their C++ serialization.

All validation happens while reading a spec file, so that every error names
the file; the C++ serialization does not fail.
"""

from __future__ import annotations

import math
import re
import textwrap
import tomllib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    StringConstraints,
    ValidationError,
    model_validator,
)

if TYPE_CHECKING:
    from pathlib import Path


# A spec file is a fenced TOML block followed by the Markdown doc string. The
# fence may be longer than three backticks if the TOML contains a fence itself.
_FILE_PATTERN = re.compile(r"\A(`{3,})toml\n(.*?)^\1\n(.*)\Z", re.DOTALL | re.MULTILINE)


# Maximum length in bytes of a single string literal piece; MSVC rejects longer ones.
_MAX_LITERAL = 8000

# Delimiter of the raw string literals; spec files must not contain it.
_RAW_DELIMITER = ')STR"'


class SpecError(Exception):
    """A spec file is invalid."""


# Fields of `Spec` that are taken from each file instead of being inherited.
_NOT_INHERITED = frozenset(
    {"name", "domain", "shape_inference", "data_propagation", "function_body_builders"}
)


def cpp_str(s: str) -> str:
    """Return a C++ expression of type `const char*` for `s`."""
    if re.fullmatch(r"[^\"\\\n]{0,200}", s):
        return f'"{s}"'
    return " ".join(f'R"STR({p})STR"' for p in _split(s, _MAX_LITERAL))


def _split(s: str, max_bytes: int) -> list[str]:
    """Split `s` into pieces of at most `max_bytes` bytes in UTF-8."""
    pieces = [""]
    size = 0
    for char in s:
        n = len(char.encode())
        if size + n > max_bytes:
            pieces.append("")
            size = 0
        pieces[-1] += char
        size += n
    return pieces


def cpp_bool(value: bool) -> str:
    return "true" if value else "false"


Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_]\w*$")]
Domain = Literal["ai.onnx", "ai.onnx.ml", "ai.onnx.preview", "ai.onnx.preview.training"]
# A type string such as `tensor(float)`.
TypeString = Annotated[str, StringConstraints(pattern=r"^\w+\([\w(), ]*\)$")]
# A C++ function name relative to `ONNX_NAMESPACE`: either unqualified or in
# `defs::<area>::utils`, such as `defs::math::utils::EinsumShapeInference_opset12`.
FunctionName = Annotated[str, StringConstraints(pattern=r"^(defs::\w+::utils::)?\w+$")]

# Header declaring a function `defs::<area>::utils::<name>`.
_UTILS_FUNCTION = re.compile(r"defs::(\w+)::utils::\w+")
# Headers declaring the unqualified functions in `ONNX_NAMESPACE`.
_GLOBAL_HEADERS = ("onnx/defs/data_propagators.h", "onnx/defs/shape_inference.h")

AttributeType = Literal[
    "FLOAT",
    "INT",
    "STRING",
    "TENSOR",
    "GRAPH",
    "SPARSE_TENSOR",
    "TYPE_PROTO",
    "FLOATS",
    "INTS",
    "STRINGS",
    "TENSORS",
    "GRAPHS",
    "SPARSE_TENSORS",
    "TYPE_PROTOS",
]

_CPP_ENUMS = {
    "single": "OpSchema::Single",
    "optional": "OpSchema::Optional",
    "variadic": "OpSchema::Variadic",
    "unknown": "OpSchema::Unknown",
    "differentiable": "OpSchema::Differentiable",
    "non_differentiable": "OpSchema::NonDifferentiable",
}
_CPP_DETERMINISM = {
    "deterministic": "Deterministic",
    "non_deterministic": "NonDeterministic",
}
_CPP_ELEMENT_TYPES = {"INT": "int64_t", "FLOAT": "float", "STRING": "std::string"}

# Python types a default value may have, by attribute type.
_DEFAULT_TYPES: dict[str, tuple[type, ...]] = {
    "INT": (int,),
    "FLOAT": (float, int),
    "STRING": (str,),
    "INTS": (int,),
    "FLOATS": (float, int),
    "STRINGS": (str,),
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FormalParameter(_Model):
    name: str
    # A type parameter such as `T` or a type string such as `tensor(int64)`.
    type: Identifier | TypeString
    desc: str = ""
    option: Literal["single", "optional", "variadic"] = "single"
    homogeneous: bool = True
    min_arity: NonNegativeInt = 1
    differentiation: Literal["unknown", "differentiable", "non_differentiable"] = (
        "unknown"
    )

    def cpp_call(self, method: Literal["Input", "Output"], index: int) -> str:
        args = [
            str(index),
            cpp_str(self.name),
            cpp_str(self.desc),
            cpp_str(self.type),
            _CPP_ENUMS[self.option],
            cpp_bool(self.homogeneous),
            str(self.min_arity),
            _CPP_ENUMS[self.differentiation],
        ]
        return f"{method}({', '.join(args)})"


class Attribute(_Model):
    type: AttributeType
    desc: str = ""
    required: bool = False
    default: int | float | str | list[int] | list[float] | list[str] | None = None

    @model_validator(mode="after")
    def _check_default(self) -> Attribute:
        if self.default is None:
            return self
        if self.required:
            raise ValueError("a required attribute must not have a default")
        allowed = _DEFAULT_TYPES.get(self.type)
        if allowed is None:
            raise ValueError(f"attributes of type `{self.type}` cannot have a default")
        values = self.default if isinstance(self.default, list) else [self.default]
        is_list = self.type.endswith("S")
        if isinstance(self.default, list) != is_list or not all(
            isinstance(v, allowed) and not isinstance(v, bool) for v in values
        ):
            raise ValueError(
                f"default `{self.default!r}` does not match type `{self.type}`"
            )
        if not all(math.isfinite(v) for v in values if isinstance(v, float)):
            raise ValueError(f"default `{self.default!r}` is not finite")
        return self

    def cpp_call(self, name: str) -> str:
        return f"Attr({cpp_str(name)}, {cpp_str(self.desc)}, AttributeProto::{self.type}, {self._cpp_default()})"

    def _cpp_default(self) -> str:
        if self.default is None:
            return cpp_bool(self.required)
        if isinstance(self.default, list):
            element = self.type.removesuffix("S")
            values = ", ".join(_cpp_scalar(element, v) for v in self.default)
            return f"std::vector<{_CPP_ELEMENT_TYPES[element]}>{{{values}}}"
        return _cpp_scalar(self.type, self.default)


def _cpp_scalar(type_: str, value: float | str) -> str:
    match type_:
        case "INT":
            return f"static_cast<int64_t>({value})"
        case "FLOAT":
            return f"static_cast<float>({float(value)!r})"
        case _:
            return f"std::string({cpp_str(str(value))})"


class TypeConstraint(_Model):
    """Allowed types of a type parameter.

    `types` is either a list of type strings or a type alias, i.e., the name
    of an `OpSchema` type helper such as `all_float_types_ir4`.
    """

    types: Identifier | list[TypeString]
    desc: str = ""

    def cpp_call(self, name: str) -> str:
        if isinstance(self.types, str):
            types = f"OpSchema::{self.types}()"
        else:
            types = "{" + ", ".join(f'"{t}"' for t in self.types) + "}"
        return f"TypeConstraint({cpp_str(name)}, {types}, {cpp_str(self.desc)})"


class Spec(_Model):
    """The TOML part of a spec file.

    `name` and `domain` identify the operator and are required in every file.
    `shape_inference` names the type and shape inference function or is
    `False` if there is none. `data_propagation` optionally names the data
    propagation function, and `function_body_builders` the context-dependent
    function body builders by opset version. These five fields are not
    inherited. The defaults of the other fields apply to the first version of
    an operator. Later versions inherit every field not set in their file from
    the previous version.
    """

    name: Identifier
    domain: Domain
    support_level: Literal["common", "experimental"] = "common"
    deprecated: bool = False
    node_determinism: Literal["unknown", "deterministic", "non_deterministic"] = (
        "unknown"
    )
    num_outputs: list[NonNegativeInt] | None = None
    allow_unchecked_attributes: bool = False
    shape_inference: FunctionName | Literal[False] | None = None
    data_propagation: FunctionName | None = None
    function_body_builders: dict[int, FunctionName] = {}
    attributes: dict[str, Attribute] = {}
    inputs: list[FormalParameter] = []
    outputs: list[FormalParameter] = []
    type_constraints: dict[str, TypeConstraint] = {}
    function_bodies: dict[int, str] = {}

    def inherit(self, child: Spec) -> Spec:
        """Return `child` with all inherited fields not set in its file taken from `self`."""
        keys = child.model_fields_set | _NOT_INHERITED
        return self.model_copy(update={k: getattr(child, k) for k in keys})

    def functions(self) -> list[str]:
        """Return the names of the C++ functions used by this spec."""
        names = [self.shape_inference, self.data_propagation]
        builders = list(self.function_body_builders.values())
        return [n for n in names if isinstance(n, str)] + builders

    def cpp_calls(self) -> list[str]:
        """Return the `OpSchema` builder calls that set this spec."""
        calls = []
        if self.support_level == "experimental":
            calls.append("SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)")
        if self.deprecated:
            calls.append("Deprecate()")
        if self.node_determinism != "unknown":
            calls.append(
                f"SetNodeDeterminism(OpSchema::NodeDeterminism::{_CPP_DETERMINISM[self.node_determinism]})"
            )
        if self.num_outputs is not None:
            calls.append(f"NumOutputs({{{', '.join(map(str, self.num_outputs))}}})")
        if self.allow_unchecked_attributes:
            calls.append("AllowUncheckedAttributes()")
        calls += [a.cpp_call(name) for name, a in self.attributes.items()]
        calls += [p.cpp_call("Input", i) for i, p in enumerate(self.inputs)]
        calls += [p.cpp_call("Output", i) for i, p in enumerate(self.outputs)]
        calls += [tc.cpp_call(name) for name, tc in self.type_constraints.items()]
        calls += [
            f"FunctionBody({cpp_str(body)}, {opset})"
            for opset, body in self.function_bodies.items()
        ]
        calls += [
            f"SetContextDependentFunctionBodyBuilder({name}, {opset})"
            for opset, name in self.function_body_builders.items()
        ]
        if isinstance(self.shape_inference, str):
            calls.append(f"TypeAndShapeInferenceFunction({self.shape_inference})")
        if self.data_propagation is not None:
            calls.append(f"PartialDataPropagationFunction({self.data_propagation})")
        return calls


@dataclass(frozen=True)
class Version:
    """A fully resolved version of an operator."""

    since_version: int
    source: str
    spec: Spec
    doc: str

    @property
    def function_name(self) -> str:
        return f"{self.spec.name}_ver{self.since_version}"

    def cpp_calls(self) -> list[str]:
        """Return the `OpSchema` builder calls that set this version."""
        if not self.doc:
            return self.spec.cpp_calls()
        doc = cpp_str("\n" + self.doc + "\n")
        return [f"SetDoc({doc})", *self.spec.cpp_calls()]


@dataclass(frozen=True)
class Operator:
    """All resolved versions of an operator, in ascending order."""

    domain: str
    name: str
    versions: list[Version]

    @property
    def namespace(self) -> str:
        """Return the C++ namespace of the generated functions."""
        return self.domain.replace(".", "_")

    @property
    def is_ml(self) -> bool:
        return self.domain == "ai.onnx.ml"

    @property
    def includes(self) -> list[str]:
        """Return the headers declaring the C++ functions used by any version."""
        headers: set[str] = set()
        for name in (n for v in self.versions for n in v.spec.functions()):
            if (match := _UTILS_FUNCTION.fullmatch(name)) is not None:
                headers.add(f"onnx/defs/{match.group(1)}/utils.h")
            else:
                headers.update(_GLOBAL_HEADERS)
        return sorted(headers)


def _dedent(doc: str) -> str:
    """Remove the common indentation of all lines but the first."""
    first, _, rest = doc.strip().partition("\n")
    return f"{first}\n{textwrap.dedent(rest)}".strip()


def parse_file(path: Path) -> tuple[Spec, str]:
    """Return the spec and the doc string of a spec file."""
    text = path.read_text()
    if _RAW_DELIMITER in text:
        raise SpecError(f"{path}: contains the raw string delimiter '{_RAW_DELIMITER}'")
    match = _FILE_PATTERN.match(text)
    if match is None:
        raise SpecError(f"{path}: expected a leading '```toml' block")
    try:
        spec = Spec.model_validate(tomllib.loads(match.group(2)))
    except (tomllib.TOMLDecodeError, ValidationError) as e:
        raise SpecError(f"{path}: {e}") from e
    return spec, _dedent(match.group(3))


def _since_version(path: Path) -> int:
    match = re.fullmatch(r"v(\d+)", path.parent.name)
    if match is None:
        raise SpecError(f"{path}: expected a parent directory named 'v<N>'")
    return int(match.group(1))


def resolve(paths: list[Path], root: Path) -> Operator:
    """Resolve the inheritance chain of the spec files of one operator.

    Each file must be located at `<domain>/v<N>/<lowercase name>.md`.
    """
    versions: list[Version] = []
    spec: Spec | None = None
    doc = ""
    for path in sorted(paths, key=_since_version):
        child, child_doc = parse_file(path)
        if (path.parent.parent.name, path.stem) != (child.domain, child.name.lower()):
            raise SpecError(
                f"{path}: path does not match domain `{child.domain}` and name `{child.name}`"
            )
        if spec is not None and child.name != spec.name:
            raise SpecError(f"{path}: name `{child.name}` differs from `{spec.name}`")
        spec = child if spec is None else spec.inherit(child)
        doc = child_doc or doc
        versions.append(
            Version(
                since_version=_since_version(path),
                source=path.relative_to(root).as_posix(),
                spec=spec,
                doc=doc,
            )
        )
    if spec is None:
        raise SpecError("no spec files given")
    return Operator(domain=spec.domain, name=spec.name, versions=versions)
