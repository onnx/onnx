# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
"""Graph utilities for checking whether an ONNX proto message is legal."""

from __future__ import annotations

__all__ = [
    "check_attribute",
    "check_function",
    "check_graph",
    "check_model",
    "check_node",
    "check_sparse_tensor",
    "check_tensor",
    "check_value_info",
    "summarize_operators",
    "DEFAULT_CONTEXT",
    "LEXICAL_SCOPE_CONTEXT",
    "OperatorKind",
    "OperatorSummary",
    "OperatorUsage",
    "ValidationError",
    "C",
    "MAXIMUM_PROTOBUF",
]

import collections
import dataclasses
import enum
import os
from typing import TYPE_CHECKING

import onnx.defs
import onnx.onnx_cpp2py_export.checker as C  # noqa: N812
from onnx.onnx_pb import IR_VERSION

if TYPE_CHECKING:
    from collections.abc import Iterable

    from google.protobuf.message import Message

# Maximum single-protobuf size; matches the C++ parser limit in proto_utils.h (2 GiB - 1 byte)
MAXIMUM_PROTOBUF = 2147483647


# NB: Please don't edit this context!
DEFAULT_CONTEXT = C.CheckerContext()
DEFAULT_CONTEXT.ir_version = IR_VERSION
# TODO: Maybe ONNX-ML should also be defaulted?
DEFAULT_CONTEXT.opset_imports = {"": onnx.defs.onnx_opset_version()}

LEXICAL_SCOPE_CONTEXT = C.LexicalScopeContext()


def _ensure_proto_type(proto: Message, proto_type: type[Message]) -> None:
    if not isinstance(proto, proto_type):
        raise TypeError(
            f"The proto message needs to be of type '{proto_type.__name__}'"
        )


def check_value_info(
    value_info: onnx.ValueInfoProto, ctx: C.CheckerContext = DEFAULT_CONTEXT
) -> None:
    _ensure_proto_type(value_info, onnx.ValueInfoProto)
    return C.check_value_info(value_info.SerializeToString(), ctx)


def check_tensor(
    tensor: onnx.TensorProto, ctx: C.CheckerContext = DEFAULT_CONTEXT
) -> None:
    _ensure_proto_type(tensor, onnx.TensorProto)
    return C.check_tensor(tensor.SerializeToString(), ctx)


def check_attribute(
    attr: onnx.AttributeProto,
    ctx: C.CheckerContext = DEFAULT_CONTEXT,
    lexical_scope_ctx: C.LexicalScopeContext = LEXICAL_SCOPE_CONTEXT,
) -> None:
    _ensure_proto_type(attr, onnx.AttributeProto)
    return C.check_attribute(attr.SerializeToString(), ctx, lexical_scope_ctx)


def check_node(
    node: onnx.NodeProto,
    ctx: C.CheckerContext = DEFAULT_CONTEXT,
    lexical_scope_ctx: C.LexicalScopeContext = LEXICAL_SCOPE_CONTEXT,
) -> None:
    _ensure_proto_type(node, onnx.NodeProto)
    return C.check_node(node.SerializeToString(), ctx, lexical_scope_ctx)


def check_function(
    function: onnx.FunctionProto,
    ctx: C.CheckerContext | None = None,
    lexical_scope_ctx: C.LexicalScopeContext = LEXICAL_SCOPE_CONTEXT,
) -> None:
    _ensure_proto_type(function, onnx.FunctionProto)
    if ctx is None:
        ctx = C.CheckerContext()
        ctx.ir_version = onnx.helper.find_min_ir_version_for(
            function.opset_import, ignore_unknown=True
        )
        ctx.opset_imports = {
            domain_version.domain: domain_version.version
            for domain_version in function.opset_import
        }
    C.check_function(function.SerializeToString(), ctx, lexical_scope_ctx)


def check_graph(
    graph: onnx.GraphProto,
    ctx: C.CheckerContext = DEFAULT_CONTEXT,
    lexical_scope_ctx: C.LexicalScopeContext = LEXICAL_SCOPE_CONTEXT,
) -> None:
    _ensure_proto_type(graph, onnx.GraphProto)
    return C.check_graph(graph.SerializeToString(), ctx, lexical_scope_ctx)


def check_sparse_tensor(
    sparse: onnx.SparseTensorProto, ctx: C.CheckerContext = DEFAULT_CONTEXT
) -> None:
    _ensure_proto_type(sparse, onnx.SparseTensorProto)
    C.check_sparse_tensor(sparse.SerializeToString(), ctx)


def check_model(
    model: onnx.ModelProto | str | bytes | os.PathLike,
    full_check: bool = False,
    skip_opset_compatibility_check: bool = False,
    check_custom_domain: bool = False,
) -> None:
    """Check the consistency of a model.

    An exception will be raised if the model's ir_version is not set
    properly or is higher than checker's ir_version, or if the model
    has duplicate keys in metadata_props.

    If IR version >= 3, the model must specify opset_import.
    If IR version < 3, the model cannot have any opset_import specified.

    Args:
        model: Model to check. If model is a path, the function checks model
            path first. If the model bytes size is larger than 2GB, function
            should be called using model path.
        full_check: If True, the function also runs shape inference check.
        skip_opset_compatibility_check: If True, the function skips the check for
            opset compatibility.
        check_custom_domain: If True, the function will check all domains. Otherwise
            only check built-in domains.
    """
    # If model is a path instead of ModelProto
    if isinstance(model, (str, os.PathLike)):
        C.check_model_path(
            os.fspath(model),
            full_check,
            skip_opset_compatibility_check,
            check_custom_domain,
        )
    else:
        protobuf_string = (
            model if isinstance(model, bytes) else model.SerializeToString()
        )
        # If the protobuf is larger than 2GiB,
        # remind users should use the model path to check
        if len(protobuf_string) > MAXIMUM_PROTOBUF:
            raise ValueError(
                "This protobuf of onnx model is too large (>2GiB). Call check_model with model path instead."
            )
        C.check_model(
            protobuf_string,
            full_check,
            skip_opset_compatibility_check,
            check_custom_domain,
        )


ValidationError = C.ValidationError


class OperatorKind(enum.Enum):
    """How the operator a node refers to is defined. See :func:`summarize_operators`."""

    STANDARD = "standard"
    """An operator with a schema registered in the current process in the
    ``ai.onnx`` or ``ai.onnx.ml`` domain."""
    PREVIEW = "preview"
    """An operator of a preview domain, such as ``ai.onnx.preview.training``."""
    FUNCTION = "function"
    """A model-local function, which the model defines itself."""
    CUSTOM = "custom"
    """An operator of another domain whose schema is registered in this process,
    for example with :func:`onnx.defs.register_schema`."""
    UNKNOWN = "unknown"
    """Neither a registered schema nor a model-local function defines the operator,
    or the model does not import an opset for its domain. This includes schemas
    unavailable in the current build."""


@dataclasses.dataclass(frozen=True)
class OperatorUsage:
    """An operator used by a model, and how often."""

    domain: str
    """The operator's domain; ``""`` for ``ai.onnx``."""
    op_type: str
    overload: str
    """The overload of a model-local function; usually ``""``."""
    opset_version: int | None
    """The opset version imported for the domain where the operator is used, or
    None if none is imported."""
    kind: OperatorKind
    count: int
    """The number of nodes that use the operator."""


@dataclasses.dataclass(frozen=True)
class OperatorSummary:
    """The operators a model uses. See :func:`summarize_operators`."""

    operators: tuple[OperatorUsage, ...]

    @property
    def uses_only_standard_operators(self) -> bool:
        """True if every operator the model uses is a standard operator or a
        model-local function, which is itself built only from standard operators.
        """
        return all(
            op.kind in (OperatorKind.STANDARD, OperatorKind.FUNCTION)
            for op in self.operators
        )

    @property
    def all_operators_defined(self) -> bool:
        """True if a registered schema or a model-local function defines every
        operator the model uses.
        """
        return all(op.kind is not OperatorKind.UNKNOWN for op in self.operators)


_STANDARD_DOMAINS = frozenset({onnx.defs.ONNX_DOMAIN, onnx.defs.ONNX_ML_DOMAIN})
_PREVIEW_DOMAINS = frozenset(
    {onnx.defs.AI_ONNX_PREVIEW_DOMAIN, onnx.defs.AI_ONNX_PREVIEW_TRAINING_DOMAIN}
)


def _normalize_domain(domain: str) -> str:
    # "ai.onnx" and "" both name the default domain.
    return onnx.defs.ONNX_DOMAIN if domain == "ai.onnx" else domain


def _opset_versions(opset_import: Iterable[onnx.OperatorSetIdProto]) -> dict[str, int]:
    return {_normalize_domain(opset.domain): opset.version for opset in opset_import}


def _subgraphs(attr: onnx.AttributeProto) -> list[onnx.GraphProto]:
    return [attr.g, *attr.graphs] if attr.HasField("g") else list(attr.graphs)


def _schema_kind(op_type: str, domain: str, opset_version: int | None) -> OperatorKind:
    if opset_version is None or not onnx.defs.has(op_type, opset_version, domain):
        return OperatorKind.UNKNOWN
    if domain in _STANDARD_DOMAINS:
        return OperatorKind.STANDARD
    if domain in _PREVIEW_DOMAINS:
        return OperatorKind.PREVIEW
    return OperatorKind.CUSTOM


def summarize_operators(model: onnx.ModelProto) -> OperatorSummary:
    """List the operators a model uses and how each is defined.

    :func:`check_model` accepts operators of any domain the model imports, so a
    valid model may still depend on operators that only a particular runtime
    implements. This summary shows which operators those are. It does not check
    the model; use :func:`check_model` for that.

    Every node is classified as an :class:`OperatorKind`: a standard operator,
    a preview operator, a model-local function, an operator of another domain
    whose schema is registered in this process (see
    :func:`onnx.defs.register_schema`), or an unknown operator. Nodes in
    subgraphs are included. The bodies of model-local functions the model calls
    are included once each, so a function counts as standard only if its body
    is. Functions the model never calls are left out.

    Classification reflects the schema registry of the current process. An
    operator whose schema is unavailable in the current build is unknown; for
    example, this applies to ``ai.onnx.ml`` operators in a build configured with
    ``ONNX_ML=0``.

    Args:
        model: The model to summarize.

    Returns:
        The operators, with the number of nodes using each, sorted by domain,
        operator, overload and opset version.
    """
    functions = {
        (_normalize_domain(f.domain), f.name, f.overload): f for f in model.functions
    }
    counts: collections.Counter[tuple[str, str, str, int | None, OperatorKind]] = (
        collections.Counter()
    )
    called: set[tuple[str, str, str]] = set()
    # Each entry is a list of nodes and the opset versions they resolve against.
    pending: list[tuple[Iterable[onnx.NodeProto], dict[str, int]]] = [
        (model.graph.node, _opset_versions(model.opset_import))
    ]
    while pending:
        nodes, opset_versions = pending.pop()
        for node in nodes:
            domain = _normalize_domain(node.domain)
            opset_version = opset_versions.get(domain)
            key = (domain, node.op_type, node.overload)
            function = functions.get(key)
            if function is not None:
                kind = OperatorKind.FUNCTION
                if key not in called:
                    called.add(key)
                    function_opsets = _opset_versions(function.opset_import)
                    pending.append((function.node, function_opsets))
                    # Graph-valued attribute defaults run in the function's scope.
                    pending.extend(
                        (graph.node, function_opsets)
                        for attr in function.attribute_proto
                        for graph in _subgraphs(attr)
                    )
            else:
                kind = _schema_kind(node.op_type, domain, opset_version)
            counts[(domain, node.op_type, node.overload, opset_version, kind)] += 1
            for attr in node.attribute:
                pending.extend(
                    (graph.node, opset_versions) for graph in _subgraphs(attr)
                )

    operators = (
        OperatorUsage(domain, op_type, overload, opset_version, kind, count)
        for (domain, op_type, overload, opset_version, kind), count in counts.items()
    )
    return OperatorSummary(
        tuple(
            sorted(
                operators,
                key=lambda op: (
                    op.domain,
                    op.op_type,
                    op.overload,
                    -1 if op.opset_version is None else op.opset_version,
                    op.kind.value,
                ),
            )
        )
    )
