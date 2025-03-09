# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections import namedtuple
from typing import TYPE_CHECKING, Any, NewType

import onnx.checker
import onnx.onnx_cpp2py_export.checker as c_checker
from onnx import IR_VERSION, ModelProto, NodeProto

if TYPE_CHECKING:
    from collections.abc import Sequence


class DeviceType:
    """Describes device type."""

    _Type = NewType("_Type", int)
    CPU: _Type = _Type(0)
    CUDA: _Type = _Type(1)


class Device:
    """Describes device type and device id
    syntax: device_type:device_id(optional)
    example: 'CPU', 'CUDA', 'CUDA:1'
    """

    def __init__(self, device: str) -> None:
        options = device.split(":")
        self.type: DeviceType = getattr(DeviceType, options[0])
        self.device_id: int = 0
        if len(options) > 1:
            self.device_id = int(options[1])


def namedtupledict(
    typename: str, field_names: Sequence[str], *args: Any, **kwargs: Any
) -> type[tuple[Any, ...]]:
    field_names_map = {n: i for i, n in enumerate(field_names)}
    # Some output names are invalid python identifier, e.g. "0"
    kwargs.setdefault("rename", True)
    data = namedtuple(typename, field_names, *args, **kwargs)  # type: ignore  # noqa: PYI024

    def getitem(self: Any, key: Any) -> Any:
        if isinstance(key, str):
            key = field_names_map[key]
        return super(type(self), self).__getitem__(key)  # type: ignore

    data.__getitem__ = getitem  # type: ignore[assignment]
    return data


class BackendRep:
    """BackendRep is the handle that a Backend returns after preparing to execute
    a model repeatedly. Users will then pass inputs to the run function of
    BackendRep to retrieve the corresponding results.
    """

    def run(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:  # noqa: ARG002
        """Abstract function."""
        return (None,)


class Backend:
    """Backend is the entity that will take an ONNX model with inputs,
    perform a computation, and then return the output.

    For one-off execution, users can use run_node and run_model to obtain results quickly.

    For repeated execution, users should use prepare, in which the Backend
    does all of the preparation work for executing the model repeatedly
    (e.g., loading initializers), and returns a BackendRep handle.
    """

    @classmethod
    def is_compatible(
        cls,
        *args: Any,  # noqa: ARG003
        **kwargs: Any,  # noqa: ARG003
    ) -> bool:
        # Return whether the model is compatible with the backend.
        return True

    @classmethod
    def prepare(
        cls,
        model: ModelProto,
        *args: Any,  # noqa: ARG003
        **kwargs: Any,  # noqa: ARG003
    ) -> BackendRep | None:
        # TODO Remove Optional from return type
        onnx.checker.check_model(model)
        return None

    @classmethod
    def run_model(
        cls, model: ModelProto, inputs: Any, *args, **kwargs: Any
    ) -> tuple[Any, ...]:
        backend = cls.prepare(model, *args, **kwargs)
        assert backend is not None
        return backend.run(inputs)

    @classmethod
    def run_node(
        cls,
        node: NodeProto,
        *args: Any,  # noqa: ARG003
        **kwargs: Any,
    ) -> tuple[Any, ...] | None:
        """Simple run one operator and return the results.

        Args:
            node: The node proto.
            args: Other arguments.
            kwargs: Other keyword arguments.
        """
        # TODO Remove Optional from return type
        if "opset_version" in kwargs:
            special_context = c_checker.CheckerContext()
            special_context.ir_version = IR_VERSION
            special_context.opset_imports = {"": kwargs["opset_version"]}
            onnx.checker.check_node(node, special_context)
        else:
            onnx.checker.check_node(node)
        return None

    @classmethod
    def supports_device(cls, device: str) -> bool:  # noqa: ARG003
        """Checks whether the backend is compiled with particular device support.
        In particular it's used in the testing suite.
        """
        return True
