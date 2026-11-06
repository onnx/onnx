# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import functools
import warnings
from typing import TYPE_CHECKING, ParamSpec, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

_P = ParamSpec("_P")
_R = TypeVar("_R")


def deprecated(message: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Warn at the caller of a deprecated function, preserving its signature."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        function.__dict__["__deprecated__"] = message

        @functools.wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return function(*args, **kwargs)

        return wrapped

    return decorate
