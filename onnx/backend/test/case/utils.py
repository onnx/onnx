# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

import numpy as np

from onnx import ONNX_ML

if TYPE_CHECKING:
    from types import ModuleType

all_numeric_dtypes = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
]


def import_recursive(package: ModuleType) -> None:
    """Takes a package and imports all modules underneath it."""
    pkg_dir = package.__path__
    module_location = package.__name__
    for _module_loader, name, ispkg in pkgutil.iter_modules(pkg_dir):
        module_name = f"{module_location}.{name}"  # Module/package
        if not ONNX_ML and module_name.startswith(
            "onnx.backend.test.case.node.ai_onnx_ml"
        ):
            continue

        module = importlib.import_module(module_name)
        if ispkg:
            import_recursive(module)
