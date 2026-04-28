# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

"""PEP 517 build backend for onnx

This is a thin wrapper over setuptools' PEP 517 build backend that
automatically adds ``cmake`` to build dependencies if there is no CMake
executable in PATH.  This approach ensures that the package uses system
CMake (that may contain downstream patches) when one is available,
and pulls in the CMake package from PyPI when it is not.
"""

from __future__ import annotations

import shutil

from setuptools.build_meta import (
    build_editable,
    build_sdist,
    build_wheel,
    get_requires_for_build_sdist,
    prepare_metadata_for_build_editable,
    prepare_metadata_for_build_wheel,
)
from setuptools.build_meta import (
    get_requires_for_build_editable as _get_requires_for_build_editable,
)
from setuptools.build_meta import (
    get_requires_for_build_wheel as _get_requires_for_build_wheel,
)

__all__ = [
    "build_editable",
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_editable",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_editable",
    "prepare_metadata_for_build_wheel",
]


def _get_cmake_dep() -> list[str]:
    if shutil.which("cmake3") or shutil.which("cmake"):
        return []
    return ["cmake>=3.18"]


def get_requires_for_build_editable(*args, **kwargs) -> list[str]:
    return _get_requires_for_build_editable(*args, **kwargs) + _get_cmake_dep()


def get_requires_for_build_wheel(*args, **kwargs) -> list[str]:
    return _get_requires_for_build_wheel(*args, **kwargs) + _get_cmake_dep()
