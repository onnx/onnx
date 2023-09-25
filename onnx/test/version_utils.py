# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

try:
    from packaging.version import parse as version
except ImportError:
    from distutils.version import StrictVersion as version  # noqa: N813


def numpy_older_than(ver: str) -> bool:
    """Returns True if the numpy version is older than the given version."""
    import numpy  # pylint: disable=import-outside-toplevel

    return version(numpy.__version__) < version(ver)
