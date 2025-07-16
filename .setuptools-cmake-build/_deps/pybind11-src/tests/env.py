from __future__ import annotations

import platform
import sys
import sysconfig

import pytest

LINUX = sys.platform.startswith("linux")
MACOS = sys.platform.startswith("darwin")
WIN = sys.platform.startswith("win32") or sys.platform.startswith("cygwin")

CPYTHON = platform.python_implementation() == "CPython"
PYPY = platform.python_implementation() == "PyPy"
PY_GIL_DISABLED = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def deprecated_call():
    """
    pytest.deprecated_call() seems broken in pytest<3.9.x; concretely, it
    doesn't work on CPython 3.8.0 with pytest==3.3.2 on Ubuntu 18.04 (#2922).

    This is a narrowed reimplementation of the following PR :(
    https://github.com/pytest-dev/pytest/pull/4104
    """
    # TODO: Remove this when testing requires pytest>=3.9.
    pieces = pytest.__version__.split(".")
    pytest_major_minor = (int(pieces[0]), int(pieces[1]))
    if pytest_major_minor < (3, 9):
        return pytest.warns((DeprecationWarning, PendingDeprecationWarning))
    return pytest.deprecated_call()
