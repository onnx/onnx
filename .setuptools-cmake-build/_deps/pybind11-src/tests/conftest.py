"""pytest configuration

Extends output capture as needed by pybind11: ignore constructors, optional unordered lines.
Adds docstring and exceptions message sanitizers.
"""

from __future__ import annotations

import contextlib
import difflib
import gc
import multiprocessing
import re
import sys
import textwrap
import traceback

import pytest

# Early diagnostic for failed imports
try:
    import pybind11_tests
except Exception:
    # pytest does not show the traceback without this.
    traceback.print_exc()
    raise


@pytest.fixture(scope="session", autouse=True)
def use_multiprocessing_forkserver_on_linux():
    if sys.platform != "linux":
        # The default on Windows and macOS is "spawn": If it's not broken, don't fix it.
        return

    # Full background: https://github.com/pybind/pybind11/issues/4105#issuecomment-1301004592
    # In a nutshell: fork() after starting threads == flakiness in the form of deadlocks.
    # It is actually a well-known pitfall, unfortunately without guard rails.
    # "forkserver" is more performant than "spawn" (~9s vs ~13s for tests/test_gil_scoped.py,
    # visit the issuecomment link above for details).
    multiprocessing.set_start_method("forkserver")


_long_marker = re.compile(r"([0-9])L")
_hexadecimal = re.compile(r"0x[0-9a-fA-F]+")

# Avoid collecting Python3 only files
collect_ignore = []


def _strip_and_dedent(s):
    """For triple-quote strings"""
    return textwrap.dedent(s.lstrip("\n").rstrip())


def _split_and_sort(s):
    """For output which does not require specific line order"""
    return sorted(_strip_and_dedent(s).splitlines())


def _make_explanation(a, b):
    """Explanation for a failed assert -- the a and b arguments are List[str]"""
    return ["--- actual / +++ expected"] + [
        line.strip("\n") for line in difflib.ndiff(a, b)
    ]


class Output:
    """Basic output post-processing and comparison"""

    def __init__(self, string):
        self.string = string
        self.explanation = []

    def __str__(self):
        return self.string

    def __eq__(self, other):
        # Ignore constructor/destructor output which is prefixed with "###"
        a = [
            line
            for line in self.string.strip().splitlines()
            if not line.startswith("###")
        ]
        b = _strip_and_dedent(other).splitlines()
        if a == b:
            return True
        self.explanation = _make_explanation(a, b)
        return False


class Unordered(Output):
    """Custom comparison for output without strict line ordering"""

    def __eq__(self, other):
        a = _split_and_sort(self.string)
        b = _split_and_sort(other)
        if a == b:
            return True
        self.explanation = _make_explanation(a, b)
        return False


class Capture:
    def __init__(self, capfd):
        self.capfd = capfd
        self.out = ""
        self.err = ""

    def __enter__(self):
        self.capfd.readouterr()
        return self

    def __exit__(self, *args):
        self.out, self.err = self.capfd.readouterr()

    def __eq__(self, other):
        a = Output(self.out)
        b = other
        if a == b:
            return True
        self.explanation = a.explanation
        return False

    def __str__(self):
        return self.out

    def __contains__(self, item):
        return item in self.out

    @property
    def unordered(self):
        return Unordered(self.out)

    @property
    def stderr(self):
        return Output(self.err)


@pytest.fixture
def capture(capsys):
    """Extended `capsys` with context manager and custom equality operators"""
    return Capture(capsys)


class SanitizedString:
    def __init__(self, sanitizer):
        self.sanitizer = sanitizer
        self.string = ""
        self.explanation = []

    def __call__(self, thing):
        self.string = self.sanitizer(thing)
        return self

    def __eq__(self, other):
        a = self.string
        b = _strip_and_dedent(other)
        if a == b:
            return True
        self.explanation = _make_explanation(a.splitlines(), b.splitlines())
        return False


def _sanitize_general(s):
    s = s.strip()
    s = s.replace("pybind11_tests.", "m.")
    return _long_marker.sub(r"\1", s)


def _sanitize_docstring(thing):
    s = thing.__doc__
    return _sanitize_general(s)


@pytest.fixture
def doc():
    """Sanitize docstrings and add custom failure explanation"""
    return SanitizedString(_sanitize_docstring)


def _sanitize_message(thing):
    s = str(thing)
    s = _sanitize_general(s)
    return _hexadecimal.sub("0", s)


@pytest.fixture
def msg():
    """Sanitize messages and add custom failure explanation"""
    return SanitizedString(_sanitize_message)


def pytest_assertrepr_compare(op, left, right):  # noqa: ARG001
    """Hook to insert custom failure explanation"""
    if hasattr(left, "explanation"):
        return left.explanation
    return None


def gc_collect():
    """Run the garbage collector twice (needed when running
    reference counting tests with PyPy)"""
    gc.collect()
    gc.collect()


def pytest_configure():
    pytest.suppress = contextlib.suppress
    pytest.gc_collect = gc_collect


def pytest_report_header(config):
    del config  # Unused.
    assert (
        pybind11_tests.compiler_info is not None
    ), "Please update pybind11_tests.cpp if this assert fails."
    return (
        "C++ Info:"
        f" {pybind11_tests.compiler_info}"
        f" {pybind11_tests.cpp_std}"
        f" {pybind11_tests.PYBIND11_INTERNALS_ID}"
        f" PYBIND11_SIMPLE_GIL_MANAGEMENT={pybind11_tests.PYBIND11_SIMPLE_GIL_MANAGEMENT}"
        f" PYBIND11_NUMPY_1_ONLY={pybind11_tests.PYBIND11_NUMPY_1_ONLY}"
    )
