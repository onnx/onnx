from __future__ import annotations

import os

import pytest

import env  # noqa: F401
from pybind11_tests import eval_ as m


def test_evals(capture):
    with capture:
        assert m.test_eval_statements()
    assert capture == "Hello World!"

    assert m.test_eval()
    assert m.test_eval_single_statement()

    assert m.test_eval_failure()


@pytest.mark.xfail("env.PYPY", raises=RuntimeError)
def test_eval_file():
    filename = os.path.join(os.path.dirname(__file__), "test_eval_call.py")
    assert m.test_eval_file(filename)

    assert m.test_eval_file_failure()


def test_eval_empty_globals():
    assert "__builtins__" in m.eval_empty_globals(None)

    g = {}
    assert "__builtins__" in m.eval_empty_globals(g)
    assert "__builtins__" in g


def test_eval_closure():
    global_, local = m.test_eval_closure()

    assert global_["closure_value"] == 42
    assert local["closure_value"] == 0

    assert "local_value" not in global_
    assert local["local_value"] == 0

    assert "func_global" not in global_
    assert local["func_global"]() == 42

    assert "func_local" not in global_
    with pytest.raises(NameError):
        local["func_local"]()
