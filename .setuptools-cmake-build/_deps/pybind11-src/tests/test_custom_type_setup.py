from __future__ import annotations

import gc
import weakref

import pytest

import env  # noqa: F401
from pybind11_tests import custom_type_setup as m


@pytest.fixture
def gc_tester():
    """Tests that an object is garbage collected.

    Assumes that any unreferenced objects are fully collected after calling
    `gc.collect()`.  That is true on CPython, but does not appear to reliably
    hold on PyPy.
    """

    weak_refs = []

    def add_ref(obj):
        # PyPy does not support `gc.is_tracked`.
        if hasattr(gc, "is_tracked"):
            assert gc.is_tracked(obj)
        weak_refs.append(weakref.ref(obj))

    yield add_ref

    gc.collect()
    for ref in weak_refs:
        assert ref() is None


# PyPy does not seem to reliably garbage collect.
@pytest.mark.skipif("env.PYPY")
def test_self_cycle(gc_tester):
    obj = m.OwnsPythonObjects()
    obj.value = obj
    gc_tester(obj)


# PyPy does not seem to reliably garbage collect.
@pytest.mark.skipif("env.PYPY")
def test_indirect_cycle(gc_tester):
    obj = m.OwnsPythonObjects()
    obj_list = [obj]
    obj.value = obj_list
    gc_tester(obj)
