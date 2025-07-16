from __future__ import annotations

from pybind11_tests import unnamed_namespace_b as m


def test_have_attr_any_struct():
    assert hasattr(m, "unnamed_namespace_b_any_struct")
