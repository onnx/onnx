from __future__ import annotations

from pybind11_tests import union_ as m


def test_union():
    instance = m.TestUnion()

    instance.as_uint = 10
    assert instance.as_int == 10
