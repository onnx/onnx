from __future__ import annotations

import pytest

from pybind11_tests import vector_unique_ptr_member as m


@pytest.mark.parametrize("num_elems", range(3))
def test_create(num_elems):
    vo = m.VectorOwner.Create(num_elems)
    assert vo.data_size() == num_elems


def test_cast():
    vo = m.VectorOwner.Create(0)
    assert m.py_cast_VectorOwner_ptr(vo) is vo
