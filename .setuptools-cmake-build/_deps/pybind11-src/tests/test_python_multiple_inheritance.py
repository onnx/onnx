# Adapted from:
# https://github.com/google/clif/blob/5718e4d0807fd3b6a8187dde140069120b81ecef/clif/testing/python/python_multiple_inheritance_test.py
from __future__ import annotations

from pybind11_tests import python_multiple_inheritance as m


class PC(m.CppBase):
    pass


class PPCC(PC, m.CppDrvd):
    pass


def test_PC():
    d = PC(11)
    assert d.get_base_value() == 11
    d.reset_base_value(13)
    assert d.get_base_value() == 13


def test_PPCC():
    d = PPCC(11)
    assert d.get_drvd_value() == 33
    d.reset_drvd_value(55)
    assert d.get_drvd_value() == 55

    assert d.get_base_value() == 11
    assert d.get_base_value_from_drvd() == 11
    d.reset_base_value(20)
    assert d.get_base_value() == 20
    assert d.get_base_value_from_drvd() == 20
    d.reset_base_value_from_drvd(30)
    assert d.get_base_value() == 30
    assert d.get_base_value_from_drvd() == 30
