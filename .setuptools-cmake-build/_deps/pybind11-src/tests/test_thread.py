from __future__ import annotations

import sys
import threading

import pytest

from pybind11_tests import thread as m


class Thread(threading.Thread):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.e = None

    def run(self):
        try:
            for i in range(10):
                self.fn(i, i)
        except Exception as e:
            self.e = e

    def join(self):
        super().join()
        if self.e:
            raise self.e


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
def test_implicit_conversion():
    a = Thread(m.test)
    b = Thread(m.test)
    c = Thread(m.test)
    for x in [a, b, c]:
        x.start()
    for x in [c, b, a]:
        x.join()


@pytest.mark.skipif(sys.platform.startswith("emscripten"), reason="Requires threads")
def test_implicit_conversion_no_gil():
    a = Thread(m.test_no_gil)
    b = Thread(m.test_no_gil)
    c = Thread(m.test_no_gil)
    for x in [a, b, c]:
        x.start()
    for x in [c, b, a]:
        x.join()
