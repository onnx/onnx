Limitations
###########

Design choices
^^^^^^^^^^^^^^

pybind11 strives to be a general solution to binding generation, but it also has
certain limitations:

- pybind11 casts away ``const``-ness in function arguments and return values.
  This is in line with the Python language, which has no concept of ``const``
  values. This means that some additional care is needed to avoid bugs that
  would be caught by the type checker in a traditional C++ program.

- The NumPy interface ``pybind11::array`` greatly simplifies accessing
  numerical data from C++ (and vice versa), but it's not a full-blown array
  class like ``Eigen::Array`` or ``boost.multi_array``. ``Eigen`` objects are
  directly supported, however, with ``pybind11/eigen.h``.

Large but useful features could be implemented in pybind11 but would lead to a
significant increase in complexity. Pybind11 strives to be simple and compact.
Users who require large new features are encouraged to write an extension to
pybind11; see `pybind11_json <https://github.com/pybind/pybind11_json>`_ for an
example.


Known bugs
^^^^^^^^^^

These are issues that hopefully will one day be fixed, but currently are
unsolved. If you know how to help with one of these issues, contributions
are welcome!

- Intel 20.2 is currently having an issue with the test suite.
  `#2573 <https://github.com/pybind/pybind11/pull/2573>`_

- Debug mode Python does not support 1-5 tests in the test suite currently.
  `#2422 <https://github.com/pybind/pybind11/pull/2422>`_

- PyPy3 7.3.1 and 7.3.2 have issues with several tests on 32-bit Windows.

Known limitations
^^^^^^^^^^^^^^^^^

These are issues that are probably solvable, but have not been fixed yet. A
clean, well written patch would likely be accepted to solve them.

- Type casters are not kept alive recursively.
  `#2527 <https://github.com/pybind/pybind11/issues/2527>`_
  One consequence is that containers of ``char *`` are currently not supported.
  `#2245 <https://github.com/pybind/pybind11/issues/2245>`_

Python 3.9.0 warning
^^^^^^^^^^^^^^^^^^^^

Combining older versions of pybind11 (< 2.6.0) with Python on exactly 3.9.0
will trigger undefined behavior that typically manifests as crashes during
interpreter shutdown (but could also destroy your data. **You have been
warned**).

This issue was `fixed in Python <https://github.com/python/cpython/pull/22670>`_.
As a mitigation for this bug, pybind11 2.6.0 or newer includes a workaround
specifically when Python 3.9.0 is detected at runtime, leaking about 50 bytes
of memory when a callback function is garbage collected.  For reference, the
pybind11 test suite has about 2,000 such callbacks, but only 49 are garbage
collected before the end-of-process. Wheels (even if built with Python 3.9.0)
will correctly avoid the leak when run in Python 3.9.1, and this does not
affect other 3.X versions.
