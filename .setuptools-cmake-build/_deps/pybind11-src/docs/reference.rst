.. _reference:

.. warning::

    Please be advised that the reference documentation discussing pybind11
    internals is currently incomplete. Please refer to the previous sections
    and the pybind11 header files for the nitty gritty details.

Reference
#########

.. _macros:

Macros
======

.. doxygendefine:: PYBIND11_MODULE

.. _core_types:

Convenience classes for arbitrary Python types
==============================================

Common member functions
-----------------------

.. doxygenclass:: object_api
    :members:

Without reference counting
--------------------------

.. doxygenclass:: handle
    :members:

With reference counting
-----------------------

.. doxygenclass:: object
    :members:

.. doxygenfunction:: reinterpret_borrow

.. doxygenfunction:: reinterpret_steal

Convenience classes for specific Python types
=============================================

.. doxygenclass:: module_
    :members:

.. doxygengroup:: pytypes
    :members:

Convenience functions converting to Python types
================================================

.. doxygenfunction:: make_tuple(Args&&...)

.. doxygenfunction:: make_iterator(Iterator, Sentinel, Extra &&...)
.. doxygenfunction:: make_iterator(Type &, Extra&&...)

.. doxygenfunction:: make_key_iterator(Iterator, Sentinel, Extra &&...)
.. doxygenfunction:: make_key_iterator(Type &, Extra&&...)

.. doxygenfunction:: make_value_iterator(Iterator, Sentinel, Extra &&...)
.. doxygenfunction:: make_value_iterator(Type &, Extra&&...)

.. _extras:

Passing extra arguments to ``def`` or ``class_``
================================================

.. doxygengroup:: annotations
    :members:

Embedding the interpreter
=========================

.. doxygendefine:: PYBIND11_EMBEDDED_MODULE

.. doxygenfunction:: initialize_interpreter

.. doxygenfunction:: finalize_interpreter

.. doxygenclass:: scoped_interpreter

Redirecting C++ streams
=======================

.. doxygenclass:: scoped_ostream_redirect

.. doxygenclass:: scoped_estream_redirect

.. doxygenfunction:: add_ostream_redirect

Python built-in functions
=========================

.. doxygengroup:: python_builtins
    :members:

Inheritance
===========

See :doc:`/classes` and :doc:`/advanced/classes` for more detail.

.. doxygendefine:: PYBIND11_OVERRIDE

.. doxygendefine:: PYBIND11_OVERRIDE_PURE

.. doxygendefine:: PYBIND11_OVERRIDE_NAME

.. doxygendefine:: PYBIND11_OVERRIDE_PURE_NAME

.. doxygenfunction:: get_override

Exceptions
==========

.. doxygenclass:: error_already_set
    :members:

.. doxygenclass:: builtin_exception
    :members:

Literals
========

.. doxygennamespace:: literals
