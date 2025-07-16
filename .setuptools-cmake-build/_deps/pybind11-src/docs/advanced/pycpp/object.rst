Python types
############

.. _wrappers:

Available wrappers
==================

All major Python types are available as thin C++ wrapper classes. These
can also be used as function parameters -- see :ref:`python_objects_as_args`.

Available types include :class:`handle`, :class:`object`, :class:`bool_`,
:class:`int_`, :class:`float_`, :class:`str`, :class:`bytes`, :class:`tuple`,
:class:`list`, :class:`dict`, :class:`slice`, :class:`none`, :class:`capsule`,
:class:`iterable`, :class:`iterator`, :class:`function`, :class:`buffer`,
:class:`array`, and :class:`array_t`.

.. warning::

    Be sure to review the :ref:`pytypes_gotchas` before using this heavily in
    your C++ API.

.. _instantiating_compound_types:

Instantiating compound Python types from C++
============================================

Dictionaries can be initialized in the :class:`dict` constructor:

.. code-block:: cpp

    using namespace pybind11::literals; // to bring in the `_a` literal
    py::dict d("spam"_a=py::none(), "eggs"_a=42);

A tuple of python objects can be instantiated using :func:`py::make_tuple`:

.. code-block:: cpp

    py::tuple tup = py::make_tuple(42, py::none(), "spam");

Each element is converted to a supported Python type.

A `simple namespace`_ can be instantiated using

.. code-block:: cpp

    using namespace pybind11::literals;  // to bring in the `_a` literal
    py::object SimpleNamespace = py::module_::import("types").attr("SimpleNamespace");
    py::object ns = SimpleNamespace("spam"_a=py::none(), "eggs"_a=42);

Attributes on a namespace can be modified with the :func:`py::delattr`,
:func:`py::getattr`, and :func:`py::setattr` functions. Simple namespaces can
be useful as lightweight stand-ins for class instances.

.. _simple namespace: https://docs.python.org/3/library/types.html#types.SimpleNamespace

.. _casting_back_and_forth:

Casting back and forth
======================

In this kind of mixed code, it is often necessary to convert arbitrary C++
types to Python, which can be done using :func:`py::cast`:

.. code-block:: cpp

    MyClass *cls = ...;
    py::object obj = py::cast(cls);

The reverse direction uses the following syntax:

.. code-block:: cpp

    py::object obj = ...;
    MyClass *cls = obj.cast<MyClass *>();

When conversion fails, both directions throw the exception :class:`cast_error`.

.. _python_libs:

Accessing Python libraries from C++
===================================

It is also possible to import objects defined in the Python standard
library or available in the current Python environment (``sys.path``) and work
with these in C++.

This example obtains a reference to the Python ``Decimal`` class.

.. code-block:: cpp

    // Equivalent to "from decimal import Decimal"
    py::object Decimal = py::module_::import("decimal").attr("Decimal");

.. code-block:: cpp

    // Try to import scipy
    py::object scipy = py::module_::import("scipy");
    return scipy.attr("__version__");


.. _calling_python_functions:

Calling Python functions
========================

It is also possible to call Python classes, functions and methods
via ``operator()``.

.. code-block:: cpp

    // Construct a Python object of class Decimal
    py::object pi = Decimal("3.14159");

.. code-block:: cpp

    // Use Python to make our directories
    py::object os = py::module_::import("os");
    py::object makedirs = os.attr("makedirs");
    makedirs("/tmp/path/to/somewhere");

One can convert the result obtained from Python to a pure C++ version
if a ``py::class_`` or type conversion is defined.

.. code-block:: cpp

    py::function f = <...>;
    py::object result_py = f(1234, "hello", some_instance);
    MyClass &result = result_py.cast<MyClass>();

.. _calling_python_methods:

Calling Python methods
========================

To call an object's method, one can again use ``.attr`` to obtain access to the
Python method.

.. code-block:: cpp

    // Calculate e^π in decimal
    py::object exp_pi = pi.attr("exp")();
    py::print(py::str(exp_pi));

In the example above ``pi.attr("exp")`` is a *bound method*: it will always call
the method for that same instance of the class. Alternately one can create an
*unbound method* via the Python class (instead of instance) and pass the ``self``
object explicitly, followed by other arguments.

.. code-block:: cpp

    py::object decimal_exp = Decimal.attr("exp");

    // Compute the e^n for n=0..4
    for (int n = 0; n < 5; n++) {
        py::print(decimal_exp(Decimal(n));
    }

Keyword arguments
=================

Keyword arguments are also supported. In Python, there is the usual call syntax:

.. code-block:: python

    def f(number, say, to):
        ...  # function code


    f(1234, say="hello", to=some_instance)  # keyword call in Python

In C++, the same call can be made using:

.. code-block:: cpp

    using namespace pybind11::literals; // to bring in the `_a` literal
    f(1234, "say"_a="hello", "to"_a=some_instance); // keyword call in C++

Unpacking arguments
===================

Unpacking of ``*args`` and ``**kwargs`` is also possible and can be mixed with
other arguments:

.. code-block:: cpp

    // * unpacking
    py::tuple args = py::make_tuple(1234, "hello", some_instance);
    f(*args);

    // ** unpacking
    py::dict kwargs = py::dict("number"_a=1234, "say"_a="hello", "to"_a=some_instance);
    f(**kwargs);

    // mixed keywords, * and ** unpacking
    py::tuple args = py::make_tuple(1234);
    py::dict kwargs = py::dict("to"_a=some_instance);
    f(*args, "say"_a="hello", **kwargs);

Generalized unpacking according to PEP448_ is also supported:

.. code-block:: cpp

    py::dict kwargs1 = py::dict("number"_a=1234);
    py::dict kwargs2 = py::dict("to"_a=some_instance);
    f(**kwargs1, "say"_a="hello", **kwargs2);

.. seealso::

    The file :file:`tests/test_pytypes.cpp` contains a complete
    example that demonstrates passing native Python types in more detail. The
    file :file:`tests/test_callbacks.cpp` presents a few examples of calling
    Python functions from C++, including keywords arguments and unpacking.

.. _PEP448: https://www.python.org/dev/peps/pep-0448/

.. _implicit_casting:

Implicit casting
================

When using the C++ interface for Python types, or calling Python functions,
objects of type :class:`object` are returned. It is possible to invoke implicit
conversions to subclasses like :class:`dict`. The same holds for the proxy objects
returned by ``operator[]`` or ``obj.attr()``.
Casting to subtypes improves code readability and allows values to be passed to
C++ functions that require a specific subtype rather than a generic :class:`object`.

.. code-block:: cpp

    #include <pybind11/numpy.h>
    using namespace pybind11::literals;

    py::module_ os = py::module_::import("os");
    py::module_ path = py::module_::import("os.path");  // like 'import os.path as path'
    py::module_ np = py::module_::import("numpy");  // like 'import numpy as np'

    py::str curdir_abs = path.attr("abspath")(path.attr("curdir"));
    py::print(py::str("Current directory: ") + curdir_abs);
    py::dict environ = os.attr("environ");
    py::print(environ["HOME"]);
    py::array_t<float> arr = np.attr("ones")(3, "dtype"_a="float32");
    py::print(py::repr(arr + py::int_(1)));

These implicit conversions are available for subclasses of :class:`object`; there
is no need to call ``obj.cast()`` explicitly as for custom classes, see
:ref:`casting_back_and_forth`.

.. note::
    If a trivial conversion via move constructor is not possible, both implicit and
    explicit casting (calling ``obj.cast()``) will attempt a "rich" conversion.
    For instance, ``py::list env = os.attr("environ");`` will succeed and is
    equivalent to the Python code ``env = list(os.environ)`` that produces a
    list of the dict keys.

..  TODO: Adapt text once PR #2349 has landed

Handling exceptions
===================

Python exceptions from wrapper classes will be thrown as a ``py::error_already_set``.
See :ref:`Handling exceptions from Python in C++
<handling_python_exceptions_cpp>` for more information on handling exceptions
raised when calling C++ wrapper classes.

.. _pytypes_gotchas:

Gotchas
=======

Default-Constructed Wrappers
----------------------------

When a wrapper type is default-constructed, it is **not** a valid Python object (i.e. it is not ``py::none()``). It is simply the same as
``PyObject*`` null pointer. To check for this, use
``static_cast<bool>(my_wrapper)``.

Assigning py::none() to wrappers
--------------------------------

You may be tempted to use types like ``py::str`` and ``py::dict`` in C++
signatures (either pure C++, or in bound signatures), and assign them default
values of ``py::none()``. However, in a best case scenario, it will fail fast
because ``None`` is not convertible to that type (e.g. ``py::dict``), or in a
worse case scenario, it will silently work but corrupt the types you want to
work with (e.g. ``py::str(py::none())`` will yield ``"None"`` in Python).
