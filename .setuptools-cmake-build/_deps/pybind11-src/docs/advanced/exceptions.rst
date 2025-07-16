Exceptions
##########

Built-in C++ to Python exception translation
============================================

When Python calls C++ code through pybind11, pybind11 provides a C++ exception handler
that will trap C++ exceptions, translate them to the corresponding Python exception,
and raise them so that Python code can handle them.

pybind11 defines translations for ``std::exception`` and its standard
subclasses, and several special exception classes that translate to specific
Python exceptions. Note that these are not actually Python exceptions, so they
cannot be examined using the Python C API. Instead, they are pure C++ objects
that pybind11 will translate the corresponding Python exception when they arrive
at its exception handler.

.. tabularcolumns:: |p{0.5\textwidth}|p{0.45\textwidth}|

+--------------------------------------+--------------------------------------+
|  Exception thrown by C++             |  Translated to Python exception type |
+======================================+======================================+
| :class:`std::exception`              | ``RuntimeError``                     |
+--------------------------------------+--------------------------------------+
| :class:`std::bad_alloc`              | ``MemoryError``                      |
+--------------------------------------+--------------------------------------+
| :class:`std::domain_error`           | ``ValueError``                       |
+--------------------------------------+--------------------------------------+
| :class:`std::invalid_argument`       | ``ValueError``                       |
+--------------------------------------+--------------------------------------+
| :class:`std::length_error`           | ``ValueError``                       |
+--------------------------------------+--------------------------------------+
| :class:`std::out_of_range`           | ``IndexError``                       |
+--------------------------------------+--------------------------------------+
| :class:`std::range_error`            | ``ValueError``                       |
+--------------------------------------+--------------------------------------+
| :class:`std::overflow_error`         | ``OverflowError``                    |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::stop_iteration`    | ``StopIteration`` (used to implement |
|                                      | custom iterators)                    |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::index_error`       | ``IndexError`` (used to indicate out |
|                                      | of bounds access in ``__getitem__``, |
|                                      | ``__setitem__``, etc.)               |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::key_error`         | ``KeyError`` (used to indicate out   |
|                                      | of bounds access in ``__getitem__``, |
|                                      | ``__setitem__`` in dict-like         |
|                                      | objects, etc.)                       |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::value_error`       | ``ValueError`` (used to indicate     |
|                                      | wrong value passed in                |
|                                      | ``container.remove(...)``)           |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::type_error`        | ``TypeError``                        |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::buffer_error`      | ``BufferError``                      |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::import_error`      | ``ImportError``                      |
+--------------------------------------+--------------------------------------+
| :class:`pybind11::attribute_error`   | ``AttributeError``                   |
+--------------------------------------+--------------------------------------+
| Any other exception                  | ``RuntimeError``                     |
+--------------------------------------+--------------------------------------+

Exception translation is not bidirectional. That is, *catching* the C++
exceptions defined above will not trap exceptions that originate from
Python. For that, catch :class:`pybind11::error_already_set`. See :ref:`below
<handling_python_exceptions_cpp>` for further details.

There is also a special exception :class:`cast_error` that is thrown by
:func:`handle::call` when the input arguments cannot be converted to Python
objects.

Registering custom translators
==============================

If the default exception conversion policy described above is insufficient,
pybind11 also provides support for registering custom exception translators.
Similar to pybind11 classes, exception translators can be local to the module
they are defined in or global to the entire python session.  To register a simple
exception conversion that translates a C++ exception into a new Python exception
using the C++ exception's ``what()`` method, a helper function is available:

.. code-block:: cpp

    py::register_exception<CppExp>(module, "PyExp");

This call creates a Python exception class with the name ``PyExp`` in the given
module and automatically converts any encountered exceptions of type ``CppExp``
into Python exceptions of type ``PyExp``.

A matching function is available for registering a local exception translator:

.. code-block:: cpp

    py::register_local_exception<CppExp>(module, "PyExp");


It is possible to specify base class for the exception using the third
parameter, a ``handle``:

.. code-block:: cpp

    py::register_exception<CppExp>(module, "PyExp", PyExc_RuntimeError);
    py::register_local_exception<CppExp>(module, "PyExp", PyExc_RuntimeError);

Then ``PyExp`` can be caught both as ``PyExp`` and ``RuntimeError``.

The class objects of the built-in Python exceptions are listed in the Python
documentation on `Standard Exceptions <https://docs.python.org/3/c-api/exceptions.html#standard-exceptions>`_.
The default base class is ``PyExc_Exception``.

When more advanced exception translation is needed, the functions
``py::register_exception_translator(translator)`` and
``py::register_local_exception_translator(translator)`` can be used to register
functions that can translate arbitrary exception types (and which may include
additional logic to do so).  The functions takes a stateless callable (e.g. a
function pointer or a lambda function without captured variables) with the call
signature ``void(std::exception_ptr)``.

When a C++ exception is thrown, the registered exception translators are tried
in reverse order of registration (i.e. the last registered translator gets the
first shot at handling the exception). All local translators will be tried
before a global translator is tried.

Inside the translator, ``std::rethrow_exception`` should be used within
a try block to re-throw the exception.  One or more catch clauses to catch
the appropriate exceptions should then be used with each clause using
``py::set_error()`` (see below).

To declare a custom Python exception type, declare a ``py::exception`` variable
and use this in the associated exception translator (note: it is often useful
to make this a static declaration when using it inside a lambda expression
without requiring capturing).

The following example demonstrates this for a hypothetical exception classes
``MyCustomException`` and ``OtherException``: the first is translated to a
custom python exception ``MyCustomError``, while the second is translated to a
standard python RuntimeError:

.. code-block:: cpp

    PYBIND11_CONSTINIT static py::gil_safe_call_once_and_store<py::object> exc_storage;
    exc_storage.call_once_and_store_result(
        [&]() { return py::exception<MyCustomException>(m, "MyCustomError"); });
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const MyCustomException &e) {
            py::set_error(exc_storage.get_stored(), e.what());
        } catch (const OtherException &e) {
            py::set_error(PyExc_RuntimeError, e.what());
        }
    });

Multiple exceptions can be handled by a single translator, as shown in the
example above. If the exception is not caught by the current translator, the
previously registered one gets a chance.

If none of the registered exception translators is able to handle the
exception, it is handled by the default converter as described in the previous
section.

.. seealso::

    The file :file:`tests/test_exceptions.cpp` contains examples
    of various custom exception translators and custom exception types.

.. note::

    Call ``py::set_error()`` for every exception caught in a custom exception
    translator.  Failure to do so will cause Python to crash with ``SystemError:
    error return without exception set``.

    Exceptions that you do not plan to handle should simply not be caught, or
    may be explicitly (re-)thrown to delegate it to the other,
    previously-declared existing exception translators.

    Note that ``libc++`` and ``libstdc++`` `behave differently under macOS
    <https://stackoverflow.com/questions/19496643/using-clang-fvisibility-hidden-and-typeinfo-and-type-erasure/28827430>`_
    with ``-fvisibility=hidden``. Therefore exceptions that are used across ABI
    boundaries need to be explicitly exported, as exercised in
    ``tests/test_exceptions.h``. See also:
    "Problems with C++ exceptions" under `GCC Wiki <https://gcc.gnu.org/wiki/Visibility>`_.


Local vs Global Exception Translators
=====================================

When a global exception translator is registered, it will be applied across all
modules in the reverse order of registration. This can create behavior where the
order of module import influences how exceptions are translated.

If module1 has the following translator:

.. code-block:: cpp

      py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::invalid_argument &e) {
            py::set_error(PyExc_ArgumentError, "module1 handled this");
        }
      }

and module2 has the following similar translator:

.. code-block:: cpp

      py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const std::invalid_argument &e) {
            py::set_error(PyExc_ArgumentError, "module2 handled this");
        }
      }

then which translator handles the invalid_argument will be determined by the
order that module1 and module2 are imported. Since exception translators are
applied in the reverse order of registration, which ever module was imported
last will "win" and that translator will be applied.

If there are multiple pybind11 modules that share exception types (either
standard built-in or custom) loaded into a single python instance and
consistent error handling behavior is needed, then local translators should be
used.

Changing the previous example to use ``register_local_exception_translator``
would mean that when invalid_argument is thrown in the module2 code, the
module2 translator will always handle it, while in module1, the module1
translator will do the same.

.. _handling_python_exceptions_cpp:

Handling exceptions from Python in C++
======================================

When C++ calls Python functions, such as in a callback function or when
manipulating Python objects, and Python raises an ``Exception``, pybind11
converts the Python exception into a C++ exception of type
:class:`pybind11::error_already_set` whose payload contains a C++ string textual
summary and the actual Python exception. ``error_already_set`` is used to
propagate Python exception back to Python (or possibly, handle them in C++).

.. tabularcolumns:: |p{0.5\textwidth}|p{0.45\textwidth}|

+--------------------------------------+--------------------------------------+
|  Exception raised in Python          |  Thrown as C++ exception type        |
+======================================+======================================+
| Any Python ``Exception``             | :class:`pybind11::error_already_set` |
+--------------------------------------+--------------------------------------+

For example:

.. code-block:: cpp

    try {
        // open("missing.txt", "r")
        auto file = py::module_::import("io").attr("open")("missing.txt", "r");
        auto text = file.attr("read")();
        file.attr("close")();
    } catch (py::error_already_set &e) {
        if (e.matches(PyExc_FileNotFoundError)) {
            py::print("missing.txt not found");
        } else if (e.matches(PyExc_PermissionError)) {
            py::print("missing.txt found but not accessible");
        } else {
            throw;
        }
    }

Note that C++ to Python exception translation does not apply here, since that is
a method for translating C++ exceptions to Python, not vice versa. The error raised
from Python is always ``error_already_set``.

This example illustrates this behavior:

.. code-block:: cpp

    try {
        py::eval("raise ValueError('The Ring')");
    } catch (py::value_error &boromir) {
        // Boromir never gets the ring
        assert(false);
    } catch (py::error_already_set &frodo) {
        // Frodo gets the ring
        py::print("I will take the ring");
    }

    try {
        // py::value_error is a request for pybind11 to raise a Python exception
        throw py::value_error("The ball");
    } catch (py::error_already_set &cat) {
        // cat won't catch the ball since
        // py::value_error is not a Python exception
        assert(false);
    } catch (py::value_error &dog) {
        // dog will catch the ball
        py::print("Run Spot run");
        throw;  // Throw it again (pybind11 will raise ValueError)
    }

Handling errors from the Python C API
=====================================

Where possible, use :ref:`pybind11 wrappers <wrappers>` instead of calling
the Python C API directly. When calling the Python C API directly, in
addition to manually managing reference counts, one must follow the pybind11
error protocol, which is outlined here.

After calling the Python C API, if Python returns an error,
``throw py::error_already_set();``, which allows pybind11 to deal with the
exception and pass it back to the Python interpreter. This includes calls to
the error setting functions such as ``py::set_error()``.

.. code-block:: cpp

    py::set_error(PyExc_TypeError, "C API type error demo");
    throw py::error_already_set();

    // But it would be easier to simply...
    throw py::type_error("pybind11 wrapper type error");

Alternately, to ignore the error, call `PyErr_Clear
<https://docs.python.org/3/c-api/exceptions.html#c.PyErr_Clear>`_.

Any Python error must be thrown or cleared, or Python/pybind11 will be left in
an invalid state.

Chaining exceptions ('raise from')
==================================

Python has a mechanism for indicating that exceptions were caused by other
exceptions:

.. code-block:: py

    try:
        print(1 / 0)
    except Exception as exc:
        raise RuntimeError("could not divide by zero") from exc

To do a similar thing in pybind11, you can use the ``py::raise_from`` function. It
sets the current python error indicator, so to continue propagating the exception
you should ``throw py::error_already_set()``.

.. code-block:: cpp

    try {
        py::eval("print(1 / 0"));
    } catch (py::error_already_set &e) {
        py::raise_from(e, PyExc_RuntimeError, "could not divide by zero");
        throw py::error_already_set();
    }

.. versionadded:: 2.8

.. _unraisable_exceptions:

Handling unraisable exceptions
==============================

If a Python function invoked from a C++ destructor or any function marked
``noexcept(true)`` (collectively, "noexcept functions") throws an exception, there
is no way to propagate the exception, as such functions may not throw.
Should they throw or fail to catch any exceptions in their call graph,
the C++ runtime calls ``std::terminate()`` to abort immediately.

Similarly, Python exceptions raised in a class's ``__del__`` method do not
propagate, but are logged by Python as an unraisable error. In Python 3.8+, a
`system hook is triggered
<https://docs.python.org/3/library/sys.html#sys.unraisablehook>`_
and an auditing event is logged.

Any noexcept function should have a try-catch block that traps
class:`error_already_set` (or any other exception that can occur). Note that
pybind11 wrappers around Python exceptions such as
:class:`pybind11::value_error` are *not* Python exceptions; they are C++
exceptions that pybind11 catches and converts to Python exceptions. Noexcept
functions cannot propagate these exceptions either. A useful approach is to
convert them to Python exceptions and then ``discard_as_unraisable`` as shown
below.

.. code-block:: cpp

    void nonthrowing_func() noexcept(true) {
        try {
            // ...
        } catch (py::error_already_set &eas) {
            // Discard the Python error using Python APIs, using the C++ magic
            // variable __func__. Python already knows the type and value and of the
            // exception object.
            eas.discard_as_unraisable(__func__);
        } catch (const std::exception &e) {
            // Log and discard C++ exceptions.
            third_party::log(e);
        }
    }

.. versionadded:: 2.6
