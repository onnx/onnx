.. _basics:

First steps
###########

This sections demonstrates the basic features of pybind11. Before getting
started, make sure that development environment is set up to compile the
included set of test cases.


Compiling the test cases
========================

Linux/macOS
-----------

On Linux  you'll need to install the **python-dev** or **python3-dev** packages as
well as **cmake**. On macOS, the included python version works out of the box,
but **cmake** must still be installed.

After installing the prerequisites, run

.. code-block:: bash

   mkdir build
   cd build
   cmake ..
   make check -j 4

The last line will both compile and run the tests.

Windows
-------

On Windows, only **Visual Studio 2017** and newer are supported.

.. Note::

    To use the C++17 in Visual Studio 2017 (MSVC 14.1), pybind11 requires the flag
    ``/permissive-`` to be passed to the compiler `to enforce standard conformance`_. When
    building with Visual Studio 2019, this is not strictly necessary, but still advised.

..  _`to enforce standard conformance`: https://docs.microsoft.com/en-us/cpp/build/reference/permissive-standards-conformance?view=vs-2017

To compile and run the tests:

.. code-block:: batch

   mkdir build
   cd build
   cmake ..
   cmake --build . --config Release --target check

This will create a Visual Studio project, compile and run the target, all from the
command line.

.. Note::

    If all tests fail, make sure that the Python binary and the testcases are compiled
    for the same processor type and bitness (i.e. either **i386** or **x86_64**). You
    can specify **x86_64** as the target architecture for the generated Visual Studio
    project using ``cmake -A x64 ..``.

.. seealso::

    Advanced users who are already familiar with Boost.Python may want to skip
    the tutorial and look at the test cases in the :file:`tests` directory,
    which exercise all features of pybind11.

Header and namespace conventions
================================

For brevity, all code examples assume that the following two lines are present:

.. code-block:: cpp

    #include <pybind11/pybind11.h>

    namespace py = pybind11;

.. note::

    ``pybind11/pybind11.h`` includes ``Python.h``, as such it must be the first file
    included in any source file or header for `the same reasons as Python.h`_.

.. _`the same reasons as Python.h`: https://docs.python.org/3/extending/extending.html#a-simple-example

Some features may require additional headers, but those will be specified as needed.

.. _simple_example:

Creating bindings for a simple function
=======================================

Let's start by creating Python bindings for an extremely simple function, which
adds two numbers and returns their result:

.. code-block:: cpp

    int add(int i, int j) {
        return i + j;
    }

For simplicity [#f1]_, we'll put both this function and the binding code into
a file named :file:`example.cpp` with the following contents:

.. code-block:: cpp

    #include <pybind11/pybind11.h>

    int add(int i, int j) {
        return i + j;
    }

    PYBIND11_MODULE(example, m) {
        m.doc() = "pybind11 example plugin"; // optional module docstring

        m.def("add", &add, "A function that adds two numbers");
    }

.. [#f1] In practice, implementation and binding code will generally be located
         in separate files.

The :func:`PYBIND11_MODULE` macro creates a function that will be called when an
``import`` statement is issued from within Python. The module name (``example``)
is given as the first macro argument (it should not be in quotes). The second
argument (``m``) defines a variable of type :class:`py::module_ <module>` which
is the main interface for creating bindings. The method :func:`module_::def`
generates binding code that exposes the ``add()`` function to Python.

.. note::

    Notice how little code was needed to expose our function to Python: all
    details regarding the function's parameters and return value were
    automatically inferred using template metaprogramming. This overall
    approach and the used syntax are borrowed from Boost.Python, though the
    underlying implementation is very different.

pybind11 is a header-only library, hence it is not necessary to link against
any special libraries and there are no intermediate (magic) translation steps.
On Linux, the above example can be compiled using the following command:

.. code-block:: bash

    $ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)

.. note::

    If you used :ref:`include_as_a_submodule` to get the pybind11 source, then
    use ``$(python3-config --includes) -Iextern/pybind11/include`` instead of
    ``$(python3 -m pybind11 --includes)`` in the above compilation, as
    explained in :ref:`building_manually`.

For more details on the required compiler flags on Linux and macOS, see
:ref:`building_manually`. For complete cross-platform compilation instructions,
refer to the :ref:`compiling` page.

The `python_example`_ and `cmake_example`_ repositories are also a good place
to start. They are both complete project examples with cross-platform build
systems. The only difference between the two is that `python_example`_ uses
Python's ``setuptools`` to build the module, while `cmake_example`_ uses CMake
(which may be preferable for existing C++ projects).

.. _python_example: https://github.com/pybind/python_example
.. _cmake_example: https://github.com/pybind/cmake_example

Building the above C++ code will produce a binary module file that can be
imported to Python. Assuming that the compiled module is located in the
current directory, the following interactive Python session shows how to
load and execute the example:

.. code-block:: pycon

    $ python
    Python 3.9.10 (main, Jan 15 2022, 11:48:04)
    [Clang 13.0.0 (clang-1300.0.29.3)] on darwin
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import example
    >>> example.add(1, 2)
    3
    >>>

.. _keyword_args:

Keyword arguments
=================

With a simple code modification, it is possible to inform Python about the
names of the arguments ("i" and "j" in this case).

.. code-block:: cpp

    m.def("add", &add, "A function which adds two numbers",
          py::arg("i"), py::arg("j"));

:class:`arg` is one of several special tag classes which can be used to pass
metadata into :func:`module_::def`. With this modified binding code, we can now
call the function using keyword arguments, which is a more readable alternative
particularly for functions taking many parameters:

.. code-block:: pycon

    >>> import example
    >>> example.add(i=1, j=2)
    3L

The keyword names also appear in the function signatures within the documentation.

.. code-block:: pycon

    >>> help(example)

    ....

    FUNCTIONS
        add(...)
            Signature : (i: int, j: int) -> int

            A function which adds two numbers

A shorter notation for named arguments is also available:

.. code-block:: cpp

    // regular notation
    m.def("add1", &add, py::arg("i"), py::arg("j"));
    // shorthand
    using namespace pybind11::literals;
    m.def("add2", &add, "i"_a, "j"_a);

The :var:`_a` suffix forms a C++11 literal which is equivalent to :class:`arg`.
Note that the literal operator must first be made visible with the directive
``using namespace pybind11::literals``. This does not bring in anything else
from the ``pybind11`` namespace except for literals.

.. _default_args:

Default arguments
=================

Suppose now that the function to be bound has default arguments, e.g.:

.. code-block:: cpp

    int add(int i = 1, int j = 2) {
        return i + j;
    }

Unfortunately, pybind11 cannot automatically extract these parameters, since they
are not part of the function's type information. However, they are simple to specify
using an extension of :class:`arg`:

.. code-block:: cpp

    m.def("add", &add, "A function which adds two numbers",
          py::arg("i") = 1, py::arg("j") = 2);

The default values also appear within the documentation.

.. code-block:: pycon

    >>> help(example)

    ....

    FUNCTIONS
        add(...)
            Signature : (i: int = 1, j: int = 2) -> int

            A function which adds two numbers

The shorthand notation is also available for default arguments:

.. code-block:: cpp

    // regular notation
    m.def("add1", &add, py::arg("i") = 1, py::arg("j") = 2);
    // shorthand
    m.def("add2", &add, "i"_a=1, "j"_a=2);

Exporting variables
===================

To expose a value from C++, use the ``attr`` function to register it in a
module as shown below. Built-in types and general objects (more on that later)
are automatically converted when assigned as attributes, and can be explicitly
converted using the function ``py::cast``.

.. code-block:: cpp

    PYBIND11_MODULE(example, m) {
        m.attr("the_answer") = 42;
        py::object world = py::cast("World");
        m.attr("what") = world;
    }

These are then accessible from Python:

.. code-block:: pycon

    >>> import example
    >>> example.the_answer
    42
    >>> example.what
    'World'

.. _supported_types:

Supported data types
====================

A large number of data types are supported out of the box and can be used
seamlessly as functions arguments, return values or with ``py::cast`` in general.
For a full overview, see the :doc:`advanced/cast/index` section.
