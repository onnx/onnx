Miscellaneous
#############

.. _macro_notes:

General notes regarding convenience macros
==========================================

pybind11 provides a few convenience macros such as
:func:`PYBIND11_DECLARE_HOLDER_TYPE` and ``PYBIND11_OVERRIDE_*``. Since these
are "just" macros that are evaluated in the preprocessor (which has no concept
of types), they *will* get confused by commas in a template argument; for
example, consider:

.. code-block:: cpp

    PYBIND11_OVERRIDE(MyReturnType<T1, T2>, Class<T3, T4>, func)

The limitation of the C preprocessor interprets this as five arguments (with new
arguments beginning after each comma) rather than three.  To get around this,
there are two alternatives: you can use a type alias, or you can wrap the type
using the ``PYBIND11_TYPE`` macro:

.. code-block:: cpp

    // Version 1: using a type alias
    using ReturnType = MyReturnType<T1, T2>;
    using ClassType = Class<T3, T4>;
    PYBIND11_OVERRIDE(ReturnType, ClassType, func);

    // Version 2: using the PYBIND11_TYPE macro:
    PYBIND11_OVERRIDE(PYBIND11_TYPE(MyReturnType<T1, T2>),
                      PYBIND11_TYPE(Class<T3, T4>), func)

The ``PYBIND11_MAKE_OPAQUE`` macro does *not* require the above workarounds.

.. _gil:

Global Interpreter Lock (GIL)
=============================

The Python C API dictates that the Global Interpreter Lock (GIL) must always
be held by the current thread to safely access Python objects. As a result,
when Python calls into C++ via pybind11 the GIL must be held, and pybind11
will never implicitly release the GIL.

.. code-block:: cpp

    void my_function() {
        /* GIL is held when this function is called from Python */
    }

    PYBIND11_MODULE(example, m) {
        m.def("my_function", &my_function);
    }

pybind11 will ensure that the GIL is held when it knows that it is calling
Python code. For example, if a Python callback is passed to C++ code via
``std::function``, when C++ code calls the function the built-in wrapper
will acquire the GIL before calling the Python callback. Similarly, the
``PYBIND11_OVERRIDE`` family of macros will acquire the GIL before calling
back into Python.

When writing C++ code that is called from other C++ code, if that code accesses
Python state, it must explicitly acquire and release the GIL.

The classes :class:`gil_scoped_release` and :class:`gil_scoped_acquire` can be
used to acquire and release the global interpreter lock in the body of a C++
function call. In this way, long-running C++ code can be parallelized using
multiple Python threads, **but great care must be taken** when any
:class:`gil_scoped_release` appear: if there is any way that the C++ code
can access Python objects, :class:`gil_scoped_acquire` should be used to
reacquire the GIL. Taking :ref:`overriding_virtuals` as an example, this
could be realized as follows (important changes highlighted):

.. code-block:: cpp
    :emphasize-lines: 8,30,31

    class PyAnimal : public Animal {
    public:
        /* Inherit the constructors */
        using Animal::Animal;

        /* Trampoline (need one for each virtual function) */
        std::string go(int n_times) {
            /* PYBIND11_OVERRIDE_PURE will acquire the GIL before accessing Python state */
            PYBIND11_OVERRIDE_PURE(
                std::string, /* Return type */
                Animal,      /* Parent class */
                go,          /* Name of function */
                n_times      /* Argument(s) */
            );
        }
    };

    PYBIND11_MODULE(example, m) {
        py::class_<Animal, PyAnimal> animal(m, "Animal");
        animal
            .def(py::init<>())
            .def("go", &Animal::go);

        py::class_<Dog>(m, "Dog", animal)
            .def(py::init<>());

        m.def("call_go", [](Animal *animal) -> std::string {
            // GIL is held when called from Python code. Release GIL before
            // calling into (potentially long-running) C++ code
            py::gil_scoped_release release;
            return call_go(animal);
        });
    }

The ``call_go`` wrapper can also be simplified using the ``call_guard`` policy
(see :ref:`call_policies`) which yields the same result:

.. code-block:: cpp

    m.def("call_go", &call_go, py::call_guard<py::gil_scoped_release>());


Common Sources Of Global Interpreter Lock Errors
==================================================================

Failing to properly hold the Global Interpreter Lock (GIL) is one of the
more common sources of bugs within code that uses pybind11. If you are
running into GIL related errors, we highly recommend you consult the
following checklist.

- Do you have any global variables that are pybind11 objects or invoke
  pybind11 functions in either their constructor or destructor? You are generally
  not allowed to invoke any Python function in a global static context. We recommend
  using lazy initialization and then intentionally leaking at the end of the program.

- Do you have any pybind11 objects that are members of other C++ structures? One
  commonly overlooked requirement is that pybind11 objects have to increase their reference count
  whenever their copy constructor is called. Thus, you need to be holding the GIL to invoke
  the copy constructor of any C++ class that has a pybind11 member. This can sometimes be very
  tricky to track for complicated programs Think carefully when you make a pybind11 object
  a member in another struct.

- C++ destructors that invoke Python functions can be particularly troublesome as
  destructors can sometimes get invoked in weird and unexpected circumstances as a result
  of exceptions.

- You should try running your code in a debug build. That will enable additional assertions
  within pybind11 that will throw exceptions on certain GIL handling errors
  (reference counting operations).

Binding sequence data types, iterators, the slicing protocol, etc.
==================================================================

Please refer to the supplemental example for details.

.. seealso::

    The file :file:`tests/test_sequences_and_iterators.cpp` contains a
    complete example that shows how to bind a sequence data type, including
    length queries (``__len__``), iterators (``__iter__``), the slicing
    protocol and other kinds of useful operations.


Partitioning code over multiple extension modules
=================================================

It's straightforward to split binding code over multiple extension modules,
while referencing types that are declared elsewhere. Everything "just" works
without any special precautions. One exception to this rule occurs when
extending a type declared in another extension module. Recall the basic example
from Section :ref:`inheritance`.

.. code-block:: cpp

    py::class_<Pet> pet(m, "Pet");
    pet.def(py::init<const std::string &>())
       .def_readwrite("name", &Pet::name);

    py::class_<Dog>(m, "Dog", pet /* <- specify parent */)
        .def(py::init<const std::string &>())
        .def("bark", &Dog::bark);

Suppose now that ``Pet`` bindings are defined in a module named ``basic``,
whereas the ``Dog`` bindings are defined somewhere else. The challenge is of
course that the variable ``pet`` is not available anymore though it is needed
to indicate the inheritance relationship to the constructor of ``class_<Dog>``.
However, it can be acquired as follows:

.. code-block:: cpp

    py::object pet = (py::object) py::module_::import("basic").attr("Pet");

    py::class_<Dog>(m, "Dog", pet)
        .def(py::init<const std::string &>())
        .def("bark", &Dog::bark);

Alternatively, you can specify the base class as a template parameter option to
``class_``, which performs an automated lookup of the corresponding Python
type. Like the above code, however, this also requires invoking the ``import``
function once to ensure that the pybind11 binding code of the module ``basic``
has been executed:

.. code-block:: cpp

    py::module_::import("basic");

    py::class_<Dog, Pet>(m, "Dog")
        .def(py::init<const std::string &>())
        .def("bark", &Dog::bark);

Naturally, both methods will fail when there are cyclic dependencies.

Note that pybind11 code compiled with hidden-by-default symbol visibility (e.g.
via the command line flag ``-fvisibility=hidden`` on GCC/Clang), which is
required for proper pybind11 functionality, can interfere with the ability to
access types defined in another extension module.  Working around this requires
manually exporting types that are accessed by multiple extension modules;
pybind11 provides a macro to do just this:

.. code-block:: cpp

    class PYBIND11_EXPORT Dog : public Animal {
        ...
    };

Note also that it is possible (although would rarely be required) to share arbitrary
C++ objects between extension modules at runtime. Internal library data is shared
between modules using capsule machinery [#f6]_ which can be also utilized for
storing, modifying and accessing user-defined data. Note that an extension module
will "see" other extensions' data if and only if they were built with the same
pybind11 version. Consider the following example:

.. code-block:: cpp

    auto data = reinterpret_cast<MyData *>(py::get_shared_data("mydata"));
    if (!data)
        data = static_cast<MyData *>(py::set_shared_data("mydata", new MyData(42)));

If the above snippet was used in several separately compiled extension modules,
the first one to be imported would create a ``MyData`` instance and associate
a ``"mydata"`` key with a pointer to it. Extensions that are imported later
would be then able to access the data behind the same pointer.

.. [#f6] https://docs.python.org/3/extending/extending.html#using-capsules

Module Destructors
==================

pybind11 does not provide an explicit mechanism to invoke cleanup code at
module destruction time. In rare cases where such functionality is required, it
is possible to emulate it using Python capsules or weak references with a
destruction callback.

.. code-block:: cpp

    auto cleanup_callback = []() {
        // perform cleanup here -- this function is called with the GIL held
    };

    m.add_object("_cleanup", py::capsule(cleanup_callback));

This approach has the potential downside that instances of classes exposed
within the module may still be alive when the cleanup callback is invoked
(whether this is acceptable will generally depend on the application).

Alternatively, the capsule may also be stashed within a type object, which
ensures that it not called before all instances of that type have been
collected:

.. code-block:: cpp

    auto cleanup_callback = []() { /* ... */ };
    m.attr("BaseClass").attr("_cleanup") = py::capsule(cleanup_callback);

Both approaches also expose a potentially dangerous ``_cleanup`` attribute in
Python, which may be undesirable from an API standpoint (a premature explicit
call from Python might lead to undefined behavior). Yet another approach that
avoids this issue involves weak reference with a cleanup callback:

.. code-block:: cpp

    // Register a callback function that is invoked when the BaseClass object is collected
    py::cpp_function cleanup_callback(
        [](py::handle weakref) {
            // perform cleanup here -- this function is called with the GIL held

            weakref.dec_ref(); // release weak reference
        }
    );

    // Create a weak reference with a cleanup callback and initially leak it
    (void) py::weakref(m.attr("BaseClass"), cleanup_callback).release();

.. note::

    PyPy does not garbage collect objects when the interpreter exits. An alternative
    approach (which also works on CPython) is to use the :py:mod:`atexit` module [#f7]_,
    for example:

    .. code-block:: cpp

        auto atexit = py::module_::import("atexit");
        atexit.attr("register")(py::cpp_function([]() {
            // perform cleanup here -- this function is called with the GIL held
        }));

    .. [#f7] https://docs.python.org/3/library/atexit.html


Generating documentation using Sphinx
=====================================

Sphinx [#f4]_ has the ability to inspect the signatures and documentation
strings in pybind11-based extension modules to automatically generate beautiful
documentation in a variety formats. The python_example repository [#f5]_ contains a
simple example repository which uses this approach.

There are two potential gotchas when using this approach: first, make sure that
the resulting strings do not contain any :kbd:`TAB` characters, which break the
docstring parsing routines. You may want to use C++11 raw string literals,
which are convenient for multi-line comments. Conveniently, any excess
indentation will be automatically be removed by Sphinx. However, for this to
work, it is important that all lines are indented consistently, i.e.:

.. code-block:: cpp

    // ok
    m.def("foo", &foo, R"mydelimiter(
        The foo function

        Parameters
        ----------
    )mydelimiter");

    // *not ok*
    m.def("foo", &foo, R"mydelimiter(The foo function

        Parameters
        ----------
    )mydelimiter");

By default, pybind11 automatically generates and prepends a signature to the docstring of a function
registered with ``module_::def()`` and ``class_::def()``. Sometimes this
behavior is not desirable, because you want to provide your own signature or remove
the docstring completely to exclude the function from the Sphinx documentation.
The class ``options`` allows you to selectively suppress auto-generated signatures:

.. code-block:: cpp

    PYBIND11_MODULE(example, m) {
        py::options options;
        options.disable_function_signatures();

        m.def("add", [](int a, int b) { return a + b; }, "A function which adds two numbers");
    }

pybind11 also appends all members of an enum to the resulting enum docstring.
This default behavior can be disabled by using the ``disable_enum_members_docstring()``
function of the ``options`` class.

With ``disable_user_defined_docstrings()`` all user defined docstrings of
``module_::def()``, ``class_::def()`` and ``enum_()`` are disabled, but the
function signatures and enum members are included in the docstring, unless they
are disabled separately.

Note that changes to the settings affect only function bindings created during the
lifetime of the ``options`` instance. When it goes out of scope at the end of the module's init function,
the default settings are restored to prevent unwanted side effects.

.. [#f4] http://www.sphinx-doc.org
.. [#f5] http://github.com/pybind/python_example

.. _avoiding-cpp-types-in-docstrings:

Avoiding C++ types in docstrings
================================

Docstrings are generated at the time of the declaration, e.g. when ``.def(...)`` is called.
At this point parameter and return types should be known to pybind11.
If a custom type is not exposed yet through a ``py::class_`` constructor or a custom type caster,
its C++ type name will be used instead to generate the signature in the docstring:

.. code-block:: text

     |  __init__(...)
     |      __init__(self: example.Foo, arg0: ns::Bar) -> None
                                              ^^^^^^^


This limitation can be circumvented by ensuring that C++ classes are registered with pybind11
before they are used as a parameter or return type of a function:

.. code-block:: cpp

    PYBIND11_MODULE(example, m) {

        auto pyFoo = py::class_<ns::Foo>(m, "Foo");
        auto pyBar = py::class_<ns::Bar>(m, "Bar");

        pyFoo.def(py::init<const ns::Bar&>());
        pyBar.def(py::init<const ns::Foo&>());
    }

Setting inner type hints in docstrings
======================================

When you use pybind11 wrappers for ``list``, ``dict``, and other generic python
types, the docstring will just display the generic type. You can convey the
inner types in the docstring by using a special 'typed' version of the generic
type.

.. code-block:: cpp

    PYBIND11_MODULE(example, m) {
        m.def("pass_list_of_str", [](py::typing::List<py::str> arg) {
            // arg can be used just like py::list
        ));
    }

The resulting docstring will be ``pass_list_of_str(arg0: list[str]) -> None``.

The following special types are available in ``pybind11/typing.h``:

* ``py::Tuple<Args...>``
* ``py::Dict<K, V>``
* ``py::List<V>``
* ``py::Set<V>``
* ``py::Callable<Signature>``

.. warning:: Just like in python, these are merely hints. They don't actually
             enforce the types of their contents at runtime or compile time.
