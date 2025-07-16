Frequently asked questions
##########################

"ImportError: dynamic module does not define init function"
===========================================================

1. Make sure that the name specified in PYBIND11_MODULE is identical to the
filename of the extension library (without suffixes such as ``.so``).

2. If the above did not fix the issue, you are likely using an incompatible
version of Python that does not match what you compiled with.

"Symbol not found: ``__Py_ZeroStruct`` / ``_PyInstanceMethod_Type``"
========================================================================

See the first answer.

"SystemError: dynamic module not initialized properly"
======================================================

See the first answer.

The Python interpreter immediately crashes when importing my module
===================================================================

See the first answer.

.. _faq_reference_arguments:

Limitations involving reference arguments
=========================================

In C++, it's fairly common to pass arguments using mutable references or
mutable pointers, which allows both read and write access to the value
supplied by the caller. This is sometimes done for efficiency reasons, or to
realize functions that have multiple return values. Here are two very basic
examples:

.. code-block:: cpp

    void increment(int &i) { i++; }
    void increment_ptr(int *i) { (*i)++; }

In Python, all arguments are passed by reference, so there is no general
issue in binding such code from Python.

However, certain basic Python types (like ``str``, ``int``, ``bool``,
``float``, etc.) are **immutable**. This means that the following attempt
to port the function to Python doesn't have the same effect on the value
provided by the caller -- in fact, it does nothing at all.

.. code-block:: python

    def increment(i):
        i += 1  # nope..

pybind11 is also affected by such language-level conventions, which means that
binding ``increment`` or ``increment_ptr`` will also create Python functions
that don't modify their arguments.

Although inconvenient, one workaround is to encapsulate the immutable types in
a custom type that does allow modifications.

An other alternative involves binding a small wrapper lambda function that
returns a tuple with all output arguments (see the remainder of the
documentation for examples on binding lambda functions). An example:

.. code-block:: cpp

    int foo(int &i) { i++; return 123; }

and the binding code

.. code-block:: cpp

   m.def("foo", [](int i) { int rv = foo(i); return std::make_tuple(rv, i); });


How can I reduce the build time?
================================

It's good practice to split binding code over multiple files, as in the
following example:

:file:`example.cpp`:

.. code-block:: cpp

    void init_ex1(py::module_ &);
    void init_ex2(py::module_ &);
    /* ... */

    PYBIND11_MODULE(example, m) {
        init_ex1(m);
        init_ex2(m);
        /* ... */
    }

:file:`ex1.cpp`:

.. code-block:: cpp

    void init_ex1(py::module_ &m) {
        m.def("add", [](int a, int b) { return a + b; });
    }

:file:`ex2.cpp`:

.. code-block:: cpp

    void init_ex2(py::module_ &m) {
        m.def("sub", [](int a, int b) { return a - b; });
    }

:command:`python`:

.. code-block:: pycon

    >>> import example
    >>> example.add(1, 2)
    3
    >>> example.sub(1, 1)
    0

As shown above, the various ``init_ex`` functions should be contained in
separate files that can be compiled independently from one another, and then
linked together into the same final shared object.  Following this approach
will:

1. reduce memory requirements per compilation unit.

2. enable parallel builds (if desired).

3. allow for faster incremental builds. For instance, when a single class
   definition is changed, only a subset of the binding code will generally need
   to be recompiled.

"recursive template instantiation exceeded maximum depth of 256"
================================================================

If you receive an error about excessive recursive template evaluation, try
specifying a larger value, e.g. ``-ftemplate-depth=1024`` on GCC/Clang. The
culprit is generally the generation of function signatures at compile time
using C++14 template metaprogramming.

.. _`faq:hidden_visibility`:

"'SomeClass' declared with greater visibility than the type of its field 'SomeClass::member' [-Wattributes]"
============================================================================================================

This error typically indicates that you are compiling without the required
``-fvisibility`` flag.  pybind11 code internally forces hidden visibility on
all internal code, but if non-hidden (and thus *exported*) code attempts to
include a pybind type (for example, ``py::object`` or ``py::list``) you can run
into this warning.

To avoid it, make sure you are specifying ``-fvisibility=hidden`` when
compiling pybind code.

As to why ``-fvisibility=hidden`` is necessary, because pybind modules could
have been compiled under different versions of pybind itself, it is also
important that the symbols defined in one module do not clash with the
potentially-incompatible symbols defined in another.  While Python extension
modules are usually loaded with localized symbols (under POSIX systems
typically using ``dlopen`` with the ``RTLD_LOCAL`` flag), this Python default
can be changed, but even if it isn't it is not always enough to guarantee
complete independence of the symbols involved when not using
``-fvisibility=hidden``.

Additionally, ``-fvisibility=hidden`` can deliver considerably binary size
savings. (See the following section for more details.)


.. _`faq:symhidden`:

How can I create smaller binaries?
==================================

To do its job, pybind11 extensively relies on a programming technique known as
*template metaprogramming*, which is a way of performing computation at compile
time using type information. Template metaprogramming usually instantiates code
involving significant numbers of deeply nested types that are either completely
removed or reduced to just a few instructions during the compiler's optimization
phase. However, due to the nested nature of these types, the resulting symbol
names in the compiled extension library can be extremely long. For instance,
the included test suite contains the following symbol:

.. only:: html

    .. code-block:: none

        _​_​Z​N​8​p​y​b​i​n​d​1​1​1​2​c​p​p​_​f​u​n​c​t​i​o​n​C​1​I​v​8​E​x​a​m​p​l​e​2​J​R​N​S​t​3​_​_​1​6​v​e​c​t​o​r​I​N​S​3​_​1​2​b​a​s​i​c​_​s​t​r​i​n​g​I​w​N​S​3​_​1​1​c​h​a​r​_​t​r​a​i​t​s​I​w​E​E​N​S​3​_​9​a​l​l​o​c​a​t​o​r​I​w​E​E​E​E​N​S​8​_​I​S​A​_​E​E​E​E​E​J​N​S​_​4​n​a​m​e​E​N​S​_​7​s​i​b​l​i​n​g​E​N​S​_​9​i​s​_​m​e​t​h​o​d​E​A​2​8​_​c​E​E​E​M​T​0​_​F​T​_​D​p​T​1​_​E​D​p​R​K​T​2​_

.. only:: not html

    .. code-block:: cpp

        __ZN8pybind1112cpp_functionC1Iv8Example2JRNSt3__16vectorINS3_12basic_stringIwNS3_11char_traitsIwEENS3_9allocatorIwEEEENS8_ISA_EEEEEJNS_4nameENS_7siblingENS_9is_methodEA28_cEEEMT0_FT_DpT1_EDpRKT2_

which is the mangled form of the following function type:

.. code-block:: cpp

    pybind11::cpp_function::cpp_function<void, Example2, std::__1::vector<std::__1::basic_string<wchar_t, std::__1::char_traits<wchar_t>, std::__1::allocator<wchar_t> >, std::__1::allocator<std::__1::basic_string<wchar_t, std::__1::char_traits<wchar_t>, std::__1::allocator<wchar_t> > > >&, pybind11::name, pybind11::sibling, pybind11::is_method, char [28]>(void (Example2::*)(std::__1::vector<std::__1::basic_string<wchar_t, std::__1::char_traits<wchar_t>, std::__1::allocator<wchar_t> >, std::__1::allocator<std::__1::basic_string<wchar_t, std::__1::char_traits<wchar_t>, std::__1::allocator<wchar_t> > > >&), pybind11::name const&, pybind11::sibling const&, pybind11::is_method const&, char const (&) [28])

The memory needed to store just the mangled name of this function (196 bytes)
is larger than the actual piece of code (111 bytes) it represents! On the other
hand, it's silly to even give this function a name -- after all, it's just a
tiny cog in a bigger piece of machinery that is not exposed to the outside
world. So we'll generally only want to export symbols for those functions which
are actually called from the outside.

This can be achieved by specifying the parameter ``-fvisibility=hidden`` to GCC
and Clang, which sets the default symbol visibility to *hidden*, which has a
tremendous impact on the final binary size of the resulting extension library.
(On Visual Studio, symbols are already hidden by default, so nothing needs to
be done there.)

In addition to decreasing binary size, ``-fvisibility=hidden`` also avoids
potential serious issues when loading multiple modules and is required for
proper pybind operation.  See the previous FAQ entry for more details.

How can I properly handle Ctrl-C in long-running functions?
===========================================================

Ctrl-C is received by the Python interpreter, and holds it until the GIL
is released, so a long-running function won't be interrupted.

To interrupt from inside your function, you can use the ``PyErr_CheckSignals()``
function, that will tell if a signal has been raised on the Python side.  This
function merely checks a flag, so its impact is negligible. When a signal has
been received, you must either explicitly interrupt execution by throwing
``py::error_already_set`` (which will propagate the existing
``KeyboardInterrupt``), or clear the error (which you usually will not want):

.. code-block:: cpp

    PYBIND11_MODULE(example, m)
    {
        m.def("long running_func", []()
        {
            for (;;) {
                if (PyErr_CheckSignals() != 0)
                    throw py::error_already_set();
                // Long running iteration
            }
        });
    }

What is a highly conclusive and simple way to find memory leaks (e.g. in pybind11 bindings)?
============================================================================================

Use ``while True`` & ``top`` (Linux, macOS).

For example, locally change tests/test_type_caster_pyobject_ptr.py like this:

.. code-block:: diff

     def test_return_list_pyobject_ptr_reference():
    +  while True:
         vec_obj = m.return_list_pyobject_ptr_reference(ValueHolder)
         assert [e.value for e in vec_obj] == [93, 186]
         # Commenting out the next `assert` will leak the Python references.
         # An easy way to see evidence of the leaks:
         # Insert `while True:` as the first line of this function and monitor the
         # process RES (Resident Memory Size) with the Unix top command.
    -    assert m.dec_ref_each_pyobject_ptr(vec_obj) == 2
    +    # assert m.dec_ref_each_pyobject_ptr(vec_obj) == 2

Then run the test as you would normally do, which will go into the infinite loop.

**In another shell, but on the same machine** run:

.. code-block:: bash

    top

This will show:

.. code-block::

        PID USER      PR  NI    VIRT    RES    SHR S  %CPU  %MEM     TIME+ COMMAND
    1266095 rwgk      20   0 5207496 611372  45696 R 100.0   0.3   0:08.01 test_type_caste

Look for the number under ``RES`` there. You'll see it going up very quickly.

**Don't forget to Ctrl-C the test command** before your machine becomes
unresponsive due to swapping.

This method only takes a couple minutes of effort and is very conclusive.
What you want to see is that the ``RES`` number is stable after a couple
seconds.

CMake doesn't detect the right Python version
=============================================

The CMake-based build system will try to automatically detect the installed
version of Python and link against that. When this fails, or when there are
multiple versions of Python and it finds the wrong one, delete
``CMakeCache.txt`` and then add ``-DPYTHON_EXECUTABLE=$(which python)`` to your
CMake configure line. (Replace ``$(which python)`` with a path to python if
your prefer.)

You can alternatively try ``-DPYBIND11_FINDPYTHON=ON``, which will activate the
new CMake FindPython support instead of pybind11's custom search. Requires
CMake 3.12+, and 3.15+ or 3.18.2+ are even better. You can set this in your
``CMakeLists.txt`` before adding or finding pybind11, as well.

Inconsistent detection of Python version in CMake and pybind11
==============================================================

The functions ``find_package(PythonInterp)`` and ``find_package(PythonLibs)``
provided by CMake for Python version detection are modified by pybind11 due to
unreliability and limitations that make them unsuitable for pybind11's needs.
Instead pybind11 provides its own, more reliable Python detection CMake code.
Conflicts can arise, however, when using pybind11 in a project that *also* uses
the CMake Python detection in a system with several Python versions installed.

This difference may cause inconsistencies and errors if *both* mechanisms are
used in the same project.

There are three possible solutions:

1. Avoid using ``find_package(PythonInterp)`` and ``find_package(PythonLibs)``
   from CMake and rely on pybind11 in detecting Python version. If this is not
   possible, the CMake machinery should be called *before* including pybind11.
2. Set ``PYBIND11_FINDPYTHON`` to ``True`` or use ``find_package(Python
   COMPONENTS Interpreter Development)`` on modern CMake (3.12+, 3.15+ better,
   3.18.2+ best). Pybind11 in these cases uses the new CMake FindPython instead
   of the old, deprecated search tools, and these modules are much better at
   finding the correct Python. If FindPythonLibs/Interp are not available
   (CMake 3.27+), then this will be ignored and FindPython will be used.
3. Set ``PYBIND11_NOPYTHON`` to ``TRUE``. Pybind11 will not search for Python.
   However, you will have to use the target-based system, and do more setup
   yourself, because it does not know about or include things that depend on
   Python, like ``pybind11_add_module``. This might be ideal for integrating
   into an existing system, like scikit-build's Python helpers.

How to cite this project?
=========================

We suggest the following BibTeX template to cite pybind11 in scientific
discourse:

.. code-block:: bash

    @misc{pybind11,
       author = {Wenzel Jakob and Jason Rhinelander and Dean Moldovan},
       year = {2017},
       note = {https://github.com/pybind/pybind11},
       title = {pybind11 -- Seamless operability between C++11 and Python}
    }
