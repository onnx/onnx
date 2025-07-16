.. _compiling:

Build systems
#############

For an overview of Python packaging including compiled packaging with a pybind11
example, along with a cookiecutter that includes several pybind11 options, see
the `Scientific Python Development Guide`_.

.. _Scientific Python Development Guide: https://learn.scientific-python.org/development/guides/packaging-compiled/

.. scikit-build-core:

Modules with CMake
==================

A Python extension module can be created with just a few lines of code:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.15...3.29)
    project(example LANGUAGES CXX)

    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(example example.cpp)
    install(TARGETS example DESTINATION .)

(You use the ``add_subdirectory`` instead, see the example in :ref:`cmake`.) In
this example, the code is located in a file named :file:`example.cpp`.  Either
method will import the pybind11 project which provides the
``pybind11_add_module`` function. It will take care of all the details needed
to build a Python extension module on any platform.

To build with pip, build, cibuildwheel, uv, or other Python tools, you can
add a ``pyproject.toml`` file like this:

.. code-block:: toml

    [build-system]
    requires = ["scikit-build-core", "pybind11"]
    build-backend = "scikit_build_core.build"

    [project]
    name = "example"
    version = "0.1.0"

You don't need setuptools files like ``MANIFEST.in``, ``setup.py``, or
``setup.cfg``, as this is not setuptools. See `scikit-build-core`_ for details.
For projects you plan to upload to PyPI, be sure to fill out the ``[project]``
table with other important metadata as well (see `Writing pyproject.toml`_).

A working sample project can be found in the [scikit_build_example]_
repository. An older and harder-to-maintain method is in [cmake_example]_. More
details about our cmake support can be found below in :ref:`cmake`.

.. _scikit-build-core: https://scikit-build-core.readthedocs.io

.. [scikit_build_example] https://github.com/pybind/scikit_build_example

.. [cmake_example] https://github.com/pybind/cmake_example

.. _modules-meson-python:

Modules with meson-python
=========================

You can also build a package with `Meson`_ using `meson-python`_, if you prefer
that. Your ``meson.build`` file would look something like this:

.. _meson-example:

.. code-block:: meson

   project(
       'example',
       'cpp',
       version: '0.1.0',
       default_options: [
           'cpp_std=c++11',
       ],
   )

   py = import('python').find_installation(pure: false)
   pybind11_dep = dependency('pybind11')

   py.extension_module('example',
       'example.cpp',
       install: true,
       dependencies : [pybind11_dep],
   )


And you would need a ``pyproject.toml`` file like this:

.. code-block:: toml

   [build-system]
   requires = ["meson-python", "pybind11"]
   build-backend = "mesonpy"

Meson-python *requires* your project to be in git (or mercurial) as it uses it
for the SDist creation. For projects you plan to upload to PyPI, be sure to fill out the
``[project]`` table as well (see `Writing pyproject.toml`_).


.. _Writing pyproject.toml: https://packaging.python.org/en/latest/guides/writing-pyproject-toml

.. _meson: https://mesonbuild.com

.. _meson-python: https://meson-python.readthedocs.io/en/latest

.. _build-setuptools:

Modules with setuptools
=======================

For projects on PyPI, a historically popular option is setuptools. Sylvain
Corlay has kindly provided an example project which shows how to set up
everything, including automatic generation of documentation using Sphinx.
Please refer to the [python_example]_ repository.

.. [python_example] https://github.com/pybind/python_example

A helper file is provided with pybind11 that can simplify usage with setuptools.

To use pybind11 inside your ``setup.py``, you have to have some system to
ensure that ``pybind11`` is installed when you build your package. There are
four possible ways to do this, and pybind11 supports all four: You can ask all
users to install pybind11 beforehand (bad), you can use
:ref:`setup_helpers-pep518` (good), ``setup_requires=`` (discouraged), or you
can :ref:`setup_helpers-copy-manually` (works but you have to manually sync
your copy to get updates). Third party packagers like conda-forge generally
strongly prefer the ``pyproject.toml`` method, as it gives them control over
the ``pybind11`` version, and they may apply patches, etc.

An example of a ``setup.py`` using pybind11's helpers:

.. code-block:: python

    from glob import glob
    from setuptools import setup
    from pybind11.setup_helpers import Pybind11Extension

    ext_modules = [
        Pybind11Extension(
            "python_example",
            sorted(glob("src/*.cpp")),  # Sort source files for reproducibility
        ),
    ]

    setup(..., ext_modules=ext_modules)

If you want to do an automatic search for the highest supported C++ standard,
that is supported via a ``build_ext`` command override; it will only affect
``Pybind11Extensions``:

.. code-block:: python

    from glob import glob
    from setuptools import setup
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    ext_modules = [
        Pybind11Extension(
            "python_example",
            sorted(glob("src/*.cpp")),
        ),
    ]

    setup(..., cmdclass={"build_ext": build_ext}, ext_modules=ext_modules)

If you have single-file extension modules that are directly stored in the
Python source tree (``foo.cpp`` in the same directory as where a ``foo.py``
would be located), you can also generate ``Pybind11Extensions`` using
``setup_helpers.intree_extensions``: ``intree_extensions(["path/to/foo.cpp",
...])`` returns a list of ``Pybind11Extensions`` which can be passed to
``ext_modules``, possibly after further customizing their attributes
(``libraries``, ``include_dirs``, etc.).  By doing so, a ``foo.*.so`` extension
module will be generated and made available upon installation.

``intree_extension`` will automatically detect if you are using a ``src``-style
layout (as long as no namespace packages are involved), but you can also
explicitly pass ``package_dir`` to it (as in ``setuptools.setup``).

Since pybind11 does not require NumPy when building, a light-weight replacement
for NumPy's parallel compilation distutils tool is included. Use it like this:

.. code-block:: python

    from pybind11.setup_helpers import ParallelCompile

    # Optional multithreaded build
    ParallelCompile("NPY_NUM_BUILD_JOBS").install()

    setup(...)

The argument is the name of an environment variable to control the number of
threads, such as ``NPY_NUM_BUILD_JOBS`` (as used by NumPy), though you can set
something different if you want; ``CMAKE_BUILD_PARALLEL_LEVEL`` is another choice
a user might expect. You can also pass ``default=N`` to set the default number
of threads (0 will take the number of threads available) and ``max=N``, the
maximum number of threads; if you have a large extension you may want set this
to a memory dependent number.

If you are developing rapidly and have a lot of C++ files, you may want to
avoid rebuilding files that have not changed. For simple cases were you are
using ``pip install -e .`` and do not have local headers, you can skip the
rebuild if an object file is newer than its source (headers are not checked!)
with the following:

.. code-block:: python

    from pybind11.setup_helpers import ParallelCompile, naive_recompile

    ParallelCompile("NPY_NUM_BUILD_JOBS", needs_recompile=naive_recompile).install()


If you have a more complex build, you can implement a smarter function and pass
it to ``needs_recompile``, or you can use [Ccache]_ instead. ``CXX="cache g++"
pip install -e .`` would be the way to use it with GCC, for example. Unlike the
simple solution, this even works even when not compiling in editable mode, but
it does require Ccache to be installed.

Keep in mind that Pip will not even attempt to rebuild if it thinks it has
already built a copy of your code, which it deduces from the version number.
One way to avoid this is to use [setuptools_scm]_, which will generate a
version number that includes the number of commits since your last tag and a
hash for a dirty directory. Another way to force a rebuild is purge your cache
or use Pip's ``--no-cache-dir`` option.

You also need a ``MANIFEST.in`` file to include all relevant files so that you
can make an SDist. If you use `pypa-build`_, that will build an SDist then a
wheel from that SDist by default, so you can look inside those files (wheels
are just zip files with a ``.whl`` extension) to make sure you aren't missing
files.  `check-manifest`_ (setuptools specific) or `check-sdist`_ (general) are
CLI tools that can compare the SDist contents with your source control.

.. [Ccache] https://ccache.dev

.. [setuptools_scm] https://github.com/pypa/setuptools_scm

.. _setup_helpers-pep518:

Build requirements
------------------

With a ``pyproject.toml`` file, you can ensure that ``pybind11`` is available
during the compilation of your project.  When this file exists, Pip will make a
new virtual environment, download just the packages listed here in
``requires=``, and build a wheel (binary Python package). It will then throw
away the environment, and install your wheel.

Your ``pyproject.toml`` file will likely look something like this:

.. code-block:: toml

    [build-system]
    requires = ["setuptools", "pybind11"]
    build-backend = "setuptools.build_meta"

.. _PEP 517: https://www.python.org/dev/peps/pep-0517/
.. _cibuildwheel: https://cibuildwheel.pypa.io
.. _pypa-build: https://build.pypa.io/en/latest/
.. _check-manifest: https://pypi.io/project/check-manifest
.. _check-sdist: https://pypi.io/project/check-sdist

.. _setup_helpers-copy-manually:

Copy manually
-------------

You can also copy ``setup_helpers.py`` directly to your project; it was
designed to be usable standalone, like the old example ``setup.py``. You can
set ``include_pybind11=False`` to skip including the pybind11 package headers,
so you can use it with git submodules and a specific git version. If you use
this, you will need to import from a local file in ``setup.py`` and ensure the
helper file is part of your MANIFEST.


Closely related, if you include pybind11 as a subproject, you can run the
``setup_helpers.py`` inplace. If loaded correctly, this should even pick up
the correct include for pybind11, though you can turn it off as shown above if
you want to input it manually.

Suggested usage if you have pybind11 as a submodule in ``extern/pybind11``:

.. code-block:: python

    DIR = os.path.abspath(os.path.dirname(__file__))

    sys.path.append(os.path.join(DIR, "extern", "pybind11"))
    from pybind11.setup_helpers import Pybind11Extension  # noqa: E402

    del sys.path[-1]


.. versionchanged:: 2.6

    Added ``setup_helpers`` file.

Building with cppimport
========================

[cppimport]_ is a small Python import hook that determines whether there is a C++
source file whose name matches the requested module. If there is, the file is
compiled as a Python extension using pybind11 and placed in the same folder as
the C++ source file. Python is then able to find the module and load it.

.. [cppimport] https://github.com/tbenthompson/cppimport



.. _cmake:

Building with CMake
===================

For C++ codebases that have an existing CMake-based build system, a Python
extension module can be created with just a few lines of code, as seen above in
the module section. Pybind11 currently supports a lower minimum if you don't
use the modern FindPython, though be aware that CMake 3.27 removed the old
mechanism, so pybind11 will automatically switch if the old mechanism is not
available. Please opt into the new mechanism if at all possible. Our default
may change in future versions. This is the minimum required:



.. versionchanged:: 2.6
   CMake 3.4+ is required.

.. versionchanged:: 2.11
   CMake 3.5+ is required.


Further information can be found at :doc:`cmake/index`.

pybind11_add_module
-------------------

To ease the creation of Python extension modules, pybind11 provides a CMake
function with the following signature:

.. code-block:: cmake

    pybind11_add_module(<name> [MODULE | SHARED] [EXCLUDE_FROM_ALL]
                        [NO_EXTRAS] [THIN_LTO] [OPT_SIZE] source1 [source2 ...])

This function behaves very much like CMake's builtin ``add_library`` (in fact,
it's a wrapper function around that command). It will add a library target
called ``<name>`` to be built from the listed source files. In addition, it
will take care of all the Python-specific compiler and linker flags as well
as the OS- and Python-version-specific file extension. The produced target
``<name>`` can be further manipulated with regular CMake commands.

``MODULE`` or ``SHARED`` may be given to specify the type of library. If no
type is given, ``MODULE`` is used by default which ensures the creation of a
Python-exclusive module. Specifying ``SHARED`` will create a more traditional
dynamic library which can also be linked from elsewhere. ``EXCLUDE_FROM_ALL``
removes this target from the default build (see CMake docs for details).

Since pybind11 is a template library, ``pybind11_add_module`` adds compiler
flags to ensure high quality code generation without bloat arising from long
symbol names and duplication of code in different translation units. It
sets default visibility to *hidden*, which is required for some pybind11
features and functionality when attempting to load multiple pybind11 modules
compiled under different pybind11 versions.  It also adds additional flags
enabling LTO (Link Time Optimization) and strip unneeded symbols. See the
:ref:`FAQ entry <faq:symhidden>` for a more detailed explanation. These
latter optimizations are never applied in ``Debug`` mode.  If ``NO_EXTRAS`` is
given, they will always be disabled, even in ``Release`` mode. However, this
will result in code bloat and is generally not recommended.

As stated above, LTO is enabled by default. Some newer compilers also support
different flavors of LTO such as `ThinLTO`_. Setting ``THIN_LTO`` will cause
the function to prefer this flavor if available. The function falls back to
regular LTO if ``-flto=thin`` is not available. If
``CMAKE_INTERPROCEDURAL_OPTIMIZATION`` is set (either ``ON`` or ``OFF``), then
that will be respected instead of the built-in flag search.

.. note::

   If you want to set the property form on targets or the
   ``CMAKE_INTERPROCEDURAL_OPTIMIZATION_<CONFIG>`` versions of this, you should
   still use ``set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)`` (otherwise a
   no-op) to disable pybind11's ipo flags.

The ``OPT_SIZE`` flag enables size-based optimization equivalent to the
standard ``/Os`` or ``-Os`` compiler flags and the ``MinSizeRel`` build type,
which avoid optimizations that can substantially increase the size of the
resulting binary. This flag is particularly useful in projects that are split
into performance-critical parts and associated bindings. In this case, we can
compile the project in release mode (and hence, optimize performance globally),
and specify ``OPT_SIZE`` for the binding target, where size might be the main
concern as performance is often less critical here. A ~25% size reduction has
been observed in practice. This flag only changes the optimization behavior at
a per-target level and takes precedence over the global CMake build type
(``Release``, ``RelWithDebInfo``) except for ``Debug`` builds, where
optimizations remain disabled.

.. _ThinLTO: http://clang.llvm.org/docs/ThinLTO.html

Configuration variables
-----------------------

By default, pybind11 will compile modules with the compiler default or the
minimum standard required by pybind11, whichever is higher.  You can set the
standard explicitly with
`CMAKE_CXX_STANDARD <https://cmake.org/cmake/help/latest/variable/CMAKE_CXX_STANDARD.html>`_:

.. code-block:: cmake

    set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
    set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
    set(CMAKE_CXX_EXTENSIONS OFF)  # optional, keep compiler extensions off

The variables can also be set when calling CMake from the command line using
the ``-D<variable>=<value>`` flag. You can also manually set ``CXX_STANDARD``
on a target or use ``target_compile_features`` on your targets - anything that
CMake supports.

Classic Python support: The target Python version can be selected by setting
``PYBIND11_PYTHON_VERSION`` or an exact Python installation can be specified
with ``PYTHON_EXECUTABLE``.  For example:

.. code-block:: bash

    cmake -DPYBIND11_PYTHON_VERSION=3.7 ..

    # Another method:
    cmake -DPYTHON_EXECUTABLE=/path/to/python ..

    # This often is a good way to get the current Python, works in environments:
    cmake -DPYTHON_EXECUTABLE=$(python3 -c "import sys; print(sys.executable)") ..


find_package vs. add_subdirectory
---------------------------------

For CMake-based projects that don't include the pybind11 repository internally,
an external installation can be detected through ``find_package(pybind11)``.
See the `Config file`_ docstring for details of relevant CMake variables.

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.4...3.18)
    project(example LANGUAGES CXX)

    find_package(pybind11 REQUIRED)
    pybind11_add_module(example example.cpp)

Note that ``find_package(pybind11)`` will only work correctly if pybind11
has been correctly installed on the system, e. g. after downloading or cloning
the pybind11 repository  :

.. code-block:: bash

    # Classic CMake
    cd pybind11
    mkdir build
    cd build
    cmake ..
    make install

    # CMake 3.15+
    cd pybind11
    cmake -S . -B build
    cmake --build build -j 2  # Build on 2 cores
    cmake --install build

Once detected, the aforementioned ``pybind11_add_module`` can be employed as
before. The function usage and configuration variables are identical no matter
if pybind11 is added as a subdirectory or found as an installed package. You
can refer to the same [cmake_example]_ repository for a full sample project
-- just swap out ``add_subdirectory`` for ``find_package``.

.. _Config file: https://github.com/pybind/pybind11/blob/master/tools/pybind11Config.cmake.in


.. _find-python-mode:

FindPython mode
---------------

CMake 3.12+ (3.15+ recommended, 3.18.2+ ideal) added a new module called
FindPython that had a highly improved search algorithm and modern targets
and tools. If you use FindPython, pybind11 will detect this and use the
existing targets instead:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.15...3.22)
    project(example LANGUAGES CXX)

    find_package(Python 3.7 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    # or add_subdirectory(pybind11)

    pybind11_add_module(example example.cpp)

You can also use the targets (as listed below) with FindPython. If you define
``PYBIND11_FINDPYTHON``, pybind11 will perform the FindPython step for you
(mostly useful when building pybind11's own tests, or as a way to change search
algorithms from the CMake invocation, with ``-DPYBIND11_FINDPYTHON=ON``.

.. warning::

    If you use FindPython to multi-target Python versions, use the individual
    targets listed below, and avoid targets that directly include Python parts.

There are `many ways to hint or force a discovery of a specific Python
installation <https://cmake.org/cmake/help/latest/module/FindPython.html>`_),
setting ``Python_ROOT_DIR`` may be the most common one (though with
virtualenv/venv support, and Conda support, this tends to find the correct
Python version more often than the old system did).

.. warning::

    When the Python libraries (i.e. ``libpythonXX.a`` and ``libpythonXX.so``
    on Unix) are not available, as is the case on a manylinux image, the
    ``Development`` component will not be resolved by ``FindPython``. When not
    using the embedding functionality, CMake 3.18+ allows you to specify
    ``Development.Module`` instead of ``Development`` to resolve this issue.

.. versionadded:: 2.6

Advanced: interface library targets
-----------------------------------

Pybind11 supports modern CMake usage patterns with a set of interface targets,
available in all modes. The targets provided are:

   ``pybind11::headers``
     Just the pybind11 headers and minimum compile requirements

   ``pybind11::pybind11``
     Python headers + ``pybind11::headers``

   ``pybind11::python_link_helper``
     Just the "linking" part of pybind11:module

   ``pybind11::module``
     Everything for extension modules - ``pybind11::pybind11`` + ``Python::Module`` (FindPython CMake 3.15+) or ``pybind11::python_link_helper``

   ``pybind11::embed``
     Everything for embedding the Python interpreter - ``pybind11::pybind11`` + ``Python::Python`` (FindPython) or Python libs

   ``pybind11::lto`` / ``pybind11::thin_lto``
     An alternative to `INTERPROCEDURAL_OPTIMIZATION` for adding link-time optimization.

   ``pybind11::windows_extras``
     ``/bigobj`` and ``/mp`` for MSVC.

   ``pybind11::opt_size``
     ``/Os`` for MSVC, ``-Os`` for other compilers. Does nothing for debug builds.

Two helper functions are also provided:

    ``pybind11_strip(target)``
      Strips a target (uses ``CMAKE_STRIP`` after the target is built)

    ``pybind11_extension(target)``
      Sets the correct extension (with SOABI) for a target.

You can use these targets to build complex applications. For example, the
``add_python_module`` function is identical to:

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.5...3.29)
    project(example LANGUAGES CXX)

    find_package(pybind11 REQUIRED)  # or add_subdirectory(pybind11)

    add_library(example MODULE main.cpp)

    target_link_libraries(example PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)

    pybind11_extension(example)
    if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
        # Strip unnecessary sections of the binary on Linux/macOS
        pybind11_strip(example)
    endif()

    set_target_properties(example PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                             CUDA_VISIBILITY_PRESET "hidden")

Instead of setting properties, you can set ``CMAKE_*`` variables to initialize these correctly.

.. warning::

    Since pybind11 is a metatemplate library, it is crucial that certain
    compiler flags are provided to ensure high quality code generation. In
    contrast to the ``pybind11_add_module()`` command, the CMake interface
    provides a *composable* set of targets to ensure that you retain flexibility.
    It can be especially important to provide or set these properties; the
    :ref:`FAQ <faq:symhidden>` contains an explanation on why these are needed.

.. versionadded:: 2.6

.. _nopython-mode:

Advanced: NOPYTHON mode
-----------------------

If you want complete control, you can set ``PYBIND11_NOPYTHON`` to completely
disable Python integration (this also happens if you run ``FindPython2`` and
``FindPython3`` without running ``FindPython``). This gives you complete
freedom to integrate into an existing system (like `Scikit-Build's
<https://scikit-build.readthedocs.io>`_ ``PythonExtensions``).
``pybind11_add_module`` and ``pybind11_extension`` will be unavailable, and the
targets will be missing any Python specific behavior.

.. versionadded:: 2.6

Embedding the Python interpreter
--------------------------------

In addition to extension modules, pybind11 also supports embedding Python into
a C++ executable or library. In CMake, simply link with the ``pybind11::embed``
target. It provides everything needed to get the interpreter running. The Python
headers and libraries are attached to the target. Unlike ``pybind11::module``,
there is no need to manually set any additional properties here. For more
information about usage in C++, see :doc:`/advanced/embedding`.

.. code-block:: cmake

    cmake_minimum_required(VERSION 3.5...3.29)
    project(example LANGUAGES CXX)

    find_package(pybind11 REQUIRED)  # or add_subdirectory(pybind11)

    add_executable(example main.cpp)
    target_link_libraries(example PRIVATE pybind11::embed)

.. _building_manually:

Building manually
=================

pybind11 is a header-only library, hence it is not necessary to link against
any special libraries and there are no intermediate (magic) translation steps.

On Linux, you can compile an example such as the one given in
:ref:`simple_example` using the following command:

.. code-block:: bash

    $ c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)

The ``python3 -m pybind11 --includes`` command fetches the include paths for
both pybind11 and Python headers. This assumes that pybind11 has been installed
using ``pip`` or ``conda``. If it hasn't, you can also manually specify
``-I <path-to-pybind11>/include`` together with the Python includes path
``python3-config --includes``.

On macOS: the build command is almost the same but it also requires passing
the ``-undefined dynamic_lookup`` flag so as to ignore missing symbols when
building the module:

.. code-block:: bash

    $ c++ -O3 -Wall -shared -std=c++11 -undefined dynamic_lookup $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)

In general, it is advisable to include several additional build parameters
that can considerably reduce the size of the created binary. Refer to section
:ref:`cmake` for a detailed example of a suitable cross-platform CMake-based
build system that works on all platforms including Windows.

.. note::

    On Linux and macOS, it's better to (intentionally) not link against
    ``libpython``. The symbols will be resolved when the extension library
    is loaded into a Python binary. This is preferable because you might
    have several different installations of a given Python version (e.g. the
    system-provided Python, and one that ships with a piece of commercial
    software). In this way, the plugin will work with both versions, instead
    of possibly importing a second Python library into a process that already
    contains one (which will lead to a segfault).


Building with Bazel
===================

You can build with the Bazel build system using the `pybind11_bazel
<https://github.com/pybind/pybind11_bazel>`_ repository.

Building with Meson
===================

You can use Meson, which has support for ``pybind11`` as a dependency (internally
relying on our ``pkg-config`` support). See the :ref:`module example above <meson-example>`.


Generating binding code automatically
=====================================

The ``Binder`` project is a tool for automatic generation of pybind11 binding
code by introspecting existing C++ codebases using LLVM/Clang. See the
[binder]_ documentation for details.

.. [binder] http://cppbinder.readthedocs.io/en/latest/about.html

[AutoWIG]_ is a Python library that wraps automatically compiled libraries into
high-level languages. It parses C++ code using LLVM/Clang technologies and
generates the wrappers using the Mako templating engine. The approach is automatic,
extensible, and applies to very complex C++ libraries, composed of thousands of
classes or incorporating modern meta-programming constructs.

.. [AutoWIG] https://github.com/StatisKit/AutoWIG

[robotpy-build]_ is a is a pure python, cross platform build tool that aims to
simplify creation of python wheels for pybind11 projects, and provide
cross-project dependency management. Additionally, it is able to autogenerate
customizable pybind11-based wrappers by parsing C++ header files.

.. [robotpy-build] https://robotpy-build.readthedocs.io

[litgen]_ is an automatic python bindings generator with a focus on generating
documented and discoverable bindings: bindings will nicely reproduce the documentation
found in headers. It is based on srcML (srcml.org), a highly scalable, multi-language
parsing tool with a developer centric approach. The API that you want to expose to python
must be C++14 compatible (but your implementation can use more modern constructs).

.. [litgen] https://pthom.github.io/litgen
