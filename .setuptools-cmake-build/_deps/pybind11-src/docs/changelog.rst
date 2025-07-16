.. _changelog:

Changelog
#########

Starting with version 1.8.0, pybind11 releases use a `semantic versioning
<http://semver.org>`_ policy.

Changes will be added here periodically from the "Suggested changelog entry"
block in pull request descriptions.


IN DEVELOPMENT
--------------

Changes will be summarized here periodically.

New Features:

* Support for Python 3.7 was removed. (Official end-of-life: 2023-06-27).
  `#5191 <https://github.com/pybind/pybind11/pull/5191>`_

* stl.h ``list|set|map_caster`` were made more user friendly: it is no longer
  necessary to explicitly convert Python iterables to ``tuple()``, ``set()``,
  or ``map()`` in many common situations.
  `#4686 <https://github.com/pybind/pybind11/pull/4686>`_

* Support for CMake older than 3.15 removed. CMake 3.15-3.30 supported.
  `#5304 <https://github.com/pybind/pybind11/pull/5304>`_

* The ``array_caster`` in pybind11/stl.h was enhanced to support value types that are not default-constructible.
  `#5305 <https://github.com/pybind/pybind11/pull/5305>`_

* Added ``py::warnings`` namespace with ``py::warnings::warn`` and ``py::warnings::new_warning_type`` that provides the interface for Python warnings.
  `#5291 <https://github.com/pybind/pybind11/pull/5291>`_

Version 2.13.6 (September 13, 2024)
-----------------------------------

New Features:

* A new ``self._pybind11_conduit_v1_()`` method is automatically added to all
  ``py::class_``-wrapped types, to enable type-safe interoperability between
  different independent Python/C++ bindings systems, including pybind11
  versions with different ``PYBIND11_INTERNALS_VERSION``'s. Supported on
  pybind11 2.11.2, 2.12.1, and 2.13.6+.
  `#5296 <https://github.com/pybind/pybind11/pull/5296>`_


Bug fixes:

* Using ``__cpp_nontype_template_args`` instead of ``__cpp_nontype_template_parameter_class``.
  `#5330 <https://github.com/pybind/pybind11/pull/5330>`_

* Properly translate C++ exception to Python exception when creating Python buffer from wrapped object.
  `#5324 <https://github.com/pybind/pybind11/pull/5324>`_


Documentation:

* Adds an answer (FAQ) for "What is a highly conclusive and simple way to find memory leaks?".
  `#5340 <https://github.com/pybind/pybind11/pull/5340>`_


Version 2.13.5 (August 22, 2024)
--------------------------------

Bug fixes:

* Fix includes when using Windows long paths (``\\?\`` prefix).
  `#5321 <https://github.com/pybind/pybind11/pull/5321>`_

* Support ``-Wpedantic`` in C++20 mode.
  `#5322 <https://github.com/pybind/pybind11/pull/5322>`_

* Fix and test ``<ranges>`` support for ``py::tuple`` and ``py::list``.
  `#5314 <https://github.com/pybind/pybind11/pull/5314>`_

Version 2.13.4 (August 14, 2024)
--------------------------------

Bug fixes:

* Fix paths with spaces, including on Windows.
  (Replaces regression from `#5302 <https://github.com/pybind/pybind11/pull/5302>`_)
  `#4874 <https://github.com/pybind/pybind11/pull/4874>`_

Documentation:

* Remove repetitive words.
  `#5308 <https://github.com/pybind/pybind11/pull/5308>`_


Version 2.13.3 (August 13, 2024)
--------------------------------

Bug fixes:

* Quote paths from pybind11-config
  `#5302 <https://github.com/pybind/pybind11/pull/5302>`_


* Fix typo in Emscripten support when in config mode (CMake)
  `#5301 <https://github.com/pybind/pybind11/pull/5301>`_


Version 2.13.2 (August 13, 2024)
--------------------------------

New Features:

* A ``pybind11::detail::type_caster_std_function_specializations`` feature was added, to support specializations for
  ``std::function``'s with return types that require custom to-Python conversion behavior (to primary use case is to catch and
  convert exceptions).
  `#4597 <https://github.com/pybind/pybind11/pull/4597>`_


Changes:


* Use ``PyMutex`` instead of ``std::mutex`` for internal locking in the free-threaded build.
  `#5219 <https://github.com/pybind/pybind11/pull/5219>`_

* Add a special type annotation for C++ empty tuple.
  `#5214 <https://github.com/pybind/pybind11/pull/5214>`_

* When compiling for WebAssembly, add the required exception flags (CMake 3.13+).
  `#5298 <https://github.com/pybind/pybind11/pull/5298>`_

Bug fixes:

* Make ``gil_safe_call_once_and_store`` thread-safe in free-threaded CPython.
  `#5246 <https://github.com/pybind/pybind11/pull/5246>`_

* A missing ``#include <algorithm>`` in pybind11/typing.h was added to fix build errors (in case user code does not already depend
  on that include).
  `#5208 <https://github.com/pybind/pybind11/pull/5208>`_

* Fix regression introduced in #5201 for GCC<10.3 in C++20 mode.
  `#5205 <https://github.com/pybind/pybind11/pull/5205>`_


.. fix(cmake)

* Remove extra = when assigning flto value in the case for Clang in CMake.
  `#5207 <https://github.com/pybind/pybind11/pull/5207>`_


Tests:

* Adding WASM testing to our CI (Pyodide / Emscripten via scikit-build-core).
  `#4745 <https://github.com/pybind/pybind11/pull/4745>`_

* clang-tidy (in GitHub Actions) was updated from clang 15 to clang 18.
  `#5272 <https://github.com/pybind/pybind11/pull/5272>`_


Version 2.13.1 (June 26, 2024)
------------------------------

New Features:

* Add support for ``Typing.Callable[..., T]``.
  `#5202 <https://github.com/pybind/pybind11/pull/5202>`_

Bug fixes:

* Avoid aligned allocation in free-threaded build in order to support macOS
  versions before 10.14.
  `#5200 <https://github.com/pybind/pybind11/pull/5200>`_

Version 2.13.0 (June 25, 2024)
------------------------------

New Features:

* Support free-threaded CPython (3.13t). Add ``py::mod_gil_not_used()`` tag to
  indicate if a module supports running with the GIL disabled.
  `#5148 <https://github.com/pybind/pybind11/pull/5148>`_

* Support for Python 3.6 was removed. (Official end-of-life: 2021-12-23).
  `#5177 <https://github.com/pybind/pybind11/pull/5177>`_

* ``py::list`` gained a ``.clear()`` method.
  `#5153 <https://github.com/pybind/pybind11/pull/5153>`_


.. feat(types)

* Support for ``Union``, ``Optional``, ``type[T]``, ``typing.TypeGuard``,
  ``typing.TypeIs``, ``typing.Never``, ``typing.NoReturn`` and
  ``typing.Literal`` was added to ``pybind11/typing.h``.
  `#5166 <https://github.com/pybind/pybind11/pull/5166>`_
  `#5165 <https://github.com/pybind/pybind11/pull/5165>`_
  `#5194 <https://github.com/pybind/pybind11/pull/5194>`_
  `#5193 <https://github.com/pybind/pybind11/pull/5193>`_
  `#5192 <https://github.com/pybind/pybind11/pull/5192>`_


.. feat(cmake)

* In CMake, if ``PYBIND11_USE_CROSSCOMPILING`` is enabled, then
  ``CMAKE_CROSSCOMPILING`` will be respected and will keep pybind11 from
  accessing the interpreter during configuration. Several CMake variables will
  be required in this case, but can be deduced from the environment variable
  ``SETUPTOOLS_EXT_SUFFIX``. The default (currently ``OFF``) may be changed in
  the future.
  `#5083 <https://github.com/pybind/pybind11/pull/5083>`_


Bug fixes:

* A refcount bug (leading to heap-use-after-free) involving trampoline
  functions with ``PyObject *`` return type was fixed.
  `#5156 <https://github.com/pybind/pybind11/pull/5156>`_

* Return ``py::ssize_t`` from ``.ref_count()`` instead of ``int``.
  `#5139 <https://github.com/pybind/pybind11/pull/5139>`_

* A subtle bug involving C++ types with unusual ``operator&`` overrides
  was fixed.
  `#5189 <https://github.com/pybind/pybind11/pull/5189>`_

* Support Python 3.13 with minor fix, add to CI.
  `#5127 <https://github.com/pybind/pybind11/pull/5127>`_


.. fix(cmake)

* Fix mistake affecting old cmake and old boost.
  `#5149 <https://github.com/pybind/pybind11/pull/5149>`_


Documentation:

* Build docs updated to feature scikit-build-core and meson-python, and updated
  setuptools instructions.
  `#5168 <https://github.com/pybind/pybind11/pull/5168>`_


Tests:

* Avoid immortal objects in tests.
  `#5150 <https://github.com/pybind/pybind11/pull/5150>`_


CI:

* Compile against Python 3.13t in CI.

* Use ``macos-13`` (Intel) for CI jobs for now (will drop Python 3.7 soon).
  `#5109 <https://github.com/pybind/pybind11/pull/5109>`_

* Releases now have artifact attestations, visible at
  https://github.com/pybind/pybind11/attestations.
  `#5196 <https://github.com/pybind/pybind11/pull/5196>`_

Other:

* Some cleanup in preparation for 3.13 support.
  `#5137 <https://github.com/pybind/pybind11/pull/5137>`_

* Avoid a warning by ensuring an iterator end check is included in release mode.
  `#5129 <https://github.com/pybind/pybind11/pull/5129>`_

* Bump max cmake to 3.29.
  `#5075 <https://github.com/pybind/pybind11/pull/5075>`_

* Update docs and noxfile.
  `#5071 <https://github.com/pybind/pybind11/pull/5071>`_

Version 2.12.1 (September 13, 2024)
-----------------------------------

New Features:

* A new ``self._pybind11_conduit_v1_()`` method is automatically added to all
  ``py::class_``-wrapped types, to enable type-safe interoperability between
  different independent Python/C++ bindings systems, including pybind11
  versions with different ``PYBIND11_INTERNALS_VERSION``'s. Supported on
  pybind11 2.11.2, 2.12.1, and 2.13.6+.
  `#5296 <https://github.com/pybind/pybind11/pull/5296>`_


Version 2.12.0 (March 27, 2024)
-------------------------------

New Features:

* ``pybind11`` now supports compiling for
  `NumPy 2 <https://numpy.org/devdocs/numpy_2_0_migration_guide.html>`_. Most
  code shouldn't change (see :ref:`upgrade-guide-2.12` for details). However,
  if you experience issues you can define ``PYBIND11_NUMPY_1_ONLY`` to disable
  the new support for now, but this will be removed in the future.
  `#5050 <https://github.com/pybind/pybind11/pull/5050>`_

* ``pybind11/gil_safe_call_once.h`` was added (it needs to be included
  explicitly). The primary use case is GIL-safe initialization of C++
  ``static`` variables.
  `#4877 <https://github.com/pybind/pybind11/pull/4877>`_

* Support move-only iterators in ``py::make_iterator``,
  ``py::make_key_iterator``, ``py::make_value_iterator``.
  `#4834 <https://github.com/pybind/pybind11/pull/4834>`_

* Two simple ``py::set_error()`` functions were added and the documentation was
  updated accordingly. In particular, ``py::exception<>::operator()`` was
  deprecated (use one of the new functions instead). The documentation for
  ``py::exception<>`` was further updated to not suggest code that may result
  in undefined behavior.
  `#4772 <https://github.com/pybind/pybind11/pull/4772>`_

Bug fixes:

* Removes potential for Undefined Behavior during process teardown.
  `#4897 <https://github.com/pybind/pybind11/pull/4897>`_

* Improve compatibility with the nvcc compiler (especially CUDA 12.1/12.2).
  `#4893 <https://github.com/pybind/pybind11/pull/4893>`_

* ``pybind11/numpy.h`` now imports NumPy's ``multiarray`` and ``_internal``
  submodules with paths depending on the installed version of NumPy (for
  compatibility with NumPy 2).
  `#4857 <https://github.com/pybind/pybind11/pull/4857>`_

* Builtins collections names in docstrings are now consistently rendered in
  lowercase (list, set, dict, tuple), in accordance with PEP 585.
  `#4833 <https://github.com/pybind/pybind11/pull/4833>`_

* Added ``py::typing::Iterator<T>``, ``py::typing::Iterable<T>``.
  `#4832 <https://github.com/pybind/pybind11/pull/4832>`_

* Render ``py::function`` as ``Callable`` in docstring.
  `#4829 <https://github.com/pybind/pybind11/pull/4829>`_

* Also bump ``PYBIND11_INTERNALS_VERSION`` for MSVC, which unlocks two new
  features without creating additional incompatibilities.
  `#4819 <https://github.com/pybind/pybind11/pull/4819>`_

* Guard against crashes/corruptions caused by modules built with different MSVC
  versions.
  `#4779 <https://github.com/pybind/pybind11/pull/4779>`_

* A long-standing bug in the handling of Python multiple inheritance was fixed.
  See PR #4762 for the rather complex details.
  `#4762 <https://github.com/pybind/pybind11/pull/4762>`_

* Fix ``bind_map`` with ``using`` declarations.
  `#4952 <https://github.com/pybind/pybind11/pull/4952>`_

* Qualify ``py::detail::concat`` usage to avoid ADL selecting one from
  somewhere else, such as modernjson's concat.
  `#4955 <https://github.com/pybind/pybind11/pull/4955>`_

* Use new PyCode API on Python 3.12+.
  `#4916 <https://github.com/pybind/pybind11/pull/4916>`_

* Minor cleanup from warnings reported by Clazy.
  `#4988 <https://github.com/pybind/pybind11/pull/4988>`_

* Remove typing and duplicate ``class_`` for ``KeysView``/``ValuesView``/``ItemsView``.
  `#4985 <https://github.com/pybind/pybind11/pull/4985>`_

* Use ``PyObject_VisitManagedDict()`` and ``PyObject_ClearManagedDict()`` on Python 3.13 and newer.
  `#4973 <https://github.com/pybind/pybind11/pull/4973>`_

* Update ``make_static_property_type()`` to make it compatible with Python 3.13.
  `#4971 <https://github.com/pybind/pybind11/pull/4971>`_

.. fix(types)

* Render typed iterators for ``make_iterator``, ``make_key_iterator``,
  ``make_value_iterator``.
  `#4876 <https://github.com/pybind/pybind11/pull/4876>`_

* Add several missing type name specializations.
  `#5073 <https://github.com/pybind/pybind11/pull/5073>`_

* Change docstring render for ``py::buffer``, ``py::sequence`` and
  ``py::handle`` (to ``Buffer``, ``Sequence``, ``Any``).
  `#4831 <https://github.com/pybind/pybind11/pull/4831>`_

* Fixed ``base_enum.__str__`` docstring.
  `#4827 <https://github.com/pybind/pybind11/pull/4827>`_

* Enforce single line docstring signatures.
  `#4735 <https://github.com/pybind/pybind11/pull/4735>`_

* Special 'typed' wrappers now available in ``typing.h`` to annotate tuple, dict,
  list, set, and function.
  `#4259 <https://github.com/pybind/pybind11/pull/4259>`_

* Create ``handle_type_name`` specialization to type-hint variable length tuples.
  `#5051 <https://github.com/pybind/pybind11/pull/5051>`_

.. fix(build)

* Setting ``PYBIND11_FINDPYTHON`` to OFF will force the old FindPythonLibs mechanism to be used.
  `#5042 <https://github.com/pybind/pybind11/pull/5042>`_

* Skip empty ``PYBIND11_PYTHON_EXECUTABLE_LAST`` for the first cmake run.
  `#4856 <https://github.com/pybind/pybind11/pull/4856>`_

* Fix FindPython mode exports & avoid ``pkg_resources`` if
  ``importlib.metadata`` available.
  `#4941 <https://github.com/pybind/pybind11/pull/4941>`_

* ``Python_ADDITIONAL_VERSIONS`` (classic search) now includes 3.12.
  `#4909 <https://github.com/pybind/pybind11/pull/4909>`_

* ``pybind11.pc`` is now relocatable by default as long as install destinations
  are not absolute paths.
  `#4830 <https://github.com/pybind/pybind11/pull/4830>`_

* Correctly detect CMake FindPython removal when used as a subdirectory.
  `#4806 <https://github.com/pybind/pybind11/pull/4806>`_

* Don't require the libs component on CMake 3.18+ when using
  PYBIND11_FINDPYTHON (fixes manylinux builds).
  `#4805 <https://github.com/pybind/pybind11/pull/4805>`_

* ``pybind11_strip`` is no longer automatically applied when
  ``CMAKE_BUILD_TYPE`` is unset.
  `#4780 <https://github.com/pybind/pybind11/pull/4780>`_

* Support ``DEBUG_POSFIX`` correctly for debug builds.
  `#4761 <https://github.com/pybind/pybind11/pull/4761>`_

* Hardcode lto/thin lto for Emscripten cross-compiles.
  `#4642 <https://github.com/pybind/pybind11/pull/4642>`_

* Upgrade maximum supported CMake version to 3.27 to fix CMP0148 warnings.
  `#4786 <https://github.com/pybind/pybind11/pull/4786>`_

Documentation:

* Small fix to grammar in ``functions.rst``.
  `#4791 <https://github.com/pybind/pybind11/pull/4791>`_

* Remove upper bound in example pyproject.toml for setuptools.
  `#4774 <https://github.com/pybind/pybind11/pull/4774>`_

CI:

* CI: Update NVHPC to 23.5 and Ubuntu 20.04.
  `#4764 <https://github.com/pybind/pybind11/pull/4764>`_

* Test on PyPy 3.10.
  `#4714 <https://github.com/pybind/pybind11/pull/4714>`_

Other:

* Use Ruff formatter instead of Black.
  `#4912 <https://github.com/pybind/pybind11/pull/4912>`_

* An ``assert()`` was added to help Coverty avoid generating a false positive.
  `#4817 <https://github.com/pybind/pybind11/pull/4817>`_

Version 2.11.2 (September 13, 2024)
-----------------------------------

New Features:

* A new ``self._pybind11_conduit_v1_()`` method is automatically added to all
  ``py::class_``-wrapped types, to enable type-safe interoperability between
  different independent Python/C++ bindings systems, including pybind11
  versions with different ``PYBIND11_INTERNALS_VERSION``'s. Supported on
  pybind11 2.11.2, 2.12.1, and 2.13.6+.
  `#5296 <https://github.com/pybind/pybind11/pull/5296>`_


Version 2.11.1 (July 17, 2023)
------------------------------

Changes:

* ``PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF`` is now provided as an option
  for disabling the default-on ``PyGILState_Check()``'s in
  ``pybind11::handle``'s ``inc_ref()`` & ``dec_ref()``.
  `#4753 <https://github.com/pybind/pybind11/pull/4753>`_

* ``PYBIND11_ASSERT_GIL_HELD_INCREF_DECREF`` was disabled for PyPy in general
  (not just PyPy Windows).
  `#4751 <https://github.com/pybind/pybind11/pull/4751>`_


Version 2.11.0 (July 14, 2023)
------------------------------

New features:

* The newly added ``pybind11::detail::is_move_constructible`` trait can be
  specialized for cases in which ``std::is_move_constructible`` does not work
  as needed. This is very similar to the long-established
  ``pybind11::detail::is_copy_constructible``.
  `#4631 <https://github.com/pybind/pybind11/pull/4631>`_

* Introduce ``recursive_container_traits``.
  `#4623 <https://github.com/pybind/pybind11/pull/4623>`_

* ``pybind11/type_caster_pyobject_ptr.h`` was added to support automatic
  wrapping of APIs that make use of ``PyObject *``. This header needs to
  included explicitly (i.e. it is not included implicitly
  with ``pybind/pybind11.h``).
  `#4601 <https://github.com/pybind/pybind11/pull/4601>`_

* ``format_descriptor<>`` & ``npy_format_descriptor<>`` ``PyObject *``
  specializations were added. The latter enables ``py::array_t<PyObject *>``
  to/from-python conversions.
  `#4674 <https://github.com/pybind/pybind11/pull/4674>`_

* ``buffer_info`` gained an ``item_type_is_equivalent_to<T>()`` member
  function.
  `#4674 <https://github.com/pybind/pybind11/pull/4674>`_

* The ``capsule`` API gained a user-friendly constructor
  (``py::capsule(ptr, "name", dtor)``).
  `#4720 <https://github.com/pybind/pybind11/pull/4720>`_

Changes:

* ``PyGILState_Check()``'s in ``pybind11::handle``'s ``inc_ref()`` &
  ``dec_ref()`` are now enabled by default again.
  `#4246 <https://github.com/pybind/pybind11/pull/4246>`_

* ``py::initialize_interpreter()`` using ``PyConfig_InitPythonConfig()``
  instead of ``PyConfig_InitIsolatedConfig()``, to obtain complete
  ``sys.path``.
  `#4473 <https://github.com/pybind/pybind11/pull/4473>`_

* Cast errors now always include Python type information, even if
  ``PYBIND11_DETAILED_ERROR_MESSAGES`` is not defined. This increases binary
  sizes slightly (~1.5%) but the error messages are much more informative.
  `#4463 <https://github.com/pybind/pybind11/pull/4463>`_

* The docstring generation for the ``std::array``-list caster was fixed.
  Previously, signatures included the size of the list in a non-standard,
  non-spec compliant way. The new format conforms to PEP 593.
  **Tooling for processing the docstrings may need to be updated accordingly.**
  `#4679 <https://github.com/pybind/pybind11/pull/4679>`_

* Setter return values (which are inaccessible for all practical purposes) are
  no longer converted to Python (only to be discarded).
  `#4621 <https://github.com/pybind/pybind11/pull/4621>`_

* Allow lambda specified to function definition to be ``noexcept(true)``
  in C++17.
  `#4593 <https://github.com/pybind/pybind11/pull/4593>`_

* Get rid of recursive template instantiations for concatenating type
  signatures on C++17 and higher.
  `#4587 <https://github.com/pybind/pybind11/pull/4587>`_

* Compatibility with Python 3.12 (beta). Note that the minimum pybind11
  ABI version for Python 3.12 is version 5. (The default ABI version
  for Python versions up to and including 3.11 is still version 4.).
  `#4570 <https://github.com/pybind/pybind11/pull/4570>`_

* With ``PYBIND11_INTERNALS_VERSION 5`` (default for Python 3.12+), MSVC builds
  use ``std::hash<std::type_index>`` and ``std::equal_to<std::type_index>``
  instead of string-based type comparisons. This resolves issues when binding
  types defined in the unnamed namespace.
  `#4319 <https://github.com/pybind/pybind11/pull/4319>`_

* Python exception ``__notes__`` (introduced with Python 3.11) are now added to
  the ``error_already_set::what()`` output.
  `#4678 <https://github.com/pybind/pybind11/pull/4678>`_

Build system improvements:

* CMake 3.27 support was added, CMake 3.4 support was dropped.
  FindPython will be used if ``FindPythonInterp`` is not present.
  `#4719 <https://github.com/pybind/pybind11/pull/4719>`_

* Update clang-tidy to 15 in CI.
  `#4387 <https://github.com/pybind/pybind11/pull/4387>`_

* Moved the linting framework over to Ruff.
  `#4483 <https://github.com/pybind/pybind11/pull/4483>`_

* Skip ``lto`` checks and target generation when
  ``CMAKE_INTERPROCEDURAL_OPTIMIZATION`` is defined.
  `#4643 <https://github.com/pybind/pybind11/pull/4643>`_

* No longer inject ``-stdlib=libc++``, not needed for modern Pythons
  (macOS 10.9+).
  `#4639 <https://github.com/pybind/pybind11/pull/4639>`_

* PyPy 3.10 support was added, PyPy 3.7 support was dropped.
  `#4728 <https://github.com/pybind/pybind11/pull/4728>`_

* Testing with Python 3.12 beta releases was added.
  `#4713 <https://github.com/pybind/pybind11/pull/4713>`_


Version 2.10.4 (Mar 16, 2023)
-----------------------------

Changes:

* ``python3 -m pybind11`` gained a ``--version`` option (prints the version and
  exits).
  `#4526 <https://github.com/pybind/pybind11/pull/4526>`_

Bug Fixes:

* Fix a warning when pydebug is enabled on Python 3.11.
  `#4461 <https://github.com/pybind/pybind11/pull/4461>`_

* Ensure ``gil_scoped_release`` RAII is non-copyable.
  `#4490 <https://github.com/pybind/pybind11/pull/4490>`_

* Ensure the tests dir does not show up with new versions of setuptools.
  `#4510 <https://github.com/pybind/pybind11/pull/4510>`_

* Better stacklevel for a warning in setuptools helpers.
  `#4516 <https://github.com/pybind/pybind11/pull/4516>`_

Version 2.10.3 (Jan 3, 2023)
----------------------------

Changes:

* Temporarily made our GIL status assertions (added in 2.10.2) disabled by
  default (re-enable manually by defining
  ``PYBIND11_ASSERT_GIL_HELD_INCREF_DECREF``, will be enabled in 2.11).
  `#4432 <https://github.com/pybind/pybind11/pull/4432>`_

* Improved error messages when ``inc_ref``/``dec_ref`` are called with an
  invalid GIL state.
  `#4427 <https://github.com/pybind/pybind11/pull/4427>`_
  `#4436 <https://github.com/pybind/pybind11/pull/4436>`_

Bug Fixes:

* Some minor touchups found by static analyzers.
  `#4440 <https://github.com/pybind/pybind11/pull/4440>`_


Version 2.10.2 (Dec 20, 2022)
-----------------------------

Changes:

* ``scoped_interpreter`` constructor taking ``PyConfig``.
  `#4330 <https://github.com/pybind/pybind11/pull/4330>`_

* ``pybind11/eigen/tensor.h`` adds converters to and from ``Eigen::Tensor`` and
  ``Eigen::TensorMap``.
  `#4201 <https://github.com/pybind/pybind11/pull/4201>`_

* ``PyGILState_Check()``'s  were integrated to ``pybind11::handle``
  ``inc_ref()`` & ``dec_ref()``. The added GIL checks are guarded by
  ``PYBIND11_ASSERT_GIL_HELD_INCREF_DECREF``, which is the default only if
  ``NDEBUG`` is not defined. (Made non-default in 2.10.3, will be active in 2.11)
  `#4246 <https://github.com/pybind/pybind11/pull/4246>`_

* Add option for enable/disable enum members in docstring.
  `#2768 <https://github.com/pybind/pybind11/pull/2768>`_

* Fixed typing of ``KeysView``, ``ValuesView`` and ``ItemsView`` in ``bind_map``.
  `#4353 <https://github.com/pybind/pybind11/pull/4353>`_

Bug fixes:

* Bug fix affecting only Python 3.6 under very specific, uncommon conditions:
  move ``PyEval_InitThreads()`` call to the correct location.
  `#4350 <https://github.com/pybind/pybind11/pull/4350>`_

* Fix segfault bug when passing foreign native functions to functional.h.
  `#4254 <https://github.com/pybind/pybind11/pull/4254>`_

Build system improvements:

* Support setting PYTHON_LIBRARIES manually for Windows ARM cross-compilation
  (classic mode).
  `#4406 <https://github.com/pybind/pybind11/pull/4406>`_

* Extend IPO/LTO detection for ICX (a.k.a IntelLLVM) compiler.
  `#4402 <https://github.com/pybind/pybind11/pull/4402>`_

* Allow calling ``find_package(pybind11 CONFIG)`` multiple times from separate
  directories in the same CMake project and properly link Python (new mode).
  `#4401 <https://github.com/pybind/pybind11/pull/4401>`_

* ``multiprocessing_set_spawn`` in pytest fixture for added safety.
  `#4377 <https://github.com/pybind/pybind11/pull/4377>`_

* Fixed a bug in two pybind11/tools cmake scripts causing "Unknown arguments specified" errors.
  `#4327 <https://github.com/pybind/pybind11/pull/4327>`_



Version 2.10.1 (Oct 31, 2022)
-----------------------------

This is the first version to fully support embedding the newly released Python 3.11.

Changes:

* Allow ``pybind11::capsule`` constructor to take null destructor pointers.
  `#4221 <https://github.com/pybind/pybind11/pull/4221>`_

* ``embed.h`` was changed so that ``PYTHONPATH`` is used also with Python 3.11
  (established behavior).
  `#4119 <https://github.com/pybind/pybind11/pull/4119>`_

* A ``PYBIND11_SIMPLE_GIL_MANAGEMENT`` option was added (cmake, C++ define),
  along with many additional tests in ``test_gil_scoped.py``. The option may be
  useful to try when debugging GIL-related issues, to determine if the more
  complex default implementation is or is not to blame. See #4216 for
  background. WARNING: Please be careful to not create ODR violations when
  using the option: everything that is linked together with mutual symbol
  visibility needs to be rebuilt.
  `#4216 <https://github.com/pybind/pybind11/pull/4216>`_

* ``PYBIND11_EXPORT_EXCEPTION`` was made non-empty only under macOS. This makes
  Linux builds safer, and enables the removal of warning suppression pragmas for
  Windows.
  `#4298 <https://github.com/pybind/pybind11/pull/4298>`_

Bug fixes:

* Fixed a bug where ``UnicodeDecodeError`` was not propagated from various
  ``py::str`` ctors when decoding surrogate utf characters.
  `#4294 <https://github.com/pybind/pybind11/pull/4294>`_

* Revert perfect forwarding for ``make_iterator``. This broke at least one
  valid use case. May revisit later.
  `#4234 <https://github.com/pybind/pybind11/pull/4234>`_

* Fix support for safe casts to ``void*`` (regression in 2.10.0).
  `#4275 <https://github.com/pybind/pybind11/pull/4275>`_

* Fix ``char8_t`` support (regression in 2.9).
  `#4278 <https://github.com/pybind/pybind11/pull/4278>`_

* Unicode surrogate character in Python exception message leads to process
  termination in ``error_already_set::what()``.
  `#4297 <https://github.com/pybind/pybind11/pull/4297>`_

* Fix MSVC 2019 v.1924 & C++14 mode error for ``overload_cast``.
  `#4188 <https://github.com/pybind/pybind11/pull/4188>`_

* Make augmented assignment operators non-const for the object-api. Behavior
  was previously broken for augmented assignment operators.
  `#4065 <https://github.com/pybind/pybind11/pull/4065>`_

* Add proper error checking to C++ bindings for Python list append and insert.
  `#4208 <https://github.com/pybind/pybind11/pull/4208>`_

* Work-around for Nvidia's CUDA nvcc compiler in versions 11.4.0 - 11.8.0.
  `#4220 <https://github.com/pybind/pybind11/pull/4220>`_

* A workaround for PyPy was added in the ``py::error_already_set``
  implementation, related to PR `#1895 <https://github.com/pybind/pybind11/pull/1895>`_
  released with v2.10.0.
  `#4079 <https://github.com/pybind/pybind11/pull/4079>`_

* Fixed compiler errors when C++23 ``std::forward_like`` is available.
  `#4136 <https://github.com/pybind/pybind11/pull/4136>`_

* Properly raise exceptions in contains methods (like when an object in unhashable).
  `#4209 <https://github.com/pybind/pybind11/pull/4209>`_

* Further improve another error in exception handling.
  `#4232 <https://github.com/pybind/pybind11/pull/4232>`_

* ``get_local_internals()`` was made compatible with
  ``finalize_interpreter()``, fixing potential freezes during interpreter
  finalization.
  `#4192 <https://github.com/pybind/pybind11/pull/4192>`_

Performance and style:

* Reserve space in set and STL map casters if possible. This will prevent
  unnecessary rehashing / resizing by knowing the number of keys ahead of time
  for Python to C++ casting. This improvement will greatly speed up the casting
  of large unordered maps and sets.
  `#4194 <https://github.com/pybind/pybind11/pull/4194>`_

* GIL RAII scopes are non-copyable to avoid potential bugs.
  `#4183 <https://github.com/pybind/pybind11/pull/4183>`_

* Explicitly default all relevant ctors for pytypes in the ``PYBIND11_OBJECT``
  macros and enforce the clang-tidy checks ``modernize-use-equals-default`` in
  macros as well.
  `#4017 <https://github.com/pybind/pybind11/pull/4017>`_

* Optimize iterator advancement in C++ bindings.
  `#4237 <https://github.com/pybind/pybind11/pull/4237>`_

* Use the modern ``PyObject_GenericGetDict`` and ``PyObject_GenericSetDict``
  for handling dynamic attribute dictionaries.
  `#4106 <https://github.com/pybind/pybind11/pull/4106>`_

* Document that users should use ``PYBIND11_NAMESPACE`` instead of using ``pybind11`` when
  opening namespaces. Using namespace declarations and namespace qualification
  remain the same as ``pybind11``. This is done to ensure consistent symbol
  visibility.
  `#4098 <https://github.com/pybind/pybind11/pull/4098>`_

* Mark ``detail::forward_like`` as constexpr.
  `#4147 <https://github.com/pybind/pybind11/pull/4147>`_

* Optimize unpacking_collector when processing ``arg_v`` arguments.
  `#4219 <https://github.com/pybind/pybind11/pull/4219>`_

* Optimize casting C++ object to ``None``.
  `#4269 <https://github.com/pybind/pybind11/pull/4269>`_


Build system improvements:

* CMake: revert overwrite behavior, now opt-in with ``PYBIND11_PYTHONLIBS_OVERRWRITE OFF``.
  `#4195 <https://github.com/pybind/pybind11/pull/4195>`_

* Include a pkg-config file when installing pybind11, such as in the Python
  package.
  `#4077 <https://github.com/pybind/pybind11/pull/4077>`_

* Avoid stripping debug symbols when ``CMAKE_BUILD_TYPE`` is set to ``DEBUG``
  instead of ``Debug``.
  `#4078 <https://github.com/pybind/pybind11/pull/4078>`_

* Followup to `#3948 <https://github.com/pybind/pybind11/pull/3948>`_, fixing vcpkg again.
  `#4123 <https://github.com/pybind/pybind11/pull/4123>`_

Version 2.10.0 (Jul 15, 2022)
-----------------------------

Removed support for Python 2.7, Python 3.5, and MSVC 2015. Support for MSVC
2017 is limited due to availability of CI runners; we highly recommend MSVC
2019 or 2022 be used. Initial support added for Python 3.11.

New features:

* ``py::anyset`` & ``py::frozenset`` were added, with copying (cast) to
  ``std::set`` (similar to ``set``).
  `#3901 <https://github.com/pybind/pybind11/pull/3901>`_

* Support bytearray casting to string.
  `#3707 <https://github.com/pybind/pybind11/pull/3707>`_

* ``type_caster<std::monostate>`` was added. ``std::monostate`` is a tag type
  that allows ``std::variant`` to act as an optional, or allows default
  construction of a ``std::variant`` holding a non-default constructible type.
  `#3818 <https://github.com/pybind/pybind11/pull/3818>`_

* ``pybind11::capsule::set_name`` added to mutate the name of the capsule instance.
  `#3866 <https://github.com/pybind/pybind11/pull/3866>`_

* NumPy: dtype constructor from type number added, accessors corresponding to
  Python API ``dtype.num``, ``dtype.byteorder``, ``dtype.flags`` and
  ``dtype.alignment`` added.
  `#3868 <https://github.com/pybind/pybind11/pull/3868>`_


Changes:

* Python 3.6 is now the minimum supported version.
  `#3688 <https://github.com/pybind/pybind11/pull/3688>`_
  `#3719 <https://github.com/pybind/pybind11/pull/3719>`_

* The minimum version for MSVC is now 2017.
  `#3722 <https://github.com/pybind/pybind11/pull/3722>`_

* Fix issues with CPython 3.11 betas and add to supported test matrix.
  `#3923 <https://github.com/pybind/pybind11/pull/3923>`_

* ``error_already_set`` is now safer and more performant, especially for
  exceptions with long tracebacks, by delaying computation.
  `#1895 <https://github.com/pybind/pybind11/pull/1895>`_

* Improve exception handling in python ``str`` bindings.
  `#3826 <https://github.com/pybind/pybind11/pull/3826>`_

* The bindings for capsules now have more consistent exception handling.
  `#3825 <https://github.com/pybind/pybind11/pull/3825>`_

* ``PYBIND11_OBJECT_CVT`` and ``PYBIND11_OBJECT_CVT_DEFAULT`` macro can now be
  used to define classes in namespaces other than pybind11.
  `#3797 <https://github.com/pybind/pybind11/pull/3797>`_

* Error printing code now uses ``PYBIND11_DETAILED_ERROR_MESSAGES`` instead of
  requiring ``NDEBUG``, allowing use with release builds if desired.
  `#3913 <https://github.com/pybind/pybind11/pull/3913>`_

* Implicit conversion of the literal ``0`` to ``pybind11::handle`` is now disabled.
  `#4008 <https://github.com/pybind/pybind11/pull/4008>`_


Bug fixes:

* Fix exception handling when ``pybind11::weakref()`` fails.
  `#3739 <https://github.com/pybind/pybind11/pull/3739>`_

* ``module_::def_submodule`` was missing proper error handling. This is fixed now.
  `#3973 <https://github.com/pybind/pybind11/pull/3973>`_

* The behavior or ``error_already_set`` was made safer and the highly opaque
  "Unknown internal error occurred" message was replaced with a more helpful
  message.
  `#3982 <https://github.com/pybind/pybind11/pull/3982>`_

* ``error_already_set::what()`` now handles non-normalized exceptions correctly.
  `#3971 <https://github.com/pybind/pybind11/pull/3971>`_

* Support older C++ compilers where filesystem is not yet part of the standard
  library and is instead included in ``std::experimental::filesystem``.
  `#3840 <https://github.com/pybind/pybind11/pull/3840>`_

* Fix ``-Wfree-nonheap-object`` warnings produced by GCC by avoiding returning
  pointers to static objects with ``return_value_policy::take_ownership``.
  `#3946 <https://github.com/pybind/pybind11/pull/3946>`_

* Fix cast from pytype rvalue to another pytype.
  `#3949 <https://github.com/pybind/pybind11/pull/3949>`_

* Ensure proper behavior when garbage collecting classes with dynamic attributes in Python >=3.9.
  `#4051 <https://github.com/pybind/pybind11/pull/4051>`_

* A couple long-standing ``PYBIND11_NAMESPACE``
  ``__attribute__((visibility("hidden")))`` inconsistencies are now fixed
  (affects only unusual environments).
  `#4043 <https://github.com/pybind/pybind11/pull/4043>`_

* ``pybind11::detail::get_internals()`` is now resilient to in-flight Python
  exceptions.
  `#3981 <https://github.com/pybind/pybind11/pull/3981>`_

* Arrays with a dimension of size 0 are now properly converted to dynamic Eigen
  matrices (more common in NumPy 1.23).
  `#4038 <https://github.com/pybind/pybind11/pull/4038>`_

* Avoid catching unrelated errors when importing NumPy.
  `#3974 <https://github.com/pybind/pybind11/pull/3974>`_

Performance and style:

* Added an accessor overload of ``(object &&key)`` to reference steal the
  object when using python types as keys. This prevents unnecessary reference
  count overhead for attr, dictionary, tuple, and sequence look ups. Added
  additional regression tests. Fixed a performance bug the caused accessor
  assignments to potentially perform unnecessary copies.
  `#3970 <https://github.com/pybind/pybind11/pull/3970>`_

* Perfect forward all args of ``make_iterator``.
  `#3980 <https://github.com/pybind/pybind11/pull/3980>`_

* Avoid potential bug in pycapsule destructor by adding an ``error_guard`` to
  one of the dtors.
  `#3958 <https://github.com/pybind/pybind11/pull/3958>`_

* Optimize dictionary access in ``strip_padding`` for numpy.
  `#3994 <https://github.com/pybind/pybind11/pull/3994>`_

* ``stl_bind.h`` bindings now take slice args as a const-ref.
  `#3852 <https://github.com/pybind/pybind11/pull/3852>`_

* Made slice constructor more consistent, and improve performance of some
  casters by allowing reference stealing.
  `#3845 <https://github.com/pybind/pybind11/pull/3845>`_

* Change numpy dtype from_args method to use const ref.
  `#3878 <https://github.com/pybind/pybind11/pull/3878>`_

* Follow rule of three to ensure ``PyErr_Restore`` is called only once.
  `#3872 <https://github.com/pybind/pybind11/pull/3872>`_

* Added missing perfect forwarding for ``make_iterator`` functions.
  `#3860 <https://github.com/pybind/pybind11/pull/3860>`_

* Optimize c++ to python function casting by using the rvalue caster.
  `#3966 <https://github.com/pybind/pybind11/pull/3966>`_

* Optimize Eigen sparse matrix casting by removing unnecessary temporary.
  `#4064 <https://github.com/pybind/pybind11/pull/4064>`_

* Avoid potential implicit copy/assignment constructors causing double free in
  ``strdup_gaurd``.
  `#3905 <https://github.com/pybind/pybind11/pull/3905>`_

* Enable clang-tidy checks ``misc-definitions-in-headers``,
  ``modernize-loop-convert``, and ``modernize-use-nullptr``.
  `#3881 <https://github.com/pybind/pybind11/pull/3881>`_
  `#3988 <https://github.com/pybind/pybind11/pull/3988>`_


Build system improvements:

* CMake: Fix file extension on Windows with cp36 and cp37 using FindPython.
  `#3919 <https://github.com/pybind/pybind11/pull/3919>`_

* CMake: Support multiple Python targets (such as on vcpkg).
  `#3948 <https://github.com/pybind/pybind11/pull/3948>`_

* CMake: Fix issue with NVCC on Windows.
  `#3947 <https://github.com/pybind/pybind11/pull/3947>`_

* CMake: Drop the bitness check on cross compiles (like targeting WebAssembly
  via Emscripten).
  `#3959 <https://github.com/pybind/pybind11/pull/3959>`_

* Add MSVC builds in debug mode to CI.
  `#3784 <https://github.com/pybind/pybind11/pull/3784>`_

* MSVC 2022 C++20 coverage was added to GitHub Actions, including Eigen.
  `#3732 <https://github.com/pybind/pybind11/pull/3732>`_,
  `#3741 <https://github.com/pybind/pybind11/pull/3741>`_


Backend and tidying up:

* New theme for the documentation.
  `#3109 <https://github.com/pybind/pybind11/pull/3109>`_

* Remove idioms in code comments.  Use more inclusive language.
  `#3809 <https://github.com/pybind/pybind11/pull/3809>`_

* ``#include <iostream>`` was removed from the ``pybind11/stl.h`` header. Your
  project may break if it has a transitive dependency on this include. The fix
  is to "Include What You Use".
  `#3928 <https://github.com/pybind/pybind11/pull/3928>`_

* Avoid ``setup.py <command>`` usage in internal tests.
  `#3734 <https://github.com/pybind/pybind11/pull/3734>`_


Version 2.9.2 (Mar 29, 2022)
----------------------------

Changes:

* Enum now has an ``__index__`` method on Python <3.8 too.
  `#3700 <https://github.com/pybind/pybind11/pull/3700>`_

* Local internals are now cleared after finalizing the interpreter.
  `#3744 <https://github.com/pybind/pybind11/pull/3744>`_

Bug fixes:

* Better support for Python 3.11 alphas.
  `#3694 <https://github.com/pybind/pybind11/pull/3694>`_

* ``PYBIND11_TYPE_CASTER`` now uses fully qualified symbols, so it can be used
  outside of ``pybind11::detail``.
  `#3758 <https://github.com/pybind/pybind11/pull/3758>`_

* Some fixes for PyPy 3.9.
  `#3768 <https://github.com/pybind/pybind11/pull/3768>`_

* Fixed a potential memleak in PyPy in ``get_type_override``.
  `#3774 <https://github.com/pybind/pybind11/pull/3774>`_

* Fix usage of ``VISIBILITY_INLINES_HIDDEN``.
  `#3721 <https://github.com/pybind/pybind11/pull/3721>`_


Build system improvements:

* Uses ``sysconfig`` module to determine installation locations on Python >=
  3.10, instead of ``distutils`` which has been deprecated.
  `#3764 <https://github.com/pybind/pybind11/pull/3764>`_

* Support Catch 2.13.5+ (supporting GLIBC 2.34+).
  `#3679 <https://github.com/pybind/pybind11/pull/3679>`_

* Fix test failures with numpy 1.22 by ignoring whitespace when comparing
  ``str()`` of dtypes.
  `#3682 <https://github.com/pybind/pybind11/pull/3682>`_


Backend and tidying up:

* clang-tidy: added ``readability-qualified-auto``,
  ``readability-braces-around-statements``,
  ``cppcoreguidelines-prefer-member-initializer``,
  ``clang-analyzer-optin.performance.Padding``,
  ``cppcoreguidelines-pro-type-static-cast-downcast``, and
  ``readability-inconsistent-declaration-parameter-name``.
  `#3702 <https://github.com/pybind/pybind11/pull/3702>`_,
  `#3699 <https://github.com/pybind/pybind11/pull/3699>`_,
  `#3716 <https://github.com/pybind/pybind11/pull/3716>`_,
  `#3709 <https://github.com/pybind/pybind11/pull/3709>`_

* clang-format was added to the pre-commit actions, and the entire code base
  automatically reformatted (after several iterations preparing for this leap).
  `#3713 <https://github.com/pybind/pybind11/pull/3713>`_


Version 2.9.1 (Feb 2, 2022)
---------------------------

Changes:

* If possible, attach Python exception with ``py::raise_from`` to ``TypeError``
  when casting from C++ to Python. This will give additional info if Python
  exceptions occur in the caster. Adds a test case of trying to convert a set
  from C++ to Python when the hash function is not defined in Python.
  `#3605 <https://github.com/pybind/pybind11/pull/3605>`_

* Add a mapping of C++11 nested exceptions to their Python exception
  equivalent using ``py::raise_from``. This attaches the nested exceptions in
  Python using the ``__cause__`` field.
  `#3608 <https://github.com/pybind/pybind11/pull/3608>`_

* Propagate Python exception traceback using ``raise_from`` if a pybind11
  function runs out of overloads.
  `#3671 <https://github.com/pybind/pybind11/pull/3671>`_

* ``py::multiple_inheritance`` is now only needed when C++ bases are hidden
  from pybind11.
  `#3650 <https://github.com/pybind/pybind11/pull/3650>`_ and
  `#3659 <https://github.com/pybind/pybind11/pull/3659>`_


Bug fixes:

* Remove a boolean cast in ``numpy.h`` that causes MSVC C4800 warnings when
  compiling against Python 3.10 or newer.
  `#3669 <https://github.com/pybind/pybind11/pull/3669>`_

* Render ``py::bool_`` and ``py::float_`` as ``bool`` and ``float``
  respectively.
  `#3622 <https://github.com/pybind/pybind11/pull/3622>`_

Build system improvements:

* Fix CMake extension suffix computation on Python 3.10+.
  `#3663 <https://github.com/pybind/pybind11/pull/3663>`_

* Allow ``CMAKE_ARGS`` to override CMake args in pybind11's own ``setup.py``.
  `#3577 <https://github.com/pybind/pybind11/pull/3577>`_

* Remove a few deprecated c-headers.
  `#3610 <https://github.com/pybind/pybind11/pull/3610>`_

* More uniform handling of test targets.
  `#3590 <https://github.com/pybind/pybind11/pull/3590>`_

* Add clang-tidy readability check to catch potentially swapped function args.
  `#3611 <https://github.com/pybind/pybind11/pull/3611>`_


Version 2.9.0 (Dec 28, 2021)
----------------------------

This is the last version to support Python 2.7 and 3.5.

New Features:

* Allow ``py::args`` to be followed by other arguments; the remaining arguments
  are implicitly keyword-only, as if a ``py::kw_only{}`` annotation had been
  used.
  `#3402 <https://github.com/pybind/pybind11/pull/3402>`_

Changes:

* Make str/bytes/memoryview more interoperable with ``std::string_view``.
  `#3521 <https://github.com/pybind/pybind11/pull/3521>`_

* Replace ``_`` with ``const_name`` in internals, avoid defining ``pybind::_``
  if ``_`` defined as macro (common gettext usage)
  `#3423 <https://github.com/pybind/pybind11/pull/3423>`_


Bug fixes:

* Fix a rare warning about extra copy in an Eigen constructor.
  `#3486 <https://github.com/pybind/pybind11/pull/3486>`_

* Fix caching of the C++ overrides.
  `#3465 <https://github.com/pybind/pybind11/pull/3465>`_

* Add missing ``std::forward`` calls to some ``cpp_function`` overloads.
  `#3443 <https://github.com/pybind/pybind11/pull/3443>`_

* Support PyPy 7.3.7 and the PyPy3.8 beta. Test python-3.11 on PRs with the
  ``python dev`` label.
  `#3419 <https://github.com/pybind/pybind11/pull/3419>`_

* Replace usage of deprecated ``Eigen::MappedSparseMatrix`` with
  ``Eigen::Map<Eigen::SparseMatrix<...>>`` for Eigen 3.3+.
  `#3499 <https://github.com/pybind/pybind11/pull/3499>`_

* Tweaks to support Microsoft Visual Studio 2022.
  `#3497 <https://github.com/pybind/pybind11/pull/3497>`_

Build system improvements:

* Nicer CMake printout and IDE organisation for pybind11's own tests.
  `#3479 <https://github.com/pybind/pybind11/pull/3479>`_

* CMake: report version type as part of the version string to avoid a spurious
  space in the package status message.
  `#3472 <https://github.com/pybind/pybind11/pull/3472>`_

* Flags starting with ``-g`` in ``$CFLAGS`` and ``$CPPFLAGS`` are no longer
  overridden by ``.Pybind11Extension``.
  `#3436 <https://github.com/pybind/pybind11/pull/3436>`_

* Ensure ThreadPool is closed in ``setup_helpers``.
  `#3548 <https://github.com/pybind/pybind11/pull/3548>`_

* Avoid LTS on ``mips64`` and ``ppc64le`` (reported broken).
  `#3557 <https://github.com/pybind/pybind11/pull/3557>`_


v2.8.1 (Oct 27, 2021)
---------------------

Changes and additions:

* The simple namespace creation shortcut added in 2.8.0 was deprecated due to
  usage of CPython internal API, and will be removed soon. Use
  ``py::module_::import("types").attr("SimpleNamespace")``.
  `#3374 <https://github.com/pybinyyd/pybind11/pull/3374>`_

* Add C++ Exception type to throw and catch ``AttributeError``. Useful for
  defining custom ``__setattr__`` and ``__getattr__`` methods.
  `#3387 <https://github.com/pybind/pybind11/pull/3387>`_

Fixes:

* Fixed the potential for dangling references when using properties with
  ``std::optional`` types.
  `#3376 <https://github.com/pybind/pybind11/pull/3376>`_

* Modernize usage of ``PyCodeObject`` on Python 3.9+ (moving toward support for
  Python 3.11a1)
  `#3368 <https://github.com/pybind/pybind11/pull/3368>`_

* A long-standing bug in ``eigen.h`` was fixed (originally PR #3343). The bug
  was unmasked by newly added ``static_assert``'s in the Eigen 3.4.0 release.
  `#3352 <https://github.com/pybind/pybind11/pull/3352>`_

* Support multiple raw inclusion of CMake helper files (Conan.io does this for
  multi-config generators).
  `#3420 <https://github.com/pybind/pybind11/pull/3420>`_

* Fix harmless warning on upcoming CMake 3.22.
  `#3368 <https://github.com/pybind/pybind11/pull/3368>`_

* Fix 2.8.0 regression with MSVC 2017 + C++17 mode + Python 3.
  `#3407 <https://github.com/pybind/pybind11/pull/3407>`_

* Fix 2.8.0 regression that caused undefined behavior (typically
  segfaults) in ``make_key_iterator``/``make_value_iterator`` if dereferencing
  the iterator returned a temporary value instead of a reference.
  `#3348 <https://github.com/pybind/pybind11/pull/3348>`_


v2.8.0 (Oct 4, 2021)
--------------------

New features:

* Added ``py::raise_from`` to enable chaining exceptions.
  `#3215 <https://github.com/pybind/pybind11/pull/3215>`_

* Allow exception translators to be optionally registered local to a module
  instead of applying globally across all pybind11 modules. Use
  ``register_local_exception_translator(ExceptionTranslator&& translator)``
  instead of  ``register_exception_translator(ExceptionTranslator&&
  translator)`` to keep your exception remapping code local to the module.
  `#2650 <https://github.com/pybinyyd/pybind11/pull/2650>`_

* Add ``make_simple_namespace`` function for instantiating Python
  ``SimpleNamespace`` objects. **Deprecated in 2.8.1.**
  `#2840 <https://github.com/pybind/pybind11/pull/2840>`_

* ``pybind11::scoped_interpreter`` and ``initialize_interpreter`` have new
  arguments to allow ``sys.argv`` initialization.
  `#2341 <https://github.com/pybind/pybind11/pull/2341>`_

* Allow Python builtins to be used as callbacks in CPython.
  `#1413 <https://github.com/pybind/pybind11/pull/1413>`_

* Added ``view`` to view arrays with a different datatype.
  `#987 <https://github.com/pybind/pybind11/pull/987>`_

* Implemented ``reshape`` on arrays.
  `#984 <https://github.com/pybind/pybind11/pull/984>`_

* Enable defining custom ``__new__`` methods on classes by fixing bug
  preventing overriding methods if they have non-pybind11 siblings.
  `#3265 <https://github.com/pybind/pybind11/pull/3265>`_

* Add ``make_value_iterator()``, and fix ``make_key_iterator()`` to return
  references instead of copies.
  `#3293 <https://github.com/pybind/pybind11/pull/3293>`_

* Improve the classes generated by ``bind_map``: `#3310 <https://github.com/pybind/pybind11/pull/3310>`_

  * Change ``.items`` from an iterator to a dictionary view.
  * Add ``.keys`` and ``.values`` (both dictionary views).
  * Allow ``__contains__`` to take any object.

* ``pybind11::custom_type_setup`` was added, for customizing the
  ``PyHeapTypeObject`` corresponding to a class, which may be useful for
  enabling garbage collection support, among other things.
  `#3287 <https://github.com/pybind/pybind11/pull/3287>`_


Changes:

* Set ``__file__`` constant when running ``eval_file`` in an embedded interpreter.
  `#3233 <https://github.com/pybind/pybind11/pull/3233>`_

* Python objects and (C++17) ``std::optional`` now accepted in ``py::slice``
  constructor.
  `#1101 <https://github.com/pybind/pybind11/pull/1101>`_

* The pybind11 proxy types ``str``, ``bytes``, ``bytearray``, ``tuple``,
  ``list`` now consistently support passing ``ssize_t`` values for sizes and
  indexes. Previously, only ``size_t`` was accepted in several interfaces.
  `#3219 <https://github.com/pybind/pybind11/pull/3219>`_

* Avoid evaluating ``PYBIND11_TLS_REPLACE_VALUE`` arguments more than once.
  `#3290 <https://github.com/pybind/pybind11/pull/3290>`_

Fixes:

* Bug fix: enum value's ``__int__`` returning non-int when underlying type is
  bool or of char type.
  `#1334 <https://github.com/pybind/pybind11/pull/1334>`_

* Fixes bug in setting error state in Capsule's pointer methods.
  `#3261 <https://github.com/pybind/pybind11/pull/3261>`_

* A long-standing memory leak in ``py::cpp_function::initialize`` was fixed.
  `#3229 <https://github.com/pybind/pybind11/pull/3229>`_

* Fixes thread safety for some ``pybind11::type_caster`` which require lifetime
  extension, such as for ``std::string_view``.
  `#3237 <https://github.com/pybind/pybind11/pull/3237>`_

* Restore compatibility with gcc 4.8.4 as distributed by ubuntu-trusty, linuxmint-17.
  `#3270 <https://github.com/pybind/pybind11/pull/3270>`_


Build system improvements:

* Fix regression in CMake Python package config: improper use of absolute path.
  `#3144 <https://github.com/pybind/pybind11/pull/3144>`_

* Cached Python version information could become stale when CMake was re-run
  with a different Python version. The build system now detects this and
  updates this information.
  `#3299 <https://github.com/pybind/pybind11/pull/3299>`_

* Specified UTF8-encoding in setup.py calls of open().
  `#3137 <https://github.com/pybind/pybind11/pull/3137>`_

* Fix a harmless warning from CMake 3.21 with the classic Python discovery.
  `#3220 <https://github.com/pybind/pybind11/pull/3220>`_

* Eigen repo and version can now be specified as cmake options.
  `#3324 <https://github.com/pybind/pybind11/pull/3324>`_


Backend and tidying up:

* Reduced thread-local storage required for keeping alive temporary data for
  type conversion to one key per ABI version, rather than one key per extension
  module.  This makes the total thread-local storage required by pybind11 2
  keys per ABI version.
  `#3275 <https://github.com/pybind/pybind11/pull/3275>`_

* Optimize NumPy array construction with additional moves.
  `#3183 <https://github.com/pybind/pybind11/pull/3183>`_

* Conversion to ``std::string`` and ``std::string_view`` now avoids making an
  extra copy of the data on Python >= 3.3.
  `#3257 <https://github.com/pybind/pybind11/pull/3257>`_

* Remove const modifier from certain C++ methods on Python collections
  (``list``, ``set``, ``dict``) such as (``clear()``, ``append()``,
  ``insert()``, etc...) and annotated them with ``py-non-const``.

* Enable readability ``clang-tidy-const-return`` and remove useless consts.
  `#3254 <https://github.com/pybind/pybind11/pull/3254>`_
  `#3194 <https://github.com/pybind/pybind11/pull/3194>`_

* The clang-tidy ``google-explicit-constructor`` option was enabled.
  `#3250 <https://github.com/pybind/pybind11/pull/3250>`_

* Mark a pytype move constructor as noexcept (perf).
  `#3236 <https://github.com/pybind/pybind11/pull/3236>`_

* Enable clang-tidy check to guard against inheritance slicing.
  `#3210 <https://github.com/pybind/pybind11/pull/3210>`_

* Legacy warning suppression pragma were removed from eigen.h. On Unix
  platforms, please use -isystem for Eigen include directories, to suppress
  compiler warnings originating from Eigen headers. Note that CMake does this
  by default. No adjustments are needed for Windows.
  `#3198 <https://github.com/pybind/pybind11/pull/3198>`_

* Format pybind11 with isort consistent ordering of imports
  `#3195 <https://github.com/pybind/pybind11/pull/3195>`_

* The warnings-suppression "pragma clamp" at the top/bottom of pybind11 was
  removed, clearing the path to refactoring and IWYU cleanup.
  `#3186 <https://github.com/pybind/pybind11/pull/3186>`_

* Enable most bugprone checks in clang-tidy and fix the found potential bugs
  and poor coding styles.
  `#3166 <https://github.com/pybind/pybind11/pull/3166>`_

* Add ``clang-tidy-readability`` rules to make boolean casts explicit improving
  code readability. Also enabled other misc and readability clang-tidy checks.
  `#3148 <https://github.com/pybind/pybind11/pull/3148>`_

* Move object in ``.pop()`` for list.
  `#3116 <https://github.com/pybind/pybind11/pull/3116>`_




v2.7.1 (Aug 3, 2021)
---------------------

Minor missing functionality added:

* Allow Python builtins to be used as callbacks in CPython.
  `#1413 <https://github.com/pybind/pybind11/pull/1413>`_

Bug fixes:

* Fix regression in CMake Python package config: improper use of absolute path.
  `#3144 <https://github.com/pybind/pybind11/pull/3144>`_

* Fix Mingw64 and add to the CI testing matrix.
  `#3132 <https://github.com/pybind/pybind11/pull/3132>`_

* Specified UTF8-encoding in setup.py calls of open().
  `#3137 <https://github.com/pybind/pybind11/pull/3137>`_

* Add clang-tidy-readability rules to make boolean casts explicit improving
  code readability. Also enabled other misc and readability clang-tidy checks.
  `#3148 <https://github.com/pybind/pybind11/pull/3148>`_

* Move object in ``.pop()`` for list.
  `#3116 <https://github.com/pybind/pybind11/pull/3116>`_

Backend and tidying up:

* Removed and fixed warning suppressions.
  `#3127 <https://github.com/pybind/pybind11/pull/3127>`_
  `#3129 <https://github.com/pybind/pybind11/pull/3129>`_
  `#3135 <https://github.com/pybind/pybind11/pull/3135>`_
  `#3141 <https://github.com/pybind/pybind11/pull/3141>`_
  `#3142 <https://github.com/pybind/pybind11/pull/3142>`_
  `#3150 <https://github.com/pybind/pybind11/pull/3150>`_
  `#3152 <https://github.com/pybind/pybind11/pull/3152>`_
  `#3160 <https://github.com/pybind/pybind11/pull/3160>`_
  `#3161 <https://github.com/pybind/pybind11/pull/3161>`_


v2.7.0 (Jul 16, 2021)
---------------------

New features:

* Enable ``py::implicitly_convertible<py::none, ...>`` for
  ``py::class_``-wrapped types.
  `#3059 <https://github.com/pybind/pybind11/pull/3059>`_

* Allow function pointer extraction from overloaded functions.
  `#2944 <https://github.com/pybind/pybind11/pull/2944>`_

* NumPy: added ``.char_()`` to type which gives the NumPy public ``char``
  result, which also distinguishes types by bit length (unlike ``.kind()``).
  `#2864 <https://github.com/pybind/pybind11/pull/2864>`_

* Add ``pybind11::bytearray`` to manipulate ``bytearray`` similar to ``bytes``.
  `#2799 <https://github.com/pybind/pybind11/pull/2799>`_

* ``pybind11/stl/filesystem.h`` registers a type caster that, on C++17/Python
  3.6+, converts ``std::filesystem::path`` to ``pathlib.Path`` and any
  ``os.PathLike`` to ``std::filesystem::path``.
  `#2730 <https://github.com/pybind/pybind11/pull/2730>`_

* A ``PYBIND11_VERSION_HEX`` define was added, similar to ``PY_VERSION_HEX``.
  `#3120 <https://github.com/pybind/pybind11/pull/3120>`_



Changes:

* ``py::str`` changed to exclusively hold ``PyUnicodeObject``. Previously
  ``py::str`` could also hold ``bytes``, which is probably surprising, was
  never documented, and can mask bugs (e.g. accidental use of ``py::str``
  instead of ``py::bytes``).
  `#2409 <https://github.com/pybind/pybind11/pull/2409>`_

* Add a safety guard to ensure that the Python GIL is held when C++ calls back
  into Python via ``object_api<>::operator()`` (e.g. ``py::function``
  ``__call__``).  (This feature is available for Python 3.6+ only.)
  `#2919 <https://github.com/pybind/pybind11/pull/2919>`_

* Catch a missing ``self`` argument in calls to ``__init__()``.
  `#2914 <https://github.com/pybind/pybind11/pull/2914>`_

* Use ``std::string_view`` if available to avoid a copy when passing an object
  to a ``std::ostream``.
  `#3042 <https://github.com/pybind/pybind11/pull/3042>`_

* An important warning about thread safety was added to the ``iostream.h``
  documentation; attempts to make ``py::scoped_ostream_redirect`` thread safe
  have been removed, as it was only partially effective.
  `#2995 <https://github.com/pybind/pybind11/pull/2995>`_


Fixes:

* Performance: avoid unnecessary strlen calls.
  `#3058 <https://github.com/pybind/pybind11/pull/3058>`_

* Fix auto-generated documentation string when using ``const T`` in
  ``pyarray_t``.
  `#3020 <https://github.com/pybind/pybind11/pull/3020>`_

* Unify error messages thrown by ``simple_collector``/``unpacking_collector``.
  `#3013 <https://github.com/pybind/pybind11/pull/3013>`_

* ``pybind11::builtin_exception`` is now explicitly exported, which means the
  types included/defined in different modules are identical, and exceptions
  raised in different modules can be caught correctly. The documentation was
  updated to explain that custom exceptions that are used across module
  boundaries need to be explicitly exported as well.
  `#2999 <https://github.com/pybind/pybind11/pull/2999>`_

* Fixed exception when printing UTF-8 to a ``scoped_ostream_redirect``.
  `#2982 <https://github.com/pybind/pybind11/pull/2982>`_

* Pickle support enhancement: ``setstate`` implementation will attempt to
  ``setattr`` ``__dict__`` only if the unpickled ``dict`` object is not empty,
  to not force use of ``py::dynamic_attr()`` unnecessarily.
  `#2972 <https://github.com/pybind/pybind11/pull/2972>`_

* Allow negative timedelta values to roundtrip.
  `#2870 <https://github.com/pybind/pybind11/pull/2870>`_

* Fix unchecked errors could potentially swallow signals/other exceptions.
  `#2863 <https://github.com/pybind/pybind11/pull/2863>`_

* Add null pointer check with ``std::localtime``.
  `#2846 <https://github.com/pybind/pybind11/pull/2846>`_

* Fix the ``weakref`` constructor from ``py::object`` to create a new
  ``weakref`` on conversion.
  `#2832 <https://github.com/pybind/pybind11/pull/2832>`_

* Avoid relying on exceptions in C++17 when getting a ``shared_ptr`` holder
  from a ``shared_from_this`` class.
  `#2819 <https://github.com/pybind/pybind11/pull/2819>`_

* Allow the codec's exception to be raised instead of :code:`RuntimeError` when
  casting from :code:`py::str` to :code:`std::string`.
  `#2903 <https://github.com/pybind/pybind11/pull/2903>`_


Build system improvements:

* In ``setup_helpers.py``, test for platforms that have some multiprocessing
  features but lack semaphores, which ``ParallelCompile`` requires.
  `#3043 <https://github.com/pybind/pybind11/pull/3043>`_

* Fix ``pybind11_INCLUDE_DIR`` in case ``CMAKE_INSTALL_INCLUDEDIR`` is
  absolute.
  `#3005 <https://github.com/pybind/pybind11/pull/3005>`_

* Fix bug not respecting ``WITH_SOABI`` or ``WITHOUT_SOABI`` to CMake.
  `#2938 <https://github.com/pybind/pybind11/pull/2938>`_

* Fix the default ``Pybind11Extension`` compilation flags with a Mingw64 python.
  `#2921 <https://github.com/pybind/pybind11/pull/2921>`_

* Clang on Windows: do not pass ``/MP`` (ignored flag).
  `#2824 <https://github.com/pybind/pybind11/pull/2824>`_

* ``pybind11.setup_helpers.intree_extensions`` can be used to generate
  ``Pybind11Extension`` instances from cpp files placed in the Python package
  source tree.
  `#2831 <https://github.com/pybind/pybind11/pull/2831>`_

Backend and tidying up:

* Enable clang-tidy performance, readability, and modernization checks
  throughout the codebase to enforce best coding practices.
  `#3046 <https://github.com/pybind/pybind11/pull/3046>`_,
  `#3049 <https://github.com/pybind/pybind11/pull/3049>`_,
  `#3051 <https://github.com/pybind/pybind11/pull/3051>`_,
  `#3052 <https://github.com/pybind/pybind11/pull/3052>`_,
  `#3080 <https://github.com/pybind/pybind11/pull/3080>`_, and
  `#3094 <https://github.com/pybind/pybind11/pull/3094>`_


* Checks for common misspellings were added to the pre-commit hooks.
  `#3076 <https://github.com/pybind/pybind11/pull/3076>`_

* Changed ``Werror`` to stricter ``Werror-all`` for Intel compiler and fixed
  minor issues.
  `#2948 <https://github.com/pybind/pybind11/pull/2948>`_

* Fixed compilation with GCC < 5 when the user defines ``_GLIBCXX_USE_CXX11_ABI``.
  `#2956 <https://github.com/pybind/pybind11/pull/2956>`_

* Added nox support for easier local testing and linting of contributions.
  `#3101 <https://github.com/pybind/pybind11/pull/3101>`_ and
  `#3121 <https://github.com/pybind/pybind11/pull/3121>`_

* Avoid RTD style issue with docutils 0.17+.
  `#3119 <https://github.com/pybind/pybind11/pull/3119>`_

* Support pipx run, such as ``pipx run pybind11 --include`` for a quick compile.
  `#3117 <https://github.com/pybind/pybind11/pull/3117>`_



v2.6.2 (Jan 26, 2021)
---------------------

Minor missing functionality added:

* enum: add missing Enum.value property.
  `#2739 <https://github.com/pybind/pybind11/pull/2739>`_

* Allow thread termination to be avoided during shutdown for CPython 3.7+ via
  ``.disarm`` for ``gil_scoped_acquire``/``gil_scoped_release``.
  `#2657 <https://github.com/pybind/pybind11/pull/2657>`_

Fixed or improved behavior in a few special cases:

* Fix bug where the constructor of ``object`` subclasses would not throw on
  being passed a Python object of the wrong type.
  `#2701 <https://github.com/pybind/pybind11/pull/2701>`_

* The ``type_caster`` for integers does not convert Python objects with
  ``__int__`` anymore with ``noconvert`` or during the first round of trying
  overloads.
  `#2698 <https://github.com/pybind/pybind11/pull/2698>`_

* When casting to a C++ integer, ``__index__`` is always called and not
  considered as conversion, consistent with Python 3.8+.
  `#2801 <https://github.com/pybind/pybind11/pull/2801>`_

Build improvements:

* Setup helpers: ``extra_compile_args`` and ``extra_link_args`` automatically set by
  Pybind11Extension are now prepended, which allows them to be overridden
  by user-set ``extra_compile_args`` and ``extra_link_args``.
  `#2808 <https://github.com/pybind/pybind11/pull/2808>`_

* Setup helpers: Don't trigger unused parameter warning.
  `#2735 <https://github.com/pybind/pybind11/pull/2735>`_

* CMake: Support running with ``--warn-uninitialized`` active.
  `#2806 <https://github.com/pybind/pybind11/pull/2806>`_

* CMake: Avoid error if included from two submodule directories.
  `#2804 <https://github.com/pybind/pybind11/pull/2804>`_

* CMake: Fix ``STATIC`` / ``SHARED`` being ignored in FindPython mode.
  `#2796 <https://github.com/pybind/pybind11/pull/2796>`_

* CMake: Respect the setting for ``CMAKE_CXX_VISIBILITY_PRESET`` if defined.
  `#2793 <https://github.com/pybind/pybind11/pull/2793>`_

* CMake: Fix issue with FindPython2/FindPython3 not working with ``pybind11::embed``.
  `#2662 <https://github.com/pybind/pybind11/pull/2662>`_

* CMake: mixing local and installed pybind11's would prioritize the installed
  one over the local one (regression in 2.6.0).
  `#2716 <https://github.com/pybind/pybind11/pull/2716>`_


Bug fixes:

* Fixed segfault in multithreaded environments when using
  ``scoped_ostream_redirect``.
  `#2675 <https://github.com/pybind/pybind11/pull/2675>`_

* Leave docstring unset when all docstring-related options are disabled, rather
  than set an empty string.
  `#2745 <https://github.com/pybind/pybind11/pull/2745>`_

* The module key in builtins that pybind11 uses to store its internals changed
  from std::string to a python str type (more natural on Python 2, no change on
  Python 3).
  `#2814 <https://github.com/pybind/pybind11/pull/2814>`_

* Fixed assertion error related to unhandled (later overwritten) exception in
  CPython 3.8 and 3.9 debug builds.
  `#2685 <https://github.com/pybind/pybind11/pull/2685>`_

* Fix ``py::gil_scoped_acquire`` assert with CPython 3.9 debug build.
  `#2683 <https://github.com/pybind/pybind11/pull/2683>`_

* Fix issue with a test failing on pytest 6.2.
  `#2741 <https://github.com/pybind/pybind11/pull/2741>`_

Warning fixes:

* Fix warning modifying constructor parameter 'flag' that shadows a field of
  'set_flag' ``[-Wshadow-field-in-constructor-modified]``.
  `#2780 <https://github.com/pybind/pybind11/pull/2780>`_

* Suppressed some deprecation warnings about old-style
  ``__init__``/``__setstate__`` in the tests.
  `#2759 <https://github.com/pybind/pybind11/pull/2759>`_

Valgrind work:

* Fix invalid access when calling a pybind11 ``__init__`` on a non-pybind11
  class instance.
  `#2755 <https://github.com/pybind/pybind11/pull/2755>`_

* Fixed various minor memory leaks in pybind11's test suite.
  `#2758 <https://github.com/pybind/pybind11/pull/2758>`_

* Resolved memory leak in cpp_function initialization when exceptions occurred.
  `#2756 <https://github.com/pybind/pybind11/pull/2756>`_

* Added a Valgrind build, checking for leaks and memory-related UB, to CI.
  `#2746 <https://github.com/pybind/pybind11/pull/2746>`_

Compiler support:

* Intel compiler was not activating C++14 support due to a broken define.
  `#2679 <https://github.com/pybind/pybind11/pull/2679>`_

* Support ICC and NVIDIA HPC SDK in C++17 mode.
  `#2729 <https://github.com/pybind/pybind11/pull/2729>`_

* Support Intel OneAPI compiler (ICC 20.2) and add to CI.
  `#2573 <https://github.com/pybind/pybind11/pull/2573>`_



v2.6.1 (Nov 11, 2020)
---------------------

* ``py::exec``, ``py::eval``, and ``py::eval_file`` now add the builtins module
  as ``"__builtins__"`` to their ``globals`` argument, better matching ``exec``
  and ``eval`` in pure Python.
  `#2616 <https://github.com/pybind/pybind11/pull/2616>`_

* ``setup_helpers`` will no longer set a minimum macOS version higher than the
  current version.
  `#2622 <https://github.com/pybind/pybind11/pull/2622>`_

* Allow deleting static properties.
  `#2629 <https://github.com/pybind/pybind11/pull/2629>`_

* Seal a leak in ``def_buffer``, cleaning up the ``capture`` object after the
  ``class_`` object goes out of scope.
  `#2634 <https://github.com/pybind/pybind11/pull/2634>`_

* ``pybind11_INCLUDE_DIRS`` was incorrect, potentially causing a regression if
  it was expected to include ``PYTHON_INCLUDE_DIRS`` (please use targets
  instead).
  `#2636 <https://github.com/pybind/pybind11/pull/2636>`_

* Added parameter names to the ``py::enum_`` constructor and methods, avoiding
  ``arg0`` in the generated docstrings.
  `#2637 <https://github.com/pybind/pybind11/pull/2637>`_

* Added ``needs_recompile`` optional function to the ``ParallelCompiler``
  helper, to allow a recompile to be skipped based on a user-defined function.
  `#2643 <https://github.com/pybind/pybind11/pull/2643>`_


v2.6.0 (Oct 21, 2020)
---------------------

See :ref:`upgrade-guide-2.6` for help upgrading to the new version.

New features:

* Keyword-only arguments supported in Python 2 or 3 with ``py::kw_only()``.
  `#2100 <https://github.com/pybind/pybind11/pull/2100>`_

* Positional-only arguments supported in Python 2 or 3 with ``py::pos_only()``.
  `#2459 <https://github.com/pybind/pybind11/pull/2459>`_

* ``py::is_final()`` class modifier to block subclassing (CPython only).
  `#2151 <https://github.com/pybind/pybind11/pull/2151>`_

* Added ``py::prepend()``, allowing a function to be placed at the beginning of
  the overload chain.
  `#1131 <https://github.com/pybind/pybind11/pull/1131>`_

* Access to the type object now provided with ``py::type::of<T>()`` and
  ``py::type::of(h)``.
  `#2364 <https://github.com/pybind/pybind11/pull/2364>`_

* Perfect forwarding support for methods.
  `#2048 <https://github.com/pybind/pybind11/pull/2048>`_

* Added ``py::error_already_set::discard_as_unraisable()``.
  `#2372 <https://github.com/pybind/pybind11/pull/2372>`_

* ``py::hash`` is now public.
  `#2217 <https://github.com/pybind/pybind11/pull/2217>`_

* ``py::class_<union_type>`` is now supported. Note that writing to one data
  member of the union and reading another (type punning) is UB in C++. Thus
  pybind11-bound enums should never be used for such conversions.
  `#2320 <https://github.com/pybind/pybind11/pull/2320>`_.

* Classes now check local scope when registering members, allowing a subclass
  to have a member with the same name as a parent (such as an enum).
  `#2335 <https://github.com/pybind/pybind11/pull/2335>`_

Code correctness features:

* Error now thrown when ``__init__`` is forgotten on subclasses.
  `#2152 <https://github.com/pybind/pybind11/pull/2152>`_

* Throw error if conversion to a pybind11 type if the Python object isn't a
  valid instance of that type, such as ``py::bytes(o)`` when ``py::object o``
  isn't a bytes instance.
  `#2349 <https://github.com/pybind/pybind11/pull/2349>`_

* Throw if conversion to ``str`` fails.
  `#2477 <https://github.com/pybind/pybind11/pull/2477>`_


API changes:

* ``py::module`` was renamed ``py::module_`` to avoid issues with C++20 when
  used unqualified, but an alias ``py::module`` is provided for backward
  compatibility.
  `#2489 <https://github.com/pybind/pybind11/pull/2489>`_

* Public constructors for ``py::module_`` have been deprecated; please use
  ``pybind11::module_::create_extension_module`` if you were using the public
  constructor (fairly rare after ``PYBIND11_MODULE`` was introduced).
  `#2552 <https://github.com/pybind/pybind11/pull/2552>`_

* ``PYBIND11_OVERLOAD*`` macros and ``get_overload`` function replaced by
  correctly-named ``PYBIND11_OVERRIDE*`` and ``get_override``, fixing
  inconsistencies in the presence of a closing ``;`` in these macros.
  ``get_type_overload`` is deprecated.
  `#2325 <https://github.com/pybind/pybind11/pull/2325>`_

Packaging / building improvements:

* The Python package was reworked to be more powerful and useful.
  `#2433 <https://github.com/pybind/pybind11/pull/2433>`_

  * :ref:`build-setuptools` is easier thanks to a new
    ``pybind11.setup_helpers`` module, which provides utilities to use
    setuptools with pybind11. It can be used via PEP 518, ``setup_requires``,
    or by directly importing or copying ``setup_helpers.py`` into your project.

  * CMake configuration files are now included in the Python package. Use
    ``pybind11.get_cmake_dir()`` or ``python -m pybind11 --cmakedir`` to get
    the directory with the CMake configuration files, or include the
    site-packages location in your ``CMAKE_MODULE_PATH``. Or you can use the
    new ``pybind11[global]`` extra when you install ``pybind11``, which
    installs the CMake files and headers into your base environment in the
    standard location.

  * ``pybind11-config`` is another way to write ``python -m pybind11`` if you
    have your PATH set up.

  * Added external typing support to the helper module, code from
    ``import pybind11`` can now be type checked.
    `#2588 <https://github.com/pybind/pybind11/pull/2588>`_

* Minimum CMake required increased to 3.4.
  `#2338 <https://github.com/pybind/pybind11/pull/2338>`_ and
  `#2370 <https://github.com/pybind/pybind11/pull/2370>`_

  * Full integration with CMake's C++ standard system and compile features
    replaces ``PYBIND11_CPP_STANDARD``.

  * Generated config file is now portable to different Python/compiler/CMake
    versions.

  * Virtual environments prioritized if ``PYTHON_EXECUTABLE`` is not set
    (``venv``, ``virtualenv``, and ``conda``) (similar to the new FindPython
    mode).

  * Other CMake features now natively supported, like
    ``CMAKE_INTERPROCEDURAL_OPTIMIZATION``, ``set(CMAKE_CXX_VISIBILITY_PRESET
    hidden)``.

  * ``CUDA`` as a language is now supported.

  * Helper functions ``pybind11_strip``, ``pybind11_extension``,
    ``pybind11_find_import`` added, see :doc:`cmake/index`.

  * Optional :ref:`find-python-mode` and :ref:`nopython-mode` with CMake.
    `#2370 <https://github.com/pybind/pybind11/pull/2370>`_

* Uninstall target added.
  `#2265 <https://github.com/pybind/pybind11/pull/2265>`_ and
  `#2346 <https://github.com/pybind/pybind11/pull/2346>`_

* ``pybind11_add_module()`` now accepts an optional ``OPT_SIZE`` flag that
  switches the binding target to size-based optimization if the global build
  type can not always be fixed to ``MinSizeRel`` (except in debug mode, where
  optimizations remain disabled).  ``MinSizeRel`` or this flag reduces binary
  size quite substantially (~25% on some platforms).
  `#2463 <https://github.com/pybind/pybind11/pull/2463>`_

Smaller or developer focused features and fixes:

* Moved ``mkdoc.py`` to a new repo, `pybind11-mkdoc`_. There are no longer
  submodules in the main repo.

* ``py::memoryview`` segfault fix and update, with new
  ``py::memoryview::from_memory`` in Python 3, and documentation.
  `#2223 <https://github.com/pybind/pybind11/pull/2223>`_

* Fix for ``buffer_info`` on Python 2.
  `#2503 <https://github.com/pybind/pybind11/pull/2503>`_

* If ``__eq__`` defined but not ``__hash__``, ``__hash__`` is now set to
  ``None``.
  `#2291 <https://github.com/pybind/pybind11/pull/2291>`_

* ``py::ellipsis`` now also works on Python 2.
  `#2360 <https://github.com/pybind/pybind11/pull/2360>`_

* Pointer to ``std::tuple`` & ``std::pair`` supported in cast.
  `#2334 <https://github.com/pybind/pybind11/pull/2334>`_

* Small fixes in NumPy support. ``py::array`` now uses ``py::ssize_t`` as first
  argument type.
  `#2293 <https://github.com/pybind/pybind11/pull/2293>`_

* Added missing signature for ``py::array``.
  `#2363 <https://github.com/pybind/pybind11/pull/2363>`_

* ``unchecked_mutable_reference`` has access to operator ``()`` and ``[]`` when
  const.
  `#2514 <https://github.com/pybind/pybind11/pull/2514>`_

* ``py::vectorize`` is now supported on functions that return void.
  `#1969 <https://github.com/pybind/pybind11/pull/1969>`_

* ``py::capsule`` supports ``get_pointer`` and ``set_pointer``.
  `#1131 <https://github.com/pybind/pybind11/pull/1131>`_

* Fix crash when different instances share the same pointer of the same type.
  `#2252 <https://github.com/pybind/pybind11/pull/2252>`_

* Fix for ``py::len`` not clearing Python's error state when it fails and throws.
  `#2575 <https://github.com/pybind/pybind11/pull/2575>`_

* Bugfixes related to more extensive testing, new GitHub Actions CI.
  `#2321 <https://github.com/pybind/pybind11/pull/2321>`_

* Bug in timezone issue in Eastern hemisphere midnight fixed.
  `#2438 <https://github.com/pybind/pybind11/pull/2438>`_

* ``std::chrono::time_point`` now works when the resolution is not the same as
  the system.
  `#2481 <https://github.com/pybind/pybind11/pull/2481>`_

* Bug fixed where ``py::array_t`` could accept arrays that did not match the
  requested ordering.
  `#2484 <https://github.com/pybind/pybind11/pull/2484>`_

* Avoid a segfault on some compilers when types are removed in Python.
  `#2564 <https://github.com/pybind/pybind11/pull/2564>`_

* ``py::arg::none()`` is now also respected when passing keyword arguments.
  `#2611 <https://github.com/pybind/pybind11/pull/2611>`_

* PyPy fixes, PyPy 7.3.x now supported, including PyPy3. (Known issue with
  PyPy2 and Windows `#2596 <https://github.com/pybind/pybind11/issues/2596>`_).
  `#2146 <https://github.com/pybind/pybind11/pull/2146>`_

* CPython 3.9.0 workaround for undefined behavior (macOS segfault).
  `#2576 <https://github.com/pybind/pybind11/pull/2576>`_

* CPython 3.9 warning fixes.
  `#2253 <https://github.com/pybind/pybind11/pull/2253>`_

* Improved C++20 support, now tested in CI.
  `#2489 <https://github.com/pybind/pybind11/pull/2489>`_
  `#2599 <https://github.com/pybind/pybind11/pull/2599>`_

* Improved but still incomplete debug Python interpreter support.
  `#2025 <https://github.com/pybind/pybind11/pull/2025>`_

* NVCC (CUDA 11) now supported and tested in CI.
  `#2461 <https://github.com/pybind/pybind11/pull/2461>`_

* NVIDIA PGI compilers now supported and tested in CI.
  `#2475 <https://github.com/pybind/pybind11/pull/2475>`_

* At least Intel 18 now explicitly required when compiling with Intel.
  `#2577 <https://github.com/pybind/pybind11/pull/2577>`_

* Extensive style checking in CI, with `pre-commit`_ support. Code
  modernization, checked by clang-tidy.

* Expanded docs, including new main page, new installing section, and CMake
  helpers page, along with over a dozen new sections on existing pages.

* In GitHub, new docs for contributing and new issue templates.

.. _pre-commit: https://pre-commit.com

.. _pybind11-mkdoc: https://github.com/pybind/pybind11-mkdoc

v2.5.0 (Mar 31, 2020)
-----------------------------------------------------

* Use C++17 fold expressions in type casters, if available. This can
  improve performance during overload resolution when functions have
  multiple arguments.
  `#2043 <https://github.com/pybind/pybind11/pull/2043>`_.

* Changed include directory resolution in ``pybind11/__init__.py``
  and installation in ``setup.py``. This fixes a number of open issues
  where pybind11 headers could not be found in certain environments.
  `#1995 <https://github.com/pybind/pybind11/pull/1995>`_.

* C++20 ``char8_t`` and ``u8string`` support. `#2026
  <https://github.com/pybind/pybind11/pull/2026>`_.

* CMake: search for Python 3.9. `bb9c91
  <https://github.com/pybind/pybind11/commit/bb9c91>`_.

* Fixes for MSYS-based build environments.
  `#2087 <https://github.com/pybind/pybind11/pull/2087>`_,
  `#2053 <https://github.com/pybind/pybind11/pull/2053>`_.

* STL bindings for ``std::vector<...>::clear``. `#2074
  <https://github.com/pybind/pybind11/pull/2074>`_.

* Read-only flag for ``py::buffer``. `#1466
  <https://github.com/pybind/pybind11/pull/1466>`_.

* Exception handling during module initialization.
  `bf2b031 <https://github.com/pybind/pybind11/commit/bf2b031>`_.

* Support linking against a CPython debug build.
  `#2025 <https://github.com/pybind/pybind11/pull/2025>`_.

* Fixed issues involving the availability and use of aligned ``new`` and
  ``delete``. `#1988 <https://github.com/pybind/pybind11/pull/1988>`_,
  `759221 <https://github.com/pybind/pybind11/commit/759221>`_.

* Fixed a resource leak upon interpreter shutdown.
  `#2020 <https://github.com/pybind/pybind11/pull/2020>`_.

* Fixed error handling in the boolean caster.
  `#1976 <https://github.com/pybind/pybind11/pull/1976>`_.

v2.4.3 (Oct 15, 2019)
-----------------------------------------------------

* Adapt pybind11 to a C API convention change in Python 3.8. `#1950
  <https://github.com/pybind/pybind11/pull/1950>`_.

v2.4.2 (Sep 21, 2019)
-----------------------------------------------------

* Replaced usage of a C++14 only construct. `#1929
  <https://github.com/pybind/pybind11/pull/1929>`_.

* Made an ifdef future-proof for Python >= 4. `f3109d
  <https://github.com/pybind/pybind11/commit/f3109d>`_.

v2.4.1 (Sep 20, 2019)
-----------------------------------------------------

* Fixed a problem involving implicit conversion from enumerations to integers
  on Python 3.8. `#1780 <https://github.com/pybind/pybind11/pull/1780>`_.

v2.4.0 (Sep 19, 2019)
-----------------------------------------------------

* Try harder to keep pybind11-internal data structures separate when there
  are potential ABI incompatibilities. Fixes crashes that occurred when loading
  multiple pybind11 extensions that were e.g. compiled by GCC (libstdc++)
  and Clang (libc++).
  `#1588 <https://github.com/pybind/pybind11/pull/1588>`_ and
  `c9f5a <https://github.com/pybind/pybind11/commit/c9f5a>`_.

* Added support for ``__await__``, ``__aiter__``, and ``__anext__`` protocols.
  `#1842 <https://github.com/pybind/pybind11/pull/1842>`_.

* ``pybind11_add_module()``: don't strip symbols when compiling in
  ``RelWithDebInfo`` mode. `#1980
  <https://github.com/pybind/pybind11/pull/1980>`_.

* ``enum_``: Reproduce Python behavior when comparing against invalid values
  (e.g. ``None``, strings, etc.). Add back support for ``__invert__()``.
  `#1912 <https://github.com/pybind/pybind11/pull/1912>`_,
  `#1907 <https://github.com/pybind/pybind11/pull/1907>`_.

* List insertion operation for ``py::list``.
  Added ``.empty()`` to all collection types.
  Added ``py::set::contains()`` and ``py::dict::contains()``.
  `#1887 <https://github.com/pybind/pybind11/pull/1887>`_,
  `#1884 <https://github.com/pybind/pybind11/pull/1884>`_,
  `#1888 <https://github.com/pybind/pybind11/pull/1888>`_.

* ``py::details::overload_cast_impl`` is available in C++11 mode, can be used
  like ``overload_cast`` with an additional set of parentheses.
  `#1581 <https://github.com/pybind/pybind11/pull/1581>`_.

* Fixed ``get_include()`` on Conda.
  `#1877 <https://github.com/pybind/pybind11/pull/1877>`_.

* ``stl_bind.h``: negative indexing support.
  `#1882 <https://github.com/pybind/pybind11/pull/1882>`_.

* Minor CMake fix to add MinGW compatibility.
  `#1851 <https://github.com/pybind/pybind11/pull/1851>`_.

* GIL-related fixes.
  `#1836 <https://github.com/pybind/pybind11/pull/1836>`_,
  `8b90b <https://github.com/pybind/pybind11/commit/8b90b>`_.

* Other very minor/subtle fixes and improvements.
  `#1329 <https://github.com/pybind/pybind11/pull/1329>`_,
  `#1910 <https://github.com/pybind/pybind11/pull/1910>`_,
  `#1863 <https://github.com/pybind/pybind11/pull/1863>`_,
  `#1847 <https://github.com/pybind/pybind11/pull/1847>`_,
  `#1890 <https://github.com/pybind/pybind11/pull/1890>`_,
  `#1860 <https://github.com/pybind/pybind11/pull/1860>`_,
  `#1848 <https://github.com/pybind/pybind11/pull/1848>`_,
  `#1821 <https://github.com/pybind/pybind11/pull/1821>`_,
  `#1837 <https://github.com/pybind/pybind11/pull/1837>`_,
  `#1833 <https://github.com/pybind/pybind11/pull/1833>`_,
  `#1748 <https://github.com/pybind/pybind11/pull/1748>`_,
  `#1852 <https://github.com/pybind/pybind11/pull/1852>`_.

v2.3.0 (June 11, 2019)
-----------------------------------------------------

* Significantly reduced module binary size (10-20%) when compiled in C++11 mode
  with GCC/Clang, or in any mode with MSVC. Function signatures are now always
  precomputed at compile time (this was previously only available in C++14 mode
  for non-MSVC compilers).
  `#934 <https://github.com/pybind/pybind11/pull/934>`_.

* Add basic support for tag-based static polymorphism, where classes
  provide a method to returns the desired type of an instance.
  `#1326 <https://github.com/pybind/pybind11/pull/1326>`_.

* Python type wrappers (``py::handle``, ``py::object``, etc.)
  now support map Python's number protocol onto C++ arithmetic
  operators such as ``operator+``, ``operator/=``, etc.
  `#1511 <https://github.com/pybind/pybind11/pull/1511>`_.

* A number of improvements related to enumerations:

   1. The ``enum_`` implementation was rewritten from scratch to reduce
      code bloat. Rather than instantiating a full implementation for each
      enumeration, most code is now contained in a generic base class.
      `#1511 <https://github.com/pybind/pybind11/pull/1511>`_.

   2. The ``value()``  method of ``py::enum_`` now accepts an optional
      docstring that will be shown in the documentation of the associated
      enumeration. `#1160 <https://github.com/pybind/pybind11/pull/1160>`_.

   3. check for already existing enum value and throw an error if present.
      `#1453 <https://github.com/pybind/pybind11/pull/1453>`_.

* Support for over-aligned type allocation via C++17's aligned ``new``
  statement. `#1582 <https://github.com/pybind/pybind11/pull/1582>`_.

* Added ``py::ellipsis()`` method for slicing of multidimensional NumPy arrays
  `#1502 <https://github.com/pybind/pybind11/pull/1502>`_.

* Numerous Improvements to the ``mkdoc.py`` script for extracting documentation
  from C++ header files.
  `#1788 <https://github.com/pybind/pybind11/pull/1788>`_.

* ``pybind11_add_module()``: allow including Python as a ``SYSTEM`` include path.
  `#1416 <https://github.com/pybind/pybind11/pull/1416>`_.

* ``pybind11/stl.h`` does not convert strings to ``vector<string>`` anymore.
  `#1258 <https://github.com/pybind/pybind11/issues/1258>`_.

* Mark static methods as such to fix auto-generated Sphinx documentation.
  `#1732 <https://github.com/pybind/pybind11/pull/1732>`_.

* Re-throw forced unwind exceptions (e.g. during pthread termination).
  `#1208 <https://github.com/pybind/pybind11/pull/1208>`_.

* Added ``__contains__`` method to the bindings of maps (``std::map``,
  ``std::unordered_map``).
  `#1767 <https://github.com/pybind/pybind11/pull/1767>`_.

* Improvements to ``gil_scoped_acquire``.
  `#1211 <https://github.com/pybind/pybind11/pull/1211>`_.

* Type caster support for ``std::deque<T>``.
  `#1609 <https://github.com/pybind/pybind11/pull/1609>`_.

* Support for ``std::unique_ptr`` holders, whose deleters differ between a base and derived
  class. `#1353 <https://github.com/pybind/pybind11/pull/1353>`_.

* Construction of STL array/vector-like data structures from
  iterators. Added an ``extend()`` operation.
  `#1709 <https://github.com/pybind/pybind11/pull/1709>`_,

* CMake build system improvements for projects that include non-C++
  files (e.g. plain C, CUDA) in ``pybind11_add_module`` et al.
  `#1678 <https://github.com/pybind/pybind11/pull/1678>`_.

* Fixed asynchronous invocation and deallocation of Python functions
  wrapped in ``std::function``.
  `#1595 <https://github.com/pybind/pybind11/pull/1595>`_.

* Fixes regarding return value policy propagation in STL type casters.
  `#1603 <https://github.com/pybind/pybind11/pull/1603>`_.

* Fixed scoped enum comparisons.
  `#1571 <https://github.com/pybind/pybind11/pull/1571>`_.

* Fixed iostream redirection for code that releases the GIL.
  `#1368 <https://github.com/pybind/pybind11/pull/1368>`_,

* A number of CI-related fixes.
  `#1757 <https://github.com/pybind/pybind11/pull/1757>`_,
  `#1744 <https://github.com/pybind/pybind11/pull/1744>`_,
  `#1670 <https://github.com/pybind/pybind11/pull/1670>`_.

v2.2.4 (September 11, 2018)
-----------------------------------------------------

* Use new Python 3.7 Thread Specific Storage (TSS) implementation if available.
  `#1454 <https://github.com/pybind/pybind11/pull/1454>`_,
  `#1517 <https://github.com/pybind/pybind11/pull/1517>`_.

* Fixes for newer MSVC versions and C++17 mode.
  `#1347 <https://github.com/pybind/pybind11/pull/1347>`_,
  `#1462 <https://github.com/pybind/pybind11/pull/1462>`_.

* Propagate return value policies to type-specific casters
  when casting STL containers.
  `#1455 <https://github.com/pybind/pybind11/pull/1455>`_.

* Allow ostream-redirection of more than 1024 characters.
  `#1479 <https://github.com/pybind/pybind11/pull/1479>`_.

* Set ``Py_DEBUG`` define when compiling against a debug Python build.
  `#1438 <https://github.com/pybind/pybind11/pull/1438>`_.

* Untangle integer logic in number type caster to work for custom
  types that may only be castable to a restricted set of builtin types.
  `#1442 <https://github.com/pybind/pybind11/pull/1442>`_.

* CMake build system: Remember Python version in cache file.
  `#1434 <https://github.com/pybind/pybind11/pull/1434>`_.

* Fix for custom smart pointers: use ``std::addressof`` to obtain holder
  address instead of ``operator&``.
  `#1435 <https://github.com/pybind/pybind11/pull/1435>`_.

* Properly report exceptions thrown during module initialization.
  `#1362 <https://github.com/pybind/pybind11/pull/1362>`_.

* Fixed a segmentation fault when creating empty-shaped NumPy array.
  `#1371 <https://github.com/pybind/pybind11/pull/1371>`_.

* The version of Intel C++ compiler must be >= 2017, and this is now checked by
  the header files. `#1363 <https://github.com/pybind/pybind11/pull/1363>`_.

* A few minor typo fixes and improvements to the test suite, and
  patches that silence compiler warnings.

* Vectors now support construction from generators, as well as ``extend()`` from a
  list or generator.
  `#1496 <https://github.com/pybind/pybind11/pull/1496>`_.


v2.2.3 (April 29, 2018)
-----------------------------------------------------

* The pybind11 header location detection was replaced by a new implementation
  that no longer depends on ``pip`` internals (the recently released ``pip``
  10 has restricted access to this API).
  `#1190 <https://github.com/pybind/pybind11/pull/1190>`_.

* Small adjustment to an implementation detail to work around a compiler segmentation fault in Clang 3.3/3.4.
  `#1350 <https://github.com/pybind/pybind11/pull/1350>`_.

* The minimal supported version of the Intel compiler was >= 17.0 since
  pybind11 v2.1. This check is now explicit, and a compile-time error is raised
  if the compiler meet the requirement.
  `#1363 <https://github.com/pybind/pybind11/pull/1363>`_.

* Fixed an endianness-related fault in the test suite.
  `#1287 <https://github.com/pybind/pybind11/pull/1287>`_.

v2.2.2 (February 7, 2018)
-----------------------------------------------------

* Fixed a segfault when combining embedded interpreter
  shutdown/reinitialization with external loaded pybind11 modules.
  `#1092 <https://github.com/pybind/pybind11/pull/1092>`_.

* Eigen support: fixed a bug where Nx1/1xN numpy inputs couldn't be passed as
  arguments to Eigen vectors (which for Eigen are simply compile-time fixed
  Nx1/1xN matrices).
  `#1106 <https://github.com/pybind/pybind11/pull/1106>`_.

* Clarified to license by moving the licensing of contributions from
  ``LICENSE`` into ``CONTRIBUTING.md``: the licensing of contributions is not
  actually part of the software license as distributed.  This isn't meant to be
  a substantial change in the licensing of the project, but addresses concerns
  that the clause made the license non-standard.
  `#1109 <https://github.com/pybind/pybind11/issues/1109>`_.

* Fixed a regression introduced in 2.1 that broke binding functions with lvalue
  character literal arguments.
  `#1128 <https://github.com/pybind/pybind11/pull/1128>`_.

* MSVC: fix for compilation failures under /permissive-, and added the flag to
  the appveyor test suite.
  `#1155 <https://github.com/pybind/pybind11/pull/1155>`_.

* Fixed ``__qualname__`` generation, and in turn, fixes how class names
  (especially nested class names) are shown in generated docstrings.
  `#1171 <https://github.com/pybind/pybind11/pull/1171>`_.

* Updated the FAQ with a suggested project citation reference.
  `#1189 <https://github.com/pybind/pybind11/pull/1189>`_.

* Added fixes for deprecation warnings when compiled under C++17 with
  ``-Wdeprecated`` turned on, and add ``-Wdeprecated`` to the test suite
  compilation flags.
  `#1191 <https://github.com/pybind/pybind11/pull/1191>`_.

* Fixed outdated PyPI URLs in ``setup.py``.
  `#1213 <https://github.com/pybind/pybind11/pull/1213>`_.

* Fixed a refcount leak for arguments that end up in a ``py::args`` argument
  for functions with both fixed positional and ``py::args`` arguments.
  `#1216 <https://github.com/pybind/pybind11/pull/1216>`_.

* Fixed a potential segfault resulting from possible premature destruction of
  ``py::args``/``py::kwargs`` arguments with overloaded functions.
  `#1223 <https://github.com/pybind/pybind11/pull/1223>`_.

* Fixed ``del map[item]`` for a ``stl_bind.h`` bound stl map.
  `#1229 <https://github.com/pybind/pybind11/pull/1229>`_.

* Fixed a regression from v2.1.x where the aggregate initialization could
  unintentionally end up at a constructor taking a templated
  ``std::initializer_list<T>`` argument.
  `#1249 <https://github.com/pybind/pybind11/pull/1249>`_.

* Fixed an issue where calling a function with a keep_alive policy on the same
  nurse/patient pair would cause the internal patient storage to needlessly
  grow (unboundedly, if the nurse is long-lived).
  `#1251 <https://github.com/pybind/pybind11/issues/1251>`_.

* Various other minor fixes.

v2.2.1 (September 14, 2017)
-----------------------------------------------------

* Added ``py::module_::reload()`` member function for reloading a module.
  `#1040 <https://github.com/pybind/pybind11/pull/1040>`_.

* Fixed a reference leak in the number converter.
  `#1078 <https://github.com/pybind/pybind11/pull/1078>`_.

* Fixed compilation with Clang on host GCC < 5 (old libstdc++ which isn't fully
  C++11 compliant). `#1062 <https://github.com/pybind/pybind11/pull/1062>`_.

* Fixed a regression where the automatic ``std::vector<bool>`` caster would
  fail to compile. The same fix also applies to any container which returns
  element proxies instead of references.
  `#1053 <https://github.com/pybind/pybind11/pull/1053>`_.

* Fixed a regression where the ``py::keep_alive`` policy could not be applied
  to constructors. `#1065 <https://github.com/pybind/pybind11/pull/1065>`_.

* Fixed a nullptr dereference when loading a ``py::module_local`` type
  that's only registered in an external module.
  `#1058 <https://github.com/pybind/pybind11/pull/1058>`_.

* Fixed implicit conversion of accessors to types derived from ``py::object``.
  `#1076 <https://github.com/pybind/pybind11/pull/1076>`_.

* The ``name`` in ``PYBIND11_MODULE(name, variable)`` can now be a macro.
  `#1082 <https://github.com/pybind/pybind11/pull/1082>`_.

* Relaxed overly strict ``py::pickle()`` check for matching get and set types.
  `#1064 <https://github.com/pybind/pybind11/pull/1064>`_.

* Conversion errors now try to be more informative when it's likely that
  a missing header is the cause (e.g. forgetting ``<pybind11/stl.h>``).
  `#1077 <https://github.com/pybind/pybind11/pull/1077>`_.

v2.2.0 (August 31, 2017)
-----------------------------------------------------

* Support for embedding the Python interpreter. See the
  :doc:`documentation page </advanced/embedding>` for a
  full overview of the new features.
  `#774 <https://github.com/pybind/pybind11/pull/774>`_,
  `#889 <https://github.com/pybind/pybind11/pull/889>`_,
  `#892 <https://github.com/pybind/pybind11/pull/892>`_,
  `#920 <https://github.com/pybind/pybind11/pull/920>`_.

  .. code-block:: cpp

      #include <pybind11/embed.h>
      namespace py = pybind11;

      int main() {
          py::scoped_interpreter guard{}; // start the interpreter and keep it alive

          py::print("Hello, World!"); // use the Python API
      }

* Support for inheriting from multiple C++ bases in Python.
  `#693 <https://github.com/pybind/pybind11/pull/693>`_.

  .. code-block:: python

      from cpp_module import CppBase1, CppBase2


      class PyDerived(CppBase1, CppBase2):
          def __init__(self):
              CppBase1.__init__(self)  # C++ bases must be initialized explicitly
              CppBase2.__init__(self)

* ``PYBIND11_MODULE`` is now the preferred way to create module entry points.
  ``PYBIND11_PLUGIN`` is deprecated. See :ref:`macros` for details.
  `#879 <https://github.com/pybind/pybind11/pull/879>`_.

  .. code-block:: cpp

      // new
      PYBIND11_MODULE(example, m) {
          m.def("add", [](int a, int b) { return a + b; });
      }

      // old
      PYBIND11_PLUGIN(example) {
          py::module m("example");
          m.def("add", [](int a, int b) { return a + b; });
          return m.ptr();
      }

* pybind11's headers and build system now more strictly enforce hidden symbol
  visibility for extension modules. This should be seamless for most users,
  but see the :doc:`upgrade` if you use a custom build system.
  `#995 <https://github.com/pybind/pybind11/pull/995>`_.

* Support for ``py::module_local`` types which allow multiple modules to
  export the same C++ types without conflicts. This is useful for opaque
  types like ``std::vector<int>``. ``py::bind_vector`` and ``py::bind_map``
  now default to ``py::module_local`` if their elements are builtins or
  local types. See :ref:`module_local` for details.
  `#949 <https://github.com/pybind/pybind11/pull/949>`_,
  `#981 <https://github.com/pybind/pybind11/pull/981>`_,
  `#995 <https://github.com/pybind/pybind11/pull/995>`_,
  `#997 <https://github.com/pybind/pybind11/pull/997>`_.

* Custom constructors can now be added very easily using lambdas or factory
  functions which return a class instance by value, pointer or holder. This
  supersedes the old placement-new ``__init__`` technique.
  See :ref:`custom_constructors` for details.
  `#805 <https://github.com/pybind/pybind11/pull/805>`_,
  `#1014 <https://github.com/pybind/pybind11/pull/1014>`_.

  .. code-block:: cpp

      struct Example {
          Example(std::string);
      };

      py::class_<Example>(m, "Example")
          .def(py::init<std::string>()) // existing constructor
          .def(py::init([](int n) { // custom constructor
              return std::make_unique<Example>(std::to_string(n));
          }));

* Similarly to custom constructors, pickling support functions are now bound
  using the ``py::pickle()`` adaptor which improves type safety. See the
  :doc:`upgrade` and :ref:`pickling` for details.
  `#1038 <https://github.com/pybind/pybind11/pull/1038>`_.

* Builtin support for converting C++17 standard library types and general
  conversion improvements:

  1. C++17 ``std::variant`` is supported right out of the box. C++11/14
     equivalents (e.g. ``boost::variant``) can also be added with a simple
     user-defined specialization. See :ref:`cpp17_container_casters` for details.
     `#811 <https://github.com/pybind/pybind11/pull/811>`_,
     `#845 <https://github.com/pybind/pybind11/pull/845>`_,
     `#989 <https://github.com/pybind/pybind11/pull/989>`_.

  2. Out-of-the-box support for C++17 ``std::string_view``.
     `#906 <https://github.com/pybind/pybind11/pull/906>`_.

  3. Improved compatibility of the builtin ``optional`` converter.
     `#874 <https://github.com/pybind/pybind11/pull/874>`_.

  4. The ``bool`` converter now accepts ``numpy.bool_`` and types which
     define ``__bool__`` (Python 3.x) or ``__nonzero__`` (Python 2.7).
     `#925 <https://github.com/pybind/pybind11/pull/925>`_.

  5. C++-to-Python casters are now more efficient and move elements out
     of rvalue containers whenever possible.
     `#851 <https://github.com/pybind/pybind11/pull/851>`_,
     `#936 <https://github.com/pybind/pybind11/pull/936>`_,
     `#938 <https://github.com/pybind/pybind11/pull/938>`_.

  6. Fixed ``bytes`` to ``std::string/char*`` conversion on Python 3.
     `#817 <https://github.com/pybind/pybind11/pull/817>`_.

  7. Fixed lifetime of temporary C++ objects created in Python-to-C++ conversions.
     `#924 <https://github.com/pybind/pybind11/pull/924>`_.

* Scope guard call policy for RAII types, e.g. ``py::call_guard<py::gil_scoped_release>()``,
  ``py::call_guard<py::scoped_ostream_redirect>()``. See :ref:`call_policies` for details.
  `#740 <https://github.com/pybind/pybind11/pull/740>`_.

* Utility for redirecting C++ streams to Python (e.g. ``std::cout`` ->
  ``sys.stdout``). Scope guard ``py::scoped_ostream_redirect`` in C++ and
  a context manager in Python. See :ref:`ostream_redirect`.
  `#1009 <https://github.com/pybind/pybind11/pull/1009>`_.

* Improved handling of types and exceptions across module boundaries.
  `#915 <https://github.com/pybind/pybind11/pull/915>`_,
  `#951 <https://github.com/pybind/pybind11/pull/951>`_,
  `#995 <https://github.com/pybind/pybind11/pull/995>`_.

* Fixed destruction order of ``py::keep_alive`` nurse/patient objects
  in reference cycles.
  `#856 <https://github.com/pybind/pybind11/pull/856>`_.

* NumPy and buffer protocol related improvements:

  1. Support for negative strides in Python buffer objects/numpy arrays. This
     required changing integers from unsigned to signed for the related C++ APIs.
     Note: If you have compiler warnings enabled, you may notice some new conversion
     warnings after upgrading. These can be resolved with ``static_cast``.
     `#782 <https://github.com/pybind/pybind11/pull/782>`_.

  2. Support ``std::complex`` and arrays inside ``PYBIND11_NUMPY_DTYPE``.
     `#831 <https://github.com/pybind/pybind11/pull/831>`_,
     `#832 <https://github.com/pybind/pybind11/pull/832>`_.

  3. Support for constructing ``py::buffer_info`` and ``py::arrays`` using
     arbitrary containers or iterators instead of requiring a ``std::vector``.
     `#788 <https://github.com/pybind/pybind11/pull/788>`_,
     `#822 <https://github.com/pybind/pybind11/pull/822>`_,
     `#860 <https://github.com/pybind/pybind11/pull/860>`_.

  4. Explicitly check numpy version and require >= 1.7.0.
     `#819 <https://github.com/pybind/pybind11/pull/819>`_.

* Support for allowing/prohibiting ``None`` for specific arguments and improved
  ``None`` overload resolution order. See :ref:`none_arguments` for details.
  `#843 <https://github.com/pybind/pybind11/pull/843>`_.
  `#859 <https://github.com/pybind/pybind11/pull/859>`_.

* Added ``py::exec()`` as a shortcut for ``py::eval<py::eval_statements>()``
  and support for C++11 raw string literals as input. See :ref:`eval`.
  `#766 <https://github.com/pybind/pybind11/pull/766>`_,
  `#827 <https://github.com/pybind/pybind11/pull/827>`_.

* ``py::vectorize()`` ignores non-vectorizable arguments and supports
  member functions.
  `#762 <https://github.com/pybind/pybind11/pull/762>`_.

* Support for bound methods as callbacks (``pybind11/functional.h``).
  `#815 <https://github.com/pybind/pybind11/pull/815>`_.

* Allow aliasing pybind11 methods: ``cls.attr("foo") = cls.attr("bar")``.
  `#802 <https://github.com/pybind/pybind11/pull/802>`_.

* Don't allow mixed static/non-static overloads.
  `#804 <https://github.com/pybind/pybind11/pull/804>`_.

* Fixed overriding static properties in derived classes.
  `#784 <https://github.com/pybind/pybind11/pull/784>`_.

* Added support for write only properties.
  `#1144 <https://github.com/pybind/pybind11/pull/1144>`_.

* Improved deduction of member functions of a derived class when its bases
  aren't registered with pybind11.
  `#855 <https://github.com/pybind/pybind11/pull/855>`_.

  .. code-block:: cpp

      struct Base {
          int foo() { return 42; }
      }

      struct Derived : Base {}

      // Now works, but previously required also binding `Base`
      py::class_<Derived>(m, "Derived")
          .def("foo", &Derived::foo); // function is actually from `Base`

* The implementation of ``py::init<>`` now uses C++11 brace initialization
  syntax to construct instances, which permits binding implicit constructors of
  aggregate types. `#1015 <https://github.com/pybind/pybind11/pull/1015>`_.

    .. code-block:: cpp

        struct Aggregate {
            int a;
            std::string b;
        };

        py::class_<Aggregate>(m, "Aggregate")
            .def(py::init<int, const std::string &>());

* Fixed issues with multiple inheritance with offset base/derived pointers.
  `#812 <https://github.com/pybind/pybind11/pull/812>`_,
  `#866 <https://github.com/pybind/pybind11/pull/866>`_,
  `#960 <https://github.com/pybind/pybind11/pull/960>`_.

* Fixed reference leak of type objects.
  `#1030 <https://github.com/pybind/pybind11/pull/1030>`_.

* Improved support for the ``/std:c++14`` and ``/std:c++latest`` modes
  on MSVC 2017.
  `#841 <https://github.com/pybind/pybind11/pull/841>`_,
  `#999 <https://github.com/pybind/pybind11/pull/999>`_.

* Fixed detection of private operator new on MSVC.
  `#893 <https://github.com/pybind/pybind11/pull/893>`_,
  `#918 <https://github.com/pybind/pybind11/pull/918>`_.

* Intel C++ compiler compatibility fixes.
  `#937 <https://github.com/pybind/pybind11/pull/937>`_.

* Fixed implicit conversion of ``py::enum_`` to integer types on Python 2.7.
  `#821 <https://github.com/pybind/pybind11/pull/821>`_.

* Added ``py::hash`` to fetch the hash value of Python objects, and
  ``.def(hash(py::self))`` to provide the C++ ``std::hash`` as the Python
  ``__hash__`` method.
  `#1034 <https://github.com/pybind/pybind11/pull/1034>`_.

* Fixed ``__truediv__`` on Python 2 and ``__itruediv__`` on Python 3.
  `#867 <https://github.com/pybind/pybind11/pull/867>`_.

* ``py::capsule`` objects now support the ``name`` attribute. This is useful
  for interfacing with ``scipy.LowLevelCallable``.
  `#902 <https://github.com/pybind/pybind11/pull/902>`_.

* Fixed ``py::make_iterator``'s ``__next__()`` for past-the-end calls.
  `#897 <https://github.com/pybind/pybind11/pull/897>`_.

* Added ``error_already_set::matches()`` for checking Python exceptions.
  `#772 <https://github.com/pybind/pybind11/pull/772>`_.

* Deprecated ``py::error_already_set::clear()``. It's no longer needed
  following a simplification of the ``py::error_already_set`` class.
  `#954 <https://github.com/pybind/pybind11/pull/954>`_.

* Deprecated ``py::handle::operator==()`` in favor of ``py::handle::is()``
  `#825 <https://github.com/pybind/pybind11/pull/825>`_.

* Deprecated ``py::object::borrowed``/``py::object::stolen``.
  Use ``py::object::borrowed_t{}``/``py::object::stolen_t{}`` instead.
  `#771 <https://github.com/pybind/pybind11/pull/771>`_.

* Changed internal data structure versioning to avoid conflicts between
  modules compiled with different revisions of pybind11.
  `#1012 <https://github.com/pybind/pybind11/pull/1012>`_.

* Additional compile-time and run-time error checking and more informative messages.
  `#786 <https://github.com/pybind/pybind11/pull/786>`_,
  `#794 <https://github.com/pybind/pybind11/pull/794>`_,
  `#803 <https://github.com/pybind/pybind11/pull/803>`_.

* Various minor improvements and fixes.
  `#764 <https://github.com/pybind/pybind11/pull/764>`_,
  `#791 <https://github.com/pybind/pybind11/pull/791>`_,
  `#795 <https://github.com/pybind/pybind11/pull/795>`_,
  `#840 <https://github.com/pybind/pybind11/pull/840>`_,
  `#844 <https://github.com/pybind/pybind11/pull/844>`_,
  `#846 <https://github.com/pybind/pybind11/pull/846>`_,
  `#849 <https://github.com/pybind/pybind11/pull/849>`_,
  `#858 <https://github.com/pybind/pybind11/pull/858>`_,
  `#862 <https://github.com/pybind/pybind11/pull/862>`_,
  `#871 <https://github.com/pybind/pybind11/pull/871>`_,
  `#872 <https://github.com/pybind/pybind11/pull/872>`_,
  `#881 <https://github.com/pybind/pybind11/pull/881>`_,
  `#888 <https://github.com/pybind/pybind11/pull/888>`_,
  `#899 <https://github.com/pybind/pybind11/pull/899>`_,
  `#928 <https://github.com/pybind/pybind11/pull/928>`_,
  `#931 <https://github.com/pybind/pybind11/pull/931>`_,
  `#944 <https://github.com/pybind/pybind11/pull/944>`_,
  `#950 <https://github.com/pybind/pybind11/pull/950>`_,
  `#952 <https://github.com/pybind/pybind11/pull/952>`_,
  `#962 <https://github.com/pybind/pybind11/pull/962>`_,
  `#965 <https://github.com/pybind/pybind11/pull/965>`_,
  `#970 <https://github.com/pybind/pybind11/pull/970>`_,
  `#978 <https://github.com/pybind/pybind11/pull/978>`_,
  `#979 <https://github.com/pybind/pybind11/pull/979>`_,
  `#986 <https://github.com/pybind/pybind11/pull/986>`_,
  `#1020 <https://github.com/pybind/pybind11/pull/1020>`_,
  `#1027 <https://github.com/pybind/pybind11/pull/1027>`_,
  `#1037 <https://github.com/pybind/pybind11/pull/1037>`_.

* Testing improvements.
  `#798 <https://github.com/pybind/pybind11/pull/798>`_,
  `#882 <https://github.com/pybind/pybind11/pull/882>`_,
  `#898 <https://github.com/pybind/pybind11/pull/898>`_,
  `#900 <https://github.com/pybind/pybind11/pull/900>`_,
  `#921 <https://github.com/pybind/pybind11/pull/921>`_,
  `#923 <https://github.com/pybind/pybind11/pull/923>`_,
  `#963 <https://github.com/pybind/pybind11/pull/963>`_.

v2.1.1 (April 7, 2017)
-----------------------------------------------------

* Fixed minimum version requirement for MSVC 2015u3
  `#773 <https://github.com/pybind/pybind11/pull/773>`_.

v2.1.0 (March 22, 2017)
-----------------------------------------------------

* pybind11 now performs function overload resolution in two phases. The first
  phase only considers exact type matches, while the second allows for implicit
  conversions to take place. A special ``noconvert()`` syntax can be used to
  completely disable implicit conversions for specific arguments.
  `#643 <https://github.com/pybind/pybind11/pull/643>`_,
  `#634 <https://github.com/pybind/pybind11/pull/634>`_,
  `#650 <https://github.com/pybind/pybind11/pull/650>`_.

* Fixed a regression where static properties no longer worked with classes
  using multiple inheritance. The ``py::metaclass`` attribute is no longer
  necessary (and deprecated as of this release) when binding classes with
  static properties.
  `#679 <https://github.com/pybind/pybind11/pull/679>`_,

* Classes bound using ``pybind11`` can now use custom metaclasses.
  `#679 <https://github.com/pybind/pybind11/pull/679>`_,

* ``py::args`` and ``py::kwargs`` can now be mixed with other positional
  arguments when binding functions using pybind11.
  `#611 <https://github.com/pybind/pybind11/pull/611>`_.

* Improved support for C++11 unicode string and character types; added
  extensive documentation regarding pybind11's string conversion behavior.
  `#624 <https://github.com/pybind/pybind11/pull/624>`_,
  `#636 <https://github.com/pybind/pybind11/pull/636>`_,
  `#715 <https://github.com/pybind/pybind11/pull/715>`_.

* pybind11 can now avoid expensive copies when converting Eigen arrays to NumPy
  arrays (and vice versa). `#610 <https://github.com/pybind/pybind11/pull/610>`_.

* The "fast path" in ``py::vectorize`` now works for any full-size group of C or
  F-contiguous arrays. The non-fast path is also faster since it no longer performs
  copies of the input arguments (except when type conversions are necessary).
  `#610 <https://github.com/pybind/pybind11/pull/610>`_.

* Added fast, unchecked access to NumPy arrays via a proxy object.
  `#746 <https://github.com/pybind/pybind11/pull/746>`_.

* Transparent support for class-specific ``operator new`` and
  ``operator delete`` implementations.
  `#755 <https://github.com/pybind/pybind11/pull/755>`_.

* Slimmer and more efficient STL-compatible iterator interface for sequence types.
  `#662 <https://github.com/pybind/pybind11/pull/662>`_.

* Improved custom holder type support.
  `#607 <https://github.com/pybind/pybind11/pull/607>`_.

* ``nullptr`` to ``None`` conversion fixed in various builtin type casters.
  `#732 <https://github.com/pybind/pybind11/pull/732>`_.

* ``enum_`` now exposes its members via a special ``__members__`` attribute.
  `#666 <https://github.com/pybind/pybind11/pull/666>`_.

* ``std::vector`` bindings created using ``stl_bind.h`` can now optionally
  implement the buffer protocol. `#488 <https://github.com/pybind/pybind11/pull/488>`_.

* Automated C++ reference documentation using doxygen and breathe.
  `#598 <https://github.com/pybind/pybind11/pull/598>`_.

* Added minimum compiler version assertions.
  `#727 <https://github.com/pybind/pybind11/pull/727>`_.

* Improved compatibility with C++1z.
  `#677 <https://github.com/pybind/pybind11/pull/677>`_.

* Improved ``py::capsule`` API. Can be used to implement cleanup
  callbacks that are involved at module destruction time.
  `#752 <https://github.com/pybind/pybind11/pull/752>`_.

* Various minor improvements and fixes.
  `#595 <https://github.com/pybind/pybind11/pull/595>`_,
  `#588 <https://github.com/pybind/pybind11/pull/588>`_,
  `#589 <https://github.com/pybind/pybind11/pull/589>`_,
  `#603 <https://github.com/pybind/pybind11/pull/603>`_,
  `#619 <https://github.com/pybind/pybind11/pull/619>`_,
  `#648 <https://github.com/pybind/pybind11/pull/648>`_,
  `#695 <https://github.com/pybind/pybind11/pull/695>`_,
  `#720 <https://github.com/pybind/pybind11/pull/720>`_,
  `#723 <https://github.com/pybind/pybind11/pull/723>`_,
  `#729 <https://github.com/pybind/pybind11/pull/729>`_,
  `#724 <https://github.com/pybind/pybind11/pull/724>`_,
  `#742 <https://github.com/pybind/pybind11/pull/742>`_,
  `#753 <https://github.com/pybind/pybind11/pull/753>`_.

v2.0.1 (Jan 4, 2017)
-----------------------------------------------------

* Fix pointer to reference error in type_caster on MSVC
  `#583 <https://github.com/pybind/pybind11/pull/583>`_.

* Fixed a segmentation in the test suite due to a typo
  `cd7eac <https://github.com/pybind/pybind11/commit/cd7eac>`_.

v2.0.0 (Jan 1, 2017)
-----------------------------------------------------

* Fixed a reference counting regression affecting types with custom metaclasses
  (introduced in v2.0.0-rc1).
  `#571 <https://github.com/pybind/pybind11/pull/571>`_.

* Quenched a CMake policy warning.
  `#570 <https://github.com/pybind/pybind11/pull/570>`_.

v2.0.0-rc1 (Dec 23, 2016)
-----------------------------------------------------

The pybind11 developers are excited to issue a release candidate of pybind11
with a subsequent v2.0.0 release planned in early January next year.

An incredible amount of effort by went into pybind11 over the last ~5 months,
leading to a release that is jam-packed with exciting new features and numerous
usability improvements. The following list links PRs or individual commits
whenever applicable.

Happy Christmas!

* Support for binding C++ class hierarchies that make use of multiple
  inheritance. `#410 <https://github.com/pybind/pybind11/pull/410>`_.

* PyPy support: pybind11 now supports nightly builds of PyPy and will
  interoperate with the future 5.7 release. No code changes are necessary,
  everything "just" works as usual. Note that we only target the Python 2.7
  branch for now; support for 3.x will be added once its ``cpyext`` extension
  support catches up. A few minor features remain unsupported for the time
  being (notably dynamic attributes in custom types).
  `#527 <https://github.com/pybind/pybind11/pull/527>`_.

* Significant work on the documentation -- in particular, the monolithic
  ``advanced.rst`` file was restructured into a easier to read hierarchical
  organization. `#448 <https://github.com/pybind/pybind11/pull/448>`_.

* Many NumPy-related improvements:

  1. Object-oriented API to access and modify NumPy ``ndarray`` instances,
     replicating much of the corresponding NumPy C API functionality.
     `#402 <https://github.com/pybind/pybind11/pull/402>`_.

  2. NumPy array ``dtype`` array descriptors are now first-class citizens and
     are exposed via a new class ``py::dtype``.

  3. Structured dtypes can be registered using the ``PYBIND11_NUMPY_DTYPE()``
     macro. Special ``array`` constructors accepting dtype objects were also
     added.

     One potential caveat involving this change: format descriptor strings
     should now be accessed via ``format_descriptor::format()`` (however, for
     compatibility purposes, the old syntax ``format_descriptor::value`` will
     still work for non-structured data types). `#308
     <https://github.com/pybind/pybind11/pull/308>`_.

  4. Further improvements to support structured dtypes throughout the system.
     `#472 <https://github.com/pybind/pybind11/pull/472>`_,
     `#474 <https://github.com/pybind/pybind11/pull/474>`_,
     `#459 <https://github.com/pybind/pybind11/pull/459>`_,
     `#453 <https://github.com/pybind/pybind11/pull/453>`_,
     `#452 <https://github.com/pybind/pybind11/pull/452>`_, and
     `#505 <https://github.com/pybind/pybind11/pull/505>`_.

  5. Fast access operators. `#497 <https://github.com/pybind/pybind11/pull/497>`_.

  6. Constructors for arrays whose storage is owned by another object.
     `#440 <https://github.com/pybind/pybind11/pull/440>`_.

  7. Added constructors for ``array`` and ``array_t`` explicitly accepting shape
     and strides; if strides are not provided, they are deduced assuming
     C-contiguity. Also added simplified constructors for 1-dimensional case.

  8. Added buffer/NumPy support for ``char[N]`` and ``std::array<char, N>`` types.

  9. Added ``memoryview`` wrapper type which is constructible from ``buffer_info``.

* Eigen: many additional conversions and support for non-contiguous
  arrays/slices.
  `#427 <https://github.com/pybind/pybind11/pull/427>`_,
  `#315 <https://github.com/pybind/pybind11/pull/315>`_,
  `#316 <https://github.com/pybind/pybind11/pull/316>`_,
  `#312 <https://github.com/pybind/pybind11/pull/312>`_, and
  `#267 <https://github.com/pybind/pybind11/pull/267>`_

* Incompatible changes in ``class_<...>::class_()``:

    1. Declarations of types that provide access via the buffer protocol must
       now include the ``py::buffer_protocol()`` annotation as an argument to
       the ``class_`` constructor.

    2. Declarations of types that require a custom metaclass (i.e. all classes
       which include static properties via commands such as
       ``def_readwrite_static()``) must now include the ``py::metaclass()``
       annotation as an argument to the ``class_`` constructor.

       These two changes were necessary to make type definitions in pybind11
       future-proof, and to support PyPy via its cpyext mechanism. `#527
       <https://github.com/pybind/pybind11/pull/527>`_.


    3. This version of pybind11 uses a redesigned mechanism for instantiating
       trampoline classes that are used to override virtual methods from within
       Python. This led to the following user-visible syntax change: instead of

       .. code-block:: cpp

           py::class_<TrampolineClass>("MyClass")
             .alias<MyClass>()
             ....

       write

       .. code-block:: cpp

           py::class_<MyClass, TrampolineClass>("MyClass")
             ....

       Importantly, both the original and the trampoline class are now
       specified as an arguments (in arbitrary order) to the ``py::class_``
       template, and the ``alias<..>()`` call is gone. The new scheme has zero
       overhead in cases when Python doesn't override any functions of the
       underlying C++ class. `rev. 86d825
       <https://github.com/pybind/pybind11/commit/86d825>`_.

* Added ``eval`` and ``eval_file`` functions for evaluating expressions and
  statements from a string or file. `rev. 0d3fc3
  <https://github.com/pybind/pybind11/commit/0d3fc3>`_.

* pybind11 can now create types with a modifiable dictionary.
  `#437 <https://github.com/pybind/pybind11/pull/437>`_ and
  `#444 <https://github.com/pybind/pybind11/pull/444>`_.

* Support for translation of arbitrary C++ exceptions to Python counterparts.
  `#296 <https://github.com/pybind/pybind11/pull/296>`_ and
  `#273 <https://github.com/pybind/pybind11/pull/273>`_.

* Report full backtraces through mixed C++/Python code, better reporting for
  import errors, fixed GIL management in exception processing.
  `#537 <https://github.com/pybind/pybind11/pull/537>`_,
  `#494 <https://github.com/pybind/pybind11/pull/494>`_,
  `rev. e72d95 <https://github.com/pybind/pybind11/commit/e72d95>`_, and
  `rev. 099d6e <https://github.com/pybind/pybind11/commit/099d6e>`_.

* Support for bit-level operations, comparisons, and serialization of C++
  enumerations. `#503 <https://github.com/pybind/pybind11/pull/503>`_,
  `#508 <https://github.com/pybind/pybind11/pull/508>`_,
  `#380 <https://github.com/pybind/pybind11/pull/380>`_,
  `#309 <https://github.com/pybind/pybind11/pull/309>`_.
  `#311 <https://github.com/pybind/pybind11/pull/311>`_.

* The ``class_`` constructor now accepts its template arguments in any order.
  `#385 <https://github.com/pybind/pybind11/pull/385>`_.

* Attribute and item accessors now have a more complete interface which makes
  it possible to chain attributes as in
  ``obj.attr("a")[key].attr("b").attr("method")(1, 2, 3)``. `#425
  <https://github.com/pybind/pybind11/pull/425>`_.

* Major redesign of the default and conversion constructors in ``pytypes.h``.
  `#464 <https://github.com/pybind/pybind11/pull/464>`_.

* Added built-in support for ``std::shared_ptr`` holder type. It is no longer
  necessary to to include a declaration of the form
  ``PYBIND11_DECLARE_HOLDER_TYPE(T, std::shared_ptr<T>)`` (though continuing to
  do so won't cause an error).
  `#454 <https://github.com/pybind/pybind11/pull/454>`_.

* New ``py::overload_cast`` casting operator to select among multiple possible
  overloads of a function. An example:

    .. code-block:: cpp

        py::class_<Pet>(m, "Pet")
            .def("set", py::overload_cast<int>(&Pet::set), "Set the pet's age")
            .def("set", py::overload_cast<const std::string &>(&Pet::set), "Set the pet's name");

  This feature only works on C++14-capable compilers.
  `#541 <https://github.com/pybind/pybind11/pull/541>`_.

* C++ types are automatically cast to Python types, e.g. when assigning
  them as an attribute. For instance, the following is now legal:

    .. code-block:: cpp

        py::module m = /* ... */
        m.attr("constant") = 123;

  (Previously, a ``py::cast`` call was necessary to avoid a compilation error.)
  `#551 <https://github.com/pybind/pybind11/pull/551>`_.

* Redesigned ``pytest``-based test suite. `#321 <https://github.com/pybind/pybind11/pull/321>`_.

* Instance tracking to detect reference leaks in test suite. `#324 <https://github.com/pybind/pybind11/pull/324>`_

* pybind11 can now distinguish between multiple different instances that are
  located at the same memory address, but which have different types.
  `#329 <https://github.com/pybind/pybind11/pull/329>`_.

* Improved logic in ``move`` return value policy.
  `#510 <https://github.com/pybind/pybind11/pull/510>`_,
  `#297 <https://github.com/pybind/pybind11/pull/297>`_.

* Generalized unpacking API to permit calling Python functions from C++ using
  notation such as ``foo(a1, a2, *args, "ka"_a=1, "kb"_a=2, **kwargs)``. `#372 <https://github.com/pybind/pybind11/pull/372>`_.

* ``py::print()`` function whose behavior matches that of the native Python
  ``print()`` function. `#372 <https://github.com/pybind/pybind11/pull/372>`_.

* Added ``py::dict`` keyword constructor:``auto d = dict("number"_a=42,
  "name"_a="World");``. `#372 <https://github.com/pybind/pybind11/pull/372>`_.

* Added ``py::str::format()`` method and ``_s`` literal: ``py::str s = "1 + 2
  = {}"_s.format(3);``. `#372 <https://github.com/pybind/pybind11/pull/372>`_.

* Added ``py::repr()`` function which is equivalent to Python's builtin
  ``repr()``. `#333 <https://github.com/pybind/pybind11/pull/333>`_.

* Improved construction and destruction logic for holder types. It is now
  possible to reference instances with smart pointer holder types without
  constructing the holder if desired. The ``PYBIND11_DECLARE_HOLDER_TYPE``
  macro now accepts an optional second parameter to indicate whether the holder
  type uses intrusive reference counting.
  `#533 <https://github.com/pybind/pybind11/pull/533>`_ and
  `#561 <https://github.com/pybind/pybind11/pull/561>`_.

* Mapping a stateless C++ function to Python and back is now "for free" (i.e.
  no extra indirections or argument conversion overheads). `rev. 954b79
  <https://github.com/pybind/pybind11/commit/954b79>`_.

* Bindings for ``std::valarray<T>``.
  `#545 <https://github.com/pybind/pybind11/pull/545>`_.

* Improved support for C++17 capable compilers.
  `#562 <https://github.com/pybind/pybind11/pull/562>`_.

* Bindings for ``std::optional<t>``.
  `#475 <https://github.com/pybind/pybind11/pull/475>`_,
  `#476 <https://github.com/pybind/pybind11/pull/476>`_,
  `#479 <https://github.com/pybind/pybind11/pull/479>`_,
  `#499 <https://github.com/pybind/pybind11/pull/499>`_, and
  `#501 <https://github.com/pybind/pybind11/pull/501>`_.

* ``stl_bind.h``: general improvements and support for ``std::map`` and
  ``std::unordered_map``.
  `#490 <https://github.com/pybind/pybind11/pull/490>`_,
  `#282 <https://github.com/pybind/pybind11/pull/282>`_,
  `#235 <https://github.com/pybind/pybind11/pull/235>`_.

* The ``std::tuple``, ``std::pair``, ``std::list``, and ``std::vector`` type
  casters now accept any Python sequence type as input. `rev. 107285
  <https://github.com/pybind/pybind11/commit/107285>`_.

* Improved CMake Python detection on multi-architecture Linux.
  `#532 <https://github.com/pybind/pybind11/pull/532>`_.

* Infrastructure to selectively disable or enable parts of the automatically
  generated docstrings. `#486 <https://github.com/pybind/pybind11/pull/486>`_.

* ``reference`` and ``reference_internal`` are now the default return value
  properties for static and non-static properties, respectively. `#473
  <https://github.com/pybind/pybind11/pull/473>`_. (the previous defaults
  were ``automatic``). `#473 <https://github.com/pybind/pybind11/pull/473>`_.

* Support for ``std::unique_ptr`` with non-default deleters or no deleter at
  all (``py::nodelete``). `#384 <https://github.com/pybind/pybind11/pull/384>`_.

* Deprecated ``handle::call()`` method. The new syntax to call Python
  functions is simply ``handle()``. It can also be invoked explicitly via
  ``handle::operator<X>()``, where ``X`` is an optional return value policy.

* Print more informative error messages when ``make_tuple()`` or ``cast()``
  fail. `#262 <https://github.com/pybind/pybind11/pull/262>`_.

* Creation of holder types for classes deriving from
  ``std::enable_shared_from_this<>`` now also works for ``const`` values.
  `#260 <https://github.com/pybind/pybind11/pull/260>`_.

* ``make_iterator()`` improvements for better compatibility with various
  types (now uses prefix increment operator); it now also accepts iterators
  with different begin/end types as long as they are equality comparable.
  `#247 <https://github.com/pybind/pybind11/pull/247>`_.

* ``arg()`` now accepts a wider range of argument types for default values.
  `#244 <https://github.com/pybind/pybind11/pull/244>`_.

* Support ``keep_alive`` where the nurse object may be ``None``. `#341
  <https://github.com/pybind/pybind11/pull/341>`_.

* Added constructors for ``str`` and ``bytes`` from zero-terminated char
  pointers, and from char pointers and length. Added constructors for ``str``
  from ``bytes`` and for ``bytes`` from ``str``, which will perform UTF-8
  decoding/encoding as required.

* Many other improvements of library internals without user-visible changes


1.8.1 (July 12, 2016)
----------------------
* Fixed a rare but potentially very severe issue when the garbage collector ran
  during pybind11 type creation.

1.8.0 (June 14, 2016)
----------------------
* Redesigned CMake build system which exports a convenient
  ``pybind11_add_module`` function to parent projects.
* ``std::vector<>`` type bindings analogous to Boost.Python's ``indexing_suite``
* Transparent conversion of sparse and dense Eigen matrices and vectors (``eigen.h``)
* Added an ``ExtraFlags`` template argument to the NumPy ``array_t<>`` wrapper
  to disable an enforced cast that may lose precision, e.g. to create overloads
  for different precisions and complex vs real-valued matrices.
* Prevent implicit conversion of floating point values to integral types in
  function arguments
* Fixed incorrect default return value policy for functions returning a shared
  pointer
* Don't allow registering a type via ``class_`` twice
* Don't allow casting a ``None`` value into a C++ lvalue reference
* Fixed a crash in ``enum_::operator==`` that was triggered by the ``help()`` command
* Improved detection of whether or not custom C++ types can be copy/move-constructed
* Extended ``str`` type to also work with ``bytes`` instances
* Added a ``"name"_a`` user defined string literal that is equivalent to ``py::arg("name")``.
* When specifying function arguments via ``py::arg``, the test that verifies
  the number of arguments now runs at compile time.
* Added ``[[noreturn]]`` attribute to ``pybind11_fail()`` to quench some
  compiler warnings
* List function arguments in exception text when the dispatch code cannot find
  a matching overload
* Added ``PYBIND11_OVERLOAD_NAME`` and ``PYBIND11_OVERLOAD_PURE_NAME`` macros which
  can be used to override virtual methods whose name differs in C++ and Python
  (e.g. ``__call__`` and ``operator()``)
* Various minor ``iterator`` and ``make_iterator()`` improvements
* Transparently support ``__bool__`` on Python 2.x and Python 3.x
* Fixed issue with destructor of unpickled object not being called
* Minor CMake build system improvements on Windows
* New ``pybind11::args`` and ``pybind11::kwargs`` types to create functions which
  take an arbitrary number of arguments and keyword arguments
* New syntax to call a Python function from C++ using ``*args`` and ``*kwargs``
* The functions ``def_property_*`` now correctly process docstring arguments (these
  formerly caused a segmentation fault)
* Many ``mkdoc.py`` improvements (enumerations, template arguments, ``DOC()``
  macro accepts more arguments)
* Cygwin support
* Documentation improvements (pickling support, ``keep_alive``, macro usage)

1.7 (April 30, 2016)
----------------------
* Added a new ``move`` return value policy that triggers C++11 move semantics.
  The automatic return value policy falls back to this case whenever a rvalue
  reference is encountered
* Significantly more general GIL state routines that are used instead of
  Python's troublesome ``PyGILState_Ensure`` and ``PyGILState_Release`` API
* Redesign of opaque types that drastically simplifies their usage
* Extended ability to pass values of type ``[const] void *``
* ``keep_alive`` fix: don't fail when there is no patient
* ``functional.h``: acquire the GIL before calling a Python function
* Added Python RAII type wrappers ``none`` and ``iterable``
* Added ``*args`` and ``*kwargs`` pass-through parameters to
  ``pybind11.get_include()`` function
* Iterator improvements and fixes
* Documentation on return value policies and opaque types improved

1.6 (April 30, 2016)
----------------------
* Skipped due to upload to PyPI gone wrong and inability to recover
  (https://github.com/pypa/packaging-problems/issues/74)

1.5 (April 21, 2016)
----------------------
* For polymorphic types, use RTTI to try to return the closest type registered with pybind11
* Pickling support for serializing and unserializing C++ instances to a byte stream in Python
* Added a convenience routine ``make_iterator()`` which turns a range indicated
  by a pair of C++ iterators into a iterable Python object
* Added ``len()`` and a variadic ``make_tuple()`` function
* Addressed a rare issue that could confuse the current virtual function
  dispatcher and another that could lead to crashes in multi-threaded
  applications
* Added a ``get_include()`` function to the Python module that returns the path
  of the directory containing the installed pybind11 header files
* Documentation improvements: import issues, symbol visibility, pickling, limitations
* Added casting support for ``std::reference_wrapper<>``

1.4 (April 7, 2016)
--------------------------
* Transparent type conversion for ``std::wstring`` and ``wchar_t``
* Allow passing ``nullptr``-valued strings
* Transparent passing of ``void *`` pointers using capsules
* Transparent support for returning values wrapped in ``std::unique_ptr<>``
* Improved docstring generation for compatibility with Sphinx
* Nicer debug error message when default parameter construction fails
* Support for "opaque" types that bypass the transparent conversion layer for STL containers
* Redesigned type casting interface to avoid ambiguities that could occasionally cause compiler errors
* Redesigned property implementation; fixes crashes due to an unfortunate default return value policy
* Anaconda package generation support

1.3 (March 8, 2016)
--------------------------

* Added support for the Intel C++ compiler (v15+)
* Added support for the STL unordered set/map data structures
* Added support for the STL linked list data structure
* NumPy-style broadcasting support in ``pybind11::vectorize``
* pybind11 now displays more verbose error messages when ``arg::operator=()`` fails
* pybind11 internal data structures now live in a version-dependent namespace to avoid ABI issues
* Many, many bugfixes involving corner cases and advanced usage

1.2 (February 7, 2016)
--------------------------

* Optional: efficient generation of function signatures at compile time using C++14
* Switched to a simpler and more general way of dealing with function default
  arguments. Unused keyword arguments in function calls are now detected and
  cause errors as expected
* New ``keep_alive`` call policy analogous to Boost.Python's ``with_custodian_and_ward``
* New ``pybind11::base<>`` attribute to indicate a subclass relationship
* Improved interface for RAII type wrappers in ``pytypes.h``
* Use RAII type wrappers consistently within pybind11 itself. This
  fixes various potential refcount leaks when exceptions occur
* Added new ``bytes`` RAII type wrapper (maps to ``string`` in Python 2.7)
* Made handle and related RAII classes const correct, using them more
  consistently everywhere now
* Got rid of the ugly ``__pybind11__`` attributes on the Python side---they are
  now stored in a C++ hash table that is not visible in Python
* Fixed refcount leaks involving NumPy arrays and bound functions
* Vastly improved handling of shared/smart pointers
* Removed an unnecessary copy operation in ``pybind11::vectorize``
* Fixed naming clashes when both pybind11 and NumPy headers are included
* Added conversions for additional exception types
* Documentation improvements (using multiple extension modules, smart pointers,
  other minor clarifications)
* unified infrastructure for parsing variadic arguments in ``class_`` and cpp_function
* Fixed license text (was: ZLIB, should have been: 3-clause BSD)
* Python 3.2 compatibility
* Fixed remaining issues when accessing types in another plugin module
* Added enum comparison and casting methods
* Improved SFINAE-based detection of whether types are copy-constructible
* Eliminated many warnings about unused variables and the use of ``offsetof()``
* Support for ``std::array<>`` conversions

1.1 (December 7, 2015)
--------------------------

* Documentation improvements (GIL, wrapping functions, casting, fixed many typos)
* Generalized conversion of integer types
* Improved support for casting function objects
* Improved support for ``std::shared_ptr<>`` conversions
* Initial support for ``std::set<>`` conversions
* Fixed type resolution issue for types defined in a separate plugin module
* CMake build system improvements
* Factored out generic functionality to non-templated code (smaller code size)
* Added a code size / compile time benchmark vs Boost.Python
* Added an appveyor CI script

1.0 (October 15, 2015)
------------------------
* Initial release
