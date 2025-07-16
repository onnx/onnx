/*
    tests/test_callbacks.cpp -- callbacks

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/functional.h>

#include "constructor_stats.h"
#include "pybind11_tests.h"

#include <thread>

int dummy_function(int i) { return i + 1; }

TEST_SUBMODULE(callbacks, m) {
    // test_callbacks, test_function_signatures
    m.def("test_callback1", [](const py::object &func) { return func(); });
    m.def("test_callback2", [](const py::object &func) { return func("Hello", 'x', true, 5); });
    m.def("test_callback3", [](const std::function<int(int)> &func) {
        return "func(43) = " + std::to_string(func(43));
    });
    m.def("test_callback4",
          []() -> std::function<int(int)> { return [](int i) { return i + 1; }; });
    m.def("test_callback5",
          []() { return py::cpp_function([](int i) { return i + 1; }, py::arg("number")); });

    // test_keyword_args_and_generalized_unpacking
    m.def("test_tuple_unpacking", [](const py::function &f) {
        auto t1 = py::make_tuple(2, 3);
        auto t2 = py::make_tuple(5, 6);
        return f("positional", 1, *t1, 4, *t2);
    });

    m.def("test_dict_unpacking", [](const py::function &f) {
        auto d1 = py::dict("key"_a = "value", "a"_a = 1);
        auto d2 = py::dict();
        auto d3 = py::dict("b"_a = 2);
        return f("positional", 1, **d1, **d2, **d3);
    });

    m.def("test_keyword_args", [](const py::function &f) { return f("x"_a = 10, "y"_a = 20); });

    m.def("test_unpacking_and_keywords1", [](const py::function &f) {
        auto args = py::make_tuple(2);
        auto kwargs = py::dict("d"_a = 4);
        return f(1, *args, "c"_a = 3, **kwargs);
    });

    m.def("test_unpacking_and_keywords2", [](const py::function &f) {
        auto kwargs1 = py::dict("a"_a = 1);
        auto kwargs2 = py::dict("c"_a = 3, "d"_a = 4);
        return f("positional",
                 *py::make_tuple(1),
                 2,
                 *py::make_tuple(3, 4),
                 5,
                 "key"_a = "value",
                 **kwargs1,
                 "b"_a = 2,
                 **kwargs2,
                 "e"_a = 5);
    });

    m.def("test_unpacking_error1", [](const py::function &f) {
        auto kwargs = py::dict("x"_a = 3);
        return f("x"_a = 1, "y"_a = 2, **kwargs); // duplicate ** after keyword
    });

    m.def("test_unpacking_error2", [](const py::function &f) {
        auto kwargs = py::dict("x"_a = 3);
        return f(**kwargs, "x"_a = 1); // duplicate keyword after **
    });

    m.def("test_arg_conversion_error1",
          [](const py::function &f) { f(234, UnregisteredType(), "kw"_a = 567); });

    m.def("test_arg_conversion_error2", [](const py::function &f) {
        f(234, "expected_name"_a = UnregisteredType(), "kw"_a = 567);
    });

    // test_lambda_closure_cleanup
    struct Payload {
        Payload() { print_default_created(this); }
        ~Payload() { print_destroyed(this); }
        Payload(const Payload &) { print_copy_created(this); }
        Payload(Payload &&) noexcept { print_move_created(this); }
    };
    // Export the payload constructor statistics for testing purposes:
    m.def("payload_cstats", &ConstructorStats::get<Payload>);
    m.def("test_lambda_closure_cleanup", []() -> std::function<void()> {
        Payload p;

        // In this situation, `Func` in the implementation of
        // `cpp_function::initialize` is NOT trivially destructible.
        return [p]() {
            /* p should be cleaned up when the returned function is garbage collected */
            (void) p;
        };
    });

    class CppCallable {
    public:
        CppCallable() { track_default_created(this); }
        ~CppCallable() { track_destroyed(this); }
        CppCallable(const CppCallable &) { track_copy_created(this); }
        CppCallable(CppCallable &&) noexcept { track_move_created(this); }
        void operator()() {}
    };

    m.def("test_cpp_callable_cleanup", []() {
        // Related issue: https://github.com/pybind/pybind11/issues/3228
        // Related PR: https://github.com/pybind/pybind11/pull/3229
        py::list alive_counts;
        ConstructorStats &stat = ConstructorStats::get<CppCallable>();
        alive_counts.append(stat.alive());
        {
            CppCallable cpp_callable;
            alive_counts.append(stat.alive());
            {
                // In this situation, `Func` in the implementation of
                // `cpp_function::initialize` IS trivially destructible,
                // only `capture` is not.
                py::cpp_function py_func(cpp_callable);
                py::detail::silence_unused_warnings(py_func);
                alive_counts.append(stat.alive());
            }
            alive_counts.append(stat.alive());
            {
                py::cpp_function py_func(std::move(cpp_callable));
                py::detail::silence_unused_warnings(py_func);
                alive_counts.append(stat.alive());
            }
            alive_counts.append(stat.alive());
        }
        alive_counts.append(stat.alive());
        return alive_counts;
    });

    // test_cpp_function_roundtrip
    /* Test if passing a function pointer from C++ -> Python -> C++ yields the original pointer */
    m.def("dummy_function", &dummy_function);
    m.def("dummy_function_overloaded", [](int i, int j) { return i + j; });
    m.def("dummy_function_overloaded", &dummy_function);
    m.def("dummy_function2", [](int i, int j) { return i + j; });
    m.def(
        "roundtrip",
        [](std::function<int(int)> f, bool expect_none) {
            if (expect_none && f) {
                throw std::runtime_error("Expected None to be converted to empty std::function");
            }
            return f;
        },
        py::arg("f"),
        py::arg("expect_none") = false);
    m.def("test_dummy_function", [](const std::function<int(int)> &f) -> std::string {
        using fn_type = int (*)(int);
        const auto *result = f.target<fn_type>();
        if (!result) {
            auto r = f(1);
            return "can't convert to function pointer: eval(1) = " + std::to_string(r);
        }
        if (*result == dummy_function) {
            auto r = (*result)(1);
            return "matches dummy_function: eval(1) = " + std::to_string(r);
        }
        return "argument does NOT match dummy_function. This should never happen!";
    });

    class AbstractBase {
    public:
        // [workaround(intel)] = default does not work here
        // Defaulting this destructor results in linking errors with the Intel compiler
        // (in Debug builds only, tested with icpc (ICC) 2021.1 Beta 20200827)
        virtual ~AbstractBase() {} // NOLINT(modernize-use-equals-default)
        virtual unsigned int func() = 0;
    };
    m.def("func_accepting_func_accepting_base",
          [](const std::function<double(AbstractBase &)> &) {});

    struct MovableObject {
        bool valid = true;

        MovableObject() = default;
        MovableObject(const MovableObject &) = default;
        MovableObject &operator=(const MovableObject &) = default;
        MovableObject(MovableObject &&o) noexcept : valid(o.valid) { o.valid = false; }
        MovableObject &operator=(MovableObject &&o) noexcept {
            valid = o.valid;
            o.valid = false;
            return *this;
        }
    };
    py::class_<MovableObject>(m, "MovableObject");

    // test_movable_object
    m.def("callback_with_movable", [](const std::function<void(MovableObject &)> &f) {
        auto x = MovableObject();
        f(x);           // lvalue reference shouldn't move out object
        return x.valid; // must still return `true`
    });

    // test_bound_method_callback
    struct CppBoundMethodTest {};
    py::class_<CppBoundMethodTest>(m, "CppBoundMethodTest")
        .def(py::init<>())
        .def("triple", [](CppBoundMethodTest &, int val) { return 3 * val; });

    // This checks that builtin functions can be passed as callbacks
    // rather than throwing RuntimeError due to trying to extract as capsule
    m.def("test_sum_builtin",
          [](const std::function<double(py::iterable)> &sum_builtin, const py::iterable &i) {
              return sum_builtin(i);
          });

    // test async Python callbacks
    using callback_f = std::function<void(int)>;
    m.def("test_async_callback", [](const callback_f &f, const py::list &work) {
        // make detached thread that calls `f` with piece of work after a little delay
        auto start_f = [f](int j) {
            auto invoke_f = [f, j] {
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                f(j);
            };
            auto t = std::thread(std::move(invoke_f));
            t.detach();
        };

        // spawn worker threads
        for (auto i : work) {
            start_f(py::cast<int>(i));
        }
    });

    m.def("callback_num_times", [](const py::function &f, std::size_t num) {
        for (std::size_t i = 0; i < num; i++) {
            f();
        }
    });

    auto *custom_def = []() {
        static PyMethodDef def;
        def.ml_name = "example_name";
        def.ml_doc = "Example doc";
        def.ml_meth = [](PyObject *, PyObject *args) -> PyObject * {
            if (PyTuple_Size(args) != 1) {
                throw std::runtime_error("Invalid number of arguments for example_name");
            }
            PyObject *first = PyTuple_GetItem(args, 0);
            if (!PyLong_Check(first)) {
                throw std::runtime_error("Invalid argument to example_name");
            }
            auto result = py::cast(PyLong_AsLong(first) * 9);
            return result.release().ptr();
        };
        def.ml_flags = METH_VARARGS;
        return &def;
    }();

    // rec_capsule with name that has the same value (but not pointer) as our internal one
    // This capsule should be detected by our code as foreign and not inspected as the pointers
    // shouldn't match
    constexpr const char *rec_capsule_name
        = pybind11::detail::internals_function_record_capsule_name;
    py::capsule rec_capsule(std::malloc(1), [](void *data) { std::free(data); });
    rec_capsule.set_name(rec_capsule_name);
    m.add_object("custom_function", PyCFunction_New(custom_def, rec_capsule.ptr()));

    // This test requires a new ABI version to pass
#if PYBIND11_INTERNALS_VERSION > 4
    // rec_capsule with nullptr name
    py::capsule rec_capsule2(std::malloc(1), [](void *data) { std::free(data); });
    m.add_object("custom_function2", PyCFunction_New(custom_def, rec_capsule2.ptr()));
#else
    m.add_object("custom_function2", py::none());
#endif
}
