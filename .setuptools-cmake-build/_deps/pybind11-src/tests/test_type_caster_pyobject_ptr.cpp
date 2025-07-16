#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <pybind11/type_caster_pyobject_ptr.h>

#include "pybind11_tests.h"

#include <cstddef>
#include <string>
#include <vector>

namespace test_type_caster_pyobject_ptr {

std::vector<PyObject *> make_vector_pyobject_ptr(const py::object &ValueHolder) {
    std::vector<PyObject *> vec_obj;
    for (int i = 1; i < 3; i++) {
        vec_obj.push_back(ValueHolder(i * 93).release().ptr());
    }
    // This vector now owns the refcounts.
    return vec_obj;
}

struct WithPyObjectPtrReturn {
#if defined(__clang_major__) && __clang_major__ < 4
    WithPyObjectPtrReturn() = default;
    WithPyObjectPtrReturn(const WithPyObjectPtrReturn &) = default;
#endif
    virtual ~WithPyObjectPtrReturn() = default;
    virtual PyObject *return_pyobject_ptr() const = 0;
};

struct WithPyObjectPtrReturnTrampoline : WithPyObjectPtrReturn {
    PyObject *return_pyobject_ptr() const override {
        PYBIND11_OVERRIDE_PURE(PyObject *, WithPyObjectPtrReturn, return_pyobject_ptr,
                               /* no arguments */);
    }
};

std::string call_return_pyobject_ptr(const WithPyObjectPtrReturn *base_class_ptr) {
    PyObject *returned_obj = base_class_ptr->return_pyobject_ptr();
#if !defined(PYPY_VERSION) // It is not worth the trouble doing something special for PyPy.
    if (Py_REFCNT(returned_obj) != 1) {
        py::pybind11_fail(__FILE__ ":" PYBIND11_TOSTRING(__LINE__));
    }
#endif
    auto ret_val = py::repr(returned_obj).cast<std::string>();
    Py_DECREF(returned_obj);
    return ret_val;
}

} // namespace test_type_caster_pyobject_ptr

TEST_SUBMODULE(type_caster_pyobject_ptr, m) {
    using namespace test_type_caster_pyobject_ptr;

    m.def("cast_from_pyobject_ptr", []() {
        PyObject *ptr = PyLong_FromLongLong(6758L);
        return py::cast(ptr, py::return_value_policy::take_ownership);
    });
    m.def("cast_handle_to_pyobject_ptr", [](py::handle obj) {
        auto rc1 = obj.ref_count();
        auto *ptr = py::cast<PyObject *>(obj);
        auto rc2 = obj.ref_count();
        if (rc2 != rc1 + 1) {
            return -1;
        }
        return 100 - py::reinterpret_steal<py::object>(ptr).attr("value").cast<int>();
    });
    m.def("cast_object_to_pyobject_ptr", [](py::object obj) {
        py::handle hdl = obj;
        auto rc1 = hdl.ref_count();
        auto *ptr = py::cast<PyObject *>(std::move(obj));
        auto rc2 = hdl.ref_count();
        if (rc2 != rc1) {
            return -1;
        }
        return 300 - py::reinterpret_steal<py::object>(ptr).attr("value").cast<int>();
    });
    m.def("cast_list_to_pyobject_ptr", [](py::list lst) {
        // This is to cover types implicitly convertible to object.
        py::handle hdl = lst;
        auto rc1 = hdl.ref_count();
        auto *ptr = py::cast<PyObject *>(std::move(lst));
        auto rc2 = hdl.ref_count();
        if (rc2 != rc1) {
            return -1;
        }
        return 400 - static_cast<int>(py::len(py::reinterpret_steal<py::list>(ptr)));
    });

    m.def(
        "return_pyobject_ptr",
        []() { return PyLong_FromLongLong(2314L); },
        py::return_value_policy::take_ownership);
    m.def("pass_pyobject_ptr", [](PyObject *ptr) {
        return 200 - py::reinterpret_borrow<py::object>(ptr).attr("value").cast<int>();
    });

    m.def("call_callback_with_object_return",
          [](const std::function<py::object(int)> &cb, int value) { return cb(value); });
    m.def(
        "call_callback_with_pyobject_ptr_return",
        [](const std::function<PyObject *(int)> &cb, int value) { return cb(value); },
        py::return_value_policy::take_ownership);
    m.def(
        "call_callback_with_pyobject_ptr_arg",
        [](const std::function<int(PyObject *)> &cb, py::handle obj) { return cb(obj.ptr()); },
        py::arg("cb"), // This triggers return_value_policy::automatic_reference
        py::arg("obj"));

    m.def("cast_to_pyobject_ptr_nullptr", [](bool set_error) {
        if (set_error) {
            py::set_error(PyExc_RuntimeError, "Reflective of healthy error handling.");
        }
        PyObject *ptr = nullptr;
        py::cast(ptr);
    });

    m.def("cast_to_pyobject_ptr_non_nullptr_with_error_set", []() {
        py::set_error(PyExc_RuntimeError, "Reflective of unhealthy error handling.");
        py::cast(Py_None);
    });

    m.def("pass_list_pyobject_ptr", [](const std::vector<PyObject *> &vec_obj) {
        int acc = 0;
        for (const auto &ptr : vec_obj) {
            acc = acc * 1000 + py::reinterpret_borrow<py::object>(ptr).attr("value").cast<int>();
        }
        return acc;
    });

    m.def("return_list_pyobject_ptr_take_ownership",
          make_vector_pyobject_ptr,
          // Ownership is transferred one-by-one when the vector is converted to a Python list.
          py::return_value_policy::take_ownership);

    m.def("return_list_pyobject_ptr_reference",
          make_vector_pyobject_ptr,
          // Ownership is not transferred.
          py::return_value_policy::reference);

    m.def("dec_ref_each_pyobject_ptr", [](const std::vector<PyObject *> &vec_obj) {
        std::size_t i = 0;
        for (; i < vec_obj.size(); i++) {
            py::handle h(vec_obj[i]);
            if (static_cast<std::size_t>(h.ref_count()) < 2) {
                break; // Something is badly wrong.
            }
            h.dec_ref();
        }
        return i;
    });

    m.def("pass_pyobject_ptr_and_int", [](PyObject *, int) {});

#ifdef PYBIND11_NO_COMPILE_SECTION // Change to ifndef for manual testing.
    {
        PyObject *ptr = nullptr;
        (void) py::cast(*ptr);
    }
#endif

    py::class_<WithPyObjectPtrReturn, WithPyObjectPtrReturnTrampoline>(m, "WithPyObjectPtrReturn")
        .def(py::init<>())
        .def("return_pyobject_ptr", &WithPyObjectPtrReturn::return_pyobject_ptr);

    m.def("call_return_pyobject_ptr", call_return_pyobject_ptr);
}
