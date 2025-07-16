#pragma once
#include "pybind11_tests.h"

#include <stdexcept>

// shared exceptions for cross_module_tests

class PYBIND11_EXPORT_EXCEPTION shared_exception : public pybind11::builtin_exception {
public:
    using builtin_exception::builtin_exception;
    explicit shared_exception() : shared_exception("") {}
    void set_error() const override { py::set_error(PyExc_RuntimeError, what()); }
};
