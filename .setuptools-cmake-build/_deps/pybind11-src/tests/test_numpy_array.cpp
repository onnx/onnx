/*
    tests/test_numpy_array.cpp -- test core array functionality

    Copyright (c) 2016 Ivan Smirnov <i.s.smirnov@gmail.com>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "pybind11_tests.h"

#include <cstdint>
#include <utility>

// Size / dtype checks.
struct DtypeCheck {
    py::dtype numpy{};
    py::dtype pybind11{};
};

template <typename T>
DtypeCheck get_dtype_check(const char *name) {
    py::module_ np = py::module_::import("numpy");
    DtypeCheck check{};
    check.numpy = np.attr("dtype")(np.attr(name));
    check.pybind11 = py::dtype::of<T>();
    return check;
}

std::vector<DtypeCheck> get_concrete_dtype_checks() {
    return {// Normalization
            get_dtype_check<std::int8_t>("int8"),
            get_dtype_check<std::uint8_t>("uint8"),
            get_dtype_check<std::int16_t>("int16"),
            get_dtype_check<std::uint16_t>("uint16"),
            get_dtype_check<std::int32_t>("int32"),
            get_dtype_check<std::uint32_t>("uint32"),
            get_dtype_check<std::int64_t>("int64"),
            get_dtype_check<std::uint64_t>("uint64")};
}

struct DtypeSizeCheck {
    std::string name{};
    int size_cpp{};
    int size_numpy{};
    // For debugging.
    py::dtype dtype{};
};

template <typename T>
DtypeSizeCheck get_dtype_size_check() {
    DtypeSizeCheck check{};
    check.name = py::type_id<T>();
    check.size_cpp = sizeof(T);
    check.dtype = py::dtype::of<T>();
    check.size_numpy = check.dtype.attr("itemsize").template cast<int>();
    return check;
}

std::vector<DtypeSizeCheck> get_platform_dtype_size_checks() {
    return {
        get_dtype_size_check<short>(),
        get_dtype_size_check<unsigned short>(),
        get_dtype_size_check<int>(),
        get_dtype_size_check<unsigned int>(),
        get_dtype_size_check<long>(),
        get_dtype_size_check<unsigned long>(),
        get_dtype_size_check<long long>(),
        get_dtype_size_check<unsigned long long>(),
    };
}

// Arrays.
using arr = py::array;
using arr_t = py::array_t<uint16_t, 0>;
static_assert(std::is_same<arr_t::value_type, uint16_t>::value, "");

template <typename... Ix>
arr data(const arr &a, Ix... index) {
    return arr(a.nbytes() - a.offset_at(index...), (const uint8_t *) a.data(index...));
}

template <typename... Ix>
arr data_t(const arr_t &a, Ix... index) {
    return arr(a.size() - a.index_at(index...), a.data(index...));
}

template <typename... Ix>
arr &mutate_data(arr &a, Ix... index) {
    auto *ptr = (uint8_t *) a.mutable_data(index...);
    for (py::ssize_t i = 0; i < a.nbytes() - a.offset_at(index...); i++) {
        ptr[i] = (uint8_t) (ptr[i] * 2);
    }
    return a;
}

template <typename... Ix>
arr_t &mutate_data_t(arr_t &a, Ix... index) {
    auto ptr = a.mutable_data(index...);
    for (py::ssize_t i = 0; i < a.size() - a.index_at(index...); i++) {
        ptr[i]++;
    }
    return a;
}

template <typename... Ix>
py::ssize_t index_at(const arr &a, Ix... idx) {
    return a.index_at(idx...);
}
template <typename... Ix>
py::ssize_t index_at_t(const arr_t &a, Ix... idx) {
    return a.index_at(idx...);
}
template <typename... Ix>
py::ssize_t offset_at(const arr &a, Ix... idx) {
    return a.offset_at(idx...);
}
template <typename... Ix>
py::ssize_t offset_at_t(const arr_t &a, Ix... idx) {
    return a.offset_at(idx...);
}
template <typename... Ix>
py::ssize_t at_t(const arr_t &a, Ix... idx) {
    return a.at(idx...);
}
template <typename... Ix>
arr_t &mutate_at_t(arr_t &a, Ix... idx) {
    a.mutable_at(idx...)++;
    return a;
}

#define def_index_fn(name, type)                                                                  \
    sm.def(#name, [](type a) { return name(a); });                                                \
    sm.def(#name, [](type a, int i) { return name(a, i); });                                      \
    sm.def(#name, [](type a, int i, int j) { return name(a, i, j); });                            \
    sm.def(#name, [](type a, int i, int j, int k) { return name(a, i, j, k); });

template <typename T, typename T2>
py::handle auxiliaries(T &&r, T2 &&r2) {
    if (r.ndim() != 2) {
        throw std::domain_error("error: ndim != 2");
    }
    py::list l;
    l.append(*r.data(0, 0));
    l.append(*r2.mutable_data(0, 0));
    l.append(r.data(0, 1) == r2.mutable_data(0, 1));
    l.append(r.ndim());
    l.append(r.itemsize());
    l.append(r.shape(0));
    l.append(r.shape(1));
    l.append(r.size());
    l.append(r.nbytes());
    return l.release();
}

// note: declaration at local scope would create a dangling reference!
static int data_i = 42;

TEST_SUBMODULE(numpy_array, sm) {
    try {
        py::module_::import("numpy");
    } catch (const py::error_already_set &) {
        return;
    }

    // test_dtypes
    py::class_<DtypeCheck>(sm, "DtypeCheck")
        .def_readonly("numpy", &DtypeCheck::numpy)
        .def_readonly("pybind11", &DtypeCheck::pybind11)
        .def("__repr__", [](const DtypeCheck &self) {
            return py::str("<DtypeCheck numpy={} pybind11={}>").format(self.numpy, self.pybind11);
        });
    sm.def("get_concrete_dtype_checks", &get_concrete_dtype_checks);

    py::class_<DtypeSizeCheck>(sm, "DtypeSizeCheck")
        .def_readonly("name", &DtypeSizeCheck::name)
        .def_readonly("size_cpp", &DtypeSizeCheck::size_cpp)
        .def_readonly("size_numpy", &DtypeSizeCheck::size_numpy)
        .def("__repr__", [](const DtypeSizeCheck &self) {
            return py::str("<DtypeSizeCheck name='{}' size_cpp={} size_numpy={} dtype={}>")
                .format(self.name, self.size_cpp, self.size_numpy, self.dtype);
        });
    sm.def("get_platform_dtype_size_checks", &get_platform_dtype_size_checks);

    // test_array_attributes
    sm.def("ndim", [](const arr &a) { return a.ndim(); });
    sm.def("shape", [](const arr &a) { return arr(a.ndim(), a.shape()); });
    sm.def("shape", [](const arr &a, py::ssize_t dim) { return a.shape(dim); });
    sm.def("strides", [](const arr &a) { return arr(a.ndim(), a.strides()); });
    sm.def("strides", [](const arr &a, py::ssize_t dim) { return a.strides(dim); });
    sm.def("writeable", [](const arr &a) { return a.writeable(); });
    sm.def("size", [](const arr &a) { return a.size(); });
    sm.def("itemsize", [](const arr &a) { return a.itemsize(); });
    sm.def("nbytes", [](const arr &a) { return a.nbytes(); });
    sm.def("owndata", [](const arr &a) { return a.owndata(); });

    // test_index_offset
    def_index_fn(index_at, const arr &);
    def_index_fn(index_at_t, const arr_t &);
    def_index_fn(offset_at, const arr &);
    def_index_fn(offset_at_t, const arr_t &);
    // test_data
    def_index_fn(data, const arr &);
    def_index_fn(data_t, const arr_t &);
    // test_mutate_data, test_mutate_readonly
    def_index_fn(mutate_data, arr &);
    def_index_fn(mutate_data_t, arr_t &);
    def_index_fn(at_t, const arr_t &);
    def_index_fn(mutate_at_t, arr_t &);

    // test_make_c_f_array
    sm.def("make_f_array", [] { return py::array_t<float>({2, 2}, {4, 8}); });
    sm.def("make_c_array", [] { return py::array_t<float>({2, 2}, {8, 4}); });

    // test_empty_shaped_array
    sm.def("make_empty_shaped_array", [] { return py::array(py::dtype("f"), {}, {}); });
    // test numpy scalars (empty shape, ndim==0)
    sm.def("scalar_int", []() { return py::array(py::dtype("i"), {}, {}, &data_i); });

    // test_wrap
    sm.def("wrap", [](const py::array &a) {
        return py::array(a.dtype(),
                         {a.shape(), a.shape() + a.ndim()},
                         {a.strides(), a.strides() + a.ndim()},
                         a.data(),
                         a);
    });

    // test_numpy_view
    struct ArrayClass {
        int data[2] = {1, 2};
        ArrayClass() { py::print("ArrayClass()"); }
        ~ArrayClass() { py::print("~ArrayClass()"); }
    };
    py::class_<ArrayClass>(sm, "ArrayClass")
        .def(py::init<>())
        .def("numpy_view", [](py::object &obj) {
            py::print("ArrayClass::numpy_view()");
            auto &a = obj.cast<ArrayClass &>();
            return py::array_t<int>({2}, {4}, a.data, obj);
        });

    // test_cast_numpy_int64_to_uint64
    sm.def("function_taking_uint64", [](uint64_t) {});

    // test_isinstance
    sm.def("isinstance_untyped", [](py::object yes, py::object no) {
        return py::isinstance<py::array>(std::move(yes))
               && !py::isinstance<py::array>(std::move(no));
    });
    sm.def("isinstance_typed", [](const py::object &o) {
        return py::isinstance<py::array_t<double>>(o) && !py::isinstance<py::array_t<int>>(o);
    });

    // test_constructors
    sm.def("default_constructors", []() {
        return py::dict("array"_a = py::array(),
                        "array_t<int32>"_a = py::array_t<std::int32_t>(),
                        "array_t<double>"_a = py::array_t<double>());
    });
    sm.def("converting_constructors", [](const py::object &o) {
        return py::dict("array"_a = py::array(o),
                        "array_t<int32>"_a = py::array_t<std::int32_t>(o),
                        "array_t<double>"_a = py::array_t<double>(o));
    });

    // test_overload_resolution
    sm.def("overloaded", [](const py::array_t<double> &) { return "double"; });
    sm.def("overloaded", [](const py::array_t<float> &) { return "float"; });
    sm.def("overloaded", [](const py::array_t<int> &) { return "int"; });
    sm.def("overloaded", [](const py::array_t<unsigned short> &) { return "unsigned short"; });
    sm.def("overloaded", [](const py::array_t<long long> &) { return "long long"; });
    sm.def("overloaded",
           [](const py::array_t<std::complex<double>> &) { return "double complex"; });
    sm.def("overloaded", [](const py::array_t<std::complex<float>> &) { return "float complex"; });

    sm.def("overloaded2",
           [](const py::array_t<std::complex<double>> &) { return "double complex"; });
    sm.def("overloaded2", [](const py::array_t<double> &) { return "double"; });
    sm.def("overloaded2",
           [](const py::array_t<std::complex<float>> &) { return "float complex"; });
    sm.def("overloaded2", [](const py::array_t<float> &) { return "float"; });

    // [workaround(intel)] ICC 20/21 breaks with py::arg().stuff, using py::arg{}.stuff works.

    // Only accept the exact types:
    sm.def("overloaded3", [](const py::array_t<int> &) { return "int"; }, py::arg{}.noconvert());
    sm.def(
        "overloaded3",
        [](const py::array_t<double> &) { return "double"; },
        py::arg{}.noconvert());

    // Make sure we don't do unsafe coercion (e.g. float to int) when not using forcecast, but
    // rather that float gets converted via the safe (conversion to double) overload:
    sm.def("overloaded4", [](const py::array_t<long long, 0> &) { return "long long"; });
    sm.def("overloaded4", [](const py::array_t<double, 0> &) { return "double"; });

    // But we do allow conversion to int if forcecast is enabled (but only if no overload matches
    // without conversion)
    sm.def("overloaded5", [](const py::array_t<unsigned int> &) { return "unsigned int"; });
    sm.def("overloaded5", [](const py::array_t<double> &) { return "double"; });

    // test_greedy_string_overload
    // Issue 685: ndarray shouldn't go to std::string overload
    sm.def("issue685", [](const std::string &) { return "string"; });
    sm.def("issue685", [](const py::array &) { return "array"; });
    sm.def("issue685", [](const py::object &) { return "other"; });

    // test_array_unchecked_fixed_dims
    sm.def(
        "proxy_add2",
        [](py::array_t<double> a, double v) {
            auto r = a.mutable_unchecked<2>();
            for (py::ssize_t i = 0; i < r.shape(0); i++) {
                for (py::ssize_t j = 0; j < r.shape(1); j++) {
                    r(i, j) += v;
                }
            }
        },
        py::arg{}.noconvert(),
        py::arg());

    sm.def("proxy_init3", [](double start) {
        py::array_t<double, py::array::c_style> a({3, 3, 3});
        auto r = a.mutable_unchecked<3>();
        for (py::ssize_t i = 0; i < r.shape(0); i++) {
            for (py::ssize_t j = 0; j < r.shape(1); j++) {
                for (py::ssize_t k = 0; k < r.shape(2); k++) {
                    r(i, j, k) = start++;
                }
            }
        }
        return a;
    });
    sm.def("proxy_init3F", [](double start) {
        py::array_t<double, py::array::f_style> a({3, 3, 3});
        auto r = a.mutable_unchecked<3>();
        for (py::ssize_t k = 0; k < r.shape(2); k++) {
            for (py::ssize_t j = 0; j < r.shape(1); j++) {
                for (py::ssize_t i = 0; i < r.shape(0); i++) {
                    r(i, j, k) = start++;
                }
            }
        }
        return a;
    });
    sm.def("proxy_squared_L2_norm", [](const py::array_t<double> &a) {
        auto r = a.unchecked<1>();
        double sumsq = 0;
        for (py::ssize_t i = 0; i < r.shape(0); i++) {
            sumsq += r[i] * r(i); // Either notation works for a 1D array
        }
        return sumsq;
    });

    sm.def("proxy_auxiliaries2", [](py::array_t<double> a) {
        auto r = a.unchecked<2>();
        auto r2 = a.mutable_unchecked<2>();
        return auxiliaries(r, r2);
    });

    sm.def("proxy_auxiliaries1_const_ref", [](py::array_t<double> a) {
        const auto &r = a.unchecked<1>();
        const auto &r2 = a.mutable_unchecked<1>();
        return r(0) == r2(0) && r[0] == r2[0];
    });

    sm.def("proxy_auxiliaries2_const_ref", [](py::array_t<double> a) {
        const auto &r = a.unchecked<2>();
        const auto &r2 = a.mutable_unchecked<2>();
        return r(0, 0) == r2(0, 0);
    });

    // test_array_unchecked_dyn_dims
    // Same as the above, but without a compile-time dimensions specification:
    sm.def(
        "proxy_add2_dyn",
        [](py::array_t<double> a, double v) {
            auto r = a.mutable_unchecked();
            if (r.ndim() != 2) {
                throw std::domain_error("error: ndim != 2");
            }
            for (py::ssize_t i = 0; i < r.shape(0); i++) {
                for (py::ssize_t j = 0; j < r.shape(1); j++) {
                    r(i, j) += v;
                }
            }
        },
        py::arg{}.noconvert(),
        py::arg());
    sm.def("proxy_init3_dyn", [](double start) {
        py::array_t<double, py::array::c_style> a({3, 3, 3});
        auto r = a.mutable_unchecked();
        if (r.ndim() != 3) {
            throw std::domain_error("error: ndim != 3");
        }
        for (py::ssize_t i = 0; i < r.shape(0); i++) {
            for (py::ssize_t j = 0; j < r.shape(1); j++) {
                for (py::ssize_t k = 0; k < r.shape(2); k++) {
                    r(i, j, k) = start++;
                }
            }
        }
        return a;
    });
    sm.def("proxy_auxiliaries2_dyn", [](py::array_t<double> a) {
        return auxiliaries(a.unchecked(), a.mutable_unchecked());
    });

    sm.def("array_auxiliaries2", [](py::array_t<double> a) { return auxiliaries(a, a); });

    // test_array_failures
    // Issue #785: Uninformative "Unknown internal error" exception when constructing array from
    // empty object:
    sm.def("array_fail_test", []() { return py::array(py::object()); });
    sm.def("array_t_fail_test", []() { return py::array_t<double>(py::object()); });
    // Make sure the error from numpy is being passed through:
    sm.def("array_fail_test_negative_size", []() {
        int c = 0;
        return py::array(-1, &c);
    });

    // test_initializer_list
    // Issue (unnumbered; reported in #788): regression: initializer lists can be ambiguous
    sm.def("array_initializer_list1", []() { return py::array_t<float>(1); });
    // { 1 } also works for the above, but clang warns about it
    sm.def("array_initializer_list2", []() { return py::array_t<float>({1, 2}); });
    sm.def("array_initializer_list3", []() { return py::array_t<float>({1, 2, 3}); });
    sm.def("array_initializer_list4", []() { return py::array_t<float>({1, 2, 3, 4}); });

    // test_array_resize
    // reshape array to 2D without changing size
    sm.def("array_reshape2", [](py::array_t<double> a) {
        const auto dim_sz = (py::ssize_t) std::sqrt(a.size());
        if (dim_sz * dim_sz != a.size()) {
            throw std::domain_error(
                "array_reshape2: input array total size is not a squared integer");
        }
        a.resize({dim_sz, dim_sz});
    });

    // resize to 3D array with each dimension = N
    sm.def("array_resize3",
           [](py::array_t<double> a, size_t N, bool refcheck) { a.resize({N, N, N}, refcheck); });

    // test_array_create_and_resize
    // return 2D array with Nrows = Ncols = N
    sm.def("create_and_resize", [](size_t N) {
        py::array_t<double> a;
        a.resize({N, N});
        std::fill(a.mutable_data(), a.mutable_data() + a.size(), 42.);
        return a;
    });

    sm.def("array_view",
           [](py::array_t<uint8_t> a, const std::string &dtype) { return a.view(dtype); });

    sm.def("reshape_initializer_list",
           [](py::array_t<int> a, size_t N, size_t M, size_t O) { return a.reshape({N, M, O}); });
    sm.def("reshape_tuple", [](py::array_t<int> a, const std::vector<int> &new_shape) {
        return a.reshape(new_shape);
    });

    sm.def("index_using_ellipsis",
           [](const py::array &a) { return a[py::make_tuple(0, py::ellipsis(), 0)]; });

    // test_argument_conversions
    sm.def("accept_double", [](const py::array_t<double, 0> &) {}, py::arg("a"));
    sm.def(
        "accept_double_forcecast",
        [](const py::array_t<double, py::array::forcecast> &) {},
        py::arg("a"));
    sm.def(
        "accept_double_c_style",
        [](const py::array_t<double, py::array::c_style> &) {},
        py::arg("a"));
    sm.def(
        "accept_double_c_style_forcecast",
        [](const py::array_t<double, py::array::forcecast | py::array::c_style> &) {},
        py::arg("a"));
    sm.def(
        "accept_double_f_style",
        [](const py::array_t<double, py::array::f_style> &) {},
        py::arg("a"));
    sm.def(
        "accept_double_f_style_forcecast",
        [](const py::array_t<double, py::array::forcecast | py::array::f_style> &) {},
        py::arg("a"));
    sm.def("accept_double_noconvert", [](const py::array_t<double, 0> &) {}, "a"_a.noconvert());
    sm.def(
        "accept_double_forcecast_noconvert",
        [](const py::array_t<double, py::array::forcecast> &) {},
        "a"_a.noconvert());
    sm.def(
        "accept_double_c_style_noconvert",
        [](const py::array_t<double, py::array::c_style> &) {},
        "a"_a.noconvert());
    sm.def(
        "accept_double_c_style_forcecast_noconvert",
        [](const py::array_t<double, py::array::forcecast | py::array::c_style> &) {},
        "a"_a.noconvert());
    sm.def(
        "accept_double_f_style_noconvert",
        [](const py::array_t<double, py::array::f_style> &) {},
        "a"_a.noconvert());
    sm.def(
        "accept_double_f_style_forcecast_noconvert",
        [](const py::array_t<double, py::array::forcecast | py::array::f_style> &) {},
        "a"_a.noconvert());

    // Check that types returns correct npy format descriptor
    sm.def("test_fmt_desc_float", [](const py::array_t<float> &) {});
    sm.def("test_fmt_desc_double", [](const py::array_t<double> &) {});
    sm.def("test_fmt_desc_const_float", [](const py::array_t<const float> &) {});
    sm.def("test_fmt_desc_const_double", [](const py::array_t<const double> &) {});

    sm.def("round_trip_float", [](double d) { return d; });

    sm.def("pass_array_pyobject_ptr_return_sum_str_values",
           [](const py::array_t<PyObject *> &objs) {
               std::string sum_str_values;
               for (const auto &obj : objs) {
                   sum_str_values += py::str(obj.attr("value"));
               }
               return sum_str_values;
           });

    sm.def("pass_array_pyobject_ptr_return_as_list",
           [](const py::array_t<PyObject *> &objs) -> py::list { return objs; });

    sm.def("return_array_pyobject_ptr_cpp_loop", [](const py::list &objs) {
        py::size_t arr_size = py::len(objs);
        py::array_t<PyObject *> arr_from_list(static_cast<py::ssize_t>(arr_size));
        PyObject **data = arr_from_list.mutable_data();
        for (py::size_t i = 0; i < arr_size; i++) {
            assert(data[i] == nullptr);
            data[i] = py::cast<PyObject *>(objs[i].attr("value"));
        }
        return arr_from_list;
    });

    sm.def("return_array_pyobject_ptr_from_list",
           [](const py::list &objs) -> py::array_t<PyObject *> { return objs; });
}
