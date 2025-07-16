/*
    tests/eigen_tensor.cpp -- automatic conversion of Eigen Tensor

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/eigen/tensor.h>

PYBIND11_NAMESPACE_BEGIN(eigen_tensor_test)

namespace py = pybind11;

PYBIND11_WARNING_DISABLE_MSVC(4127)

template <typename M>
void reset_tensor(M &x) {
    for (int i = 0; i < x.dimension(0); i++) {
        for (int j = 0; j < x.dimension(1); j++) {
            for (int k = 0; k < x.dimension(2); k++) {
                x(i, j, k) = i * (5 * 2) + j * 2 + k;
            }
        }
    }
}

template <typename M>
bool check_tensor(M &x) {
    for (int i = 0; i < x.dimension(0); i++) {
        for (int j = 0; j < x.dimension(1); j++) {
            for (int k = 0; k < x.dimension(2); k++) {
                if (x(i, j, k) != (i * (5 * 2) + j * 2 + k)) {
                    return false;
                }
            }
        }
    }
    return true;
}

template <int Options>
Eigen::Tensor<double, 3, Options> &get_tensor() {
    static Eigen::Tensor<double, 3, Options> *x;

    if (!x) {
        x = new Eigen::Tensor<double, 3, Options>(3, 5, 2);
        reset_tensor(*x);
    }

    return *x;
}

template <int Options>
Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> &get_tensor_map() {
    static Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> *x;

    if (!x) {
        x = new Eigen::TensorMap<Eigen::Tensor<double, 3, Options>>(get_tensor<Options>());
    }

    return *x;
}

template <int Options>
Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options> &get_fixed_tensor() {
    static Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options> *x;

    if (!x) {
        Eigen::aligned_allocator<Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options>>
            allocator;
        x = new (allocator.allocate(1))
            Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options>();
        reset_tensor(*x);
    }

    return *x;
}

template <int Options>
const Eigen::Tensor<double, 3, Options> &get_const_tensor() {
    return get_tensor<Options>();
}

template <int Options>
struct CustomExample {
    CustomExample() : member(get_tensor<Options>()), view_member(member) {}

    Eigen::Tensor<double, 3, Options> member;
    Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> view_member;
};

template <int Options>
void init_tensor_module(pybind11::module &m) {
    const char *needed_options = "";
    if (Options == Eigen::ColMajor) {
        needed_options = "F";
    } else {
        needed_options = "C";
    }
    m.attr("needed_options") = needed_options;

    m.def("setup", []() {
        reset_tensor(get_tensor<Options>());
        reset_tensor(get_fixed_tensor<Options>());
    });

    m.def("is_ok", []() {
        return check_tensor(get_tensor<Options>()) && check_tensor(get_fixed_tensor<Options>());
    });

    py::class_<CustomExample<Options>>(m, "CustomExample", py::module_local())
        .def(py::init<>())
        .def_readonly(
            "member", &CustomExample<Options>::member, py::return_value_policy::reference_internal)
        .def_readonly("member_view",
                      &CustomExample<Options>::view_member,
                      py::return_value_policy::reference_internal);

    m.def(
        "copy_fixed_tensor",
        []() { return &get_fixed_tensor<Options>(); },
        py::return_value_policy::copy);

    m.def("copy_tensor", []() { return &get_tensor<Options>(); }, py::return_value_policy::copy);

    m.def(
        "copy_const_tensor",
        []() { return &get_const_tensor<Options>(); },
        py::return_value_policy::copy);

    m.def(
        "move_fixed_tensor_copy",
        []() -> Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options> {
            return get_fixed_tensor<Options>();
        },
        py::return_value_policy::move);

    m.def(
        "move_tensor_copy",
        []() -> Eigen::Tensor<double, 3, Options> { return get_tensor<Options>(); },
        py::return_value_policy::move);

    m.def(
        "move_const_tensor",
        []() -> const Eigen::Tensor<double, 3, Options> & { return get_const_tensor<Options>(); },
        py::return_value_policy::move);

    m.def(
        "take_fixed_tensor",
        []() {
            Eigen::aligned_allocator<
                Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options>>
                allocator;
            static auto *obj = new (allocator.allocate(1))
                Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options>(
                    get_fixed_tensor<Options>());
            return obj; // take_ownership will fail.
        },
        py::return_value_policy::take_ownership);

    m.def(
        "take_tensor",
        []() {
            static auto *obj = new Eigen::Tensor<double, 3, Options>(get_tensor<Options>());
            return obj; // take_ownership will fail.
        },
        py::return_value_policy::take_ownership);

    m.def(
        "take_const_tensor",
        []() -> const Eigen::Tensor<double, 3, Options> * {
            static auto *obj = new Eigen::Tensor<double, 3, Options>(get_tensor<Options>());
            return obj; // take_ownership will fail.
        },
        py::return_value_policy::take_ownership);

    m.def(
        "take_view_tensor",
        []() -> const Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> * {
            static auto *obj
                = new Eigen::TensorMap<Eigen::Tensor<double, 3, Options>>(get_tensor<Options>());
            return obj; // take_ownership will fail.
        },
        py::return_value_policy::take_ownership);

    m.def(
        "reference_tensor",
        []() { return &get_tensor<Options>(); },
        py::return_value_policy::reference);

    m.def(
        "reference_tensor_v2",
        []() -> Eigen::Tensor<double, 3, Options> & { return get_tensor<Options>(); },
        py::return_value_policy::reference);

    m.def(
        "reference_tensor_internal",
        []() { return &get_tensor<Options>(); },
        py::return_value_policy::reference_internal);

    m.def(
        "reference_fixed_tensor",
        []() { return &get_tensor<Options>(); },
        py::return_value_policy::reference);

    m.def(
        "reference_const_tensor",
        []() { return &get_const_tensor<Options>(); },
        py::return_value_policy::reference);

    m.def(
        "reference_const_tensor_v2",
        []() -> const Eigen::Tensor<double, 3, Options> & { return get_const_tensor<Options>(); },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_tensor",
        []() -> Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> {
            return get_tensor_map<Options>();
        },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_tensor_v2",
        // NOLINTNEXTLINE(readability-const-return-type)
        []() -> const Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> {
            return get_tensor_map<Options>(); // NOLINT(readability-const-return-type)
        },                                    // NOLINT(readability-const-return-type)
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_tensor_v3",
        []() -> Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> * {
            return &get_tensor_map<Options>();
        },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_tensor_v4",
        []() -> const Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> * {
            return &get_tensor_map<Options>();
        },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_tensor_v5",
        []() -> Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> & {
            return get_tensor_map<Options>();
        },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_tensor_v6",
        []() -> const Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> & {
            return get_tensor_map<Options>();
        },
        py::return_value_policy::reference);

    m.def(
        "reference_view_of_fixed_tensor",
        []() {
            return Eigen::TensorMap<
                Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options>>(
                get_fixed_tensor<Options>());
        },
        py::return_value_policy::reference);

    m.def("round_trip_tensor",
          [](const Eigen::Tensor<double, 3, Options> &tensor) { return tensor; });

    m.def(
        "round_trip_tensor_noconvert",
        [](const Eigen::Tensor<double, 3, Options> &tensor) { return tensor; },
        py::arg("tensor").noconvert());

    m.def("round_trip_tensor2",
          [](const Eigen::Tensor<int32_t, 3, Options> &tensor) { return tensor; });

    m.def("round_trip_fixed_tensor",
          [](const Eigen::TensorFixedSize<double, Eigen::Sizes<3, 5, 2>, Options> &tensor) {
              return tensor;
          });

    m.def(
        "round_trip_view_tensor",
        [](Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> view) { return view; },
        py::return_value_policy::reference);

    m.def(
        "round_trip_view_tensor_ref",
        [](Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> &view) { return view; },
        py::return_value_policy::reference);

    m.def(
        "round_trip_view_tensor_ptr",
        [](Eigen::TensorMap<Eigen::Tensor<double, 3, Options>> *view) { return view; },
        py::return_value_policy::reference);

    m.def(
        "round_trip_aligned_view_tensor",
        [](Eigen::TensorMap<Eigen::Tensor<double, 3, Options>, Eigen::Aligned> view) {
            return view;
        },
        py::return_value_policy::reference);

    m.def(
        "round_trip_const_view_tensor",
        [](Eigen::TensorMap<const Eigen::Tensor<double, 3, Options>> view) {
            return Eigen::Tensor<double, 3, Options>(view);
        },
        py::return_value_policy::move);

    m.def(
        "round_trip_rank_0",
        [](const Eigen::Tensor<double, 0, Options> &tensor) { return tensor; },
        py::return_value_policy::move);

    m.def(
        "round_trip_rank_0_noconvert",
        [](const Eigen::Tensor<double, 0, Options> &tensor) { return tensor; },
        py::arg("tensor").noconvert(),
        py::return_value_policy::move);

    m.def(
        "round_trip_rank_0_view",
        [](Eigen::TensorMap<Eigen::Tensor<double, 0, Options>> &tensor) { return tensor; },
        py::return_value_policy::reference);
}

void test_module(py::module_ &m) {
    auto f_style = m.def_submodule("f_style");
    auto c_style = m.def_submodule("c_style");

    init_tensor_module<Eigen::ColMajor>(f_style);
    init_tensor_module<Eigen::RowMajor>(c_style);
}

PYBIND11_NAMESPACE_END(eigen_tensor_test)
