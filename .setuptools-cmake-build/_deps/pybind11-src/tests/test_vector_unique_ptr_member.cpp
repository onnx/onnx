#include "pybind11_tests.h"

#include <cstddef>
#include <memory>
#include <vector>

namespace pybind11_tests {
namespace vector_unique_ptr_member {

struct DataType {};

// Reduced from a use case in the wild.
struct VectorOwner {
    static std::unique_ptr<VectorOwner> Create(std::size_t num_elems) {
        return std::unique_ptr<VectorOwner>(
            new VectorOwner(std::vector<std::unique_ptr<DataType>>(num_elems)));
    }

    std::size_t data_size() const { return data_.size(); }

private:
    explicit VectorOwner(std::vector<std::unique_ptr<DataType>> data) : data_(std::move(data)) {}

    const std::vector<std::unique_ptr<DataType>> data_;
};

} // namespace vector_unique_ptr_member
} // namespace pybind11_tests

namespace pybind11 {
namespace detail {

template <>
struct is_copy_constructible<pybind11_tests::vector_unique_ptr_member::VectorOwner>
    : std::false_type {};

template <>
struct is_move_constructible<pybind11_tests::vector_unique_ptr_member::VectorOwner>
    : std::false_type {};

} // namespace detail
} // namespace pybind11

using namespace pybind11_tests::vector_unique_ptr_member;

py::object py_cast_VectorOwner_ptr(VectorOwner *ptr) { return py::cast(ptr); }

TEST_SUBMODULE(vector_unique_ptr_member, m) {
    py::class_<VectorOwner>(m, "VectorOwner")
        .def_static("Create", &VectorOwner::Create)
        .def("data_size", &VectorOwner::data_size);

    m.def("py_cast_VectorOwner_ptr", py_cast_VectorOwner_ptr);
}
