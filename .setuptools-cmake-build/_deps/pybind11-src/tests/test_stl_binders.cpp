/*
    tests/test_stl_binders.cpp -- Usage of stl_binders functions

    Copyright (c) 2016 Sergey Lyskov

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

#include "pybind11_tests.h"

#include <deque>
#include <map>
#include <unordered_map>
#include <vector>

class El {
public:
    El() = delete;
    explicit El(int v) : a(v) {}

    int a;
};

std::ostream &operator<<(std::ostream &s, El const &v) {
    s << "El{" << v.a << '}';
    return s;
}

/// Issue #487: binding std::vector<E> with E non-copyable
class E_nc {
public:
    explicit E_nc(int i) : value{i} {}
    E_nc(const E_nc &) = delete;
    E_nc &operator=(const E_nc &) = delete;
    E_nc(E_nc &&) = default;
    E_nc &operator=(E_nc &&) = default;

    int value;
};

template <class Container>
Container *one_to_n(int n) {
    auto *v = new Container();
    for (int i = 1; i <= n; i++) {
        v->emplace_back(i);
    }
    return v;
}

template <class Map>
Map *times_ten(int n) {
    auto *m = new Map();
    for (int i = 1; i <= n; i++) {
        m->emplace(int(i), E_nc(10 * i));
    }
    return m;
}

template <class NestMap>
NestMap *times_hundred(int n) {
    auto *m = new NestMap();
    for (int i = 1; i <= n; i++) {
        for (int j = 1; j <= n; j++) {
            (*m)[i].emplace(int(j * 10), E_nc(100 * j));
        }
    }
    return m;
}

/*
 * Recursive data structures as test for issue #4623
 */
struct RecursiveVector : std::vector<RecursiveVector> {
    using Parent = std::vector<RecursiveVector>;
    using Parent::Parent;
};

struct RecursiveMap : std::map<int, RecursiveMap> {
    using Parent = std::map<int, RecursiveMap>;
    using Parent::Parent;
};

class UserVectorLike : private std::vector<int> {
public:
    // This is only a subset of the member functions, as needed at the time.
    using Base = std::vector<int>;
    using typename Base::const_iterator;
    using typename Base::difference_type;
    using typename Base::iterator;
    using typename Base::size_type;
    using typename Base::value_type;

    using Base::at;
    using Base::back;
    using Base::Base;
    using Base::begin;
    using Base::cbegin;
    using Base::cend;
    using Base::clear;
    using Base::empty;
    using Base::end;
    using Base::erase;
    using Base::front;
    using Base::insert;
    using Base::pop_back;
    using Base::push_back;
    using Base::reserve;
    using Base::shrink_to_fit;
    using Base::swap;
    using Base::operator[];
    using Base::capacity;
    using Base::size;
};

bool operator==(UserVectorLike const &, UserVectorLike const &) { return true; }
bool operator!=(UserVectorLike const &, UserVectorLike const &) { return false; }

class UserMapLike : private std::map<int, int> {
public:
    // This is only a subset of the member functions, as needed at the time.
    using Base = std::map<int, int>;
    using typename Base::const_iterator;
    using typename Base::iterator;
    using typename Base::key_type;
    using typename Base::mapped_type;
    using typename Base::size_type;
    using typename Base::value_type;

    using Base::at;
    using Base::Base;
    using Base::begin;
    using Base::cbegin;
    using Base::cend;
    using Base::clear;
    using Base::emplace;
    using Base::emplace_hint;
    using Base::empty;
    using Base::end;
    using Base::erase;
    using Base::find;
    using Base::insert;
    using Base::max_size;
    using Base::swap;
    using Base::operator[];
    using Base::size;
};

/*
 * Pybind11 does not catch more complicated recursion schemes, such as mutual
 * recursion.
 * In that case custom recursive_container_traits specializations need to be added,
 * thus manually telling pybind11 about the recursion.
 */
struct MutuallyRecursiveContainerPairMV;
struct MutuallyRecursiveContainerPairVM;

struct MutuallyRecursiveContainerPairMV : std::map<int, MutuallyRecursiveContainerPairVM> {};
struct MutuallyRecursiveContainerPairVM : std::vector<MutuallyRecursiveContainerPairMV> {};

namespace pybind11 {
namespace detail {
template <typename SFINAE>
struct recursive_container_traits<MutuallyRecursiveContainerPairMV, SFINAE> {
    using type_to_check_recursively = recursive_bottom;
};
template <typename SFINAE>
struct recursive_container_traits<MutuallyRecursiveContainerPairVM, SFINAE> {
    using type_to_check_recursively = recursive_bottom;
};
} // namespace detail
} // namespace pybind11

TEST_SUBMODULE(stl_binders, m) {
    // test_vector_int
    py::bind_vector<std::vector<unsigned int>>(m, "VectorInt", py::buffer_protocol());

    // test_vector_custom
    py::class_<El>(m, "El").def(py::init<int>());
    py::bind_vector<std::vector<El>>(m, "VectorEl");
    py::bind_vector<std::vector<std::vector<El>>>(m, "VectorVectorEl");

    // test_map_string_double
    py::bind_map<std::map<std::string, double>>(m, "MapStringDouble");
    py::bind_map<std::unordered_map<std::string, double>>(m, "UnorderedMapStringDouble");

    // test_map_string_double_const
    py::bind_map<std::map<std::string, double const>>(m, "MapStringDoubleConst");
    py::bind_map<std::unordered_map<std::string, double const>>(m,
                                                                "UnorderedMapStringDoubleConst");

    // test_map_view_types
    py::bind_map<std::map<std::string, float>>(m, "MapStringFloat");
    py::bind_map<std::unordered_map<std::string, float>>(m, "UnorderedMapStringFloat");

    py::bind_map<std::map<std::pair<double, int>, int32_t>>(m, "MapPairDoubleIntInt32");
    py::bind_map<std::map<std::pair<double, int>, int64_t>>(m, "MapPairDoubleIntInt64");

    py::bind_map<std::map<int, py::object>>(m, "MapIntObject");
    py::bind_map<std::map<std::string, py::object>>(m, "MapStringObject");

    py::class_<E_nc>(m, "ENC").def(py::init<int>()).def_readwrite("value", &E_nc::value);

    // test_noncopyable_containers
    py::bind_vector<std::vector<E_nc>>(m, "VectorENC");
    m.def("get_vnc", &one_to_n<std::vector<E_nc>>);
    py::bind_vector<std::deque<E_nc>>(m, "DequeENC");
    m.def("get_dnc", &one_to_n<std::deque<E_nc>>);
    py::bind_map<std::map<int, E_nc>>(m, "MapENC");
    m.def("get_mnc", &times_ten<std::map<int, E_nc>>);
    py::bind_map<std::unordered_map<int, E_nc>>(m, "UmapENC");
    m.def("get_umnc", &times_ten<std::unordered_map<int, E_nc>>);
    // Issue #1885: binding nested std::map<X, Container<E>> with E non-copyable
    py::bind_map<std::map<int, std::vector<E_nc>>>(m, "MapVecENC");
    m.def("get_nvnc", [](int n) {
        auto *m = new std::map<int, std::vector<E_nc>>();
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                (*m)[i].emplace_back(j);
            }
        }
        return m;
    });
    py::bind_map<std::map<int, std::map<int, E_nc>>>(m, "MapMapENC");
    m.def("get_nmnc", &times_hundred<std::map<int, std::map<int, E_nc>>>);
    py::bind_map<std::unordered_map<int, std::unordered_map<int, E_nc>>>(m, "UmapUmapENC");
    m.def("get_numnc", &times_hundred<std::unordered_map<int, std::unordered_map<int, E_nc>>>);

    // test_vector_buffer
    py::bind_vector<std::vector<unsigned char>>(m, "VectorUChar", py::buffer_protocol());
    // no dtype declared for this version:
    struct VUndeclStruct {
        bool w;
        uint32_t x;
        double y;
        bool z;
    };
    m.def("create_undeclstruct", [m]() mutable {
        py::bind_vector<std::vector<VUndeclStruct>>(
            m, "VectorUndeclStruct", py::buffer_protocol());
    });

    // Bind recursive container types
    py::bind_vector<RecursiveVector>(m, "RecursiveVector");
    py::bind_map<RecursiveMap>(m, "RecursiveMap");
    py::bind_map<MutuallyRecursiveContainerPairMV>(m, "MutuallyRecursiveContainerPairMV");
    py::bind_vector<MutuallyRecursiveContainerPairVM>(m, "MutuallyRecursiveContainerPairVM");

    // Bind with private inheritance + `using` directives.
    py::bind_vector<UserVectorLike>(m, "UserVectorLike");
    py::bind_map<UserMapLike>(m, "UserMapLike");

    // The rest depends on numpy:
    try {
        py::module_::import("numpy");
    } catch (...) {
        return;
    }

    // test_vector_buffer_numpy
    struct VStruct {
        bool w;
        uint32_t x;
        double y;
        bool z;
    };
    PYBIND11_NUMPY_DTYPE(VStruct, w, x, y, z);
    py::class_<VStruct>(m, "VStruct").def_readwrite("x", &VStruct::x);
    py::bind_vector<std::vector<VStruct>>(m, "VectorStruct", py::buffer_protocol());
    m.def("get_vectorstruct",
          [] { return std::vector<VStruct>{{false, 5, 3.0, true}, {true, 30, -1e4, false}}; });
}
