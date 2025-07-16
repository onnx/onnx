/*
    tests/test_stl.cpp -- STL type casters

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/stl.h>

#include "constructor_stats.h"
#include "pybind11_tests.h"

#ifndef PYBIND11_HAS_FILESYSTEM_IS_OPTIONAL
#    define PYBIND11_HAS_FILESYSTEM_IS_OPTIONAL
#endif
#include <pybind11/stl/filesystem.h>

#include <string>
#include <vector>

#if defined(PYBIND11_TEST_BOOST)
#    include <boost/optional.hpp>

namespace PYBIND11_NAMESPACE {
namespace detail {
template <typename T>
struct type_caster<boost::optional<T>> : optional_caster<boost::optional<T>> {};

template <>
struct type_caster<boost::none_t> : void_caster<boost::none_t> {};
} // namespace detail
} // namespace PYBIND11_NAMESPACE
#endif

// Test with `std::variant` in C++17 mode, or with `boost::variant` in C++11/14
#if defined(PYBIND11_HAS_VARIANT)
using std::variant;
#    define PYBIND11_TEST_VARIANT 1
#elif defined(PYBIND11_TEST_BOOST)
#    include <boost/variant.hpp>
#    define PYBIND11_TEST_VARIANT 1
using boost::variant;

namespace PYBIND11_NAMESPACE {
namespace detail {
template <typename... Ts>
struct type_caster<boost::variant<Ts...>> : variant_caster<boost::variant<Ts...>> {};

template <>
struct visit_helper<boost::variant> {
    template <typename... Args>
    static auto call(Args &&...args) -> decltype(boost::apply_visitor(args...)) {
        return boost::apply_visitor(args...);
    }
};
} // namespace detail
} // namespace PYBIND11_NAMESPACE
#endif

PYBIND11_MAKE_OPAQUE(std::vector<std::string, std::allocator<std::string>>)

/// Issue #528: templated constructor
struct TplCtorClass {
    template <typename T>
    explicit TplCtorClass(const T &) {}
    bool operator==(const TplCtorClass &) const { return true; }
};

namespace std {
template <>
struct hash<TplCtorClass> {
    size_t operator()(const TplCtorClass &) const { return 0; }
};
} // namespace std

template <template <typename> class OptionalImpl, typename T>
struct OptionalHolder {
    // NOLINTNEXTLINE(modernize-use-equals-default): breaks GCC 4.8
    OptionalHolder() {};
    bool member_initialized() const { return member && member->initialized; }
    OptionalImpl<T> member = T{};
};

enum class EnumType {
    kSet = 42,
    kUnset = 85,
};

// This is used to test that return-by-ref and return-by-copy policies are
// handled properly for optional types. This is a regression test for a dangling
// reference issue. The issue seemed to require the enum value type to
// reproduce - it didn't seem to happen if the value type is just an integer.
template <template <typename> class OptionalImpl>
class OptionalProperties {
public:
    using OptionalEnumValue = OptionalImpl<EnumType>;

    OptionalProperties() : value(EnumType::kSet) {}
    ~OptionalProperties() {
        // Reset value to detect use-after-destruction.
        // This is set to a specific value rather than nullopt to ensure that
        // the memory that contains the value gets re-written.
        value = EnumType::kUnset;
    }

    OptionalEnumValue &access_by_ref() { return value; }
    OptionalEnumValue access_by_copy() { return value; }

private:
    OptionalEnumValue value;
};

// This type mimics aspects of boost::optional from old versions of Boost,
// which exposed a dangling reference bug in Pybind11. Recent versions of
// boost::optional, as well as libstdc++'s std::optional, don't seem to be
// affected by the same issue. This is meant to be a minimal implementation
// required to reproduce the issue, not fully standard-compliant.
// See issue #3330 for more details.
template <typename T>
class ReferenceSensitiveOptional {
public:
    using value_type = T;

    ReferenceSensitiveOptional() = default;
    // NOLINTNEXTLINE(google-explicit-constructor)
    ReferenceSensitiveOptional(const T &value) : storage{value} {}
    // NOLINTNEXTLINE(google-explicit-constructor)
    ReferenceSensitiveOptional(T &&value) : storage{std::move(value)} {}
    ReferenceSensitiveOptional &operator=(const T &value) {
        storage = {value};
        return *this;
    }
    ReferenceSensitiveOptional &operator=(T &&value) {
        storage = {std::move(value)};
        return *this;
    }

    template <typename... Args>
    T &emplace(Args &&...args) {
        storage.clear();
        storage.emplace_back(std::forward<Args>(args)...);
        return storage.back();
    }

    const T &value() const noexcept {
        assert(!storage.empty());
        return storage[0];
    }

    const T &operator*() const noexcept { return value(); }

    const T *operator->() const noexcept { return &value(); }

    explicit operator bool() const noexcept { return !storage.empty(); }

private:
    std::vector<T> storage;
};

namespace PYBIND11_NAMESPACE {
namespace detail {
template <typename T>
struct type_caster<ReferenceSensitiveOptional<T>>
    : optional_caster<ReferenceSensitiveOptional<T>> {};
} // namespace detail
} // namespace PYBIND11_NAMESPACE

TEST_SUBMODULE(stl, m) {
    // test_vector
    m.def("cast_vector", []() { return std::vector<int>{1}; });
    m.def("load_vector", [](const std::vector<int> &v) { return v.at(0) == 1 && v.at(1) == 2; });
    // `std::vector<bool>` is special because it returns proxy objects instead of references
    m.def("cast_bool_vector", []() { return std::vector<bool>{true, false}; });
    m.def("load_bool_vector",
          [](const std::vector<bool> &v) { return v.at(0) == true && v.at(1) == false; });
    // Unnumbered regression (caused by #936): pointers to stl containers aren't castable
    m.def(
        "cast_ptr_vector",
        []() {
            // Using no-destructor idiom to side-step warnings from overzealous compilers.
            static auto *v = new std::vector<RValueCaster>{2};
            return v;
        },
        py::return_value_policy::reference);

    // test_deque
    m.def("cast_deque", []() { return std::deque<int>{1}; });
    m.def("load_deque", [](const std::deque<int> &v) { return v.at(0) == 1 && v.at(1) == 2; });

    // test_array
    m.def("cast_array", []() { return std::array<int, 2>{{1, 2}}; });
    m.def("load_array", [](const std::array<int, 2> &a) { return a[0] == 1 && a[1] == 2; });

    // test_valarray
    m.def("cast_valarray", []() { return std::valarray<int>{1, 4, 9}; });
    m.def("load_valarray", [](const std::valarray<int> &v) {
        return v.size() == 3 && v[0] == 1 && v[1] == 4 && v[2] == 9;
    });

    // test_map
    m.def("cast_map", []() { return std::map<std::string, std::string>{{"key", "value"}}; });
    m.def("load_map", [](const std::map<std::string, std::string> &map) {
        return map.at("key") == "value" && map.at("key2") == "value2";
    });

    // test_set
    m.def("cast_set", []() { return std::set<std::string>{"key1", "key2"}; });
    m.def("load_set", [](const std::set<std::string> &set) {
        return (set.count("key1") != 0u) && (set.count("key2") != 0u) && (set.count("key3") != 0u);
    });

    // test_recursive_casting
    m.def("cast_rv_vector", []() { return std::vector<RValueCaster>{2}; });
    m.def("cast_rv_array", []() { return std::array<RValueCaster, 3>(); });
    // NB: map and set keys are `const`, so while we technically do move them (as `const Type &&`),
    // casters don't typically do anything with that, which means they fall to the `const Type &`
    // caster.
    m.def("cast_rv_map",
          []() { return std::unordered_map<std::string, RValueCaster>{{"a", RValueCaster{}}}; });
    m.def("cast_rv_nested", []() {
        std::vector<std::array<std::list<std::unordered_map<std::string, RValueCaster>>, 2>> v;
        v.emplace_back();           // add an array
        v.back()[0].emplace_back(); // add a map to the array
        v.back()[0].back().emplace("b", RValueCaster{});
        v.back()[0].back().emplace("c", RValueCaster{});
        v.back()[1].emplace_back(); // add a map to the array
        v.back()[1].back().emplace("a", RValueCaster{});
        return v;
    });
    static std::array<RValueCaster, 2> lva;
    static std::unordered_map<std::string, RValueCaster> lvm{{"a", RValueCaster{}},
                                                             {"b", RValueCaster{}}};
    static std::unordered_map<std::string, std::vector<std::list<std::array<RValueCaster, 2>>>>
        lvn;
    lvn["a"].emplace_back();        // add a list
    lvn["a"].back().emplace_back(); // add an array
    lvn["a"].emplace_back();        // another list
    lvn["a"].back().emplace_back(); // add an array
    lvn["b"].emplace_back();        // add a list
    lvn["b"].back().emplace_back(); // add an array
    lvn["b"].back().emplace_back(); // add another array
    static std::vector<RValueCaster> lvv{2};
    m.def("cast_lv_vector", []() -> const decltype(lvv) & { return lvv; });
    m.def("cast_lv_array", []() -> const decltype(lva) & { return lva; });
    m.def("cast_lv_map", []() -> const decltype(lvm) & { return lvm; });
    m.def("cast_lv_nested", []() -> const decltype(lvn) & { return lvn; });
    // #853:
    m.def("cast_unique_ptr_vector", []() {
        std::vector<std::unique_ptr<UserType>> v;
        v.emplace_back(new UserType{7});
        v.emplace_back(new UserType{42});
        return v;
    });

    pybind11::enum_<EnumType>(m, "EnumType")
        .value("kSet", EnumType::kSet)
        .value("kUnset", EnumType::kUnset);

    // test_move_out_container
    struct MoveOutContainer {
        struct Value {
            int value;
        };
        std::list<Value> move_list() const { return {{0}, {1}, {2}}; }
    };
    py::class_<MoveOutContainer::Value>(m, "MoveOutContainerValue")
        .def_readonly("value", &MoveOutContainer::Value::value);
    py::class_<MoveOutContainer>(m, "MoveOutContainer")
        .def(py::init<>())
        .def_property_readonly("move_list", &MoveOutContainer::move_list);

    // Class that can be move- and copy-constructed, but not assigned
    struct NoAssign {
        int value;

        explicit NoAssign(int value = 0) : value(value) {}
        NoAssign(const NoAssign &) = default;
        NoAssign(NoAssign &&) = default;

        NoAssign &operator=(const NoAssign &) = delete;
        NoAssign &operator=(NoAssign &&) = delete;
    };
    py::class_<NoAssign>(m, "NoAssign", "Class with no C++ assignment operators")
        .def(py::init<>())
        .def(py::init<int>());

    struct MoveOutDetector {
        MoveOutDetector() = default;
        MoveOutDetector(const MoveOutDetector &) = default;
        MoveOutDetector(MoveOutDetector &&other) noexcept : initialized(other.initialized) {
            // steal underlying resource
            other.initialized = false;
        }
        bool initialized = true;
    };
    py::class_<MoveOutDetector>(m, "MoveOutDetector", "Class with move tracking")
        .def(py::init<>())
        .def_readonly("initialized", &MoveOutDetector::initialized);

#ifdef PYBIND11_HAS_OPTIONAL
    // test_optional
    m.attr("has_optional") = true;

    using opt_int = std::optional<int>;
    using opt_no_assign = std::optional<NoAssign>;
    m.def("double_or_zero", [](const opt_int &x) -> int { return x.value_or(0) * 2; });
    m.def("half_or_none", [](int x) -> opt_int { return x != 0 ? opt_int(x / 2) : opt_int(); });
    m.def(
        "test_nullopt",
        [](opt_int x) { return x.value_or(42); },
        py::arg_v("x", std::nullopt, "None"));
    m.def(
        "test_no_assign",
        [](const opt_no_assign &x) { return x ? x->value : 42; },
        py::arg_v("x", std::nullopt, "None"));

    m.def("nodefer_none_optional", [](std::optional<int>) { return true; });
    m.def("nodefer_none_optional", [](const py::none &) { return false; });

    using opt_holder = OptionalHolder<std::optional, MoveOutDetector>;
    py::class_<opt_holder>(m, "OptionalHolder", "Class with optional member")
        .def(py::init<>())
        .def_readonly("member", &opt_holder::member)
        .def("member_initialized", &opt_holder::member_initialized);

    using opt_props = OptionalProperties<std::optional>;
    pybind11::class_<opt_props>(m, "OptionalProperties")
        .def(pybind11::init<>())
        .def_property_readonly("access_by_ref", &opt_props::access_by_ref)
        .def_property_readonly("access_by_copy", &opt_props::access_by_copy);
#endif

#ifdef PYBIND11_HAS_EXP_OPTIONAL
    // test_exp_optional
    m.attr("has_exp_optional") = true;

    using exp_opt_int = std::experimental::optional<int>;
    using exp_opt_no_assign = std::experimental::optional<NoAssign>;
    m.def("double_or_zero_exp", [](const exp_opt_int &x) -> int { return x.value_or(0) * 2; });
    m.def("half_or_none_exp",
          [](int x) -> exp_opt_int { return x ? exp_opt_int(x / 2) : exp_opt_int(); });
    m.def(
        "test_nullopt_exp",
        [](exp_opt_int x) { return x.value_or(42); },
        py::arg_v("x", std::experimental::nullopt, "None"));
    m.def(
        "test_no_assign_exp",
        [](const exp_opt_no_assign &x) { return x ? x->value : 42; },
        py::arg_v("x", std::experimental::nullopt, "None"));

    using opt_exp_holder = OptionalHolder<std::experimental::optional, MoveOutDetector>;
    py::class_<opt_exp_holder>(m, "OptionalExpHolder", "Class with optional member")
        .def(py::init<>())
        .def_readonly("member", &opt_exp_holder::member)
        .def("member_initialized", &opt_exp_holder::member_initialized);

    using opt_exp_props = OptionalProperties<std::experimental::optional>;
    pybind11::class_<opt_exp_props>(m, "OptionalExpProperties")
        .def(pybind11::init<>())
        .def_property_readonly("access_by_ref", &opt_exp_props::access_by_ref)
        .def_property_readonly("access_by_copy", &opt_exp_props::access_by_copy);
#endif

#if defined(PYBIND11_TEST_BOOST)
    // test_boost_optional
    m.attr("has_boost_optional") = true;

    using boost_opt_int = boost::optional<int>;
    using boost_opt_no_assign = boost::optional<NoAssign>;
    m.def("double_or_zero_boost", [](const boost_opt_int &x) -> int { return x.value_or(0) * 2; });
    m.def("half_or_none_boost",
          [](int x) -> boost_opt_int { return x != 0 ? boost_opt_int(x / 2) : boost_opt_int(); });
    m.def(
        "test_nullopt_boost",
        [](boost_opt_int x) { return x.value_or(42); },
        py::arg_v("x", boost::none, "None"));
    m.def(
        "test_no_assign_boost",
        [](const boost_opt_no_assign &x) { return x ? x->value : 42; },
        py::arg_v("x", boost::none, "None"));

    using opt_boost_holder = OptionalHolder<boost::optional, MoveOutDetector>;
    py::class_<opt_boost_holder>(m, "OptionalBoostHolder", "Class with optional member")
        .def(py::init<>())
        .def_readonly("member", &opt_boost_holder::member)
        .def("member_initialized", &opt_boost_holder::member_initialized);

    using opt_boost_props = OptionalProperties<boost::optional>;
    pybind11::class_<opt_boost_props>(m, "OptionalBoostProperties")
        .def(pybind11::init<>())
        .def_property_readonly("access_by_ref", &opt_boost_props::access_by_ref)
        .def_property_readonly("access_by_copy", &opt_boost_props::access_by_copy);
#endif

    // test_refsensitive_optional
    using refsensitive_opt_int = ReferenceSensitiveOptional<int>;
    using refsensitive_opt_no_assign = ReferenceSensitiveOptional<NoAssign>;
    m.def("double_or_zero_refsensitive",
          [](const refsensitive_opt_int &x) -> int { return (x ? x.value() : 0) * 2; });
    m.def("half_or_none_refsensitive", [](int x) -> refsensitive_opt_int {
        return x != 0 ? refsensitive_opt_int(x / 2) : refsensitive_opt_int();
    });
    m.def(
        "test_nullopt_refsensitive",
        // NOLINTNEXTLINE(performance-unnecessary-value-param)
        [](refsensitive_opt_int x) { return x ? x.value() : 42; },
        py::arg_v("x", refsensitive_opt_int(), "None"));
    m.def(
        "test_no_assign_refsensitive",
        [](const refsensitive_opt_no_assign &x) { return x ? x->value : 42; },
        py::arg_v("x", refsensitive_opt_no_assign(), "None"));

    using opt_refsensitive_holder = OptionalHolder<ReferenceSensitiveOptional, MoveOutDetector>;
    py::class_<opt_refsensitive_holder>(
        m, "OptionalRefSensitiveHolder", "Class with optional member")
        .def(py::init<>())
        .def_readonly("member", &opt_refsensitive_holder::member)
        .def("member_initialized", &opt_refsensitive_holder::member_initialized);

    using opt_refsensitive_props = OptionalProperties<ReferenceSensitiveOptional>;
    pybind11::class_<opt_refsensitive_props>(m, "OptionalRefSensitiveProperties")
        .def(pybind11::init<>())
        .def_property_readonly("access_by_ref", &opt_refsensitive_props::access_by_ref)
        .def_property_readonly("access_by_copy", &opt_refsensitive_props::access_by_copy);

#ifdef PYBIND11_HAS_FILESYSTEM
    // test_fs_path
    m.attr("has_filesystem") = true;
    m.def("parent_path", [](const std::filesystem::path &p) { return p.parent_path(); });
#endif

#ifdef PYBIND11_TEST_VARIANT
    static_assert(std::is_same<py::detail::variant_caster_visitor::result_type, py::handle>::value,
                  "visitor::result_type is required by boost::variant in C++11 mode");

    struct visitor {
        using result_type = const char *;

        result_type operator()(int) { return "int"; }
        result_type operator()(const std::string &) { return "std::string"; }
        result_type operator()(double) { return "double"; }
        result_type operator()(std::nullptr_t) { return "std::nullptr_t"; }
#    if defined(PYBIND11_HAS_VARIANT)
        result_type operator()(std::monostate) { return "std::monostate"; }
#    endif
    };

    // test_variant
    m.def("load_variant", [](const variant<int, std::string, double, std::nullptr_t> &v) {
        return py::detail::visit_helper<variant>::call(visitor(), v);
    });
    m.def("load_variant_2pass", [](variant<double, int> v) {
        return py::detail::visit_helper<variant>::call(visitor(), v);
    });
    m.def("cast_variant", []() {
        using V = variant<int, std::string>;
        return py::make_tuple(V(5), V("Hello"));
    });

#    if defined(PYBIND11_HAS_VARIANT)
    // std::monostate tests.
    m.def("load_monostate_variant",
          [](const variant<std::monostate, int, std::string> &v) -> const char * {
              return py::detail::visit_helper<variant>::call(visitor(), v);
          });
    m.def("cast_monostate_variant", []() {
        using V = variant<std::monostate, int, std::string>;
        return py::make_tuple(V{}, V(5), V("Hello"));
    });
#    endif
#endif

    // #528: templated constructor
    // (no python tests: the test here is that this compiles)
    m.def("tpl_ctor_vector", [](std::vector<TplCtorClass> &) {});
    m.def("tpl_ctor_map", [](std::unordered_map<TplCtorClass, TplCtorClass> &) {});
    m.def("tpl_ctor_set", [](std::unordered_set<TplCtorClass> &) {});
#if defined(PYBIND11_HAS_OPTIONAL)
    m.def("tpl_constr_optional", [](std::optional<TplCtorClass> &) {});
#endif
#if defined(PYBIND11_HAS_EXP_OPTIONAL)
    m.def("tpl_constr_optional_exp", [](std::experimental::optional<TplCtorClass> &) {});
#endif
#if defined(PYBIND11_TEST_BOOST)
    m.def("tpl_constr_optional_boost", [](boost::optional<TplCtorClass> &) {});
#endif

    // test_vec_of_reference_wrapper
    // #171: Can't return STL structures containing reference wrapper
    m.def("return_vec_of_reference_wrapper", [](std::reference_wrapper<UserType> p4) {
        static UserType p1{1}, p2{2}, p3{3};
        return std::vector<std::reference_wrapper<UserType>>{
            std::ref(p1), std::ref(p2), std::ref(p3), p4};
    });

    // test_stl_pass_by_pointer
    m.def("stl_pass_by_pointer", [](std::vector<int> *v) { return *v; }, "v"_a = nullptr);

    // #1258: pybind11/stl.h converts string to vector<string>
    m.def("func_with_string_or_vector_string_arg_overload",
          [](const std::vector<std::string> &) { return 1; });
    m.def("func_with_string_or_vector_string_arg_overload",
          [](const std::list<std::string> &) { return 2; });
    m.def("func_with_string_or_vector_string_arg_overload", [](const std::string &) { return 3; });

    class Placeholder {
    public:
        Placeholder() { print_created(this); }
        Placeholder(const Placeholder &) = delete;
        ~Placeholder() { print_destroyed(this); }
    };
    py::class_<Placeholder>(m, "Placeholder");

    /// test_stl_vector_ownership
    m.def(
        "test_stl_ownership",
        []() {
            std::vector<Placeholder *> result;
            result.push_back(new Placeholder());
            return result;
        },
        py::return_value_policy::take_ownership);

    m.def("array_cast_sequence", [](std::array<int, 3> x) { return x; });

    /// test_issue_1561
    struct Issue1561Inner {
        std::string data;
    };
    struct Issue1561Outer {
        std::vector<Issue1561Inner> list;
    };

    py::class_<Issue1561Inner>(m, "Issue1561Inner")
        .def(py::init<std::string>())
        .def_readwrite("data", &Issue1561Inner::data);

    py::class_<Issue1561Outer>(m, "Issue1561Outer")
        .def(py::init<>())
        .def_readwrite("list", &Issue1561Outer::list);

    m.def(
        "return_vector_bool_raw_ptr",
        []() { return new std::vector<bool>(4513); },
        // Without explicitly specifying `take_ownership`, this function leaks.
        py::return_value_policy::take_ownership);
}
