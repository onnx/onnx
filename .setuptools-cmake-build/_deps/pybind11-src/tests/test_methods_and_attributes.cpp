/*
    tests/test_methods_and_attributes.cpp -- constructors, deconstructors, attribute access,
    __str__, argument and return value conventions

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "constructor_stats.h"
#include "pybind11_tests.h"

#if !defined(PYBIND11_OVERLOAD_CAST)
template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;
#endif

class ExampleMandA {
public:
    ExampleMandA() { print_default_created(this); }
    explicit ExampleMandA(int value) : value(value) { print_created(this, value); }
    ExampleMandA(const ExampleMandA &e) : value(e.value) { print_copy_created(this); }
    explicit ExampleMandA(std::string &&) {}
    ExampleMandA(ExampleMandA &&e) noexcept : value(e.value) { print_move_created(this); }
    ~ExampleMandA() { print_destroyed(this); }

    std::string toString() const { return "ExampleMandA[value=" + std::to_string(value) + "]"; }

    void operator=(const ExampleMandA &e) {
        print_copy_assigned(this);
        value = e.value;
    }
    void operator=(ExampleMandA &&e) noexcept {
        print_move_assigned(this);
        value = e.value;
    }

    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    void add1(ExampleMandA other) { value += other.value; }         // passing by value
    void add2(ExampleMandA &other) { value += other.value; }        // passing by reference
    void add3(const ExampleMandA &other) { value += other.value; }  // passing by const reference
    void add4(ExampleMandA *other) { value += other->value; }       // passing by pointer
    void add5(const ExampleMandA *other) { value += other->value; } // passing by const pointer

    void add6(int other) { value += other; }        // passing by value
    void add7(int &other) { value += other; }       // passing by reference
    void add8(const int &other) { value += other; } // passing by const reference
    // NOLINTNEXTLINE(readability-non-const-parameter) Deliberately non-const for testing
    void add9(int *other) { value += *other; }        // passing by pointer
    void add10(const int *other) { value += *other; } // passing by const pointer

    void consume_str(std::string &&) {}

    ExampleMandA self1() { return *this; }              // return by value
    ExampleMandA &self2() { return *this; }             // return by reference
    const ExampleMandA &self3() const { return *this; } // return by const reference
    ExampleMandA *self4() { return this; }              // return by pointer
    const ExampleMandA *self5() const { return this; }  // return by const pointer

    int internal1() const { return value; }        // return by value
    int &internal2() { return value; }             // return by reference
    const int &internal3() const { return value; } // return by const reference
    int *internal4() { return &value; }            // return by pointer
    const int *internal5() { return &value; }      // return by const pointer

    py::str overloaded() { return "()"; }
    py::str overloaded(int) { return "(int)"; }
    py::str overloaded(int, float) { return "(int, float)"; }
    py::str overloaded(float, int) { return "(float, int)"; }
    py::str overloaded(int, int) { return "(int, int)"; }
    py::str overloaded(float, float) { return "(float, float)"; }
    py::str overloaded(int) const { return "(int) const"; }
    py::str overloaded(int, float) const { return "(int, float) const"; }
    py::str overloaded(float, int) const { return "(float, int) const"; }
    py::str overloaded(int, int) const { return "(int, int) const"; }
    py::str overloaded(float, float) const { return "(float, float) const"; }

    static py::str overloaded(float) { return "static float"; }

    int value = 0;
};

struct TestProperties {
    int value = 1;
    static int static_value;

    int get() const { return value; }
    void set(int v) { value = v; }

    static int static_get() { return static_value; }
    static void static_set(int v) { static_value = v; }
};
int TestProperties::static_value = 1;

struct TestPropertiesOverride : TestProperties {
    int value = 99;
    static int static_value;
};
int TestPropertiesOverride::static_value = 99;

struct TestPropRVP {
    UserType v1{1};
    UserType v2{1};
    static UserType sv1;
    static UserType sv2;

    const UserType &get1() const { return v1; }
    const UserType &get2() const { return v2; }
    UserType get_rvalue() const { return v2; }
    void set1(int v) { v1.set(v); }
    void set2(int v) { v2.set(v); }
};
UserType TestPropRVP::sv1(1);
UserType TestPropRVP::sv2(1);

// Test None-allowed py::arg argument policy
class NoneTester {
public:
    int answer = 42;
};
int none1(const NoneTester &obj) { return obj.answer; }
int none2(NoneTester *obj) { return obj ? obj->answer : -1; }
int none3(std::shared_ptr<NoneTester> &obj) { return obj ? obj->answer : -1; }
int none4(std::shared_ptr<NoneTester> *obj) { return obj && *obj ? (*obj)->answer : -1; }
int none5(const std::shared_ptr<NoneTester> &obj) { return obj ? obj->answer : -1; }

// Issue #2778: implicit casting from None to object (not pointer)
class NoneCastTester {
public:
    int answer = -1;
    NoneCastTester() = default;
    explicit NoneCastTester(int v) : answer(v) {}
};

struct StrIssue {
    int val = -1;

    StrIssue() = default;
    explicit StrIssue(int i) : val{i} {}
};

// Issues #854, #910: incompatible function args when member function/pointer is in unregistered
// base class
class UnregisteredBase {
public:
    void do_nothing() const {}
    void increase_value() {
        rw_value++;
        ro_value += 0.25;
    }
    void set_int(int v) { rw_value = v; }
    int get_int() const { return rw_value; }
    double get_double() const { return ro_value; }
    int rw_value = 42;
    double ro_value = 1.25;
};
class RegisteredDerived : public UnregisteredBase {
public:
    using UnregisteredBase::UnregisteredBase;
    double sum() const { return rw_value + ro_value; }
};

// Test explicit lvalue ref-qualification
struct RefQualified {
    int value = 0;

    void refQualified(int other) & { value += other; }
    int constRefQualified(int other) const & { return value + other; }
};

// Test rvalue ref param
struct RValueRefParam {
    std::size_t func1(std::string &&s) { return s.size(); }
    std::size_t func2(std::string &&s) const { return s.size(); }
    std::size_t func3(std::string &&s) & { return s.size(); }
    std::size_t func4(std::string &&s) const & { return s.size(); }
};

namespace pybind11_tests {
namespace exercise_is_setter {

struct FieldBase {
    int int_value() const { return int_value_; }

    FieldBase &SetIntValue(int int_value) {
        int_value_ = int_value;
        return *this;
    }

private:
    int int_value_ = -99;
};

struct Field : FieldBase {};

void add_bindings(py::module &m) {
    py::module sm = m.def_submodule("exercise_is_setter");
    // NOTE: FieldBase is not wrapped, therefore ...
    py::class_<Field>(sm, "Field")
        .def(py::init<>())
        .def_property(
            "int_value",
            &Field::int_value,
            &Field::SetIntValue // ... the `FieldBase &` return value here cannot be converted.
        );
}

} // namespace exercise_is_setter
} // namespace pybind11_tests

TEST_SUBMODULE(methods_and_attributes, m) {
    // test_methods_and_attributes
    py::class_<ExampleMandA> emna(m, "ExampleMandA");
    emna.def(py::init<>())
        .def(py::init<int>())
        .def(py::init<std::string &&>())
        .def(py::init<const ExampleMandA &>())
        .def("add1", &ExampleMandA::add1)
        .def("add2", &ExampleMandA::add2)
        .def("add3", &ExampleMandA::add3)
        .def("add4", &ExampleMandA::add4)
        .def("add5", &ExampleMandA::add5)
        .def("add6", &ExampleMandA::add6)
        .def("add7", &ExampleMandA::add7)
        .def("add8", &ExampleMandA::add8)
        .def("add9", &ExampleMandA::add9)
        .def("add10", &ExampleMandA::add10)
        .def("consume_str", &ExampleMandA::consume_str)
        .def("self1", &ExampleMandA::self1)
        .def("self2", &ExampleMandA::self2)
        .def("self3", &ExampleMandA::self3)
        .def("self4", &ExampleMandA::self4)
        .def("self5", &ExampleMandA::self5)
        .def("internal1", &ExampleMandA::internal1)
        .def("internal2", &ExampleMandA::internal2)
        .def("internal3", &ExampleMandA::internal3)
        .def("internal4", &ExampleMandA::internal4)
        .def("internal5", &ExampleMandA::internal5)
#if defined(PYBIND11_OVERLOAD_CAST)
        .def("overloaded", py::overload_cast<>(&ExampleMandA::overloaded))
        .def("overloaded", py::overload_cast<int>(&ExampleMandA::overloaded))
        .def("overloaded", py::overload_cast<int, float>(&ExampleMandA::overloaded))
        .def("overloaded", py::overload_cast<float, int>(&ExampleMandA::overloaded))
        .def("overloaded", py::overload_cast<int, int>(&ExampleMandA::overloaded))
        .def("overloaded", py::overload_cast<float, float>(&ExampleMandA::overloaded))
        .def("overloaded_float", py::overload_cast<float, float>(&ExampleMandA::overloaded))
        .def("overloaded_const", py::overload_cast<int>(&ExampleMandA::overloaded, py::const_))
        .def("overloaded_const",
             py::overload_cast<int, float>(&ExampleMandA::overloaded, py::const_))
        .def("overloaded_const",
             py::overload_cast<float, int>(&ExampleMandA::overloaded, py::const_))
        .def("overloaded_const",
             py::overload_cast<int, int>(&ExampleMandA::overloaded, py::const_))
        .def("overloaded_const",
             py::overload_cast<float, float>(&ExampleMandA::overloaded, py::const_))
#else
        // Use both the traditional static_cast method and the C++11 compatible overload_cast_
        .def("overloaded", overload_cast_<>()(&ExampleMandA::overloaded))
        .def("overloaded", overload_cast_<int>()(&ExampleMandA::overloaded))
        .def("overloaded", overload_cast_<int,   float>()(&ExampleMandA::overloaded))
        .def("overloaded", static_cast<py::str (ExampleMandA::*)(float,   int)>(&ExampleMandA::overloaded))
        .def("overloaded", static_cast<py::str (ExampleMandA::*)(int,     int)>(&ExampleMandA::overloaded))
        .def("overloaded", static_cast<py::str (ExampleMandA::*)(float, float)>(&ExampleMandA::overloaded))
        .def("overloaded_float", overload_cast_<float, float>()(&ExampleMandA::overloaded))
        .def("overloaded_const", overload_cast_<int         >()(&ExampleMandA::overloaded, py::const_))
        .def("overloaded_const", overload_cast_<int,   float>()(&ExampleMandA::overloaded, py::const_))
        .def("overloaded_const", static_cast<py::str (ExampleMandA::*)(float,   int) const>(&ExampleMandA::overloaded))
        .def("overloaded_const", static_cast<py::str (ExampleMandA::*)(int,     int) const>(&ExampleMandA::overloaded))
        .def("overloaded_const", static_cast<py::str (ExampleMandA::*)(float, float) const>(&ExampleMandA::overloaded))
#endif
        // test_no_mixed_overloads
        // Raise error if trying to mix static/non-static overloads on the same name:
        .def_static("add_mixed_overloads1",
                    []() {
                        auto emna = py::reinterpret_borrow<py::class_<ExampleMandA>>(
                            py::module_::import("pybind11_tests.methods_and_attributes")
                                .attr("ExampleMandA"));
                        emna.def("overload_mixed1",
                                 static_cast<py::str (ExampleMandA::*)(int, int)>(
                                     &ExampleMandA::overloaded))
                            .def_static(
                                "overload_mixed1",
                                static_cast<py::str (*)(float)>(&ExampleMandA::overloaded));
                    })
        .def_static("add_mixed_overloads2",
                    []() {
                        auto emna = py::reinterpret_borrow<py::class_<ExampleMandA>>(
                            py::module_::import("pybind11_tests.methods_and_attributes")
                                .attr("ExampleMandA"));
                        emna.def_static("overload_mixed2",
                                        static_cast<py::str (*)(float)>(&ExampleMandA::overloaded))
                            .def("overload_mixed2",
                                 static_cast<py::str (ExampleMandA::*)(int, int)>(
                                     &ExampleMandA::overloaded));
                    })
        .def("__str__", &ExampleMandA::toString)
        .def_readwrite("value", &ExampleMandA::value);

    // test_copy_method
    // Issue #443: can't call copied methods in Python 3
    emna.attr("add2b") = emna.attr("add2");

    // test_properties, test_static_properties, test_static_cls
    py::class_<TestProperties>(m, "TestProperties")
        .def(py::init<>())
        .def_readonly("def_readonly", &TestProperties::value)
        .def_readwrite("def_readwrite", &TestProperties::value)
        .def_property("def_writeonly", nullptr, [](TestProperties &s, int v) { s.value = v; })
        .def_property("def_property_writeonly", nullptr, &TestProperties::set)
        .def_property_readonly("def_property_readonly", &TestProperties::get)
        .def_property("def_property", &TestProperties::get, &TestProperties::set)
        .def_property("def_property_impossible", nullptr, nullptr)
        .def_readonly_static("def_readonly_static", &TestProperties::static_value)
        .def_readwrite_static("def_readwrite_static", &TestProperties::static_value)
        .def_property_static("def_writeonly_static",
                             nullptr,
                             [](const py::object &, int v) { TestProperties::static_value = v; })
        .def_property_readonly_static(
            "def_property_readonly_static",
            [](const py::object &) { return TestProperties::static_get(); })
        .def_property_static(
            "def_property_writeonly_static",
            nullptr,
            [](const py::object &, int v) { return TestProperties::static_set(v); })
        .def_property_static(
            "def_property_static",
            [](const py::object &) { return TestProperties::static_get(); },
            [](const py::object &, int v) { TestProperties::static_set(v); })
        .def_property_static(
            "static_cls",
            [](py::object cls) { return cls; },
            [](const py::object &cls, const py::function &f) { f(cls); });

    py::class_<TestPropertiesOverride, TestProperties>(m, "TestPropertiesOverride")
        .def(py::init<>())
        .def_readonly("def_readonly", &TestPropertiesOverride::value)
        .def_readonly_static("def_readonly_static", &TestPropertiesOverride::static_value);

    auto static_get1 = [](const py::object &) -> const UserType & { return TestPropRVP::sv1; };
    auto static_get2 = [](const py::object &) -> const UserType & { return TestPropRVP::sv2; };
    auto static_set1 = [](const py::object &, int v) { TestPropRVP::sv1.set(v); };
    auto static_set2 = [](const py::object &, int v) { TestPropRVP::sv2.set(v); };
    auto rvp_copy = py::return_value_policy::copy;

    // test_property_return_value_policies
    py::class_<TestPropRVP>(m, "TestPropRVP")
        .def(py::init<>())
        .def_property_readonly("ro_ref", &TestPropRVP::get1)
        .def_property_readonly("ro_copy", &TestPropRVP::get2, rvp_copy)
        .def_property_readonly("ro_func", py::cpp_function(&TestPropRVP::get2, rvp_copy))
        .def_property("rw_ref", &TestPropRVP::get1, &TestPropRVP::set1)
        .def_property("rw_copy", &TestPropRVP::get2, &TestPropRVP::set2, rvp_copy)
        .def_property(
            "rw_func", py::cpp_function(&TestPropRVP::get2, rvp_copy), &TestPropRVP::set2)
        .def_property_readonly_static("static_ro_ref", static_get1)
        .def_property_readonly_static("static_ro_copy", static_get2, rvp_copy)
        .def_property_readonly_static("static_ro_func", py::cpp_function(static_get2, rvp_copy))
        .def_property_static("static_rw_ref", static_get1, static_set1)
        .def_property_static("static_rw_copy", static_get2, static_set2, rvp_copy)
        .def_property_static(
            "static_rw_func", py::cpp_function(static_get2, rvp_copy), static_set2)
        // test_property_rvalue_policy
        .def_property_readonly("rvalue", &TestPropRVP::get_rvalue)
        .def_property_readonly_static("static_rvalue",
                                      [](const py::object &) { return UserType(1); });

    // test_metaclass_override
    struct MetaclassOverride {};
    py::class_<MetaclassOverride>(m, "MetaclassOverride", py::metaclass((PyObject *) &PyType_Type))
        .def_property_readonly_static("readonly", [](const py::object &) { return 1; });

    // test_overload_ordering
    m.def("overload_order", [](const std::string &) { return 1; });
    m.def("overload_order", [](const std::string &) { return 2; });
    m.def("overload_order", [](int) { return 3; });
    m.def("overload_order", [](int) { return 4; }, py::prepend{});

#if !defined(PYPY_VERSION)
    // test_dynamic_attributes
    class DynamicClass {
    public:
        DynamicClass() { print_default_created(this); }
        DynamicClass(const DynamicClass &) = delete;
        ~DynamicClass() { print_destroyed(this); }
    };
    py::class_<DynamicClass>(m, "DynamicClass", py::dynamic_attr()).def(py::init());

    class CppDerivedDynamicClass : public DynamicClass {};
    py::class_<CppDerivedDynamicClass, DynamicClass>(m, "CppDerivedDynamicClass").def(py::init());
#endif

    // test_bad_arg_default
    // Issue/PR #648: bad arg default debugging output
#if defined(PYBIND11_DETAILED_ERROR_MESSAGES)
    m.attr("detailed_error_messages_enabled") = true;
#else
    m.attr("detailed_error_messages_enabled") = false;
#endif
    m.def("bad_arg_def_named", [] {
        auto m = py::module_::import("pybind11_tests");
        m.def(
            "should_fail",
            [](int, UnregisteredType) {},
            py::arg(),
            py::arg("a") = UnregisteredType());
    });
    m.def("bad_arg_def_unnamed", [] {
        auto m = py::module_::import("pybind11_tests");
        m.def(
            "should_fail",
            [](int, UnregisteredType) {},
            py::arg(),
            py::arg() = UnregisteredType());
    });

    // [workaround(intel)] ICC 20/21 breaks with py::arg().stuff, using py::arg{}.stuff works.

    // test_accepts_none
    py::class_<NoneTester, std::shared_ptr<NoneTester>>(m, "NoneTester").def(py::init<>());
    m.def("no_none1", &none1, py::arg{}.none(false));
    m.def("no_none2", &none2, py::arg{}.none(false));
    m.def("no_none3", &none3, py::arg{}.none(false));
    m.def("no_none4", &none4, py::arg{}.none(false));
    m.def("no_none5", &none5, py::arg{}.none(false));
    m.def("ok_none1", &none1);
    m.def("ok_none2", &none2, py::arg{}.none(true));
    m.def("ok_none3", &none3);
    m.def("ok_none4", &none4, py::arg{}.none(true));
    m.def("ok_none5", &none5);

    m.def("no_none_kwarg", &none2, "a"_a.none(false));
    m.def("no_none_kwarg_kw_only", &none2, py::kw_only(), "a"_a.none(false));

    // test_casts_none
    // Issue #2778: implicit casting from None to object (not pointer)
    py::class_<NoneCastTester>(m, "NoneCastTester")
        .def(py::init<>())
        .def(py::init<int>())
        .def(py::init([](py::none const &) { return NoneCastTester{}; }));
    py::implicitly_convertible<py::none, NoneCastTester>();
    m.def("ok_obj_or_none", [](NoneCastTester const &foo) { return foo.answer; });

    // test_str_issue
    // Issue #283: __str__ called on uninitialized instance when constructor arguments invalid
    py::class_<StrIssue>(m, "StrIssue")
        .def(py::init<int>())
        .def(py::init<>())
        .def("__str__",
             [](const StrIssue &si) { return "StrIssue[" + std::to_string(si.val) + "]"; });

    // test_unregistered_base_implementations
    //
    // Issues #854/910: incompatible function args when member function/pointer is in unregistered
    // base class The methods and member pointers below actually resolve to members/pointers in
    // UnregisteredBase; before this test/fix they would be registered via lambda with a first
    // argument of an unregistered type, and thus uncallable.
    py::class_<RegisteredDerived>(m, "RegisteredDerived")
        .def(py::init<>())
        .def("do_nothing", &RegisteredDerived::do_nothing)
        .def("increase_value", &RegisteredDerived::increase_value)
        .def_readwrite("rw_value", &RegisteredDerived::rw_value)
        .def_readonly("ro_value", &RegisteredDerived::ro_value)
        // Uncommenting the next line should trigger a static_assert:
        // .def_readwrite("fails", &UserType::value)
        // Uncommenting the next line should trigger a static_assert:
        // .def_readonly("fails", &UserType::value)
        .def_property("rw_value_prop", &RegisteredDerived::get_int, &RegisteredDerived::set_int)
        .def_property_readonly("ro_value_prop", &RegisteredDerived::get_double)
        // This one is in the registered class:
        .def("sum", &RegisteredDerived::sum);

    using Adapted
        = decltype(py::method_adaptor<RegisteredDerived>(&RegisteredDerived::do_nothing));
    static_assert(std::is_same<Adapted, void (RegisteredDerived::*)() const>::value, "");

    // test_methods_and_attributes
    py::class_<RefQualified>(m, "RefQualified")
        .def(py::init<>())
        .def_readonly("value", &RefQualified::value)
        .def("refQualified", &RefQualified::refQualified)
        .def("constRefQualified", &RefQualified::constRefQualified);

    py::class_<RValueRefParam>(m, "RValueRefParam")
        .def(py::init<>())
        .def("func1", &RValueRefParam::func1)
        .def("func2", &RValueRefParam::func2)
        .def("func3", &RValueRefParam::func3)
        .def("func4", &RValueRefParam::func4);

    pybind11_tests::exercise_is_setter::add_bindings(m);
}
