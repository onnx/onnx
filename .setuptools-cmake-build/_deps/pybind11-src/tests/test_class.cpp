/*
    tests/test_class.cpp -- test py::class_ definitions and basic functionality

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#if defined(__INTEL_COMPILER) && __cplusplus >= 201703L
// Intel compiler requires a separate header file to support aligned new operators
// and does not set the __cpp_aligned_new feature macro.
// This header needs to be included before pybind11.
#    include <aligned_new>
#endif

#include <pybind11/stl.h>

#include "constructor_stats.h"
#include "local_bindings.h"
#include "pybind11_tests.h"

#include <utility>

PYBIND11_WARNING_DISABLE_MSVC(4324)
//     warning C4324: structure was padded due to alignment specifier

// test_brace_initialization
struct NoBraceInitialization {
    explicit NoBraceInitialization(std::vector<int> v) : vec{std::move(v)} {}
    template <typename T>
    NoBraceInitialization(std::initializer_list<T> l) : vec(l) {}

    std::vector<int> vec;
};

namespace test_class {
namespace pr4220_tripped_over_this { // PR #4227

template <int>
struct SoEmpty {};

template <typename T>
std::string get_msg(const T &) {
    return "This is really only meant to exercise successful compilation.";
}

using Empty0 = SoEmpty<0x0>;

void bind_empty0(py::module_ &m) {
    py::class_<Empty0>(m, "Empty0").def(py::init<>()).def("get_msg", get_msg<Empty0>);
}

} // namespace pr4220_tripped_over_this
} // namespace test_class

TEST_SUBMODULE(class_, m) {
    m.def("obj_class_name", [](py::handle obj) { return py::detail::obj_class_name(obj.ptr()); });

    // test_instance
    struct NoConstructor {
        NoConstructor() = default;
        NoConstructor(const NoConstructor &) = default;
        NoConstructor(NoConstructor &&) = default;
        static NoConstructor *new_instance() {
            auto *ptr = new NoConstructor();
            print_created(ptr, "via new_instance");
            return ptr;
        }
        ~NoConstructor() { print_destroyed(this); }
    };
    struct NoConstructorNew {
        NoConstructorNew() = default;
        NoConstructorNew(const NoConstructorNew &) = default;
        NoConstructorNew(NoConstructorNew &&) = default;
        static NoConstructorNew *new_instance() {
            auto *ptr = new NoConstructorNew();
            print_created(ptr, "via new_instance");
            return ptr;
        }
        ~NoConstructorNew() { print_destroyed(this); }
    };

    py::class_<NoConstructor>(m, "NoConstructor")
        .def_static("new_instance", &NoConstructor::new_instance, "Return an instance");

    py::class_<NoConstructorNew>(m, "NoConstructorNew")
        .def(py::init([]() { return nullptr; })) // Need a NOOP __init__
        .def_static("__new__",
                    [](const py::object &) { return NoConstructorNew::new_instance(); });

    // test_inheritance
    class Pet {
    public:
        Pet(const std::string &name, const std::string &species)
            : m_name(name), m_species(species) {}
        std::string name() const { return m_name; }
        std::string species() const { return m_species; }

    private:
        std::string m_name;
        std::string m_species;
    };

    class Dog : public Pet {
    public:
        explicit Dog(const std::string &name) : Pet(name, "dog") {}
        std::string bark() const { return "Woof!"; }
    };

    class Rabbit : public Pet {
    public:
        explicit Rabbit(const std::string &name) : Pet(name, "parrot") {}
    };

    class Hamster : public Pet {
    public:
        explicit Hamster(const std::string &name) : Pet(name, "rodent") {}
    };

    class Chimera : public Pet {
        Chimera() : Pet("Kimmy", "chimera") {}
    };

    py::class_<Pet> pet_class(m, "Pet");
    pet_class.def(py::init<std::string, std::string>())
        .def("name", &Pet::name)
        .def("species", &Pet::species);

    /* One way of declaring a subclass relationship: reference parent's class_ object */
    py::class_<Dog>(m, "Dog", pet_class).def(py::init<std::string>());

    /* Another way of declaring a subclass relationship: reference parent's C++ type */
    py::class_<Rabbit, Pet>(m, "Rabbit").def(py::init<std::string>());

    /* And another: list parent in class template arguments */
    py::class_<Hamster, Pet>(m, "Hamster").def(py::init<std::string>());

    /* Constructors are not inherited by default */
    py::class_<Chimera, Pet>(m, "Chimera");

    m.def("pet_name_species",
          [](const Pet &pet) { return pet.name() + " is a " + pet.species(); });
    m.def("dog_bark", [](const Dog &dog) { return dog.bark(); });

    // test_automatic_upcasting
    struct BaseClass {
        BaseClass() = default;
        BaseClass(const BaseClass &) = default;
        BaseClass(BaseClass &&) = default;
        virtual ~BaseClass() = default;
    };
    struct DerivedClass1 : BaseClass {};
    struct DerivedClass2 : BaseClass {};

    py::class_<BaseClass>(m, "BaseClass").def(py::init<>());
    py::class_<DerivedClass1>(m, "DerivedClass1").def(py::init<>());
    py::class_<DerivedClass2>(m, "DerivedClass2").def(py::init<>());

    m.def("return_class_1", []() -> BaseClass * { return new DerivedClass1(); });
    m.def("return_class_2", []() -> BaseClass * { return new DerivedClass2(); });
    m.def("return_class_n", [](int n) -> BaseClass * {
        if (n == 1) {
            return new DerivedClass1();
        }
        if (n == 2) {
            return new DerivedClass2();
        }
        return new BaseClass();
    });
    m.def("return_none", []() -> BaseClass * { return nullptr; });

    // test_isinstance
    m.def("check_instances", [](const py::list &l) {
        return py::make_tuple(py::isinstance<py::tuple>(l[0]),
                              py::isinstance<py::dict>(l[1]),
                              py::isinstance<Pet>(l[2]),
                              py::isinstance<Pet>(l[3]),
                              py::isinstance<Dog>(l[4]),
                              py::isinstance<Rabbit>(l[5]),
                              py::isinstance<UnregisteredType>(l[6]));
    });

    struct Invalid {};

    // test_type
    m.def("check_type", [](int category) {
        // Currently not supported (via a fail at compile time)
        // See https://github.com/pybind/pybind11/issues/2486
        // if (category == 2)
        //     return py::type::of<int>();
        if (category == 1) {
            return py::type::of<DerivedClass1>();
        }
        return py::type::of<Invalid>();
    });

    m.def("get_type_of", [](py::object ob) { return py::type::of(std::move(ob)); });

    m.def("get_type_classic", [](py::handle h) { return h.get_type(); });

    m.def("as_type", [](const py::object &ob) { return py::type(ob); });

    // test_mismatched_holder
    struct MismatchBase1 {};
    struct MismatchDerived1 : MismatchBase1 {};

    struct MismatchBase2 {};
    struct MismatchDerived2 : MismatchBase2 {};

    m.def("mismatched_holder_1", []() {
        auto mod = py::module_::import("__main__");
        py::class_<MismatchBase1, std::shared_ptr<MismatchBase1>>(mod, "MismatchBase1");
        py::class_<MismatchDerived1, MismatchBase1>(mod, "MismatchDerived1");
    });
    m.def("mismatched_holder_2", []() {
        auto mod = py::module_::import("__main__");
        py::class_<MismatchBase2>(mod, "MismatchBase2");
        py::class_<MismatchDerived2, std::shared_ptr<MismatchDerived2>, MismatchBase2>(
            mod, "MismatchDerived2");
    });

    // test_override_static
    // #511: problem with inheritance + overwritten def_static
    struct MyBase {
        static std::unique_ptr<MyBase> make() { return std::unique_ptr<MyBase>(new MyBase()); }
    };

    struct MyDerived : MyBase {
        static std::unique_ptr<MyDerived> make() {
            return std::unique_ptr<MyDerived>(new MyDerived());
        }
    };

    py::class_<MyBase>(m, "MyBase").def_static("make", &MyBase::make);

    py::class_<MyDerived, MyBase>(m, "MyDerived")
        .def_static("make", &MyDerived::make)
        .def_static("make2", &MyDerived::make);

    // test_implicit_conversion_life_support
    struct ConvertibleFromUserType {
        int i;

        explicit ConvertibleFromUserType(UserType u) : i(u.value()) {}
    };

    py::class_<ConvertibleFromUserType>(m, "AcceptsUserType").def(py::init<UserType>());
    py::implicitly_convertible<UserType, ConvertibleFromUserType>();

    m.def("implicitly_convert_argument", [](const ConvertibleFromUserType &r) { return r.i; });
    m.def("implicitly_convert_variable", [](const py::object &o) {
        // `o` is `UserType` and `r` is a reference to a temporary created by implicit
        // conversion. This is valid when called inside a bound function because the temp
        // object is attached to the same life support system as the arguments.
        const auto &r = o.cast<const ConvertibleFromUserType &>();
        return r.i;
    });
    m.add_object("implicitly_convert_variable_fail", [&] {
        auto f = [](PyObject *, PyObject *args) -> PyObject * {
            auto o = py::reinterpret_borrow<py::tuple>(args)[0];
            try { // It should fail here because there is no life support.
                o.cast<const ConvertibleFromUserType &>();
            } catch (const py::cast_error &e) {
                return py::str(e.what()).release().ptr();
            }
            return py::str().release().ptr();
        };

        auto *def = new PyMethodDef{"f", f, METH_VARARGS, nullptr};
        py::capsule def_capsule(def,
                                [](void *ptr) { delete reinterpret_cast<PyMethodDef *>(ptr); });
        return py::reinterpret_steal<py::object>(
            PyCFunction_NewEx(def, def_capsule.ptr(), m.ptr()));
    }());

    // test_operator_new_delete
    struct HasOpNewDel {
        std::uint64_t i;
        static void *operator new(size_t s) {
            py::print("A new", s);
            return ::operator new(s);
        }
        static void *operator new(size_t s, void *ptr) {
            py::print("A placement-new", s);
            return ptr;
        }
        static void operator delete(void *p) {
            py::print("A delete");
            return ::operator delete(p);
        }
    };
    struct HasOpNewDelSize {
        std::uint32_t i;
        static void *operator new(size_t s) {
            py::print("B new", s);
            return ::operator new(s);
        }
        static void *operator new(size_t s, void *ptr) {
            py::print("B placement-new", s);
            return ptr;
        }
        static void operator delete(void *p, size_t s) {
            py::print("B delete", s);
            return ::operator delete(p);
        }
    };
    struct AliasedHasOpNewDelSize {
        std::uint64_t i;
        static void *operator new(size_t s) {
            py::print("C new", s);
            return ::operator new(s);
        }
        static void *operator new(size_t s, void *ptr) {
            py::print("C placement-new", s);
            return ptr;
        }
        static void operator delete(void *p, size_t s) {
            py::print("C delete", s);
            return ::operator delete(p);
        }
        virtual ~AliasedHasOpNewDelSize() = default;
        AliasedHasOpNewDelSize() = default;
        AliasedHasOpNewDelSize(const AliasedHasOpNewDelSize &) = delete;
    };
    struct PyAliasedHasOpNewDelSize : AliasedHasOpNewDelSize {
        PyAliasedHasOpNewDelSize() = default;
        explicit PyAliasedHasOpNewDelSize(int) {}
        std::uint64_t j;
    };
    struct HasOpNewDelBoth {
        std::uint32_t i[8];
        static void *operator new(size_t s) {
            py::print("D new", s);
            return ::operator new(s);
        }
        static void *operator new(size_t s, void *ptr) {
            py::print("D placement-new", s);
            return ptr;
        }
        static void operator delete(void *p) {
            py::print("D delete");
            return ::operator delete(p);
        }
        static void operator delete(void *p, size_t s) {
            py::print("D wrong delete", s);
            return ::operator delete(p);
        }
    };
    py::class_<HasOpNewDel>(m, "HasOpNewDel").def(py::init<>());
    py::class_<HasOpNewDelSize>(m, "HasOpNewDelSize").def(py::init<>());
    py::class_<HasOpNewDelBoth>(m, "HasOpNewDelBoth").def(py::init<>());
    py::class_<AliasedHasOpNewDelSize, PyAliasedHasOpNewDelSize> aliased(m,
                                                                         "AliasedHasOpNewDelSize");
    aliased.def(py::init<>());
    aliased.attr("size_noalias") = py::int_(sizeof(AliasedHasOpNewDelSize));
    aliased.attr("size_alias") = py::int_(sizeof(PyAliasedHasOpNewDelSize));

    // This test is actually part of test_local_bindings (test_duplicate_local), but we need a
    // definition in a different compilation unit within the same module:
    bind_local<LocalExternal, 17>(m, "LocalExternal", py::module_local());

    // test_bind_protected_functions
    class ProtectedA {
    protected:
        int foo() const { return value; }

    private:
        int value = 42;
    };

    class PublicistA : public ProtectedA {
    public:
        using ProtectedA::foo;
    };

    py::class_<ProtectedA>(m, "ProtectedA").def(py::init<>()).def("foo", &PublicistA::foo);

    class ProtectedB {
    public:
        virtual ~ProtectedB() = default;
        ProtectedB() = default;
        ProtectedB(const ProtectedB &) = delete;

    protected:
        virtual int foo() const { return value; }
        virtual void *void_foo() { return static_cast<void *>(&value); }
        virtual void *get_self() { return static_cast<void *>(this); }

    private:
        int value = 42;
    };

    class TrampolineB : public ProtectedB {
    public:
        int foo() const override { PYBIND11_OVERRIDE(int, ProtectedB, foo, ); }
        void *void_foo() override { PYBIND11_OVERRIDE(void *, ProtectedB, void_foo, ); }
        void *get_self() override { PYBIND11_OVERRIDE(void *, ProtectedB, get_self, ); }
    };

    class PublicistB : public ProtectedB {
    public:
        // [workaround(intel)] = default does not work here
        // Removing or defaulting this destructor results in linking errors with the Intel compiler
        // (in Debug builds only, tested with icpc (ICC) 2021.1 Beta 20200827)
        ~PublicistB() override {}; // NOLINT(modernize-use-equals-default)
        using ProtectedB::foo;
        using ProtectedB::get_self;
        using ProtectedB::void_foo;
    };

    m.def("read_foo", [](const void *original) {
        const int *ptr = reinterpret_cast<const int *>(original);
        return *ptr;
    });

    m.def("pointers_equal",
          [](const void *original, const void *comparison) { return original == comparison; });

    py::class_<ProtectedB, TrampolineB>(m, "ProtectedB")
        .def(py::init<>())
        .def("foo", &PublicistB::foo)
        .def("void_foo", &PublicistB::void_foo)
        .def("get_self", &PublicistB::get_self);

    // test_brace_initialization
    struct BraceInitialization {
        int field1;
        std::string field2;
    };

    py::class_<BraceInitialization>(m, "BraceInitialization")
        .def(py::init<int, const std::string &>())
        .def_readwrite("field1", &BraceInitialization::field1)
        .def_readwrite("field2", &BraceInitialization::field2);
    // We *don't* want to construct using braces when the given constructor argument maps to a
    // constructor, because brace initialization could go to the wrong place (in particular when
    // there is also an `initializer_list<T>`-accept constructor):
    py::class_<NoBraceInitialization>(m, "NoBraceInitialization")
        .def(py::init<std::vector<int>>())
        .def_readonly("vec", &NoBraceInitialization::vec);

    // test_reentrant_implicit_conversion_failure
    // #1035: issue with runaway reentrant implicit conversion
    struct BogusImplicitConversion {
        BogusImplicitConversion(const BogusImplicitConversion &) = default;
    };

    py::class_<BogusImplicitConversion>(m, "BogusImplicitConversion")
        .def(py::init<const BogusImplicitConversion &>());

    py::implicitly_convertible<int, BogusImplicitConversion>();

    // test_qualname
    // #1166: nested class docstring doesn't show nested name
    // Also related: tests that __qualname__ is set properly
    struct NestBase {};
    struct Nested {};
    py::class_<NestBase> base(m, "NestBase");
    base.def(py::init<>());
    py::class_<Nested>(base, "Nested")
        .def(py::init<>())
        .def("fn", [](Nested &, int, NestBase &, Nested &) {})
        .def("fa", [](Nested &, int, NestBase &, Nested &) {}, "a"_a, "b"_a, "c"_a);
    base.def("g", [](NestBase &, Nested &) {});
    base.def("h", []() { return NestBase(); });

    // test_error_after_conversion
    // The second-pass path through dispatcher() previously didn't
    // remember which overload was used, and would crash trying to
    // generate a useful error message

    struct NotRegistered {};
    struct StringWrapper {
        std::string str;
    };
    m.def("test_error_after_conversions", [](int) {});
    m.def("test_error_after_conversions",
          [](const StringWrapper &) -> NotRegistered { return {}; });
    py::class_<StringWrapper>(m, "StringWrapper").def(py::init<std::string>());
    py::implicitly_convertible<std::string, StringWrapper>();

#if defined(PYBIND11_CPP17)
    struct alignas(1024) Aligned {
        std::uintptr_t ptr() const { return (uintptr_t) this; }
    };
    py::class_<Aligned>(m, "Aligned").def(py::init<>()).def("ptr", &Aligned::ptr);
#endif

    // test_final
    struct IsFinal final {};
    py::class_<IsFinal>(m, "IsFinal", py::is_final());

    // test_non_final_final
    struct IsNonFinalFinal {};
    py::class_<IsNonFinalFinal>(m, "IsNonFinalFinal", py::is_final());

    // test_exception_rvalue_abort
    struct PyPrintDestructor {
        PyPrintDestructor() = default;
        ~PyPrintDestructor() { py::print("Print from destructor"); }
        void throw_something() { throw std::runtime_error("error"); }
    };
    py::class_<PyPrintDestructor>(m, "PyPrintDestructor")
        .def(py::init<>())
        .def("throw_something", &PyPrintDestructor::throw_something);

    // test_multiple_instances_with_same_pointer
    struct SamePointer {};
    static SamePointer samePointer;
    py::class_<SamePointer, std::unique_ptr<SamePointer, py::nodelete>>(m, "SamePointer")
        .def(py::init([]() { return &samePointer; }));

    struct Empty {};
    py::class_<Empty>(m, "Empty").def(py::init<>());

    // test_base_and_derived_nested_scope
    struct BaseWithNested {
        struct Nested {};
    };

    struct DerivedWithNested : BaseWithNested {
        struct Nested {};
    };

    py::class_<BaseWithNested> baseWithNested_class(m, "BaseWithNested");
    py::class_<DerivedWithNested, BaseWithNested> derivedWithNested_class(m, "DerivedWithNested");
    py::class_<BaseWithNested::Nested>(baseWithNested_class, "Nested")
        .def_static("get_name", []() { return "BaseWithNested::Nested"; });
    py::class_<DerivedWithNested::Nested>(derivedWithNested_class, "Nested")
        .def_static("get_name", []() { return "DerivedWithNested::Nested"; });

    // test_register_duplicate_class
    struct Duplicate {};
    struct OtherDuplicate {};
    struct DuplicateNested {};
    struct OtherDuplicateNested {};

    m.def("register_duplicate_class_name", [](const py::module_ &m) {
        py::class_<Duplicate>(m, "Duplicate");
        py::class_<OtherDuplicate>(m, "Duplicate");
    });
    m.def("register_duplicate_class_type", [](const py::module_ &m) {
        py::class_<OtherDuplicate>(m, "OtherDuplicate");
        py::class_<OtherDuplicate>(m, "YetAnotherDuplicate");
    });
    m.def("register_duplicate_nested_class_name", [](const py::object &gt) {
        py::class_<DuplicateNested>(gt, "DuplicateNested");
        py::class_<OtherDuplicateNested>(gt, "DuplicateNested");
    });
    m.def("register_duplicate_nested_class_type", [](const py::object &gt) {
        py::class_<OtherDuplicateNested>(gt, "OtherDuplicateNested");
        py::class_<OtherDuplicateNested>(gt, "YetAnotherDuplicateNested");
    });

    test_class::pr4220_tripped_over_this::bind_empty0(m);
}

template <int N>
class BreaksBase {
public:
    virtual ~BreaksBase() = default;
    BreaksBase() = default;
    BreaksBase(const BreaksBase &) = delete;
};
template <int N>
class BreaksTramp : public BreaksBase<N> {};
// These should all compile just fine:
using DoesntBreak1 = py::class_<BreaksBase<1>, std::unique_ptr<BreaksBase<1>>, BreaksTramp<1>>;
using DoesntBreak2 = py::class_<BreaksBase<2>, BreaksTramp<2>, std::unique_ptr<BreaksBase<2>>>;
using DoesntBreak3 = py::class_<BreaksBase<3>, std::unique_ptr<BreaksBase<3>>>;
using DoesntBreak4 = py::class_<BreaksBase<4>, BreaksTramp<4>>;
using DoesntBreak5 = py::class_<BreaksBase<5>>;
using DoesntBreak6 = py::class_<BreaksBase<6>, std::shared_ptr<BreaksBase<6>>, BreaksTramp<6>>;
using DoesntBreak7 = py::class_<BreaksBase<7>, BreaksTramp<7>, std::shared_ptr<BreaksBase<7>>>;
using DoesntBreak8 = py::class_<BreaksBase<8>, std::shared_ptr<BreaksBase<8>>>;
#define CHECK_BASE(N)                                                                             \
    static_assert(std::is_same<typename DoesntBreak##N::type, BreaksBase<(N)>>::value,            \
                  "DoesntBreak" #N " has wrong type!")
CHECK_BASE(1);
CHECK_BASE(2);
CHECK_BASE(3);
CHECK_BASE(4);
CHECK_BASE(5);
CHECK_BASE(6);
CHECK_BASE(7);
CHECK_BASE(8);
#define CHECK_ALIAS(N)                                                                            \
    static_assert(                                                                                \
        DoesntBreak##N::has_alias                                                                 \
            && std::is_same<typename DoesntBreak##N::type_alias, BreaksTramp<(N)>>::value,        \
        "DoesntBreak" #N " has wrong type_alias!")
#define CHECK_NOALIAS(N)                                                                          \
    static_assert(!DoesntBreak##N::has_alias                                                      \
                      && std::is_void<typename DoesntBreak##N::type_alias>::value,                \
                  "DoesntBreak" #N " has type alias, but shouldn't!")
CHECK_ALIAS(1);
CHECK_ALIAS(2);
CHECK_NOALIAS(3);
CHECK_ALIAS(4);
CHECK_NOALIAS(5);
CHECK_ALIAS(6);
CHECK_ALIAS(7);
CHECK_NOALIAS(8);
#define CHECK_HOLDER(N, TYPE)                                                                     \
    static_assert(std::is_same<typename DoesntBreak##N::holder_type,                              \
                               std::TYPE##_ptr<BreaksBase<(N)>>>::value,                          \
                  "DoesntBreak" #N " has wrong holder_type!")
CHECK_HOLDER(1, unique);
CHECK_HOLDER(2, unique);
CHECK_HOLDER(3, unique);
CHECK_HOLDER(4, unique);
CHECK_HOLDER(5, unique);
CHECK_HOLDER(6, shared);
CHECK_HOLDER(7, shared);
CHECK_HOLDER(8, shared);

// There's no nice way to test that these fail because they fail to compile; leave them here,
// though, so that they can be manually tested by uncommenting them (and seeing that compilation
// failures occurs).

// We have to actually look into the type: the typedef alone isn't enough to instantiate the type:
#define CHECK_BROKEN(N)                                                                           \
    static_assert(std::is_same<typename Breaks##N::type, BreaksBase<-(N)>>::value,                \
                  "Breaks1 has wrong type!");

#ifdef PYBIND11_NEVER_DEFINED_EVER
// Two holder classes:
typedef py::
    class_<BreaksBase<-1>, std::unique_ptr<BreaksBase<-1>>, std::unique_ptr<BreaksBase<-1>>>
        Breaks1;
CHECK_BROKEN(1);
// Two aliases:
typedef py::class_<BreaksBase<-2>, BreaksTramp<-2>, BreaksTramp<-2>> Breaks2;
CHECK_BROKEN(2);
// Holder + 2 aliases
typedef py::
    class_<BreaksBase<-3>, std::unique_ptr<BreaksBase<-3>>, BreaksTramp<-3>, BreaksTramp<-3>>
        Breaks3;
CHECK_BROKEN(3);
// Alias + 2 holders
typedef py::class_<BreaksBase<-4>,
                   std::unique_ptr<BreaksBase<-4>>,
                   BreaksTramp<-4>,
                   std::shared_ptr<BreaksBase<-4>>>
    Breaks4;
CHECK_BROKEN(4);
// Invalid option (not a subclass or holder)
typedef py::class_<BreaksBase<-5>, BreaksTramp<-4>> Breaks5;
CHECK_BROKEN(5);
// Invalid option: multiple inheritance not supported:
template <>
struct BreaksBase<-8> : BreaksBase<-6>, BreaksBase<-7> {};
typedef py::class_<BreaksBase<-8>, BreaksBase<-6>, BreaksBase<-7>> Breaks8;
CHECK_BROKEN(8);
#endif
