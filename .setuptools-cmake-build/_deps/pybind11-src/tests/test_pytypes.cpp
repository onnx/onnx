/*
    tests/test_pytypes.cpp -- Python type casters

    Copyright (c) 2017 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/typing.h>

#include "pybind11_tests.h"

#include <utility>

//__has_include has been part of C++17, no need to check it
#if defined(PYBIND11_CPP20) && __has_include(<ranges>)
#    if !defined(PYBIND11_COMPILER_CLANG) || __clang_major__ >= 16 // llvm/llvm-project#52696
#        define PYBIND11_TEST_PYTYPES_HAS_RANGES
#        include <ranges>
#    endif
#endif

namespace external {
namespace detail {
bool check(PyObject *o) { return PyFloat_Check(o) != 0; }

PyObject *conv(PyObject *o) {
    PyObject *ret = nullptr;
    if (PyLong_Check(o)) {
        double v = PyLong_AsDouble(o);
        if (!(v == -1.0 && PyErr_Occurred())) {
            ret = PyFloat_FromDouble(v);
        }
    } else {
        py::set_error(PyExc_TypeError, "Unexpected type");
    }
    return ret;
}

PyObject *default_constructed() { return PyFloat_FromDouble(0.0); }
} // namespace detail
class float_ : public py::object {
    PYBIND11_OBJECT_CVT(float_, py::object, external::detail::check, external::detail::conv)

    float_() : py::object(external::detail::default_constructed(), stolen_t{}) {}

    double get_value() const { return PyFloat_AsDouble(this->ptr()); }
};
} // namespace external

namespace pybind11 {
namespace detail {
template <>
struct handle_type_name<external::float_> {
    static constexpr auto name = const_name("float");
};
} // namespace detail
} // namespace pybind11

namespace implicit_conversion_from_0_to_handle {
// Uncomment to trigger compiler error. Note: Before PR #4008 this used to compile successfully.
// void expected_to_trigger_compiler_error() { py::handle(0); }
} // namespace implicit_conversion_from_0_to_handle

// Used to validate systematically that PR #4008 does/did NOT change the behavior.
void pure_compile_tests_for_handle_from_PyObject_pointers() {
    {
        PyObject *ptr = Py_None;
        py::handle{ptr};
    }
    {
        PyObject *const ptr = Py_None;
        py::handle{ptr};
    }
    // Uncomment to trigger compiler errors.
    // PyObject const *               ptr = Py_None; py::handle{ptr};
    // PyObject const *const          ptr = Py_None; py::handle{ptr};
    // PyObject volatile *            ptr = Py_None; py::handle{ptr};
    // PyObject volatile *const       ptr = Py_None; py::handle{ptr};
    // PyObject const volatile *      ptr = Py_None; py::handle{ptr};
    // PyObject const volatile *const ptr = Py_None; py::handle{ptr};
}

namespace handle_from_move_only_type_with_operator_PyObject {

// Reduced from
// https://github.com/pytorch/pytorch/blob/279634f384662b7c3a9f8bf7ccc3a6afd2f05657/torch/csrc/utils/object_ptr.h
struct operator_ncnst {
    operator_ncnst() = default;
    operator_ncnst(operator_ncnst &&) = default;
    operator PyObject *() /* */ { return Py_None; } // NOLINT(google-explicit-constructor)
};

struct operator_const {
    operator_const() = default;
    operator_const(operator_const &&) = default;
    operator PyObject *() const { return Py_None; } // NOLINT(google-explicit-constructor)
};

bool from_ncnst() {
    operator_ncnst obj;
    auto h = py::handle(obj);  // Critical part of test: does this compile?
    return h.ptr() == Py_None; // Just something.
}

bool from_const() {
    operator_const obj;
    auto h = py::handle(obj);  // Critical part of test: does this compile?
    return h.ptr() == Py_None; // Just something.
}

void m_defs(py::module_ &m) {
    m.def("handle_from_move_only_type_with_operator_PyObject_ncnst", from_ncnst);
    m.def("handle_from_move_only_type_with_operator_PyObject_const", from_const);
}

} // namespace handle_from_move_only_type_with_operator_PyObject

#if defined(PYBIND11_TYPING_H_HAS_STRING_LITERAL)
namespace literals {
enum Color { RED = 0, BLUE = 1 };

typedef py::typing::Literal<"26",
                            "0x1A",
                            "\"hello world\"",
                            "b\"hello world\"",
                            "u\"hello world\"",
                            "True",
                            "Color.RED",
                            "None">
    LiteralFoo;
} // namespace literals
namespace typevar {
typedef py::typing::TypeVar<"T"> TypeVarT;
typedef py::typing::TypeVar<"V"> TypeVarV;
} // namespace typevar
#endif

TEST_SUBMODULE(pytypes, m) {
    m.def("obj_class_name", [](py::handle obj) { return py::detail::obj_class_name(obj.ptr()); });

    handle_from_move_only_type_with_operator_PyObject::m_defs(m);

    // test_bool
    m.def("get_bool", [] { return py::bool_(false); });
    // test_int
    m.def("get_int", [] { return py::int_(0); });
    // test_iterator
    m.def("get_iterator", [] { return py::iterator(); });
    // test_iterable
    m.def("get_iterable", [] { return py::iterable(); });
    m.def("get_frozenset_from_iterable",
          [](const py::iterable &iter) { return py::frozenset(iter); });
    m.def("get_list_from_iterable", [](const py::iterable &iter) { return py::list(iter); });
    m.def("get_set_from_iterable", [](const py::iterable &iter) { return py::set(iter); });
    m.def("get_tuple_from_iterable", [](const py::iterable &iter) { return py::tuple(iter); });
    // test_float
    m.def("get_float", [] { return py::float_(0.0f); });
    // test_list
    m.def("list_no_args", []() { return py::list{}; });
    m.def("list_ssize_t", []() { return py::list{(py::ssize_t) 0}; });
    m.def("list_size_t", []() { return py::list{(py::size_t) 0}; });
    m.def("list_insert_ssize_t", [](py::list *l) { return l->insert((py::ssize_t) 1, 83); });
    m.def("list_insert_size_t", [](py::list *l) { return l->insert((py::size_t) 3, 57); });
    m.def("list_clear", [](py::list *l) { l->clear(); });
    m.def("get_list", []() {
        py::list list;
        list.append("value");
        py::print("Entry at position 0:", list[0]);
        list[0] = py::str("overwritten");
        list.insert(0, "inserted-0");
        list.insert(2, "inserted-2");
        return list;
    });
    m.def("print_list", [](const py::list &list) {
        int index = 0;
        for (auto item : list) {
            py::print("list item {}: {}"_s.format(index++, item));
        }
    });
    // test_none
    m.def("get_none", [] { return py::none(); });
    m.def("print_none", [](const py::none &none) { py::print("none: {}"_s.format(none)); });

    // test_set, test_frozenset
    m.def("get_set", []() {
        py::set set;
        set.add(py::str("key1"));
        set.add("key2");
        set.add(std::string("key3"));
        return set;
    });
    m.def("get_frozenset", []() {
        py::set set;
        set.add(py::str("key1"));
        set.add("key2");
        set.add(std::string("key3"));
        return py::frozenset(set);
    });
    m.def("print_anyset", [](const py::anyset &set) {
        for (auto item : set) {
            py::print("key:", item);
        }
    });
    m.def("anyset_size", [](const py::anyset &set) { return set.size(); });
    m.def("anyset_empty", [](const py::anyset &set) { return set.empty(); });
    m.def("anyset_contains",
          [](const py::anyset &set, const py::object &key) { return set.contains(key); });
    m.def("anyset_contains",
          [](const py::anyset &set, const char *key) { return set.contains(key); });
    m.def("set_add", [](py::set &set, const py::object &key) { set.add(key); });
    m.def("set_clear", [](py::set &set) { set.clear(); });

    // test_dict
    m.def("get_dict", []() { return py::dict("key"_a = "value"); });
    m.def("print_dict", [](const py::dict &dict) {
        for (auto item : dict) {
            py::print("key: {}, value={}"_s.format(item.first, item.second));
        }
    });
    m.def("dict_keyword_constructor", []() {
        auto d1 = py::dict("x"_a = 1, "y"_a = 2);
        auto d2 = py::dict("z"_a = 3, **d1);
        return d2;
    });
    m.def("dict_contains",
          [](const py::dict &dict, const py::object &val) { return dict.contains(val); });
    m.def("dict_contains",
          [](const py::dict &dict, const char *val) { return dict.contains(val); });

    // test_tuple
    m.def("tuple_no_args", []() { return py::tuple{}; });
    m.def("tuple_ssize_t", []() { return py::tuple{(py::ssize_t) 0}; });
    m.def("tuple_size_t", []() { return py::tuple{(py::size_t) 0}; });
    m.def("get_tuple", []() { return py::make_tuple(42, py::none(), "spam"); });

    // test_simple_namespace
    m.def("get_simple_namespace", []() {
        auto ns = py::module_::import("types").attr("SimpleNamespace")(
            "attr"_a = 42, "x"_a = "foo", "wrong"_a = 1);
        py::delattr(ns, "wrong");
        py::setattr(ns, "right", py::int_(2));
        return ns;
    });

    // test_str
    m.def("str_from_char_ssize_t", []() { return py::str{"red", (py::ssize_t) 3}; });
    m.def("str_from_char_size_t", []() { return py::str{"blue", (py::size_t) 4}; });
    m.def("str_from_string", []() { return py::str(std::string("baz")); });
    m.def("str_from_std_string_input", [](const std::string &stri) { return py::str(stri); });
    m.def("str_from_cstr_input", [](const char *c_str) { return py::str(c_str); });
    m.def("str_from_bytes", []() { return py::str(py::bytes("boo", 3)); });
    m.def("str_from_bytes_input",
          [](const py::bytes &encoded_str) { return py::str(encoded_str); });

    m.def("str_from_object", [](const py::object &obj) { return py::str(obj); });
    m.def("repr_from_object", [](const py::object &obj) { return py::repr(obj); });
    m.def("str_from_handle", [](py::handle h) { return py::str(h); });
    m.def("str_from_string_from_str",
          [](const py::str &obj) { return py::str(static_cast<std::string>(obj)); });

    m.def("str_format", []() {
        auto s1 = "{} + {} = {}"_s.format(1, 2, 3);
        auto s2 = "{a} + {b} = {c}"_s.format("a"_a = 1, "b"_a = 2, "c"_a = 3);
        return py::make_tuple(s1, s2);
    });

    // test_bytes
    m.def("bytes_from_char_ssize_t", []() { return py::bytes{"green", (py::ssize_t) 5}; });
    m.def("bytes_from_char_size_t", []() { return py::bytes{"purple", (py::size_t) 6}; });
    m.def("bytes_from_string", []() { return py::bytes(std::string("foo")); });
    m.def("bytes_from_str", []() { return py::bytes(py::str("bar", 3)); });

    // test bytearray
    m.def("bytearray_from_char_ssize_t", []() { return py::bytearray{"$%", (py::ssize_t) 2}; });
    m.def("bytearray_from_char_size_t", []() { return py::bytearray{"@$!", (py::size_t) 3}; });
    m.def("bytearray_from_string", []() { return py::bytearray(std::string("foo")); });
    m.def("bytearray_size", []() { return py::bytearray("foo").size(); });

    // test_capsule
    m.def("return_capsule_with_destructor", []() {
        py::print("creating capsule");
        return py::capsule([]() { py::print("destructing capsule"); });
    });

    m.def("return_renamed_capsule_with_destructor", []() {
        py::print("creating capsule");
        auto cap = py::capsule([]() { py::print("destructing capsule"); });
        static const char *capsule_name = "test_name1";
        py::print("renaming capsule");
        cap.set_name(capsule_name);
        return cap;
    });

    m.def("return_capsule_with_destructor_2", []() {
        py::print("creating capsule");
        return py::capsule((void *) 1234, [](void *ptr) {
            py::print("destructing capsule: {}"_s.format((size_t) ptr));
        });
    });

    m.def("return_capsule_with_destructor_3", []() {
        py::print("creating capsule");
        auto cap = py::capsule((void *) 1233, "oname", [](void *ptr) {
            py::print("destructing capsule: {}"_s.format((size_t) ptr));
        });
        py::print("original name: {}"_s.format(cap.name()));
        return cap;
    });

    m.def("return_renamed_capsule_with_destructor_2", []() {
        py::print("creating capsule");
        auto cap = py::capsule((void *) 1234, [](void *ptr) {
            py::print("destructing capsule: {}"_s.format((size_t) ptr));
        });
        static const char *capsule_name = "test_name2";
        py::print("renaming capsule");
        cap.set_name(capsule_name);
        return cap;
    });

    m.def("return_capsule_with_name_and_destructor", []() {
        auto capsule = py::capsule((void *) 12345, "pointer type description", [](PyObject *ptr) {
            if (ptr) {
                const auto *name = PyCapsule_GetName(ptr);
                py::print("destructing capsule ({}, '{}')"_s.format(
                    (size_t) PyCapsule_GetPointer(ptr, name), name));
            }
        });

        capsule.set_pointer((void *) 1234);

        // Using get_pointer<T>()
        void *contents1 = static_cast<void *>(capsule);
        void *contents2 = capsule.get_pointer();
        void *contents3 = capsule.get_pointer<void>();

        auto result1 = reinterpret_cast<size_t>(contents1);
        auto result2 = reinterpret_cast<size_t>(contents2);
        auto result3 = reinterpret_cast<size_t>(contents3);

        py::print(
            "created capsule ({}, '{}')"_s.format(result1 & result2 & result3, capsule.name()));
        return capsule;
    });

    m.def("return_capsule_with_explicit_nullptr_dtor", []() {
        py::print("creating capsule with explicit nullptr dtor");
        return py::capsule(reinterpret_cast<void *>(1234),
                           static_cast<void (*)(void *)>(nullptr)); // PR #4221
    });

    // test_accessors
    m.def("accessor_api", [](const py::object &o) {
        auto d = py::dict();

        d["basic_attr"] = o.attr("basic_attr");

        auto l = py::list();
        for (auto item : o.attr("begin_end")) {
            l.append(item);
        }
        d["begin_end"] = l;

        d["operator[object]"] = o.attr("d")["operator[object]"_s];
        d["operator[char *]"] = o.attr("d")["operator[char *]"];

        d["attr(object)"] = o.attr("sub").attr("attr_obj");
        d["attr(char *)"] = o.attr("sub").attr("attr_char");
        try {
            o.attr("sub").attr("missing").ptr();
        } catch (const py::error_already_set &) {
            d["missing_attr_ptr"] = "raised"_s;
        }
        try {
            o.attr("missing").attr("doesn't matter");
        } catch (const py::error_already_set &) {
            d["missing_attr_chain"] = "raised"_s;
        }

        d["is_none"] = o.attr("basic_attr").is_none();

        d["operator()"] = o.attr("func")(1);
        d["operator*"] = o.attr("func")(*o.attr("begin_end"));

        // Test implicit conversion
        py::list implicit_list = o.attr("begin_end");
        d["implicit_list"] = implicit_list;
        py::dict implicit_dict = o.attr("__dict__");
        d["implicit_dict"] = implicit_dict;

        return d;
    });

    m.def("tuple_accessor", [](const py::tuple &existing_t) {
        try {
            existing_t[0] = 1;
        } catch (const py::error_already_set &) {
            // --> Python system error
            // Only new tuples (refcount == 1) are mutable
            auto new_t = py::tuple(3);
            for (size_t i = 0; i < new_t.size(); ++i) {
                new_t[i] = i;
            }
            return new_t;
        }
        return py::tuple();
    });

    m.def("accessor_assignment", []() {
        auto l = py::list(1);
        l[0] = 0;

        auto d = py::dict();
        d["get"] = l[0];
        auto var = l[0];
        d["deferred_get"] = var;
        l[0] = 1;
        d["set"] = l[0];
        var = 99; // this assignment should not overwrite l[0]
        d["deferred_set"] = l[0];
        d["var"] = var;

        return d;
    });

    m.def("accessor_moves", []() { // See PR #3970
        py::list return_list;
#ifdef PYBIND11_HANDLE_REF_DEBUG
        py::int_ py_int_0(0);
        py::int_ py_int_42(42);
        py::str py_str_count("count");

        auto tup = py::make_tuple(0);

        py::sequence seq(tup);

        py::list lst;
        lst.append(0);

#    define PYBIND11_LOCAL_DEF(...)                                                               \
        {                                                                                         \
            std::size_t inc_refs = py::handle::inc_ref_counter();                                 \
            __VA_ARGS__;                                                                          \
            inc_refs = py::handle::inc_ref_counter() - inc_refs;                                  \
            return_list.append(inc_refs);                                                         \
        }

        PYBIND11_LOCAL_DEF(tup[py_int_0])    // l-value (to have a control)
        PYBIND11_LOCAL_DEF(tup[py::int_(0)]) // r-value

        PYBIND11_LOCAL_DEF(tup.attr(py_str_count))     // l-value
        PYBIND11_LOCAL_DEF(tup.attr(py::str("count"))) // r-value

        PYBIND11_LOCAL_DEF(seq[py_int_0])    // l-value
        PYBIND11_LOCAL_DEF(seq[py::int_(0)]) // r-value

        PYBIND11_LOCAL_DEF(seq.attr(py_str_count))     // l-value
        PYBIND11_LOCAL_DEF(seq.attr(py::str("count"))) // r-value

        PYBIND11_LOCAL_DEF(lst[py_int_0])    // l-value
        PYBIND11_LOCAL_DEF(lst[py::int_(0)]) // r-value

        PYBIND11_LOCAL_DEF(lst.attr(py_str_count))     // l-value
        PYBIND11_LOCAL_DEF(lst.attr(py::str("count"))) // r-value

        auto lst_acc = lst[py::int_(0)];
        lst_acc = py::int_(42);                    // Detaches lst_acc from lst.
        PYBIND11_LOCAL_DEF(lst_acc = py_int_42)    // l-value
        PYBIND11_LOCAL_DEF(lst_acc = py::int_(42)) // r-value
#    undef PYBIND11_LOCAL_DEF
#endif
        return return_list;
    });

    // test_constructors
    m.def("default_constructors", []() {
        return py::dict("bytes"_a = py::bytes(),
                        "bytearray"_a = py::bytearray(),
                        "str"_a = py::str(),
                        "bool"_a = py::bool_(),
                        "int"_a = py::int_(),
                        "float"_a = py::float_(),
                        "tuple"_a = py::tuple(),
                        "list"_a = py::list(),
                        "dict"_a = py::dict(),
                        "set"_a = py::set());
    });

    m.def("converting_constructors", [](const py::dict &d) {
        return py::dict("bytes"_a = py::bytes(d["bytes"]),
                        "bytearray"_a = py::bytearray(d["bytearray"]),
                        "str"_a = py::str(d["str"]),
                        "bool"_a = py::bool_(d["bool"]),
                        "int"_a = py::int_(d["int"]),
                        "float"_a = py::float_(d["float"]),
                        "tuple"_a = py::tuple(d["tuple"]),
                        "list"_a = py::list(d["list"]),
                        "dict"_a = py::dict(d["dict"]),
                        "set"_a = py::set(d["set"]),
                        "frozenset"_a = py::frozenset(d["frozenset"]),
                        "memoryview"_a = py::memoryview(d["memoryview"]));
    });

    m.def("cast_functions", [](const py::dict &d) {
        // When converting between Python types, obj.cast<T>() should be the same as T(obj)
        return py::dict("bytes"_a = d["bytes"].cast<py::bytes>(),
                        "bytearray"_a = d["bytearray"].cast<py::bytearray>(),
                        "str"_a = d["str"].cast<py::str>(),
                        "bool"_a = d["bool"].cast<py::bool_>(),
                        "int"_a = d["int"].cast<py::int_>(),
                        "float"_a = d["float"].cast<py::float_>(),
                        "tuple"_a = d["tuple"].cast<py::tuple>(),
                        "list"_a = d["list"].cast<py::list>(),
                        "dict"_a = d["dict"].cast<py::dict>(),
                        "set"_a = d["set"].cast<py::set>(),
                        "frozenset"_a = d["frozenset"].cast<py::frozenset>(),
                        "memoryview"_a = d["memoryview"].cast<py::memoryview>());
    });

    m.def("convert_to_pybind11_str", [](const py::object &o) { return py::str(o); });

    m.def("nonconverting_constructor",
          [](const std::string &type, py::object value, bool move) -> py::object {
              if (type == "bytes") {
                  return move ? py::bytes(std::move(value)) : py::bytes(value);
              }
              if (type == "none") {
                  return move ? py::none(std::move(value)) : py::none(value);
              }
              if (type == "ellipsis") {
                  return move ? py::ellipsis(std::move(value)) : py::ellipsis(value);
              }
              if (type == "type") {
                  return move ? py::type(std::move(value)) : py::type(value);
              }
              throw std::runtime_error("Invalid type");
          });

    m.def("get_implicit_casting", []() {
        py::dict d;
        d["char*_i1"] = "abc";
        const char *c2 = "abc";
        d["char*_i2"] = c2;
        d["char*_e"] = py::cast(c2);
        d["char*_p"] = py::str(c2);

        d["int_i1"] = 42;
        int i = 42;
        d["int_i2"] = i;
        i++;
        d["int_e"] = py::cast(i);
        i++;
        d["int_p"] = py::int_(i);

        d["str_i1"] = std::string("str");
        std::string s2("str1");
        d["str_i2"] = s2;
        s2[3] = '2';
        d["str_e"] = py::cast(s2);
        s2[3] = '3';
        d["str_p"] = py::str(s2);

        py::list l(2);
        l[0] = 3;
        l[1] = py::cast(6);
        l.append(9);
        l.append(py::cast(12));
        l.append(py::int_(15));

        return py::dict("d"_a = d, "l"_a = l);
    });

    // test_print
    m.def("print_function", []() {
        py::print("Hello, World!");
        py::print(1, 2.0, "three", true, std::string("-- multiple args"));
        auto args = py::make_tuple("and", "a", "custom", "separator");
        py::print("*args", *args, "sep"_a = "-");
        py::print("no new line here", "end"_a = " -- ");
        py::print("next print");

        auto py_stderr = py::module_::import("sys").attr("stderr");
        py::print("this goes to stderr", "file"_a = py_stderr);

        py::print("flush", "flush"_a = true);

        py::print(
            "{a} + {b} = {c}"_s.format("a"_a = "py::print", "b"_a = "str.format", "c"_a = "this"));
    });

    m.def("print_failure", []() { py::print(42, UnregisteredType()); });

    m.def("hash_function", [](py::object obj) { return py::hash(std::move(obj)); });

    m.def("obj_contains",
          [](py::object &obj, const py::object &key) { return obj.contains(key); });

    m.def("test_number_protocol", [](const py::object &a, const py::object &b) {
        py::list l;
        l.append(a.equal(b));
        l.append(a.not_equal(b));
        l.append(a < b);
        l.append(a <= b);
        l.append(a > b);
        l.append(a >= b);
        l.append(a + b);
        l.append(a - b);
        l.append(a * b);
        l.append(a / b);
        l.append(a | b);
        l.append(a & b);
        l.append(a ^ b);
        l.append(a >> b);
        l.append(a << b);
        return l;
    });

    m.def("test_list_slicing", [](const py::list &a) { return a[py::slice(0, -1, 2)]; });

    // See #2361
    m.def("issue2361_str_implicit_copy_none", []() {
        py::str is_this_none = py::none();
        return is_this_none;
    });
    m.def("issue2361_dict_implicit_copy_none", []() {
        py::dict is_this_none = py::none();
        return is_this_none;
    });

    m.def("test_memoryview_object", [](const py::buffer &b) { return py::memoryview(b); });

    m.def("test_memoryview_buffer_info",
          [](const py::buffer &b) { return py::memoryview(b.request()); });

    m.def("test_memoryview_from_buffer", [](bool is_unsigned) {
        static const int16_t si16[] = {3, 1, 4, 1, 5};
        static const uint16_t ui16[] = {2, 7, 1, 8};
        if (is_unsigned) {
            return py::memoryview::from_buffer(ui16, {4}, {sizeof(uint16_t)});
        }
        return py::memoryview::from_buffer(si16, {5}, {sizeof(int16_t)});
    });

    m.def("test_memoryview_from_buffer_nativeformat", []() {
        static const char *format = "@i";
        static const int32_t arr[] = {4, 7, 5};
        return py::memoryview::from_buffer(arr, sizeof(int32_t), format, {3}, {sizeof(int32_t)});
    });

    m.def("test_memoryview_from_buffer_empty_shape", []() {
        static const char *buf = "";
        return py::memoryview::from_buffer(buf, 1, "B", {}, {});
    });

    m.def("test_memoryview_from_buffer_invalid_strides", []() {
        static const char *buf = "\x02\x03\x04";
        return py::memoryview::from_buffer(buf, 1, "B", {3}, {});
    });

    m.def("test_memoryview_from_buffer_nullptr", []() {
        return py::memoryview::from_buffer(static_cast<void *>(nullptr), 1, "B", {}, {});
    });

    m.def("test_memoryview_from_memory", []() {
        const char *buf = "\xff\xe1\xab\x37";
        return py::memoryview::from_memory(buf, static_cast<py::ssize_t>(strlen(buf)));
    });

    // test_builtin_functions
    m.def("get_len", [](py::handle h) { return py::len(h); });

#ifdef PYBIND11_STR_LEGACY_PERMISSIVE
    m.attr("PYBIND11_STR_LEGACY_PERMISSIVE") = true;
#endif

    m.def("isinstance_pybind11_bytes",
          [](py::object o) { return py::isinstance<py::bytes>(std::move(o)); });
    m.def("isinstance_pybind11_str",
          [](py::object o) { return py::isinstance<py::str>(std::move(o)); });

    m.def("pass_to_pybind11_bytes", [](py::bytes b) { return py::len(std::move(b)); });
    m.def("pass_to_pybind11_str", [](py::str s) { return py::len(std::move(s)); });
    m.def("pass_to_std_string", [](const std::string &s) { return s.size(); });

    // test_weakref
    m.def("weakref_from_handle", [](py::handle h) { return py::weakref(h); });
    m.def("weakref_from_handle_and_function",
          [](py::handle h, py::function f) { return py::weakref(h, std::move(f)); });
    m.def("weakref_from_object", [](const py::object &o) { return py::weakref(o); });
    m.def("weakref_from_object_and_function",
          [](py::object o, py::function f) { return py::weakref(std::move(o), std::move(f)); });

// See PR #3263 for background (https://github.com/pybind/pybind11/pull/3263):
// pytypes.h could be changed to enforce the "most correct" user code below, by removing
// `const` from iterator `reference` using type aliases, but that will break existing
// user code.
#if (defined(__APPLE__) && defined(__clang__)) || defined(PYPY_VERSION)
// This is "most correct" and enforced on these platforms.
#    define PYBIND11_AUTO_IT auto it
#else
    // This works on many platforms and is (unfortunately) reflective of existing user code.
    // NOLINTNEXTLINE(bugprone-macro-parentheses)
#    define PYBIND11_AUTO_IT auto &it
#endif

    m.def("tuple_iterator", []() {
        auto tup = py::make_tuple(5, 7);
        int tup_sum = 0;
        for (PYBIND11_AUTO_IT : tup) {
            tup_sum += it.cast<int>();
        }
        return tup_sum;
    });

    m.def("dict_iterator", []() {
        py::dict dct;
        dct[py::int_(3)] = 5;
        dct[py::int_(7)] = 11;
        int kv_sum = 0;
        for (PYBIND11_AUTO_IT : dct) {
            kv_sum += it.first.cast<int>() * 100 + it.second.cast<int>();
        }
        return kv_sum;
    });

    m.def("passed_iterator", [](const py::iterator &py_it) {
        int elem_sum = 0;
        for (PYBIND11_AUTO_IT : py_it) {
            elem_sum += it.cast<int>();
        }
        return elem_sum;
    });

#undef PYBIND11_AUTO_IT

    // Tests below this line are for pybind11 IMPLEMENTATION DETAILS:

    m.def("sequence_item_get_ssize_t", [](const py::object &o) {
        return py::detail::accessor_policies::sequence_item::get(o, (py::ssize_t) 1);
    });
    m.def("sequence_item_set_ssize_t", [](const py::object &o) {
        auto s = py::str{"peppa", 5};
        py::detail::accessor_policies::sequence_item::set(o, (py::ssize_t) 1, s);
    });
    m.def("sequence_item_get_size_t", [](const py::object &o) {
        return py::detail::accessor_policies::sequence_item::get(o, (py::size_t) 2);
    });
    m.def("sequence_item_set_size_t", [](const py::object &o) {
        auto s = py::str{"george", 6};
        py::detail::accessor_policies::sequence_item::set(o, (py::size_t) 2, s);
    });
    m.def("list_item_get_ssize_t", [](const py::object &o) {
        return py::detail::accessor_policies::list_item::get(o, (py::ssize_t) 3);
    });
    m.def("list_item_set_ssize_t", [](const py::object &o) {
        auto s = py::str{"rebecca", 7};
        py::detail::accessor_policies::list_item::set(o, (py::ssize_t) 3, s);
    });
    m.def("list_item_get_size_t", [](const py::object &o) {
        return py::detail::accessor_policies::list_item::get(o, (py::size_t) 4);
    });
    m.def("list_item_set_size_t", [](const py::object &o) {
        auto s = py::str{"richard", 7};
        py::detail::accessor_policies::list_item::set(o, (py::size_t) 4, s);
    });
    m.def("tuple_item_get_ssize_t", [](const py::object &o) {
        return py::detail::accessor_policies::tuple_item::get(o, (py::ssize_t) 5);
    });
    m.def("tuple_item_set_ssize_t", []() {
        auto s0 = py::str{"emely", 5};
        auto s1 = py::str{"edmond", 6};
        auto o = py::tuple{2};
        py::detail::accessor_policies::tuple_item::set(o, (py::ssize_t) 0, s0);
        py::detail::accessor_policies::tuple_item::set(o, (py::ssize_t) 1, s1);
        return o;
    });
    m.def("tuple_item_get_size_t", [](const py::object &o) {
        return py::detail::accessor_policies::tuple_item::get(o, (py::size_t) 6);
    });
    m.def("tuple_item_set_size_t", []() {
        auto s0 = py::str{"candy", 5};
        auto s1 = py::str{"cat", 3};
        auto o = py::tuple{2};
        py::detail::accessor_policies::tuple_item::set(o, (py::size_t) 1, s1);
        py::detail::accessor_policies::tuple_item::set(o, (py::size_t) 0, s0);
        return o;
    });

    m.def("square_float_", [](const external::float_ &x) -> double {
        double v = x.get_value();
        return v * v;
    });

    m.def("tuple_rvalue_getter", [](const py::tuple &tup) {
        // tests accessing tuple object with rvalue int
        for (size_t i = 0; i < tup.size(); i++) {
            auto o = py::handle(tup[py::int_(i)]);
            if (!o) {
                throw py::value_error("tuple is malformed");
            }
        }
        return tup;
    });
    m.def("list_rvalue_getter", [](const py::list &l) {
        // tests accessing list with rvalue int
        for (size_t i = 0; i < l.size(); i++) {
            auto o = py::handle(l[py::int_(i)]);
            if (!o) {
                throw py::value_error("list is malformed");
            }
        }
        return l;
    });
    m.def("populate_dict_rvalue", [](int population) {
        auto d = py::dict();
        for (int i = 0; i < population; i++) {
            d[py::int_(i)] = py::int_(i);
        }
        return d;
    });
    m.def("populate_obj_str_attrs", [](py::object &o, int population) {
        for (int i = 0; i < population; i++) {
            o.attr(py::str(py::int_(i))) = py::str(py::int_(i));
        }
        return o;
    });

    // testing immutable object augmented assignment: #issue 3812
    m.def("inplace_append", [](py::object &a, const py::object &b) {
        a += b;
        return a;
    });
    m.def("inplace_subtract", [](py::object &a, const py::object &b) {
        a -= b;
        return a;
    });
    m.def("inplace_multiply", [](py::object &a, const py::object &b) {
        a *= b;
        return a;
    });
    m.def("inplace_divide", [](py::object &a, const py::object &b) {
        a /= b;
        return a;
    });
    m.def("inplace_or", [](py::object &a, const py::object &b) {
        a |= b;
        return a;
    });
    m.def("inplace_and", [](py::object &a, const py::object &b) {
        a &= b;
        return a;
    });
    m.def("inplace_lshift", [](py::object &a, const py::object &b) {
        a <<= b;
        return a;
    });
    m.def("inplace_rshift", [](py::object &a, const py::object &b) {
        a >>= b;
        return a;
    });

    m.def("annotate_tuple_float_str", [](const py::typing::Tuple<py::float_, py::str> &) {});
    m.def("annotate_tuple_empty", [](const py::typing::Tuple<> &) {});
    m.def("annotate_tuple_variable_length",
          [](const py::typing::Tuple<py::float_, py::ellipsis> &) {});
    m.def("annotate_dict_str_int", [](const py::typing::Dict<py::str, int> &) {});
    m.def("annotate_list_int", [](const py::typing::List<int> &) {});
    m.def("annotate_set_str", [](const py::typing::Set<std::string> &) {});
    m.def("annotate_iterable_str", [](const py::typing::Iterable<std::string> &) {});
    m.def("annotate_iterator_int", [](const py::typing::Iterator<int> &) {});
    m.def("annotate_fn",
          [](const py::typing::Callable<int(py::typing::List<py::str>, py::str)> &) {});

    m.def("annotate_fn_only_return", [](const py::typing::Callable<int(py::ellipsis)> &) {});
    m.def("annotate_type", [](const py::typing::Type<int> &t) -> py::type { return t; });

    m.def("annotate_union",
          [](py::typing::List<py::typing::Union<py::str, py::int_, py::object>> l,
             py::str a,
             py::int_ b,
             py::object c) -> py::typing::List<py::typing::Union<py::str, py::int_, py::object>> {
              l.append(a);
              l.append(b);
              l.append(c);
              return l;
          });

    m.def("union_typing_only",
          [](py::typing::List<py::typing::Union<py::str>> &l)
              -> py::typing::List<py::typing::Union<py::int_>> { return l; });

    m.def("annotate_union_to_object",
          [](py::typing::Union<int, py::str> &o) -> py::object { return o; });

    m.def("annotate_optional",
          [](py::list &list) -> py::typing::List<py::typing::Optional<py::str>> {
              list.append(py::str("hi"));
              list.append(py::none());
              return list;
          });

    m.def("annotate_type_guard", [](py::object &o) -> py::typing::TypeGuard<py::str> {
        return py::isinstance<py::str>(o);
    });
    m.def("annotate_type_is",
          [](py::object &o) -> py::typing::TypeIs<py::str> { return py::isinstance<py::str>(o); });

    m.def("annotate_no_return", []() -> py::typing::NoReturn { throw 0; });
    m.def("annotate_never", []() -> py::typing::Never { throw 0; });

    m.def("annotate_optional_to_object",
          [](py::typing::Optional<int> &o) -> py::object { return o; });

#if defined(PYBIND11_TYPING_H_HAS_STRING_LITERAL)
    py::enum_<literals::Color>(m, "Color")
        .value("RED", literals::Color::RED)
        .value("BLUE", literals::Color::BLUE);

    m.def("annotate_literal", [](literals::LiteralFoo &o) -> py::object { return o; });
    m.def("annotate_generic_containers",
          [](const py::typing::List<typevar::TypeVarT> &l) -> py::typing::List<typevar::TypeVarV> {
              return l;
          });

    m.def("annotate_listT_to_T",
          [](const py::typing::List<typevar::TypeVarT> &l) -> typevar::TypeVarT { return l[0]; });
    m.def("annotate_object_to_T", [](const py::object &o) -> typevar::TypeVarT { return o; });
    m.attr("defined_PYBIND11_TYPING_H_HAS_STRING_LITERAL") = true;
#else
    m.attr("defined_PYBIND11_TYPING_H_HAS_STRING_LITERAL") = false;
#endif

#if defined(PYBIND11_TEST_PYTYPES_HAS_RANGES)

    // test_tuple_ranges
    m.def("tuple_iterator_default_initialization", []() {
        using TupleIterator = decltype(std::declval<py::tuple>().begin());
        static_assert(std::random_access_iterator<TupleIterator>);
        return TupleIterator{} == TupleIterator{};
    });

    m.def("transform_tuple_plus_one", [](py::tuple &tpl) {
        py::list ret{};
        for (auto it : tpl | std::views::transform([](auto &o) { return py::cast<int>(o) + 1; })) {
            ret.append(py::int_(it));
        }
        return ret;
    });

    // test_list_ranges
    m.def("list_iterator_default_initialization", []() {
        using ListIterator = decltype(std::declval<py::list>().begin());
        static_assert(std::random_access_iterator<ListIterator>);
        return ListIterator{} == ListIterator{};
    });

    m.def("transform_list_plus_one", [](py::list &lst) {
        py::list ret{};
        for (auto it : lst | std::views::transform([](auto &o) { return py::cast<int>(o) + 1; })) {
            ret.append(py::int_(it));
        }
        return ret;
    });

    // test_dict_ranges
    m.def("dict_iterator_default_initialization", []() {
        using DictIterator = decltype(std::declval<py::dict>().begin());
        static_assert(std::forward_iterator<DictIterator>);
        return DictIterator{} == DictIterator{};
    });

    m.def("transform_dict_plus_one", [](py::dict &dct) {
        py::list ret{};
        for (auto it : dct | std::views::transform([](auto &o) {
                           return std::pair{py::cast<int>(o.first) + 1,
                                            py::cast<int>(o.second) + 1};
                       })) {
            ret.append(py::make_tuple(py::int_(it.first), py::int_(it.second)));
        }
        return ret;
    });

    m.attr("defined_PYBIND11_TEST_PYTYPES_HAS_RANGES") = true;
#else
    m.attr("defined_PYBIND11_TEST_PYTYPES_HAS_RANGES") = false;
#endif
}
