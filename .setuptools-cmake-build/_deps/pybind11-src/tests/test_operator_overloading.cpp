/*
    tests/test_operator_overloading.cpp -- operator overloading

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "constructor_stats.h"
#include "pybind11_tests.h"

#include <functional>

class Vector2 {
public:
    Vector2(float x, float y) : x(x), y(y) { print_created(this, toString()); }
    Vector2(const Vector2 &v) : x(v.x), y(v.y) { print_copy_created(this); }
    Vector2(Vector2 &&v) noexcept : x(v.x), y(v.y) {
        print_move_created(this);
        v.x = v.y = 0;
    }
    Vector2 &operator=(const Vector2 &v) {
        x = v.x;
        y = v.y;
        print_copy_assigned(this);
        return *this;
    }
    Vector2 &operator=(Vector2 &&v) noexcept {
        x = v.x;
        y = v.y;
        v.x = v.y = 0;
        print_move_assigned(this);
        return *this;
    }
    ~Vector2() { print_destroyed(this); }

    std::string toString() const {
        return "[" + std::to_string(x) + ", " + std::to_string(y) + "]";
    }

    Vector2 operator-() const { return Vector2(-x, -y); }
    Vector2 operator+(const Vector2 &v) const { return Vector2(x + v.x, y + v.y); }
    Vector2 operator-(const Vector2 &v) const { return Vector2(x - v.x, y - v.y); }
    Vector2 operator-(float value) const { return Vector2(x - value, y - value); }
    Vector2 operator+(float value) const { return Vector2(x + value, y + value); }
    Vector2 operator*(float value) const { return Vector2(x * value, y * value); }
    Vector2 operator/(float value) const { return Vector2(x / value, y / value); }
    Vector2 operator*(const Vector2 &v) const { return Vector2(x * v.x, y * v.y); }
    Vector2 operator/(const Vector2 &v) const { return Vector2(x / v.x, y / v.y); }
    Vector2 &operator+=(const Vector2 &v) {
        x += v.x;
        y += v.y;
        return *this;
    }
    Vector2 &operator-=(const Vector2 &v) {
        x -= v.x;
        y -= v.y;
        return *this;
    }
    Vector2 &operator*=(float v) {
        x *= v;
        y *= v;
        return *this;
    }
    Vector2 &operator/=(float v) {
        x /= v;
        y /= v;
        return *this;
    }
    Vector2 &operator*=(const Vector2 &v) {
        x *= v.x;
        y *= v.y;
        return *this;
    }
    Vector2 &operator/=(const Vector2 &v) {
        x /= v.x;
        y /= v.y;
        return *this;
    }

    friend Vector2 operator+(float f, const Vector2 &v) { return Vector2(f + v.x, f + v.y); }
    friend Vector2 operator-(float f, const Vector2 &v) { return Vector2(f - v.x, f - v.y); }
    friend Vector2 operator*(float f, const Vector2 &v) { return Vector2(f * v.x, f * v.y); }
    friend Vector2 operator/(float f, const Vector2 &v) { return Vector2(f / v.x, f / v.y); }

    bool operator==(const Vector2 &v) const { return x == v.x && y == v.y; }
    bool operator!=(const Vector2 &v) const { return x != v.x || y != v.y; }

private:
    float x, y;
};

class C1 {};
class C2 {};

int operator+(const C1 &, const C1 &) { return 11; }
int operator+(const C2 &, const C2 &) { return 22; }
int operator+(const C2 &, const C1 &) { return 21; }
int operator+(const C1 &, const C2 &) { return 12; }

struct HashMe {
    std::string member;
};

bool operator==(const HashMe &lhs, const HashMe &rhs) { return lhs.member == rhs.member; }

// Note: Specializing explicit within `namespace std { ... }` is done due to a
// bug in GCC<7. If you are supporting compilers later than this, consider
// specializing `using template<> struct std::hash<...>` in the global
// namespace instead, per this recommendation:
// https://en.cppreference.com/w/cpp/language/extending_std#Adding_template_specializations
namespace std {
template <>
struct hash<Vector2> {
    // Not a good hash function, but easy to test
    size_t operator()(const Vector2 &) { return 4; }
};

// HashMe has a hash function in C++ but no `__hash__` for Python.
template <>
struct hash<HashMe> {
    std::size_t operator()(const HashMe &selector) const {
        return std::hash<std::string>()(selector.member);
    }
};
} // namespace std

// Not a good abs function, but easy to test.
std::string abs(const Vector2 &) { return "abs(Vector2)"; }

// clang 7.0.0 and Apple LLVM 10.0.1 introduce `-Wself-assign-overloaded` to
// `-Wall`, which is used here for overloading (e.g. `py::self += py::self `).
// Here, we suppress the warning
// Taken from: https://github.com/RobotLocomotion/drake/commit/aaf84b46
// TODO(eric): This could be resolved using a function / functor (e.g. `py::self()`).
#if defined(__APPLE__) && defined(__clang__)
#    if (__clang_major__ >= 10)
PYBIND11_WARNING_DISABLE_CLANG("-Wself-assign-overloaded")
#    endif
#elif defined(__clang__)
#    if (__clang_major__ >= 7)
PYBIND11_WARNING_DISABLE_CLANG("-Wself-assign-overloaded")
#    endif
#endif

TEST_SUBMODULE(operators, m) {

    // test_operator_overloading
    py::class_<Vector2>(m, "Vector2")
        .def(py::init<float, float>())
        .def(py::self + py::self)
        .def(py::self + float())
        .def(py::self - py::self)
        .def(py::self - float())
        .def(py::self * float())
        .def(py::self / float())
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self += py::self)
        .def(py::self -= py::self)
        .def(py::self *= float())
        .def(py::self /= float())
        .def(py::self *= py::self)
        .def(py::self /= py::self)
        .def(float() + py::self)
        .def(float() - py::self)
        .def(float() * py::self)
        .def(float() / py::self)
        .def(-py::self)
        .def("__str__", &Vector2::toString)
        .def("__repr__", &Vector2::toString)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::hash(py::self))
        // N.B. See warning about usage of `py::detail::abs(py::self)` in
        // `operators.h`.
        .def("__abs__", [](const Vector2 &v) { return abs(v); });

    m.attr("Vector") = m.attr("Vector2");

    // test_operators_notimplemented
    // #393: need to return NotSupported to ensure correct arithmetic operator behavior
    py::class_<C1>(m, "C1").def(py::init<>()).def(py::self + py::self);

    py::class_<C2>(m, "C2")
        .def(py::init<>())
        .def(py::self + py::self)
        .def("__add__", [](const C2 &c2, const C1 &c1) { return c2 + c1; })
        .def("__radd__", [](const C2 &c2, const C1 &c1) { return c1 + c2; });

    // test_nested
    // #328: first member in a class can't be used in operators
    struct NestABase {
        int value = -2;
    };
    py::class_<NestABase>(m, "NestABase")
        .def(py::init<>())
        .def_readwrite("value", &NestABase::value);

    struct NestA : NestABase {
        int value = 3;
        NestA &operator+=(int i) {
            value += i;
            return *this;
        }
    };
    py::class_<NestA>(m, "NestA")
        .def(py::init<>())
        .def(py::self += int())
        .def(
            "as_base",
            [](NestA &a) -> NestABase & { return (NestABase &) a; },
            py::return_value_policy::reference_internal);
    m.def("get_NestA", [](const NestA &a) { return a.value; });

    struct NestB {
        NestA a;
        int value = 4;
        NestB &operator-=(int i) {
            value -= i;
            return *this;
        }
    };
    py::class_<NestB>(m, "NestB")
        .def(py::init<>())
        .def(py::self -= int())
        .def_readwrite("a", &NestB::a);
    m.def("get_NestB", [](const NestB &b) { return b.value; });

    struct NestC {
        NestB b;
        int value = 5;
        NestC &operator*=(int i) {
            value *= i;
            return *this;
        }
    };
    py::class_<NestC>(m, "NestC")
        .def(py::init<>())
        .def(py::self *= int())
        .def_readwrite("b", &NestC::b);
    m.def("get_NestC", [](const NestC &c) { return c.value; });

    // test_overriding_eq_reset_hash
    // #2191 Overriding __eq__ should set __hash__ to None
    struct Comparable {
        int value;
        bool operator==(const Comparable &rhs) const { return value == rhs.value; }
    };

    struct Hashable : Comparable {
        explicit Hashable(int value) : Comparable{value} {};
        size_t hash() const { return static_cast<size_t>(value); }
    };

    struct Hashable2 : Hashable {
        using Hashable::Hashable;
    };

    py::class_<Comparable>(m, "Comparable").def(py::init<int>()).def(py::self == py::self);

    py::class_<Hashable>(m, "Hashable")
        .def(py::init<int>())
        .def(py::self == py::self)
        .def("__hash__", &Hashable::hash);

    // define __hash__ before __eq__
    py::class_<Hashable2>(m, "Hashable2")
        .def("__hash__", &Hashable::hash)
        .def(py::init<int>())
        .def(py::self == py::self);

    // define __eq__ but not __hash__
    py::class_<HashMe>(m, "HashMe").def(py::self == py::self);

    m.def("get_unhashable_HashMe_set", []() { return std::unordered_set<HashMe>{{"one"}}; });
}
