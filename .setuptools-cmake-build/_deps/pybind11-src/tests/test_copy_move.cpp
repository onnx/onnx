/*
    tests/test_copy_move_policies.cpp -- 'copy' and 'move' return value policies
                                         and related tests

    Copyright (c) 2016 Ben North <ben@redfrontdoor.org>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/stl.h>

#include "constructor_stats.h"
#include "pybind11_tests.h"

#include <type_traits>

template <typename derived>
struct empty {
    static const derived &get_one() { return instance_; }
    static derived instance_;
};

struct lacking_copy_ctor : public empty<lacking_copy_ctor> {
    lacking_copy_ctor() = default;
    lacking_copy_ctor(const lacking_copy_ctor &other) = delete;
};

template <>
lacking_copy_ctor empty<lacking_copy_ctor>::instance_ = {};

struct lacking_move_ctor : public empty<lacking_move_ctor> {
    lacking_move_ctor() = default;
    lacking_move_ctor(const lacking_move_ctor &other) = delete;
    lacking_move_ctor(lacking_move_ctor &&other) = delete;
};

template <>
lacking_move_ctor empty<lacking_move_ctor>::instance_ = {};

/* Custom type caster move/copy test classes */
class MoveOnlyInt {
public:
    MoveOnlyInt() { print_default_created(this); }
    explicit MoveOnlyInt(int v) : value{v} { print_created(this, value); }
    MoveOnlyInt(MoveOnlyInt &&m) noexcept {
        print_move_created(this, m.value);
        std::swap(value, m.value);
    }
    MoveOnlyInt &operator=(MoveOnlyInt &&m) noexcept {
        print_move_assigned(this, m.value);
        std::swap(value, m.value);
        return *this;
    }
    MoveOnlyInt(const MoveOnlyInt &) = delete;
    MoveOnlyInt &operator=(const MoveOnlyInt &) = delete;
    ~MoveOnlyInt() { print_destroyed(this); }

    int value;
};
class MoveOrCopyInt {
public:
    MoveOrCopyInt() { print_default_created(this); }
    explicit MoveOrCopyInt(int v) : value{v} { print_created(this, value); }
    MoveOrCopyInt(MoveOrCopyInt &&m) noexcept {
        print_move_created(this, m.value);
        std::swap(value, m.value);
    }
    MoveOrCopyInt &operator=(MoveOrCopyInt &&m) noexcept {
        print_move_assigned(this, m.value);
        std::swap(value, m.value);
        return *this;
    }
    MoveOrCopyInt(const MoveOrCopyInt &c) {
        print_copy_created(this, c.value);
        // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
        value = c.value;
    }
    MoveOrCopyInt &operator=(const MoveOrCopyInt &c) {
        print_copy_assigned(this, c.value);
        value = c.value;
        return *this;
    }
    ~MoveOrCopyInt() { print_destroyed(this); }

    int value;
};
class CopyOnlyInt {
public:
    CopyOnlyInt() { print_default_created(this); }
    explicit CopyOnlyInt(int v) : value{v} { print_created(this, value); }
    CopyOnlyInt(const CopyOnlyInt &c) {
        print_copy_created(this, c.value);
        // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
        value = c.value;
    }
    CopyOnlyInt &operator=(const CopyOnlyInt &c) {
        print_copy_assigned(this, c.value);
        value = c.value;
        return *this;
    }
    ~CopyOnlyInt() { print_destroyed(this); }

    int value;
};
PYBIND11_NAMESPACE_BEGIN(pybind11)
PYBIND11_NAMESPACE_BEGIN(detail)
template <>
struct type_caster<MoveOnlyInt> {
    PYBIND11_TYPE_CASTER(MoveOnlyInt, const_name("MoveOnlyInt"));
    bool load(handle src, bool) {
        value = MoveOnlyInt(src.cast<int>());
        return true;
    }
    static handle cast(const MoveOnlyInt &m, return_value_policy r, handle p) {
        return pybind11::cast(m.value, r, p);
    }
};

template <>
struct type_caster<MoveOrCopyInt> {
    PYBIND11_TYPE_CASTER(MoveOrCopyInt, const_name("MoveOrCopyInt"));
    bool load(handle src, bool) {
        value = MoveOrCopyInt(src.cast<int>());
        return true;
    }
    static handle cast(const MoveOrCopyInt &m, return_value_policy r, handle p) {
        return pybind11::cast(m.value, r, p);
    }
};

template <>
struct type_caster<CopyOnlyInt> {
protected:
    CopyOnlyInt value;

public:
    static constexpr auto name = const_name("CopyOnlyInt");
    bool load(handle src, bool) {
        value = CopyOnlyInt(src.cast<int>());
        return true;
    }
    static handle cast(const CopyOnlyInt &m, return_value_policy r, handle p) {
        return pybind11::cast(m.value, r, p);
    }
    static handle cast(const CopyOnlyInt *src, return_value_policy policy, handle parent) {
        if (!src) {
            return none().release();
        }
        return cast(*src, policy, parent);
    }
    explicit operator CopyOnlyInt *() { return &value; }
    explicit operator CopyOnlyInt &() { return value; }
    template <typename T>
    using cast_op_type = pybind11::detail::cast_op_type<T>;
};
PYBIND11_NAMESPACE_END(detail)
PYBIND11_NAMESPACE_END(pybind11)

namespace {

py::object CastUnusualOpRefConstRef(const UnusualOpRef &cref) { return py::cast(cref); }
py::object CastUnusualOpRefMovable(UnusualOpRef &&mvbl) { return py::cast(std::move(mvbl)); }

} // namespace

TEST_SUBMODULE(copy_move_policies, m) {
    // test_lacking_copy_ctor
    py::class_<lacking_copy_ctor>(m, "lacking_copy_ctor")
        .def_static("get_one", &lacking_copy_ctor::get_one, py::return_value_policy::copy);
    // test_lacking_move_ctor
    py::class_<lacking_move_ctor>(m, "lacking_move_ctor")
        .def_static("get_one", &lacking_move_ctor::get_one, py::return_value_policy::move);

    // test_move_and_copy_casts
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    m.def("move_and_copy_casts", [](const py::object &o) {
        int r = 0;
        r += py::cast<MoveOrCopyInt>(o).value; /* moves */
        r += py::cast<MoveOnlyInt>(o).value;   /* moves */
        r += py::cast<CopyOnlyInt>(o).value;   /* copies */
        auto m1(py::cast<MoveOrCopyInt>(o));   /* moves */
        auto m2(py::cast<MoveOnlyInt>(o));     /* moves */
        auto m3(py::cast<CopyOnlyInt>(o));     /* copies */
        r += m1.value + m2.value + m3.value;

        return r;
    });

    // test_move_and_copy_loads
    m.def("move_only", [](MoveOnlyInt m) { return m.value; });
    // Changing this breaks the existing test: needs careful review.
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    m.def("move_or_copy", [](MoveOrCopyInt m) { return m.value; });
    // Changing this breaks the existing test: needs careful review.
    // NOLINTNEXTLINE(performance-unnecessary-value-param)
    m.def("copy_only", [](CopyOnlyInt m) { return m.value; });
    m.def("move_pair",
          [](std::pair<MoveOnlyInt, MoveOrCopyInt> p) { return p.first.value + p.second.value; });
    m.def("move_tuple", [](std::tuple<MoveOnlyInt, MoveOrCopyInt, MoveOnlyInt> t) {
        return std::get<0>(t).value + std::get<1>(t).value + std::get<2>(t).value;
    });
    m.def("copy_tuple", [](std::tuple<CopyOnlyInt, CopyOnlyInt> t) {
        return std::get<0>(t).value + std::get<1>(t).value;
    });
    m.def("move_copy_nested",
          [](std::pair<MoveOnlyInt,
                       std::pair<std::tuple<MoveOrCopyInt, CopyOnlyInt, std::tuple<MoveOnlyInt>>,
                                 MoveOrCopyInt>> x) {
              return x.first.value + std::get<0>(x.second.first).value
                     + std::get<1>(x.second.first).value
                     + std::get<0>(std::get<2>(x.second.first)).value + x.second.second.value;
          });
    m.def("move_and_copy_cstats", []() {
        ConstructorStats::gc();
        // Reset counts to 0 so that previous tests don't affect later ones:
        auto &mc = ConstructorStats::get<MoveOrCopyInt>();
        mc.move_assignments = mc.move_constructions = mc.copy_assignments = mc.copy_constructions
            = 0;
        auto &mo = ConstructorStats::get<MoveOnlyInt>();
        mo.move_assignments = mo.move_constructions = mo.copy_assignments = mo.copy_constructions
            = 0;
        auto &co = ConstructorStats::get<CopyOnlyInt>();
        co.move_assignments = co.move_constructions = co.copy_assignments = co.copy_constructions
            = 0;
        py::dict d;
        d["MoveOrCopyInt"] = py::cast(mc, py::return_value_policy::reference);
        d["MoveOnlyInt"] = py::cast(mo, py::return_value_policy::reference);
        d["CopyOnlyInt"] = py::cast(co, py::return_value_policy::reference);
        return d;
    });
#ifdef PYBIND11_HAS_OPTIONAL
    // test_move_and_copy_load_optional
    m.attr("has_optional") = true;
    m.def("move_optional", [](std::optional<MoveOnlyInt> o) { return o->value; });
    m.def("move_or_copy_optional", [](std::optional<MoveOrCopyInt> o) { return o->value; });
    m.def("copy_optional", [](std::optional<CopyOnlyInt> o) { return o->value; });
    m.def("move_optional_tuple",
          [](std::optional<std::tuple<MoveOrCopyInt, MoveOnlyInt, CopyOnlyInt>> x) {
              return std::get<0>(*x).value + std::get<1>(*x).value + std::get<2>(*x).value;
          });
#else
    m.attr("has_optional") = false;
#endif

    // #70 compilation issue if operator new is not public - simple body added
    // but not needed on most compilers; MSVC and nvcc don't like a local
    // struct not having a method defined when declared, since it can not be
    // added later.
    struct PrivateOpNew {
        int value = 1;

    private:
        void *operator new(size_t bytes) {
            void *ptr = std::malloc(bytes);
            if (ptr) {
                return ptr;
            }
            throw std::bad_alloc{};
        }
    };
    py::class_<PrivateOpNew>(m, "PrivateOpNew").def_readonly("value", &PrivateOpNew::value);
    m.def("private_op_new_value", []() { return PrivateOpNew(); });
    m.def(
        "private_op_new_reference",
        []() -> const PrivateOpNew & {
            static PrivateOpNew x{};
            return x;
        },
        py::return_value_policy::reference);

    // test_move_fallback
    // #389: rvp::move should fall-through to copy on non-movable objects
    struct MoveIssue1 {
        int v;
        explicit MoveIssue1(int v) : v{v} {}
        MoveIssue1(const MoveIssue1 &c) = default;
        MoveIssue1(MoveIssue1 &&) = delete;
    };
    py::class_<MoveIssue1>(m, "MoveIssue1")
        .def(py::init<int>())
        .def_readwrite("value", &MoveIssue1::v);

    struct MoveIssue2 {
        int v;
        explicit MoveIssue2(int v) : v{v} {}
        MoveIssue2(MoveIssue2 &&) = default;
    };
    py::class_<MoveIssue2>(m, "MoveIssue2")
        .def(py::init<int>())
        .def_readwrite("value", &MoveIssue2::v);

    // #2742: Don't expect ownership of raw pointer to `new`ed object to be transferred with
    // `py::return_value_policy::move`
    m.def(
        "get_moveissue1",
        [](int i) { return std::unique_ptr<MoveIssue1>(new MoveIssue1(i)); },
        py::return_value_policy::move);
    m.def("get_moveissue2", [](int i) { return MoveIssue2(i); }, py::return_value_policy::move);

    // Make sure that cast from pytype rvalue to other pytype works
    m.def("get_pytype_rvalue_castissue", [](double i) { return py::float_(i).cast<py::int_>(); });

    py::class_<UnusualOpRef>(m, "UnusualOpRef");
    m.def("CallCastUnusualOpRefConstRef",
          []() { return CastUnusualOpRefConstRef(UnusualOpRef()); });
    m.def("CallCastUnusualOpRefMovable", []() { return CastUnusualOpRefMovable(UnusualOpRef()); });
}

/*
 * Rest of the file:
 * static_assert based tests for pybind11 adaptations of
 * std::is_move_constructible, std::is_copy_constructible and
 * std::is_copy_assignable (no adaptation of std::is_move_assignable).
 * Difference between pybind11 and std traits: pybind11 traits will also check
 * the contained value_types.
 */

struct NotMovable {
    NotMovable() = default;
    NotMovable(NotMovable const &) = default;
    NotMovable(NotMovable &&) = delete;
    NotMovable &operator=(NotMovable const &) = default;
    NotMovable &operator=(NotMovable &&) = delete;
};
static_assert(!std::is_move_constructible<NotMovable>::value,
              "!std::is_move_constructible<NotMovable>::value");
static_assert(std::is_copy_constructible<NotMovable>::value,
              "std::is_copy_constructible<NotMovable>::value");
static_assert(!pybind11::detail::is_move_constructible<NotMovable>::value,
              "!pybind11::detail::is_move_constructible<NotMovable>::value");
static_assert(pybind11::detail::is_copy_constructible<NotMovable>::value,
              "pybind11::detail::is_copy_constructible<NotMovable>::value");
static_assert(!std::is_move_assignable<NotMovable>::value,
              "!std::is_move_assignable<NotMovable>::value");
static_assert(std::is_copy_assignable<NotMovable>::value,
              "std::is_copy_assignable<NotMovable>::value");
// pybind11 does not have this
// static_assert(!pybind11::detail::is_move_assignable<NotMovable>::value,
//               "!pybind11::detail::is_move_assignable<NotMovable>::value");
static_assert(pybind11::detail::is_copy_assignable<NotMovable>::value,
              "pybind11::detail::is_copy_assignable<NotMovable>::value");

struct NotCopyable {
    NotCopyable() = default;
    NotCopyable(NotCopyable const &) = delete;
    NotCopyable(NotCopyable &&) = default;
    NotCopyable &operator=(NotCopyable const &) = delete;
    NotCopyable &operator=(NotCopyable &&) = default;
};
static_assert(std::is_move_constructible<NotCopyable>::value,
              "std::is_move_constructible<NotCopyable>::value");
static_assert(!std::is_copy_constructible<NotCopyable>::value,
              "!std::is_copy_constructible<NotCopyable>::value");
static_assert(pybind11::detail::is_move_constructible<NotCopyable>::value,
              "pybind11::detail::is_move_constructible<NotCopyable>::value");
static_assert(!pybind11::detail::is_copy_constructible<NotCopyable>::value,
              "!pybind11::detail::is_copy_constructible<NotCopyable>::value");
static_assert(std::is_move_assignable<NotCopyable>::value,
              "std::is_move_assignable<NotCopyable>::value");
static_assert(!std::is_copy_assignable<NotCopyable>::value,
              "!std::is_copy_assignable<NotCopyable>::value");
// pybind11 does not have this
// static_assert(!pybind11::detail::is_move_assignable<NotCopyable>::value,
//               "!pybind11::detail::is_move_assignable<NotCopyable>::value");
static_assert(!pybind11::detail::is_copy_assignable<NotCopyable>::value,
              "!pybind11::detail::is_copy_assignable<NotCopyable>::value");

struct NotCopyableNotMovable {
    NotCopyableNotMovable() = default;
    NotCopyableNotMovable(NotCopyableNotMovable const &) = delete;
    NotCopyableNotMovable(NotCopyableNotMovable &&) = delete;
    NotCopyableNotMovable &operator=(NotCopyableNotMovable const &) = delete;
    NotCopyableNotMovable &operator=(NotCopyableNotMovable &&) = delete;
};
static_assert(!std::is_move_constructible<NotCopyableNotMovable>::value,
              "!std::is_move_constructible<NotCopyableNotMovable>::value");
static_assert(!std::is_copy_constructible<NotCopyableNotMovable>::value,
              "!std::is_copy_constructible<NotCopyableNotMovable>::value");
static_assert(!pybind11::detail::is_move_constructible<NotCopyableNotMovable>::value,
              "!pybind11::detail::is_move_constructible<NotCopyableNotMovable>::value");
static_assert(!pybind11::detail::is_copy_constructible<NotCopyableNotMovable>::value,
              "!pybind11::detail::is_copy_constructible<NotCopyableNotMovable>::value");
static_assert(!std::is_move_assignable<NotCopyableNotMovable>::value,
              "!std::is_move_assignable<NotCopyableNotMovable>::value");
static_assert(!std::is_copy_assignable<NotCopyableNotMovable>::value,
              "!std::is_copy_assignable<NotCopyableNotMovable>::value");
// pybind11 does not have this
// static_assert(!pybind11::detail::is_move_assignable<NotCopyableNotMovable>::value,
//               "!pybind11::detail::is_move_assignable<NotCopyableNotMovable>::value");
static_assert(!pybind11::detail::is_copy_assignable<NotCopyableNotMovable>::value,
              "!pybind11::detail::is_copy_assignable<NotCopyableNotMovable>::value");

struct NotMovableVector : std::vector<NotMovable> {};
static_assert(std::is_move_constructible<NotMovableVector>::value,
              "std::is_move_constructible<NotMovableVector>::value");
static_assert(std::is_copy_constructible<NotMovableVector>::value,
              "std::is_copy_constructible<NotMovableVector>::value");
static_assert(!pybind11::detail::is_move_constructible<NotMovableVector>::value,
              "!pybind11::detail::is_move_constructible<NotMovableVector>::value");
static_assert(pybind11::detail::is_copy_constructible<NotMovableVector>::value,
              "pybind11::detail::is_copy_constructible<NotMovableVector>::value");
static_assert(std::is_move_assignable<NotMovableVector>::value,
              "std::is_move_assignable<NotMovableVector>::value");
static_assert(std::is_copy_assignable<NotMovableVector>::value,
              "std::is_copy_assignable<NotMovableVector>::value");
// pybind11 does not have this
// static_assert(!pybind11::detail::is_move_assignable<NotMovableVector>::value,
//               "!pybind11::detail::is_move_assignable<NotMovableVector>::value");
static_assert(pybind11::detail::is_copy_assignable<NotMovableVector>::value,
              "pybind11::detail::is_copy_assignable<NotMovableVector>::value");

struct NotCopyableVector : std::vector<NotCopyable> {};
static_assert(std::is_move_constructible<NotCopyableVector>::value,
              "std::is_move_constructible<NotCopyableVector>::value");
static_assert(std::is_copy_constructible<NotCopyableVector>::value,
              "std::is_copy_constructible<NotCopyableVector>::value");
static_assert(pybind11::detail::is_move_constructible<NotCopyableVector>::value,
              "pybind11::detail::is_move_constructible<NotCopyableVector>::value");
static_assert(!pybind11::detail::is_copy_constructible<NotCopyableVector>::value,
              "!pybind11::detail::is_copy_constructible<NotCopyableVector>::value");
static_assert(std::is_move_assignable<NotCopyableVector>::value,
              "std::is_move_assignable<NotCopyableVector>::value");
static_assert(std::is_copy_assignable<NotCopyableVector>::value,
              "std::is_copy_assignable<NotCopyableVector>::value");
// pybind11 does not have this
// static_assert(!pybind11::detail::is_move_assignable<NotCopyableVector>::value,
//               "!pybind11::detail::is_move_assignable<NotCopyableVector>::value");
static_assert(!pybind11::detail::is_copy_assignable<NotCopyableVector>::value,
              "!pybind11::detail::is_copy_assignable<NotCopyableVector>::value");

struct NotCopyableNotMovableVector : std::vector<NotCopyableNotMovable> {};
static_assert(std::is_move_constructible<NotCopyableNotMovableVector>::value,
              "std::is_move_constructible<NotCopyableNotMovableVector>::value");
static_assert(std::is_copy_constructible<NotCopyableNotMovableVector>::value,
              "std::is_copy_constructible<NotCopyableNotMovableVector>::value");
static_assert(!pybind11::detail::is_move_constructible<NotCopyableNotMovableVector>::value,
              "!pybind11::detail::is_move_constructible<NotCopyableNotMovableVector>::value");
static_assert(!pybind11::detail::is_copy_constructible<NotCopyableNotMovableVector>::value,
              "!pybind11::detail::is_copy_constructible<NotCopyableNotMovableVector>::value");
static_assert(std::is_move_assignable<NotCopyableNotMovableVector>::value,
              "std::is_move_assignable<NotCopyableNotMovableVector>::value");
static_assert(std::is_copy_assignable<NotCopyableNotMovableVector>::value,
              "std::is_copy_assignable<NotCopyableNotMovableVector>::value");
// pybind11 does not have this
// static_assert(!pybind11::detail::is_move_assignable<NotCopyableNotMovableVector>::value,
//               "!pybind11::detail::is_move_assignable<NotCopyableNotMovableVector>::value");
static_assert(!pybind11::detail::is_copy_assignable<NotCopyableNotMovableVector>::value,
              "!pybind11::detail::is_copy_assignable<NotCopyableNotMovableVector>::value");

struct NotMovableMap : std::map<int, NotMovable> {};
static_assert(std::is_move_constructible<NotMovableMap>::value,
              "std::is_move_constructible<NotMovableMap>::value");
static_assert(std::is_copy_constructible<NotMovableMap>::value,
              "std::is_copy_constructible<NotMovableMap>::value");
static_assert(!pybind11::detail::is_move_constructible<NotMovableMap>::value,
              "!pybind11::detail::is_move_constructible<NotMovableMap>::value");
static_assert(pybind11::detail::is_copy_constructible<NotMovableMap>::value,
              "pybind11::detail::is_copy_constructible<NotMovableMap>::value");
static_assert(std::is_move_assignable<NotMovableMap>::value,
              "std::is_move_assignable<NotMovableMap>::value");
static_assert(std::is_copy_assignable<NotMovableMap>::value,
              "std::is_copy_assignable<NotMovableMap>::value");
// pybind11 does not have this
// static_assert(!pybind11::detail::is_move_assignable<NotMovableMap>::value,
//               "!pybind11::detail::is_move_assignable<NotMovableMap>::value");
static_assert(pybind11::detail::is_copy_assignable<NotMovableMap>::value,
              "pybind11::detail::is_copy_assignable<NotMovableMap>::value");

struct NotCopyableMap : std::map<int, NotCopyable> {};
static_assert(std::is_move_constructible<NotCopyableMap>::value,
              "std::is_move_constructible<NotCopyableMap>::value");
static_assert(std::is_copy_constructible<NotCopyableMap>::value,
              "std::is_copy_constructible<NotCopyableMap>::value");
static_assert(pybind11::detail::is_move_constructible<NotCopyableMap>::value,
              "pybind11::detail::is_move_constructible<NotCopyableMap>::value");
static_assert(!pybind11::detail::is_copy_constructible<NotCopyableMap>::value,
              "!pybind11::detail::is_copy_constructible<NotCopyableMap>::value");
static_assert(std::is_move_assignable<NotCopyableMap>::value,
              "std::is_move_assignable<NotCopyableMap>::value");
static_assert(std::is_copy_assignable<NotCopyableMap>::value,
              "std::is_copy_assignable<NotCopyableMap>::value");
// pybind11 does not have this
// static_assert(!pybind11::detail::is_move_assignable<NotCopyableMap>::value,
//               "!pybind11::detail::is_move_assignable<NotCopyableMap>::value");
static_assert(!pybind11::detail::is_copy_assignable<NotCopyableMap>::value,
              "!pybind11::detail::is_copy_assignable<NotCopyableMap>::value");

struct NotCopyableNotMovableMap : std::map<int, NotCopyableNotMovable> {};
static_assert(std::is_move_constructible<NotCopyableNotMovableMap>::value,
              "std::is_move_constructible<NotCopyableNotMovableMap>::value");
static_assert(std::is_copy_constructible<NotCopyableNotMovableMap>::value,
              "std::is_copy_constructible<NotCopyableNotMovableMap>::value");
static_assert(!pybind11::detail::is_move_constructible<NotCopyableNotMovableMap>::value,
              "!pybind11::detail::is_move_constructible<NotCopyableNotMovableMap>::value");
static_assert(!pybind11::detail::is_copy_constructible<NotCopyableNotMovableMap>::value,
              "!pybind11::detail::is_copy_constructible<NotCopyableNotMovableMap>::value");
static_assert(std::is_move_assignable<NotCopyableNotMovableMap>::value,
              "std::is_move_assignable<NotCopyableNotMovableMap>::value");
static_assert(std::is_copy_assignable<NotCopyableNotMovableMap>::value,
              "std::is_copy_assignable<NotCopyableNotMovableMap>::value");
// pybind11 does not have this
// static_assert(!pybind11::detail::is_move_assignable<NotCopyableNotMovableMap>::value,
//               "!pybind11::detail::is_move_assignable<NotCopyableNotMovableMap>::value");
static_assert(!pybind11::detail::is_copy_assignable<NotCopyableNotMovableMap>::value,
              "!pybind11::detail::is_copy_assignable<NotCopyableNotMovableMap>::value");

struct RecursiveVector : std::vector<RecursiveVector> {};
static_assert(std::is_move_constructible<RecursiveVector>::value,
              "std::is_move_constructible<RecursiveVector>::value");
static_assert(std::is_copy_constructible<RecursiveVector>::value,
              "std::is_copy_constructible<RecursiveVector>::value");
static_assert(pybind11::detail::is_move_constructible<RecursiveVector>::value,
              "pybind11::detail::is_move_constructible<RecursiveVector>::value");
static_assert(pybind11::detail::is_copy_constructible<RecursiveVector>::value,
              "pybind11::detail::is_copy_constructible<RecursiveVector>::value");
static_assert(std::is_move_assignable<RecursiveVector>::value,
              "std::is_move_assignable<RecursiveVector>::value");
static_assert(std::is_copy_assignable<RecursiveVector>::value,
              "std::is_copy_assignable<RecursiveVector>::value");
// pybind11 does not have this
// static_assert(!pybind11::detail::is_move_assignable<RecursiveVector>::value,
//               "!pybind11::detail::is_move_assignable<RecursiveVector>::value");
static_assert(pybind11::detail::is_copy_assignable<RecursiveVector>::value,
              "pybind11::detail::is_copy_assignable<RecursiveVector>::value");

struct RecursiveMap : std::map<int, RecursiveMap> {};
static_assert(std::is_move_constructible<RecursiveMap>::value,
              "std::is_move_constructible<RecursiveMap>::value");
static_assert(std::is_copy_constructible<RecursiveMap>::value,
              "std::is_copy_constructible<RecursiveMap>::value");
static_assert(pybind11::detail::is_move_constructible<RecursiveMap>::value,
              "pybind11::detail::is_move_constructible<RecursiveMap>::value");
static_assert(pybind11::detail::is_copy_constructible<RecursiveMap>::value,
              "pybind11::detail::is_copy_constructible<RecursiveMap>::value");
static_assert(std::is_move_assignable<RecursiveMap>::value,
              "std::is_move_assignable<RecursiveMap>::value");
static_assert(std::is_copy_assignable<RecursiveMap>::value,
              "std::is_copy_assignable<RecursiveMap>::value");
// pybind11 does not have this
// static_assert(!pybind11::detail::is_move_assignable<RecursiveMap>::value,
//               "!pybind11::detail::is_move_assignable<RecursiveMap>::value");
static_assert(pybind11::detail::is_copy_assignable<RecursiveMap>::value,
              "pybind11::detail::is_copy_assignable<RecursiveMap>::value");
