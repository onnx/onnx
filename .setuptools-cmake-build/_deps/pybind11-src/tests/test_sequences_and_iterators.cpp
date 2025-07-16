/*
    tests/test_sequences_and_iterators.cpp -- supporting Pythons' sequence protocol, iterators,
    etc.

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "constructor_stats.h"
#include "pybind11_tests.h"

#include <algorithm>
#include <utility>
#include <vector>

#ifdef PYBIND11_HAS_OPTIONAL
#    include <optional>
#endif // PYBIND11_HAS_OPTIONAL

template <typename T>
class NonZeroIterator {
    const T *ptr_;

public:
    explicit NonZeroIterator(const T *ptr) : ptr_(ptr) {}

    // Make the iterator non-copyable and movable
    NonZeroIterator(const NonZeroIterator &) = delete;
    NonZeroIterator(NonZeroIterator &&) noexcept = default;
    NonZeroIterator &operator=(const NonZeroIterator &) = delete;
    NonZeroIterator &operator=(NonZeroIterator &&) noexcept = default;

    const T &operator*() const { return *ptr_; }
    NonZeroIterator &operator++() {
        ++ptr_;
        return *this;
    }
};

class NonZeroSentinel {};

template <typename A, typename B>
bool operator==(const NonZeroIterator<std::pair<A, B>> &it, const NonZeroSentinel &) {
    return !(*it).first || !(*it).second;
}

/* Iterator where dereferencing returns prvalues instead of references. */
template <typename T>
class NonRefIterator {
    const T *ptr_;

public:
    explicit NonRefIterator(const T *ptr) : ptr_(ptr) {}
    T operator*() const { return T(*ptr_); }
    NonRefIterator &operator++() {
        ++ptr_;
        return *this;
    }
    bool operator==(const NonRefIterator &other) const { return ptr_ == other.ptr_; }
};

class NonCopyableInt {
public:
    explicit NonCopyableInt(int value) : value_(value) {}
    NonCopyableInt(const NonCopyableInt &) = delete;
    NonCopyableInt(NonCopyableInt &&other) noexcept : value_(other.value_) {
        other.value_ = -1; // detect when an unwanted move occurs
    }
    NonCopyableInt &operator=(const NonCopyableInt &) = delete;
    NonCopyableInt &operator=(NonCopyableInt &&other) noexcept {
        value_ = other.value_;
        other.value_ = -1; // detect when an unwanted move occurs
        return *this;
    }
    int get() const { return value_; }
    void set(int value) { value_ = value; }
    ~NonCopyableInt() = default;

private:
    int value_;
};
using NonCopyableIntPair = std::pair<NonCopyableInt, NonCopyableInt>;

PYBIND11_MAKE_OPAQUE(std::vector<NonCopyableInt>)
PYBIND11_MAKE_OPAQUE(std::vector<NonCopyableIntPair>)

template <typename PythonType>
py::list test_random_access_iterator(PythonType x) {
    if (x.size() < 5) {
        throw py::value_error("Please provide at least 5 elements for testing.");
    }

    auto checks = py::list();
    auto assert_equal = [&checks](py::handle a, py::handle b) {
        auto result = PyObject_RichCompareBool(a.ptr(), b.ptr(), Py_EQ);
        if (result == -1) {
            throw py::error_already_set();
        }
        checks.append(result != 0);
    };

    auto it = x.begin();
    assert_equal(x[0], *it);
    assert_equal(x[0], it[0]);
    assert_equal(x[1], it[1]);

    assert_equal(x[1], *(++it));
    assert_equal(x[1], *(it++));
    assert_equal(x[2], *it);
    assert_equal(x[3], *(it += 1));
    assert_equal(x[2], *(--it));
    assert_equal(x[2], *(it--));
    assert_equal(x[1], *it);
    assert_equal(x[0], *(it -= 1));

    assert_equal(it->attr("real"), x[0].attr("real"));
    assert_equal((it + 1)->attr("real"), x[1].attr("real"));

    assert_equal(x[1], *(it + 1));
    assert_equal(x[1], *(1 + it));
    it += 3;
    assert_equal(x[1], *(it - 2));

    checks.append(static_cast<std::size_t>(x.end() - x.begin()) == x.size());
    checks.append((x.begin() + static_cast<std::ptrdiff_t>(x.size())) == x.end());
    checks.append(x.begin() < x.end());

    return checks;
}

TEST_SUBMODULE(sequences_and_iterators, m) {
    // test_sliceable
    class Sliceable {
    public:
        explicit Sliceable(int n) : size(n) {}
        int start, stop, step;
        int size;
    };
    py::class_<Sliceable>(m, "Sliceable")
        .def(py::init<int>())
        .def("__getitem__", [](const Sliceable &s, const py::slice &slice) {
            py::ssize_t start = 0, stop = 0, step = 0, slicelength = 0;
            if (!slice.compute(s.size, &start, &stop, &step, &slicelength)) {
                throw py::error_already_set();
            }
            int istart = static_cast<int>(start);
            int istop = static_cast<int>(stop);
            int istep = static_cast<int>(step);
            return std::make_tuple(istart, istop, istep);
        });

    m.def("make_forward_slice_size_t", []() { return py::slice(0, -1, 1); });
    m.def("make_reversed_slice_object",
          []() { return py::slice(py::none(), py::none(), py::int_(-1)); });
#ifdef PYBIND11_HAS_OPTIONAL
    m.attr("has_optional") = true;
    m.def("make_reversed_slice_size_t_optional_verbose",
          []() { return py::slice(std::nullopt, std::nullopt, -1); });
    // Warning: The following spelling may still compile if optional<> is not present and give
    // wrong answers. Please use with caution.
    m.def("make_reversed_slice_size_t_optional", []() { return py::slice({}, {}, -1); });
#else
    m.attr("has_optional") = false;
#endif

    // test_sequence
    class Sequence {
    public:
        explicit Sequence(size_t size) : m_size(size) {
            print_created(this, "of size", m_size);
            // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
            m_data = new float[size];
            memset(m_data, 0, sizeof(float) * size);
        }
        explicit Sequence(const std::vector<float> &value) : m_size(value.size()) {
            print_created(this, "of size", m_size, "from std::vector");
            // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
            m_data = new float[m_size];
            memcpy(m_data, &value[0], sizeof(float) * m_size);
        }
        Sequence(const Sequence &s) : m_size(s.m_size) {
            print_copy_created(this);
            // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
            m_data = new float[m_size];
            memcpy(m_data, s.m_data, sizeof(float) * m_size);
        }
        Sequence(Sequence &&s) noexcept : m_size(s.m_size), m_data(s.m_data) {
            print_move_created(this);
            s.m_size = 0;
            s.m_data = nullptr;
        }

        ~Sequence() {
            print_destroyed(this);
            delete[] m_data;
        }

        Sequence &operator=(const Sequence &s) {
            if (&s != this) {
                delete[] m_data;
                m_size = s.m_size;
                m_data = new float[m_size];
                memcpy(m_data, s.m_data, sizeof(float) * m_size);
            }
            print_copy_assigned(this);
            return *this;
        }

        Sequence &operator=(Sequence &&s) noexcept {
            if (&s != this) {
                delete[] m_data;
                m_size = s.m_size;
                m_data = s.m_data;
                s.m_size = 0;
                s.m_data = nullptr;
            }
            print_move_assigned(this);
            return *this;
        }

        bool operator==(const Sequence &s) const {
            if (m_size != s.size()) {
                return false;
            }
            for (size_t i = 0; i < m_size; ++i) {
                if (m_data[i] != s[i]) {
                    return false;
                }
            }
            return true;
        }
        bool operator!=(const Sequence &s) const { return !operator==(s); }

        float operator[](size_t index) const { return m_data[index]; }
        float &operator[](size_t index) { return m_data[index]; }

        bool contains(float v) const {
            for (size_t i = 0; i < m_size; ++i) {
                if (v == m_data[i]) {
                    return true;
                }
            }
            return false;
        }

        Sequence reversed() const {
            Sequence result(m_size);
            for (size_t i = 0; i < m_size; ++i) {
                result[m_size - i - 1] = m_data[i];
            }
            return result;
        }

        size_t size() const { return m_size; }

        const float *begin() const { return m_data; }
        const float *end() const { return m_data + m_size; }

    private:
        size_t m_size;
        float *m_data;
    };
    py::class_<Sequence>(m, "Sequence")
        .def(py::init<size_t>())
        .def(py::init<const std::vector<float> &>())
        /// Bare bones interface
        .def("__getitem__",
             [](const Sequence &s, size_t i) {
                 if (i >= s.size()) {
                     throw py::index_error();
                 }
                 return s[i];
             })
        .def("__setitem__",
             [](Sequence &s, size_t i, float v) {
                 if (i >= s.size()) {
                     throw py::index_error();
                 }
                 s[i] = v;
             })
        .def("__len__", &Sequence::size)
        /// Optional sequence protocol operations
        .def(
            "__iter__",
            [](const Sequence &s) { return py::make_iterator(s.begin(), s.end()); },
            py::keep_alive<0, 1>() /* Essential: keep object alive while iterator exists */)
        .def("__contains__", [](const Sequence &s, float v) { return s.contains(v); })
        .def("__reversed__", [](const Sequence &s) -> Sequence { return s.reversed(); })
        /// Slicing protocol (optional)
        .def("__getitem__",
             [](const Sequence &s, const py::slice &slice) -> Sequence * {
                 size_t start = 0, stop = 0, step = 0, slicelength = 0;
                 if (!slice.compute(s.size(), &start, &stop, &step, &slicelength)) {
                     throw py::error_already_set();
                 }
                 auto *seq = new Sequence(slicelength);
                 for (size_t i = 0; i < slicelength; ++i) {
                     (*seq)[i] = s[start];
                     start += step;
                 }
                 return seq;
             })
        .def("__setitem__",
             [](Sequence &s, const py::slice &slice, const Sequence &value) {
                 size_t start = 0, stop = 0, step = 0, slicelength = 0;
                 if (!slice.compute(s.size(), &start, &stop, &step, &slicelength)) {
                     throw py::error_already_set();
                 }
                 if (slicelength != value.size()) {
                     throw std::runtime_error(
                         "Left and right hand size of slice assignment have different sizes!");
                 }
                 for (size_t i = 0; i < slicelength; ++i) {
                     s[start] = value[i];
                     start += step;
                 }
             })
        /// Comparisons
        .def(py::self == py::self)
        .def(py::self != py::self)
        // Could also define py::self + py::self for concatenation, etc.
        ;

    // test_map_iterator
    // Interface of a map-like object that isn't (directly) an unordered_map, but provides some
    // basic map-like functionality.
    class StringMap {
    public:
        StringMap() = default;
        explicit StringMap(std::unordered_map<std::string, std::string> init)
            : map(std::move(init)) {}

        void set(const std::string &key, std::string val) { map[key] = std::move(val); }
        std::string get(const std::string &key) const { return map.at(key); }
        size_t size() const { return map.size(); }

    private:
        std::unordered_map<std::string, std::string> map;

    public:
        decltype(map.cbegin()) begin() const { return map.cbegin(); }
        decltype(map.cend()) end() const { return map.cend(); }
    };
    py::class_<StringMap>(m, "StringMap")
        .def(py::init<>())
        .def(py::init<std::unordered_map<std::string, std::string>>())
        .def("__getitem__",
             [](const StringMap &map, const std::string &key) {
                 try {
                     return map.get(key);
                 } catch (const std::out_of_range &) {
                     throw py::key_error("key '" + key + "' does not exist");
                 }
             })
        .def("__setitem__", &StringMap::set)
        .def("__len__", &StringMap::size)
        .def(
            "__iter__",
            [](const StringMap &map) { return py::make_key_iterator(map.begin(), map.end()); },
            py::keep_alive<0, 1>())
        .def(
            "items",
            [](const StringMap &map) { return py::make_iterator(map.begin(), map.end()); },
            py::keep_alive<0, 1>())
        .def(
            "values",
            [](const StringMap &map) { return py::make_value_iterator(map.begin(), map.end()); },
            py::keep_alive<0, 1>());

    // test_generalized_iterators
    class IntPairs {
    public:
        explicit IntPairs(std::vector<std::pair<int, int>> data) : data_(std::move(data)) {}
        const std::pair<int, int> *begin() const { return data_.data(); }
        // .end() only required for py::make_iterator(self) overload
        const std::pair<int, int> *end() const { return data_.data() + data_.size(); }

    private:
        std::vector<std::pair<int, int>> data_;
    };

    {
        // #4383 : Make sure `py::make_*iterator` functions work with move-only iterators
        using iterator_t = NonZeroIterator<std::pair<int, int>>;

        static_assert(std::is_move_assignable<iterator_t>::value, "");
        static_assert(std::is_move_constructible<iterator_t>::value, "");
        static_assert(!std::is_copy_assignable<iterator_t>::value, "");
        static_assert(!std::is_copy_constructible<iterator_t>::value, "");
    }

    py::class_<IntPairs>(m, "IntPairs")
        .def(py::init<std::vector<std::pair<int, int>>>())
        .def(
            "nonzero",
            [](const IntPairs &s) {
                return py::make_iterator(NonZeroIterator<std::pair<int, int>>(s.begin()),
                                         NonZeroSentinel());
            },
            py::keep_alive<0, 1>())
        .def(
            "nonzero_keys",
            [](const IntPairs &s) {
                return py::make_key_iterator(NonZeroIterator<std::pair<int, int>>(s.begin()),
                                             NonZeroSentinel());
            },
            py::keep_alive<0, 1>())
        .def(
            "nonzero_values",
            [](const IntPairs &s) {
                return py::make_value_iterator(NonZeroIterator<std::pair<int, int>>(s.begin()),
                                               NonZeroSentinel());
            },
            py::keep_alive<0, 1>())

        // test iterator that returns values instead of references
        .def(
            "nonref",
            [](const IntPairs &s) {
                return py::make_iterator(NonRefIterator<std::pair<int, int>>(s.begin()),
                                         NonRefIterator<std::pair<int, int>>(s.end()));
            },
            py::keep_alive<0, 1>())
        .def(
            "nonref_keys",
            [](const IntPairs &s) {
                return py::make_key_iterator(NonRefIterator<std::pair<int, int>>(s.begin()),
                                             NonRefIterator<std::pair<int, int>>(s.end()));
            },
            py::keep_alive<0, 1>())
        .def(
            "nonref_values",
            [](const IntPairs &s) {
                return py::make_value_iterator(NonRefIterator<std::pair<int, int>>(s.begin()),
                                               NonRefIterator<std::pair<int, int>>(s.end()));
            },
            py::keep_alive<0, 1>())

        // test single-argument make_iterator
        .def(
            "simple_iterator",
            [](IntPairs &self) { return py::make_iterator(self); },
            py::keep_alive<0, 1>())
        .def(
            "simple_keys",
            [](IntPairs &self) { return py::make_key_iterator(self); },
            py::keep_alive<0, 1>())
        .def(
            "simple_values",
            [](IntPairs &self) { return py::make_value_iterator(self); },
            py::keep_alive<0, 1>())

        // Test iterator with an Extra (doesn't do anything useful, so not used
        // at runtime, but tests need to be able to compile with the correct
        // overload. See PR #3293.
        .def(
            "_make_iterator_extras",
            [](IntPairs &self) { return py::make_iterator(self, py::call_guard<int>()); },
            py::keep_alive<0, 1>())
        .def(
            "_make_key_extras",
            [](IntPairs &self) { return py::make_key_iterator(self, py::call_guard<int>()); },
            py::keep_alive<0, 1>())
        .def(
            "_make_value_extras",
            [](IntPairs &self) { return py::make_value_iterator(self, py::call_guard<int>()); },
            py::keep_alive<0, 1>());

    // test_iterator_referencing
    py::class_<NonCopyableInt>(m, "NonCopyableInt")
        .def(py::init<int>())
        .def("set", &NonCopyableInt::set)
        .def("__int__", &NonCopyableInt::get);
    py::class_<std::vector<NonCopyableInt>>(m, "VectorNonCopyableInt")
        .def(py::init<>())
        .def("append",
             [](std::vector<NonCopyableInt> &vec, int value) { vec.emplace_back(value); })
        .def("__iter__", [](std::vector<NonCopyableInt> &vec) {
            return py::make_iterator(vec.begin(), vec.end());
        });
    py::class_<std::vector<NonCopyableIntPair>>(m, "VectorNonCopyableIntPair")
        .def(py::init<>())
        .def("append",
             [](std::vector<NonCopyableIntPair> &vec, const std::pair<int, int> &value) {
                 vec.emplace_back(NonCopyableInt(value.first), NonCopyableInt(value.second));
             })
        .def("keys",
             [](std::vector<NonCopyableIntPair> &vec) {
                 return py::make_key_iterator(vec.begin(), vec.end());
             })
        .def("values", [](std::vector<NonCopyableIntPair> &vec) {
            return py::make_value_iterator(vec.begin(), vec.end());
        });

#if 0
    // Obsolete: special data structure for exposing custom iterator types to python
    // kept here for illustrative purposes because there might be some use cases which
    // are not covered by the much simpler py::make_iterator

    struct PySequenceIterator {
        PySequenceIterator(const Sequence &seq, py::object ref) : seq(seq), ref(ref) { }

        float next() {
            if (index == seq.size())
                throw py::stop_iteration();
            return seq[index++];
        }

        const Sequence &seq;
        py::object ref; // keep a reference
        size_t index = 0;
    };

    py::class_<PySequenceIterator>(seq, "Iterator")
        .def("__iter__", [](PySequenceIterator &it) -> PySequenceIterator& { return it; })
        .def("__next__", &PySequenceIterator::next);

    On the actual Sequence object, the iterator would be constructed as follows:
    .def("__iter__", [](py::object s) { return PySequenceIterator(s.cast<const Sequence &>(), s); })
#endif

    // test_python_iterator_in_cpp
    m.def("object_to_list", [](const py::object &o) {
        auto l = py::list();
        for (auto item : o) {
            l.append(item);
        }
        return l;
    });

    m.def("iterator_to_list", [](py::iterator it) {
        auto l = py::list();
        while (it != py::iterator::sentinel()) {
            l.append(*it);
            ++it;
        }
        return l;
    });

    // test_sequence_length: check that Python sequences can be converted to py::sequence.
    m.def("sequence_length", [](const py::sequence &seq) { return seq.size(); });

    // Make sure that py::iterator works with std algorithms
    m.def("count_none", [](const py::object &o) {
        return std::count_if(o.begin(), o.end(), [](py::handle h) { return h.is_none(); });
    });

    m.def("find_none", [](const py::object &o) {
        auto it = std::find_if(o.begin(), o.end(), [](py::handle h) { return h.is_none(); });
        return it->is_none();
    });

    m.def("count_nonzeros", [](const py::dict &d) {
        return std::count_if(d.begin(), d.end(), [](std::pair<py::handle, py::handle> p) {
            return p.second.cast<int>() != 0;
        });
    });

    m.def("tuple_iterator", &test_random_access_iterator<py::tuple>);
    m.def("list_iterator", &test_random_access_iterator<py::list>);
    m.def("sequence_iterator", &test_random_access_iterator<py::sequence>);

    // test_iterator_passthrough
    // #181: iterator passthrough did not compile
    m.def("iterator_passthrough", [](py::iterator s) -> py::iterator {
        return py::make_iterator(std::begin(s), std::end(s));
    });

    // test_iterator_rvp
    // #388: Can't make iterators via make_iterator() with different r/v policies
    static std::vector<int> list = {1, 2, 3};
    m.def("make_iterator_1",
          []() { return py::make_iterator<py::return_value_policy::copy>(list); });
    m.def("make_iterator_2",
          []() { return py::make_iterator<py::return_value_policy::automatic>(list); });

    // test_iterator on c arrays
    // #4100: ensure lvalue required as increment operand
    class CArrayHolder {
    public:
        CArrayHolder(double x, double y, double z) {
            values[0] = x;
            values[1] = y;
            values[2] = z;
        };
        double values[3];
    };

    py::class_<CArrayHolder>(m, "CArrayHolder")
        .def(py::init<double, double, double>())
        .def(
            "__iter__",
            [](const CArrayHolder &v) { return py::make_iterator(v.values, v.values + 3); },
            py::keep_alive<0, 1>());
}
