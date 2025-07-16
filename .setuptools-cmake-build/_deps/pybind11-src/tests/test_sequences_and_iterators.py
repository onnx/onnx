from __future__ import annotations

import pytest
from pytest import approx  # noqa: PT013

from pybind11_tests import ConstructorStats
from pybind11_tests import sequences_and_iterators as m


def test_slice_constructors():
    assert m.make_forward_slice_size_t() == slice(0, -1, 1)
    assert m.make_reversed_slice_object() == slice(None, None, -1)


@pytest.mark.skipif(not m.has_optional, reason="no <optional>")
def test_slice_constructors_explicit_optional():
    assert m.make_reversed_slice_size_t_optional() == slice(None, None, -1)
    assert m.make_reversed_slice_size_t_optional_verbose() == slice(None, None, -1)


def test_generalized_iterators():
    assert list(m.IntPairs([(1, 2), (3, 4), (0, 5)]).nonzero()) == [(1, 2), (3, 4)]
    assert list(m.IntPairs([(1, 2), (2, 0), (0, 3), (4, 5)]).nonzero()) == [(1, 2)]
    assert list(m.IntPairs([(0, 3), (1, 2), (3, 4)]).nonzero()) == []

    assert list(m.IntPairs([(1, 2), (3, 4), (0, 5)]).nonzero_keys()) == [1, 3]
    assert list(m.IntPairs([(1, 2), (2, 0), (0, 3), (4, 5)]).nonzero_keys()) == [1]
    assert list(m.IntPairs([(0, 3), (1, 2), (3, 4)]).nonzero_keys()) == []

    assert list(m.IntPairs([(1, 2), (3, 4), (0, 5)]).nonzero_values()) == [2, 4]
    assert list(m.IntPairs([(1, 2), (2, 0), (0, 3), (4, 5)]).nonzero_values()) == [2]
    assert list(m.IntPairs([(0, 3), (1, 2), (3, 4)]).nonzero_values()) == []

    # __next__ must continue to raise StopIteration
    it = m.IntPairs([(0, 0)]).nonzero()
    for _ in range(3):
        with pytest.raises(StopIteration):
            next(it)

    it = m.IntPairs([(0, 0)]).nonzero_keys()
    for _ in range(3):
        with pytest.raises(StopIteration):
            next(it)


def test_nonref_iterators():
    pairs = m.IntPairs([(1, 2), (3, 4), (0, 5)])
    assert list(pairs.nonref()) == [(1, 2), (3, 4), (0, 5)]
    assert list(pairs.nonref_keys()) == [1, 3, 0]
    assert list(pairs.nonref_values()) == [2, 4, 5]


def test_generalized_iterators_simple():
    assert list(m.IntPairs([(1, 2), (3, 4), (0, 5)]).simple_iterator()) == [
        (1, 2),
        (3, 4),
        (0, 5),
    ]
    assert list(m.IntPairs([(1, 2), (3, 4), (0, 5)]).simple_keys()) == [1, 3, 0]
    assert list(m.IntPairs([(1, 2), (3, 4), (0, 5)]).simple_values()) == [2, 4, 5]


def test_iterator_doc_annotations():
    assert m.IntPairs.nonref.__doc__.endswith("-> Iterator[tuple[int, int]]\n")
    assert m.IntPairs.nonref_keys.__doc__.endswith("-> Iterator[int]\n")
    assert m.IntPairs.nonref_values.__doc__.endswith("-> Iterator[int]\n")
    assert m.IntPairs.simple_iterator.__doc__.endswith("-> Iterator[tuple[int, int]]\n")
    assert m.IntPairs.simple_keys.__doc__.endswith("-> Iterator[int]\n")
    assert m.IntPairs.simple_values.__doc__.endswith("-> Iterator[int]\n")


def test_iterator_referencing():
    """Test that iterators reference rather than copy their referents."""
    vec = m.VectorNonCopyableInt()
    vec.append(3)
    vec.append(5)
    assert [int(x) for x in vec] == [3, 5]
    # Increment everything to make sure the referents can be mutated
    for x in vec:
        x.set(int(x) + 1)
    assert [int(x) for x in vec] == [4, 6]

    vec = m.VectorNonCopyableIntPair()
    vec.append([3, 4])
    vec.append([5, 7])
    assert [int(x) for x in vec.keys()] == [3, 5]
    assert [int(x) for x in vec.values()] == [4, 7]
    for x in vec.keys():
        x.set(int(x) + 1)
    for x in vec.values():
        x.set(int(x) + 10)
    assert [int(x) for x in vec.keys()] == [4, 6]
    assert [int(x) for x in vec.values()] == [14, 17]


def test_sliceable():
    sliceable = m.Sliceable(100)
    assert sliceable[::] == (0, 100, 1)
    assert sliceable[10::] == (10, 100, 1)
    assert sliceable[:10:] == (0, 10, 1)
    assert sliceable[::10] == (0, 100, 10)
    assert sliceable[-10::] == (90, 100, 1)
    assert sliceable[:-10:] == (0, 90, 1)
    assert sliceable[::-10] == (99, -1, -10)
    assert sliceable[50:60:1] == (50, 60, 1)
    assert sliceable[50:60:-1] == (50, 60, -1)


def test_sequence():
    cstats = ConstructorStats.get(m.Sequence)

    s = m.Sequence(5)
    assert cstats.values() == ["of size", "5"]

    assert "Sequence" in repr(s)
    assert len(s) == 5
    assert s[0] == 0
    assert s[3] == 0
    assert 12.34 not in s
    s[0], s[3] = 12.34, 56.78
    assert 12.34 in s
    assert s[0] == approx(12.34, rel=1e-05)
    assert s[3] == approx(56.78, rel=1e-05)

    rev = reversed(s)
    assert cstats.values() == ["of size", "5"]

    rev2 = s[::-1]
    assert cstats.values() == ["of size", "5"]

    it = iter(m.Sequence(0))
    for _ in range(3):  # __next__ must continue to raise StopIteration
        with pytest.raises(StopIteration):
            next(it)
    assert cstats.values() == ["of size", "0"]

    expected = [0, 56.78, 0, 0, 12.34]
    assert rev == approx(expected, rel=1e-05)
    assert rev2 == approx(expected, rel=1e-05)
    assert rev == rev2

    rev[0::2] = m.Sequence([2.0, 2.0, 2.0])
    assert cstats.values() == ["of size", "3", "from std::vector"]

    assert rev == approx([2, 56.78, 2, 0, 2], rel=1e-05)

    assert cstats.alive() == 4
    del it
    assert cstats.alive() == 3
    del s
    assert cstats.alive() == 2
    del rev
    assert cstats.alive() == 1
    del rev2
    assert cstats.alive() == 0

    assert cstats.values() == []
    assert cstats.default_constructions == 0
    assert cstats.copy_constructions == 0
    assert cstats.move_constructions >= 1
    assert cstats.copy_assignments == 0
    assert cstats.move_assignments == 0


def test_sequence_length():
    """#2076: Exception raised by len(arg) should be propagated"""

    class BadLen(RuntimeError):
        pass

    class SequenceLike:
        def __getitem__(self, i):
            return None

        def __len__(self):
            raise BadLen()

    with pytest.raises(BadLen):
        m.sequence_length(SequenceLike())

    assert m.sequence_length([1, 2, 3]) == 3
    assert m.sequence_length("hello") == 5


def test_sequence_doc():
    assert m.sequence_length.__doc__.strip() == "sequence_length(arg0: Sequence) -> int"


def test_map_iterator():
    sm = m.StringMap({"hi": "bye", "black": "white"})
    assert sm["hi"] == "bye"
    assert len(sm) == 2
    assert sm["black"] == "white"

    with pytest.raises(KeyError):
        assert sm["orange"]
    sm["orange"] = "banana"
    assert sm["orange"] == "banana"

    expected = {"hi": "bye", "black": "white", "orange": "banana"}
    for k in sm:
        assert sm[k] == expected[k]
    for k, v in sm.items():
        assert v == expected[k]
    assert list(sm.values()) == [expected[k] for k in sm]

    it = iter(m.StringMap({}))
    for _ in range(3):  # __next__ must continue to raise StopIteration
        with pytest.raises(StopIteration):
            next(it)


def test_python_iterator_in_cpp():
    t = (1, 2, 3)
    assert m.object_to_list(t) == [1, 2, 3]
    assert m.object_to_list(iter(t)) == [1, 2, 3]
    assert m.iterator_to_list(iter(t)) == [1, 2, 3]

    with pytest.raises(TypeError) as excinfo:
        m.object_to_list(1)
    assert "object is not iterable" in str(excinfo.value)

    with pytest.raises(TypeError) as excinfo:
        m.iterator_to_list(1)
    assert "incompatible function arguments" in str(excinfo.value)

    def bad_next_call():
        raise RuntimeError("py::iterator::advance() should propagate errors")

    with pytest.raises(RuntimeError) as excinfo:
        m.iterator_to_list(iter(bad_next_call, None))
    assert str(excinfo.value) == "py::iterator::advance() should propagate errors"

    lst = [1, None, 0, None]
    assert m.count_none(lst) == 2
    assert m.find_none(lst) is True
    assert m.count_nonzeros({"a": 0, "b": 1, "c": 2}) == 2

    r = range(5)
    assert all(m.tuple_iterator(tuple(r)))
    assert all(m.list_iterator(list(r)))
    assert all(m.sequence_iterator(r))


def test_iterator_passthrough():
    """#181: iterator passthrough did not compile"""
    from pybind11_tests.sequences_and_iterators import iterator_passthrough

    values = [3, 5, 7, 9, 11, 13, 15]
    assert list(iterator_passthrough(iter(values))) == values


def test_iterator_rvp():
    """#388: Can't make iterators via make_iterator() with different r/v policies"""
    import pybind11_tests.sequences_and_iterators as m

    assert list(m.make_iterator_1()) == [1, 2, 3]
    assert list(m.make_iterator_2()) == [1, 2, 3]
    assert not isinstance(m.make_iterator_1(), type(m.make_iterator_2()))


def test_carray_iterator():
    """#4100: Check for proper iterator overload with C-Arrays"""
    args_gt = [float(i) for i in range(3)]
    arr_h = m.CArrayHolder(*args_gt)
    args = list(arr_h)
    assert args_gt == args
