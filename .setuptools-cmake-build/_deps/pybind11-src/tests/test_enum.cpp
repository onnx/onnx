/*
    tests/test_enums.cpp -- enumerations

    Copyright (c) 2016 Wenzel Jakob <wenzel.jakob@epfl.ch>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "pybind11_tests.h"

TEST_SUBMODULE(enums, m) {
    // test_unscoped_enum
    enum UnscopedEnum { EOne = 1, ETwo, EThree };
    py::enum_<UnscopedEnum>(m, "UnscopedEnum", py::arithmetic(), "An unscoped enumeration")
        .value("EOne", EOne, "Docstring for EOne")
        .value("ETwo", ETwo, "Docstring for ETwo")
        .value("EThree", EThree, "Docstring for EThree")
        .export_values();

    // test_scoped_enum
    enum class ScopedEnum { Two = 2, Three };
    py::enum_<ScopedEnum>(m, "ScopedEnum", py::arithmetic())
        .value("Two", ScopedEnum::Two)
        .value("Three", ScopedEnum::Three);

    m.def("test_scoped_enum", [](ScopedEnum z) {
        return "ScopedEnum::" + std::string(z == ScopedEnum::Two ? "Two" : "Three");
    });

    // test_binary_operators
    enum Flags { Read = 4, Write = 2, Execute = 1 };
    py::enum_<Flags>(m, "Flags", py::arithmetic())
        .value("Read", Flags::Read)
        .value("Write", Flags::Write)
        .value("Execute", Flags::Execute)
        .export_values();

    // test_implicit_conversion
    class ClassWithUnscopedEnum {
    public:
        enum EMode { EFirstMode = 1, ESecondMode };

        static EMode test_function(EMode mode) { return mode; }
    };
    py::class_<ClassWithUnscopedEnum> exenum_class(m, "ClassWithUnscopedEnum");
    exenum_class.def_static("test_function", &ClassWithUnscopedEnum::test_function);
    py::enum_<ClassWithUnscopedEnum::EMode>(exenum_class, "EMode")
        .value("EFirstMode", ClassWithUnscopedEnum::EFirstMode)
        .value("ESecondMode", ClassWithUnscopedEnum::ESecondMode)
        .export_values();

    // test_enum_to_int
    m.def("test_enum_to_int", [](int) {});
    m.def("test_enum_to_uint", [](uint32_t) {});
    m.def("test_enum_to_long_long", [](long long) {});

    // test_duplicate_enum_name
    enum SimpleEnum { ONE, TWO, THREE };

    m.def("register_bad_enum", [m]() {
        py::enum_<SimpleEnum>(m, "SimpleEnum")
            .value("ONE", SimpleEnum::ONE) // NOTE: all value function calls are called with the
                                           // same first parameter value
            .value("ONE", SimpleEnum::TWO)
            .value("ONE", SimpleEnum::THREE)
            .export_values();
    });

    // test_enum_scalar
    enum UnscopedUCharEnum : unsigned char {};
    enum class ScopedShortEnum : short {};
    enum class ScopedLongEnum : long {};
    enum UnscopedUInt64Enum : std::uint64_t {};
    static_assert(
        py::detail::all_of<
            std::is_same<py::enum_<UnscopedUCharEnum>::Scalar, unsigned char>,
            std::is_same<py::enum_<ScopedShortEnum>::Scalar, short>,
            std::is_same<py::enum_<ScopedLongEnum>::Scalar, long>,
            std::is_same<py::enum_<UnscopedUInt64Enum>::Scalar, std::uint64_t>>::value,
        "Error during the deduction of enum's scalar type with normal integer underlying");

    // test_enum_scalar_with_char_underlying
    enum class ScopedCharEnum : char { Zero, Positive };
    enum class ScopedWCharEnum : wchar_t { Zero, Positive };
    enum class ScopedChar32Enum : char32_t { Zero, Positive };
    enum class ScopedChar16Enum : char16_t { Zero, Positive };

    // test the scalar of char type enums according to chapter 'Character types'
    // from https://en.cppreference.com/w/cpp/language/types
    static_assert(
        py::detail::any_of<
            std::is_same<py::enum_<ScopedCharEnum>::Scalar, signed char>,  // e.g. gcc on x86
            std::is_same<py::enum_<ScopedCharEnum>::Scalar, unsigned char> // e.g. arm linux
            >::value,
        "char should be cast to either signed char or unsigned char");
    static_assert(sizeof(py::enum_<ScopedWCharEnum>::Scalar) == 2
                      || sizeof(py::enum_<ScopedWCharEnum>::Scalar) == 4,
                  "wchar_t should be either 16 bits (Windows) or 32 (everywhere else)");
    static_assert(
        py::detail::all_of<
            std::is_same<py::enum_<ScopedChar32Enum>::Scalar, std::uint_least32_t>,
            std::is_same<py::enum_<ScopedChar16Enum>::Scalar, std::uint_least16_t>>::value,
        "char32_t, char16_t (and char8_t)'s size, signedness, and alignment is determined");
#if defined(PYBIND11_HAS_U8STRING)
    enum class ScopedChar8Enum : char8_t { Zero, Positive };
    static_assert(std::is_same<py::enum_<ScopedChar8Enum>::Scalar, unsigned char>::value);
#endif

    // test_char_underlying_enum
    py::enum_<ScopedCharEnum>(m, "ScopedCharEnum")
        .value("Zero", ScopedCharEnum::Zero)
        .value("Positive", ScopedCharEnum::Positive);
    py::enum_<ScopedWCharEnum>(m, "ScopedWCharEnum")
        .value("Zero", ScopedWCharEnum::Zero)
        .value("Positive", ScopedWCharEnum::Positive);
    py::enum_<ScopedChar32Enum>(m, "ScopedChar32Enum")
        .value("Zero", ScopedChar32Enum::Zero)
        .value("Positive", ScopedChar32Enum::Positive);
    py::enum_<ScopedChar16Enum>(m, "ScopedChar16Enum")
        .value("Zero", ScopedChar16Enum::Zero)
        .value("Positive", ScopedChar16Enum::Positive);

    // test_bool_underlying_enum
    enum class ScopedBoolEnum : bool { FALSE, TRUE };

    // bool is unsigned (std::is_signed returns false) and 1-byte long, so represented with u8
    static_assert(std::is_same<py::enum_<ScopedBoolEnum>::Scalar, std::uint8_t>::value, "");

    py::enum_<ScopedBoolEnum>(m, "ScopedBoolEnum")
        .value("FALSE", ScopedBoolEnum::FALSE)
        .value("TRUE", ScopedBoolEnum::TRUE);
}
