// Copyright (c) 2024 The pybind Community.

#pragma once

#include <string>

namespace pybind11_tests {
namespace test_cpp_conduit {

struct Traveler {
    explicit Traveler(const std::string &luggage) : luggage(luggage) {}
    std::string luggage;
};

struct PremiumTraveler : Traveler {
    explicit PremiumTraveler(const std::string &luggage, int points)
        : Traveler(luggage), points(points) {}
    int points;
};

struct LonelyTraveler {};
struct VeryLonelyTraveler : LonelyTraveler {};

} // namespace test_cpp_conduit
} // namespace pybind11_tests
