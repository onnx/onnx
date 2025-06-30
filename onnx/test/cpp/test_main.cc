// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>

#include "gtest/gtest.h"

int main(int argc, char** argv) {
  std::cout << "Running main() from test_main.cc" << '\n';
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
