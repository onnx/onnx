// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string_view>

#include "gtest/gtest.h"
#include "onnx/checker.h"
#include "onnx/defs/parser.h"
#include "onnx/version_converter/convert.h"

namespace ONNX_NAMESPACE::Test {

static ModelProto ParseVersionConverterModel(std::string_view model_text) {
  ModelProto model;
  auto status = OnnxParser::Parse(model, model_text);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  return model;
}

#ifndef ONNX_NO_EXCEPTIONS
TEST(VersionConverterTest, RejectsSignedBitShiftIntermediateWithoutValueInfo) {
  auto model = ParseVersionConverterModel(R"ONNX(
    <
      ir_version: 13,
      opset_import: ["" : 28]
    >
    bitshift (float[2] X, float[2] Y) => (float[2] Z) {
      X_int = Cast <to = 6> (X)
      Y_int = Cast <to = 6> (Y)
      shifted = BitShift <direction = "RIGHT"> (X_int, Y_int)
      Z = Cast <to = 1> (shifted)
    }
  )ONNX");

  EXPECT_THROW(version_conversion::ConvertVersion(model, 27), assert_error);
}
#endif

TEST(VersionConverterTest, AllowsUnsignedBitShiftIntermediateWithoutValueInfo) {
  auto model = ParseVersionConverterModel(R"ONNX(
    <
      ir_version: 13,
      opset_import: ["" : 28]
    >
    bitshift (float[2] X, float[2] Y) => (float[2] Z) {
      X_int = Cast <to = 12> (X)
      Y_int = Cast <to = 12> (Y)
      shifted = BitShift <direction = "RIGHT"> (X_int, Y_int)
      Z = Cast <to = 1> (shifted)
    }
  )ONNX");

  auto converted = version_conversion::ConvertVersion(model, 27);
  EXPECT_EQ(converted.opset_import(0).version(), 27);
  checker::check_model(converted, true, true);
}

} // namespace ONNX_NAMESPACE::Test
