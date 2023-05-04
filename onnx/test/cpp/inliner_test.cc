// Copyright (c) ONNX Project Contributors

/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include "gtest/gtest.h"
#include "onnx/checker.h"
#include "onnx/common/constants.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/printer.h"
#include "onnx/defs/schema.h"
#include "onnx/inliner/inliner.h"
#include "onnx/shape_inference/implementation.h"

namespace ONNX_NAMESPACE {
namespace Test {

static void InlineFunctions(ModelProto& model, const char* input) {
  OnnxParser parser(input);
  auto status = parser.Parse(model);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  checker::check_model(model);
  shape_inference::InferShapes(model);

  // std::cout << ProtoToString(model) << "\n";
  inliner::InlineLocalFunctions(model);
  // std::cout << ProtoToString(model) << "\n";

  // The following will ensure basic sanity checks hold after inlining, including
  // absence of duplicate names (multiple assignments to same name).
  checker::check_model(model);
  shape_inference::InferShapes(model);
}

TEST(FunctionInliner, BasicTest) {
  const char* code = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 10, "local" : 1 ]
>
agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N, 10] C)
{
  T = local.foo (X, W, B)
  C = local.square(T)
}

<
  opset_import: [ "" : 10 ],
  domain: "local",
  doc_string: "Function foo."
>
foo (x, w, b) => (c) {
  T = MatMul(x, w)
  S = Add(T, b)
  c = Softmax(S)
}

<
  opset_import: [ "" : 10 ],
  domain: "local",
  doc_string: "Function square."
>
square (x) => (y) {
  y = Mul (x, x)
}
)ONNX";

  ModelProto model;
  InlineFunctions(model, code);
  auto num_nodes = model.graph().node_size();
  ASSERT_EQ(num_nodes, 4);
  auto num_functions = model.functions_size();
  ASSERT_EQ(num_functions, 0);
}

TEST(FunctionInliner, Nested) {
  const char* code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  Y = local.foo (X)
}

<opset_import: [ "" : 17, "local" : 1 ], domain: "local">
foo (x) => (y) {
  temp = Add(x, x)
  y = local.bar(temp)
}

<opset_import: [ "" : 17 ], domain: "local">
bar (x) => (y) {
  y = Mul (x, x)
}
)ONNX";

  ModelProto model;
  InlineFunctions(model, code);
  auto num_nodes = model.graph().node_size();
  ASSERT_EQ(num_nodes, 2);
  auto num_functions = model.functions_size();
  ASSERT_EQ(num_functions, 0);
}

TEST(FunctionInliner, Renaming) {
  const char* code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  temp = local.foo (X)
  temp__1 = Mul (temp, temp)
  Y = Abs (temp__1)
}

<opset_import: [ "" : 17, "local" : 1 ], domain: "local">
foo (x) => (y) {
  temp = Add(x, x)
  y = Neg (temp)
}
)ONNX";

  ModelProto model;
  // Check that renaming handles accidental collision of names: when "temp" in "foo" is
  // inlined, it will be renamed into something distinct from "temp" and "temp__1" as
  // both these names occur in the main graph.
  InlineFunctions(model, code);
}

TEST(FunctionInliner, TwoCallsToSameFunction) {
  const char* code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  temp = local.foo (X)
  Y = local.foo (temp)
}

<opset_import: [ "" : 17, "local" : 1 ], domain: "local">
foo (x) => (y) {
  temp = Add(x, x)
  y = Neg (temp)
}
)ONNX";

  ModelProto model;
  // The call below will check that multiple assignments to same name does not happen
  // after inlining two calls to same function.
  InlineFunctions(model, code);
}

TEST(FunctionInliner, OpsetMismatch) {
  const char* code = R"ONNX(
<ir_version: 8, opset_import: [ "" : 17, "local" : 1 ]>
agraph (float[N] X) => (float[N] Y)
{
  temp = local.foo (X)
  Y = local.bar (temp)
}

<opset_import: [ "" : 18], domain: "local">
foo (x) => (y) {
  y = Add(x, x)
}

<opset_import: [ "" : 17], domain: "local">
bar (x) => (y) {
  y = Add(x, x)
}
)ONNX";

  ModelProto model;
  InlineFunctions(model, code);

  // The first node's call, to foo, must not be inlined.
  auto& first_node = model.graph().node(0);
  // Check that it is still a call to foo
  ASSERT_EQ(first_node.op_type(), "foo");

  // The second node's call, to bar, must be inlined.
  auto& second_node = model.graph().node(1);
  // Check that it is a call to Add
  ASSERT_EQ(second_node.op_type(), "Add");

  // The non-inlined foo must still be in the function list.
  ASSERT_EQ(model.functions_size(), 1);
  ASSERT_EQ(model.functions(0).name(), "foo");
}

} // namespace Test
} // namespace ONNX_NAMESPACE
