// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "onnx/checker.h"
#include "onnx/defs/function.h"
#include "onnx/defs/parser.h"
#include "onnx/inliner/inliner.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/version_converter/convert.h"

namespace ONNX_NAMESPACE::Test {

#ifndef ONNX_NO_EXCEPTIONS
TEST(OpsetVersionRange, SchemaFunctionInliner) {
  const char* code = R"(
    <ir_version: 8, opset_import: ["" : 18]>
    softmax_graph (float[2] X) => (float[2] Y) { Y = Softmax(X) }
  )";
  ModelProto model;
  const auto status = OnnxParser::Parse(model, code);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  auto selected = inliner::FunctionIdSet::Create({{"", "Softmax"}});
  for (int64_t version : {int64_t{18}, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()}) {
    model.mutable_opset_import(0)->set_version(version);
    ModelProto decoded = model;
    // Valid representable-version control is exercised before range errors.
    if (version == 18) {
      EXPECT_NO_THROW(inliner::InlineSelectedFunctions(decoded, *selected, OpSchemaRegistry::Instance()));
      EXPECT_GT(decoded.graph().node_size(), 1);
    } else {
      EXPECT_THROW(
          inliner::InlineSelectedFunctions(decoded, *selected, OpSchemaRegistry::Instance()), checker::ValidationError);
    }
  }
}

TEST(OpsetVersionRange, InlinerConversionTarget) {
  const char* code = R"(
    <ir_version: 8, opset_import: ["" : 10, "local" : 1]>
    local_graph (float[2] X) => (float[2] Y) { Y = local.copy(X) }
    <opset_import: ["" : 10], domain: "local">
    copy (X) => (Y) { Y = Identity(X) }
  )";
  ModelProto model;
  const auto status = OnnxParser::Parse(model, code);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  model.mutable_opset_import(0)->set_version(std::numeric_limits<int64_t>::max());
  ModelProto decoded = model;
  EXPECT_THROW(inliner::InlineLocalFunctions(decoded, true), checker::ValidationError);
}
TEST(OpsetVersionRange, ShapeInference) {
  const char* code = R"(
    <ir_version: 8, opset_import: ["" : 18]>
    identity_graph (float[2] X) => (float[] Y) { Y = Identity(X) }
  )";
  ModelProto model;
  const auto status = OnnxParser::Parse(model, code);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  for (int64_t version : {int64_t{18}, std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()}) {
    model.mutable_opset_import(0)->set_version(version);
    ModelProto decoded = model;
    if (version == 18) {
      EXPECT_NO_THROW(shape_inference::InferShapes(decoded));
      const auto& shape = decoded.graph().output(0).type().tensor_type().shape();
      ASSERT_EQ(shape.dim_size(), 1);
      EXPECT_EQ(shape.dim(0).dim_value(), 2);
    } else {
      EXPECT_THROW(shape_inference::InferShapes(decoded), checker::ValidationError);
    }
  }
}

TEST(OpsetVersionRange, SharedHelperBoundsAndDomain) {
  for (int version : {std::numeric_limits<int>::min(), -1, 0, 18, std::numeric_limits<int>::max()}) {
    EXPECT_EQ(checker::checked_opset_version(version, "com.vendor"), version);
  }
  for (int64_t version :
       {std::numeric_limits<int64_t>::min(),
        int64_t{std::numeric_limits<int>::min()} - 1,
        int64_t{std::numeric_limits<int>::max()} + 1,
        std::numeric_limits<int64_t>::max()}) {
    try {
      checker::checked_opset_version(version, "com.vendor");
      FAIL() << "Expected an out-of-range version to be rejected";
    } catch (const checker::ValidationError& error) {
      EXPECT_NE(std::string(error.what()).find("com.vendor"), std::string::npos);
      EXPECT_NE(std::string(error.what()).find(std::to_string(version)), std::string::npos);
    }
  }
}

TEST(OpsetVersionRange, LocalFunctionImportsBeforeMutation) {
  const char* code = R"(
    <ir_version: 8, opset_import: ["" : 18, "local" : 1]>
    local_graph (float[2] X) => (float[2] Y) { Y = local.copy(X) }
    <opset_import: ["" : 18], domain: "local">
    copy (X) => (Y) { Y = Identity(X) }
  )";
  ModelProto model;
  const auto status = OnnxParser::Parse(model, code);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  auto selected = inliner::FunctionIdSet::Create({{"local", "copy"}});
  auto unselected = inliner::FunctionIdSet::Create({});
  for (bool unused_domain : {false, true}) {
    for (int64_t version : {std::numeric_limits<int64_t>::min(), std::numeric_limits<int64_t>::max()}) {
      ModelProto invalid = model;
      auto* function = invalid.mutable_functions(0);
      auto* import = unused_domain ? function->add_opset_import() : function->mutable_opset_import(0);
      import->set_domain(unused_domain ? "com.vendor" : "");
      import->set_version(version);
      const auto before = invalid.SerializeAsString();
      for (bool convert_version : {false, true}) {
        ModelProto copy = invalid;
        EXPECT_THROW(inliner::InlineLocalFunctions(copy, convert_version), checker::ValidationError);
        EXPECT_EQ(copy.SerializeAsString(), before);
      }
      for (const auto* selection : {selected.get(), unselected.get()}) {
        ModelProto copy = invalid;
        EXPECT_THROW(inliner::InlineSelectedFunctions(copy, *selection), checker::ValidationError);
        EXPECT_EQ(copy.SerializeAsString(), before);
      }
      // Both inference modes reject local imports before per-node handling.
      for (int error_mode : {0, 1}) {
        ModelProto copy = invalid;
        EXPECT_THROW(
            shape_inference::InferShapes(copy, OpSchemaRegistry::Instance(), ShapeInferenceOptions(false, error_mode)),
            checker::ValidationError);
        EXPECT_EQ(copy.SerializeAsString(), before);
      }
      EXPECT_THROW(version_conversion::ConvertVersion(invalid, int64_t{19}), checker::ValidationError);
      // The same policy applies when the function is never called.
      invalid.mutable_graph()->mutable_node(0)->set_op_type("Identity");
      invalid.mutable_graph()->mutable_node(0)->clear_domain();
      EXPECT_THROW(shape_inference::InferShapes(invalid), checker::ValidationError);
    }
  }
  ModelProto converted = model;
  converted.mutable_functions(0)->mutable_opset_import(0)->set_version(17);
  EXPECT_NO_THROW(inliner::InlineLocalFunctions(converted, true));
  ASSERT_EQ(converted.graph().node_size(), 1);
  EXPECT_EQ(converted.graph().node(0).op_type(), "Identity");
  EXPECT_EQ(converted.functions_size(), 0);
}

TEST(OpsetVersionRange, UnusedModelDomain) {
  const char* code = R"(
    <ir_version: 8, opset_import: ["" : 18]>
    identity_graph (float[2] X) => (float[2] Y) { Y = Identity(X) }
  )";
  ModelProto model;
  const auto status = OnnxParser::Parse(model, code);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  auto* import = model.add_opset_import();
  import->set_domain("com.vendor");
  import->set_version(std::numeric_limits<int64_t>::max());
  const auto before = model.SerializeAsString();
  EXPECT_THROW(checker::check_model(model), checker::ValidationError);
  EXPECT_THROW(shape_inference::InferShapes(model), checker::ValidationError);
  EXPECT_THROW(inliner::InlineLocalFunctions(model, true), checker::ValidationError);
  auto selected = inliner::FunctionIdSet::Create({});
  EXPECT_THROW(inliner::InlineSelectedFunctions(model, *selected), checker::ValidationError);
  EXPECT_THROW(version_conversion::ConvertVersion(model, int64_t{19}), checker::ValidationError);
  EXPECT_EQ(model.SerializeAsString(), before);
}

TEST(OpsetVersionRange, VersionConversionInt64AndLegacyEntryPoints) {
  const char* code = R"(
    <ir_version: 8, opset_import: ["" : 18]>
    identity_graph (float[2] X) => (float[2] Y) { Y = Identity(X) }
  )";
  ModelProto model;
  const auto status = OnnxParser::Parse(model, code);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  ModelProto (*legacy)(const ModelProto&, int) = &version_conversion::ConvertVersion;
  ModelProto (*wide)(const ModelProto&, int64_t) = &version_conversion::ConvertVersion;
  EXPECT_EQ(legacy(model, 19).opset_import(0).version(), 19);
  EXPECT_EQ(wide(model, int64_t{19}).opset_import(0).version(), 19);
  version_conversion::DefaultVersionConverter converter;
  for (int64_t version :
       {std::numeric_limits<int64_t>::min(),
        int64_t{std::numeric_limits<int>::min()} - 1,
        int64_t{std::numeric_limits<int>::max()} + 1,
        std::numeric_limits<int64_t>::max()}) {
    EXPECT_THROW(wide(model, version), checker::ValidationError);
    EXPECT_THROW(converter.convert_version(model, OpSetID(18), OpSetID(version)), checker::ValidationError);
    ModelProto invalid = model;
    invalid.mutable_opset_import(0)->set_version(version);
    EXPECT_THROW(wide(invalid, int64_t{19}), checker::ValidationError);
    EXPECT_THROW(converter.convert_version(invalid, OpSetID(version), OpSetID(19)), checker::ValidationError);
  }
}

TEST(OpsetVersionRange, FunctionExpansion) {
  const auto* schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 18, "");
  ASSERT_NE(schema, nullptr);
  ASSERT_NE(schema->GetFunction(), nullptr);
  FunctionProto function = *schema->GetFunction();
  NodeProto node;
  node.set_op_type("MeanVarianceNormalization");
  node.add_input("X");
  node.add_output("Y");
  GraphProto graph;
  EXPECT_NO_THROW(FunctionExpandHelper(node, function, graph, "test"));
  EXPECT_GT(graph.node_size(), 0);
  const auto before = graph.SerializeAsString();
  for (bool unused_domain : {false, true}) {
    FunctionProto invalid = function;
    auto* import = unused_domain ? invalid.add_opset_import() : invalid.mutable_opset_import(0);
    import->set_domain(unused_domain ? "com.vendor" : "");
    import->set_version(std::numeric_limits<int64_t>::max());
    EXPECT_THROW(FunctionExpandHelper(node, invalid, graph, "test"), checker::ValidationError);
    EXPECT_EQ(graph.SerializeAsString(), before);
  }
}

TEST(OpsetVersionRange, FunctionDeterminism) {
  NodeProto node;
  node.set_op_type("Identity");
  node.add_input("X");
  node.add_output("Y");
  OperatorSetIdProto import;
  import.set_version(18);
  OpSchema schema;
  schema.SetName("LocalIdentity").SetDomain("local").SinceVersion(18).FunctionBody({node}, {import});
  EXPECT_EQ(schema.GetNodeDeterminism(), OpSchema::NodeDeterminism::Deterministic);
  for (bool unused_domain : {false, true}) {
    auto imports = std::vector<OperatorSetIdProto>{import};
    if (unused_domain) {
      imports.emplace_back();
      imports.back().set_domain("com.vendor");
    }
    imports.back().set_version(std::numeric_limits<int64_t>::max());
    OpSchema invalid;
    invalid.SetName("InvalidLocalIdentity").SetDomain("local").SinceVersion(18).FunctionBody({node}, imports);
    // Determinism lookup preserves its existing non-validating behavior.
    EXPECT_NO_THROW(invalid.GetNodeDeterminism());
    if (unused_domain) {
      EXPECT_EQ(invalid.GetNodeDeterminism(), OpSchema::NodeDeterminism::Deterministic);
    }
  }
}
#endif // ONNX_NO_EXCEPTIONS

} // namespace ONNX_NAMESPACE::Test
