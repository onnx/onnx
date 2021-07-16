/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include "gtest/gtest.h"
#include "onnx/checker.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/shape_inference.h"
#include "onnx/onnx_pb.h"

#include "onnx/shape_inference/implementation.h"

using namespace ONNX_NAMESPACE::shape_inference;

namespace ONNX_NAMESPACE {

namespace Test {

inline bool CompareShape(const TensorShapeProto* A, const TensorShapeProto* B) {
    if (A == nullptr ||  B == nullptr) {
        fail_check("The compared shapes should not be nullptr.");
        return false;
    }
    if (A->dim_size() != B->dim_size()) {
        fail_check("The compared sizes of dim are different.");
        return false;
    }
    for (int i = 0; i < A->dim_size() ; ++i) {
        if (A->dim(i).has_dim_value() != B->dim(i).has_dim_value() || 
                A->dim(i).dim_value() != B->dim(i).dim_value()) {
            fail_check("The compared dim values are different.");
            return false;
        }
    }
    return true;
}

void RunDataPropagationTest(const char* graphCode, int domain_version = 15) {
  // Parses the graph from graphCode
  GraphProto graph;
  OnnxParser parser(graphCode);
  auto status = parser.Parse(graph);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_TRUE(parser.EndOfInput()) << "Extra unparsed input unexpected.";

  // Constructs name to TypeProto map from value_info, input, output
  std::unordered_map<std::string, TypeProto*> valueTypesByName;
  for (auto& vi : *graph.mutable_value_info()) {
    if (vi.has_type()) {
      valueTypesByName[vi.name()] = vi.mutable_type();
    }
  }
  for (auto& vi : *graph.mutable_input()) {
    if (vi.has_type()) {
      valueTypesByName[vi.name()] = vi.mutable_type();
    }
  }
  for (auto& vi : *graph.mutable_output()) {
    if (vi.has_type()) {
      valueTypesByName[vi.name()] = vi.mutable_type();
    }
  }

  // Constructs name to TensorProto map from initializer
  std::unordered_map<std::string, const TensorProto*> inputDataByName;
  for (const auto& tp : graph.initializer()) {
    inputDataByName[tp.name()] = &tp;
  }
  // Collects data from constant nodes
  for (const auto& n : graph.node()) {
    if (n.op_type() != "Constant" || n.output().size() != 1) {
      continue;
    }
    for (const auto& attr : n.attribute()) {
      if (attr.name() == "value") {
        if (attr.type() == AttributeProto::TENSOR && attr.has_t()) {
          inputDataByName[n.output(0)] = &attr.t();
        }
      }
    }
  }

  // Runs data propagation on each node
  std::unordered_map<std::string, TensorShapeProto> generatedShapeDataByName;
  auto* schemaRegistry = OpSchemaRegistry::Instance();
  const TensorShapeProto* propagatedShape;
  for (auto n: graph.node()) {
    DataPropagationContextImpl dataPropagationCtx(
        n, valueTypesByName, inputDataByName, generatedShapeDataByName);
    const auto schema = schemaRegistry->GetSchema(n.op_type(), domain_version, n.domain());
    EXPECT_TRUE(schema->has_data_propagation_function());
    schema->GetDataPropagationFunction()(dataPropagationCtx);
    propagatedShape = dataPropagationCtx.getInputData(0);
  }
  // Expects the input data of final node (from the output of previous node)
  // is same as the given output shape
  auto* outputShape = graph.mutable_output(0)
    ->mutable_type()->mutable_tensor_type()->mutable_shape();
  EXPECT_TRUE(CompareShape(propagatedShape, outputShape));
}

TEST(DataPropagationImplTest, ShapeTest) {
  const char* code = R"ONNX(
agraph (int32[2,5] x) => (int32[2,5] z)
{
    y = Shape(x)
    z = Cast(y)
}
)ONNX";

  RunDataPropagationTest(code);
}

TEST(DataPropagationImplTest, CastTest) {
  const char* code = R"ONNX(
agraph (int32[2,5] x) => (int32[2,5] z)
{
    y = Shape(x)
    z = Cast(y)
}
)ONNX";

  RunDataPropagationTest(code);
}

TEST(DataPropagationImplTest, SqueezeTest) {
  const char* code = R"ONNX(
agraph (int32[2,5] x) => (int32[2,5] w)
{
    y = Shape(x)
    z = Squeeze(y)
    w = Cast(z)
}
)ONNX";

  RunDataPropagationTest(code);
}

TEST(DataPropagationImplTest, UnsqueezeTest) {
  const char* code = R"ONNX(
agraph (int32[2,5] x) => (int32[2,5] w)
{
    y = Shape(x)
    z = Unsqueeze(y)
    w = Cast(z)
}
)ONNX";

  RunDataPropagationTest(code);
}

} // namespace Test
} // namespace ONNX_NAMESPACE
