#include <iostream>
#include "gtest/gtest.h"
#include "onnx/checker.h"
#include "onnx/common/constants.h"
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"

using namespace ONNX_NAMESPACE::checker;

namespace ONNX_NAMESPACE {
namespace Test {

// Utilities. TODO: Turn them into reusable ONNX utilities for use by

TensorProto ToTensor(double value, TensorProto_DataType elem_type) {
  TensorProto t;
  t.set_data_type(elem_type);
  switch (elem_type) {
    case TensorProto_DataType::TensorProto_DataType_FLOAT:
      t.add_float_data((float)value);
      break;
    case TensorProto_DataType::TensorProto_DataType_DOUBLE:
      t.add_double_data(value);
      break;
    // case TensorProto_DataType::TensorProto_DataType_FLOAT16:
    //   t.add_int32_data(onnxruntime::math::floatToHalf((float)value));
    //   break;
    default:
      assert(false);
  }

  return t;
}

void BuildNodes(FunctionProto& functionProto, const std::vector<FunctionBodyHelper::NodeDef>& node_defs) {
  for (size_t i = 0; i < node_defs.size(); i++) {
    const FunctionBodyHelper::NodeDef& node = node_defs[i];
    auto* np = functionProto.add_node();

    np->set_op_type(node.op_type);
    for (const auto& inp : node.inputs) {
      np->add_input(inp);
    }
    for (const auto& o : node.outputs) {
      np->add_output(o);
    }
    for (const auto& attr : node.attributes) {
      *(np->add_attribute()) = attr.proto;
    }
  }
}

bool BuildFunctionProto(
    FunctionProto& functionProto,
    const OpSchema& schema,
    const std::vector<FunctionBodyHelper::NodeDef>& node_defs) {
  BuildNodes(functionProto, node_defs);
  schema.BuildFunction(functionProto);
  return true;
}

static bool
BuildTwiceFunctionBody(const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
  // Create a scalar-tensor constant 2.0 of input-type:
  auto* tp = ctx.getInputType(0);
  if ((tp == nullptr) || (!tp->has_tensor_type()))
    return false;
  auto elem_type = (TensorProto_DataType)tp->tensor_type().elem_type();
  auto two_as_tensor = ToTensor(2.0, elem_type);

  std::vector<FunctionBodyHelper::NodeDef> body{// nodes: {outputs, op, inputs, attributes}
                                                {{"Two"}, "Constant", {}, {{"value", two_as_tensor}}},
                                                {{"Y"}, "Mul", {"X", "Two"}}};

  return BuildFunctionProto(functionProto, schema, body);
}

void RegisterTwiceSchema() {
  ONNX_NAMESPACE::OpSchema schema;
  schema.SetName("CustomFunTwice")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(12)
      .SetDoc("This operator returns an output tensor that is twice the input tensor.")
      .Input(0, "X", "Input tensor", "T", OpSchema::Single)
      .Output(0, "Y", "Output tensor", "T", OpSchema::Single)
      .TypeConstraint("T", {"tensor(float)", "tensor(double)"}, "Type of the input and output values")
      .SetContextDependentFunctionBodyBuilder(BuildTwiceFunctionBody);
  ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce unused(schema);
  (void)unused;
}

TEST(FunctionAPITest, TypeContextTest) {
  RegisterTwiceSchema();

  const auto* schema = OpSchemaRegistry::Schema("CustomFunTwice", 12, "");
  EXPECT_TRUE(schema);
  EXPECT_FALSE(schema->HasFunction());
  EXPECT_TRUE(schema->HasContextDependentFunction());

  NodeProto nodeProto;
  nodeProto.set_op_type("CustomFunTwice");
  nodeProto.add_input("X");
  nodeProto.add_output("Y");

  TypeProto floatTypeProto;
  floatTypeProto.mutable_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_FLOAT);

  FunctionBodyBuildContextImpl ctx(nodeProto, {floatTypeProto});
  FunctionProto fnProto;
  EXPECT_TRUE(schema->BuildContextDependentFunction(ctx, fnProto));
  EXPECT_EQ(fnProto.node_size(), 2);

  LexicalScopeContext lexicalScope;
  CheckerContext checkerCtx;
  checkerCtx.set_ir_version(7);
  check_function(fnProto, checkerCtx, lexicalScope);
}

} // namespace Test
} // namespace ONNX_NAMESPACE