/*
 * SPDX-License-Identifier: Apache-2.0
 */

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

// A monomorphic context-dependent function test-case.
static bool
BuildFloatFunctionBody(const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
  // Create a scalar-tensor constant 2.0 of float type:
  auto two_as_tensor = ToTensor(2.0, TensorProto_DataType::TensorProto_DataType_FLOAT);

  std::vector<FunctionBodyHelper::NodeDef> body{// nodes: {outputs, op, inputs, attributes}
                                                {{"Two"}, "Constant", {}, {{"value", two_as_tensor}}},
                                                {{"Y"}, "Mul", {"X", "Two"}}};

  return BuildFunctionProto(functionProto, schema, body);
}

void RegisterCustomFuncFloatSchema() {
  ONNX_NAMESPACE::OpSchema schema;
  schema.SetName("CustomFuncFloat")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(12)
      .SetDoc("This operator returns an output tensor that is twice the input tensor.")
      .Input(0, "X", "Input tensor", "T", OpSchema::Single)
      .Output(0, "Y", "Output tensor", "T", OpSchema::Single)
      .TypeConstraint("T", {"tensor(float)"}, "Type of the input and output values")
      .SetContextDependentFunctionBodyBuilder(BuildFloatFunctionBody);
  ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce unused(schema);
  (void)unused;
}

// Test for Context dependant function without type context
TEST(FunctionAPITest, ContextDependentFunctionTest) {
  RegisterCustomFuncFloatSchema();

  const auto* schema = OpSchemaRegistry::Schema("CustomFuncFloat", 12, ONNX_DOMAIN);
  EXPECT_TRUE(schema);
  EXPECT_FALSE(schema->HasFunction());
  EXPECT_TRUE(schema->HasContextDependentFunction());

  NodeProto nodeProto;
  nodeProto.set_op_type("CustomFuncFloat");
  nodeProto.add_input("X");
  nodeProto.add_output("Y");

  FunctionBodyBuildContextImpl ctx(nodeProto);
  FunctionProto fnProto;
  EXPECT_TRUE(schema->BuildContextDependentFunction(ctx, fnProto));
  EXPECT_EQ(fnProto.node_size(), 2);

  LexicalScopeContext lexicalScope;
  CheckerContext checkerCtx;
  std::unordered_map<std::string, int> opset_imports({{ONNX_DOMAIN, 12}});
  checkerCtx.set_opset_imports(opset_imports);
  checkerCtx.set_ir_version(7);
  check_function(fnProto, checkerCtx, lexicalScope);
}

// A polymorphic context-dependent function test-case.

static bool
BuildFunctionBody(const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
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

void RegisterCustomFunctionSchema() {
  ONNX_NAMESPACE::OpSchema schema;
  schema.SetName("CustomFunction")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(12)
      .SetDoc("This operator returns an output tensor that is twice the input tensor.")
      .Input(0, "X", "Input tensor", "T", OpSchema::Single)
      .Output(0, "Y", "Output tensor", "T", OpSchema::Single)
      .TypeConstraint("T", {"tensor(float)", "tensor(double)"}, "Type of the input and output values")
      .SetContextDependentFunctionBodyBuilder(BuildFunctionBody);
  ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce unused(schema);
  (void)unused;
}

void RegisterVersionedLogSoftmaxFunctionSchema(const std::string& op_type, bool with_missing_version) {
  // SinceVersion of a function operator tells the last opset version where its semantic is defined.
  // Function bodies and context dependent function builders also have versions.
  // A function's version specifies its valid range since its since_version till the since_version
  // of the next function of the same funtion op.
  ONNX_NAMESPACE::OpSchema schema;
  schema.SetName(op_type)
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(13)
      .SetDoc("A variation of LogSoftMax that takes axes as input instead of attribute.")
      .Attr("axis", "", AttributeProto::INT, static_cast<int64_t>(-1))
      .Input(0, "input", "Input tensor", "T", OpSchema::Single)
      .Output(0, "output", "Output tensor", "T", OpSchema::Single)
      .TypeConstraint("T", {"tensor(float)", "tensor(double)"}, "Type of the input and output values")
      .SetContextDependentFunctionBodyBuilder(
          [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
            const int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
            FunctionBuilder builder(functionProto);
            builder.Const1D("axes", axis)
                .Add("X_ReduceMax = ReduceMax <keepdims = 1> (input)", "axes", std::vector<int64_t>({axis}))
                .Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    X_Log = Log (X_ReduceSum)
                    output = Sub (X_Sub, X_Log)
                )");
            schema.BuildFunction(functionProto);
            return true;
          },
          13);
  if (!with_missing_version) {
    schema.SetContextDependentFunctionBodyBuilder(
        [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
          const int64_t axis = ctx.getAttribute("axis") != nullptr ? ctx.getAttribute("axis")->i() : -1;
          FunctionBuilder builder(functionProto);
          builder.Const1D("axes", axis).Add("X_ReduceMax = ReduceMax <keepdims = 1> (input, axes)").Add(R"(
                    X_Sub = Sub (input, X_ReduceMax)
                    X_Exp = Exp (X_Sub)
                    X_ReduceSum = ReduceSum <keepdims = 1> (X_Exp, axes)
                    X_Log = Log (X_ReduceSum)
                    output = Sub (X_Sub, X_Log)
                )");
          schema.BuildFunction(functionProto);
          return true;
        },
        18);
  }

  ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce unused(schema);
  (void)unused;
}

void BuildLogSoftmaxFunction(
    const OpSchema& schema,
    const std::string& op_type,
    FunctionProto& fnProto,
    int requested_opset_version) {
  NodeProto nodeProto;
  nodeProto.set_op_type(op_type);
  nodeProto.add_input("input");
  nodeProto.add_output("output");

  TypeProto floatTypeProto;
  floatTypeProto.mutable_tensor_type()->set_elem_type(TensorProto_DataType::TensorProto_DataType_FLOAT);

  FunctionBodyBuildContextImpl ctx(nodeProto, {floatTypeProto});
  schema.BuildContextDependentFunctionWithOpsetVersion(ctx, fnProto, requested_opset_version, "", true);
}

TEST(FunctionAPITest, VersionedFunctionWithMissingVersionTest) {
  std::string op_type = "VersionedLogSoftMax";
  RegisterVersionedLogSoftmaxFunctionSchema(op_type, true);

  // The function op's since version is 13. there is no new version after version 13.
  // As a result, all OpSchema shall point to the same one.
  const auto* schema13 = OpSchemaRegistry::Schema(op_type, 13, ONNX_DOMAIN);
  EXPECT_TRUE(schema13);
  const auto* schema17 = OpSchemaRegistry::Schema(op_type, 17, ONNX_DOMAIN);
  EXPECT_TRUE(schema17 == schema13);
  const auto* schema18 = OpSchemaRegistry::Schema(op_type, 18, ONNX_DOMAIN);
  EXPECT_TRUE(schema18 == schema13);

  // VersionedLogSoftMax has 1 function body for opset version 13.
  // It is ok to request a function body for a model with opset import version 13.
  FunctionProto function_proto13;
  BuildLogSoftmaxFunction(*schema13, op_type, function_proto13, 13);
  ASSERT_TRUE(function_proto13.name() == op_type);

  try {
    // It may not be ok to request a function body for a model with opset import version 14, 15, 16, and 17.
    // This is because Sub(14) exists and will be used, instead of Sub(13), to execute the function body.
    FunctionProto function_proto17;
    BuildLogSoftmaxFunction(*schema17, op_type, function_proto17, 17);
    FAIL() << "Expect runtime_error failure in building function for VersionedLogSoftMax with opset version 17";
  } catch (std::runtime_error err) {
    std::cout << err.what();
    EXPECT_TRUE(
        std::string(err.what())
            .find("Operator (Sub) of version 17 is used but the op has been updated at least once since version 13.") !=
        std::string::npos);
  } catch (...) {
    FAIL()
        << "Expect runtime_error failure in getting function op schema for VersionedLogSoftMax with opset version 17";
  }

  try {
    // It may not be ok to request a function body for a model with opset import version 18.
    // This is because Sub(14) exists and will be used, instead of Sub(13), to execute the function body.
    // This is also because ReduceMax(18) exists and will be used, instead of ReduceMax(13), to execute the function
    // body.
    FunctionProto function_proto18;
    BuildLogSoftmaxFunction(*schema18, op_type, function_proto18, 18);
    FAIL() << "Expect runtime_error failure in building function for VersionedLogSoftMax with opset version 18";
  } catch (std::runtime_error err) {
    std::cout << err.what();
    EXPECT_TRUE(
        std::string(err.what())
            .find(
                "Operator (ReduceMax) of version 18 is used but the op has been updated at least once since version 13.") !=
        std::string::npos);
    EXPECT_TRUE(
        std::string(err.what())
            .find("Operator (Sub) of version 18 is used but the op has been updated at least once since version 13.") !=
        std::string::npos);
  } catch (...) {
    FAIL()
        << "Expect runtime_error failure in getting function op schema for VersionedLogSoftMax with opset version 18";
  }
}

TEST(FunctionAPITest, VersionedFunctionTest) {
  std::string op_type = "VersionedLogSoftMax2";
  RegisterVersionedLogSoftmaxFunctionSchema(op_type, false);

  // The function op's since version is 13. there is no new version after version 13.
  // As a result, all OpSchema shall point to the same one.
  const auto* schema13 = OpSchemaRegistry::Schema(op_type, 13, ONNX_DOMAIN);
  EXPECT_TRUE(schema13);
  const auto* schema17 = OpSchemaRegistry::Schema(op_type, 17, ONNX_DOMAIN);
  EXPECT_TRUE(schema17 == schema13);
  const auto* schema18 = OpSchemaRegistry::Schema(op_type, 18, ONNX_DOMAIN);
  EXPECT_TRUE(schema18 == schema13);

  // VersionedLogSoftMax has 2 function bodies: for opset version 13 and 18.
  // It is ok to request a function body for a model with opset import version 13.
  FunctionProto function_proto13;
  BuildLogSoftmaxFunction(*schema13, op_type, function_proto13, 13);
  ASSERT_TRUE(function_proto13.name() == op_type);

  try {
    // It may not be ok to request a function body for a model with opset import version 14, 15, 16, and 17.
    // This is because Sub(14) exists and will be used, instead of Sub(13), to execute the function body.
    FunctionProto function_proto17;
    BuildLogSoftmaxFunction(*schema17, op_type, function_proto17, 17);
    FAIL() << "Expect runtime_error failure in building function for VersionedLogSoftMax2 with opset version 17";
  } catch (std::runtime_error err) {
    std::cout << err.what();
    EXPECT_TRUE(
        std::string(err.what())
            .find("Operator (Sub) of version 17 is used but the op has been updated at least once since version 13.") !=
        std::string::npos);
  } catch (...) {
    FAIL()
        << "Expect runtime_error failure in getting function op schema for VersionedLogSoftMax2 with opset version 17";
  }

  // It is ok to request a function body for a model with opset import version 18.
  FunctionProto function_proto18;
  BuildLogSoftmaxFunction(*schema13, op_type, function_proto18, 18);
  ASSERT_TRUE(function_proto18.name() == op_type);
}

TEST(FunctionAPITest, VersionedFunctionBodyTest) {
  // This test illustrate issues of ONNX function ops.
  // It is over simplified in that only one primary op (Sub) is used in function body.
  // ONNX opset     1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18
  // MySub:            2                    9                               // MySub function op is created at opset 2.
  //                                                                        // Its semantic is updated at opset 7
  // Body Ideal:       2           6  7     9          13 14    16          // Ideally function body shall be provided
  //                                                                        // each time there is any version bump of
  //                                                                        // used primary ops. It will be more
  //                                                                        // frequent
  //                                                                        // if more primary ops are used.
  // Body Real:        2                    9                   16          // In real life, we seldom add function body
  //                                                                        // due to primary op update
  // Sub:           1              6  7                13 14                // Version bumps of Sub
  // Model:            y  y  y  y  n  n  n  y  y  y  y n  n  n  y  y  y     // Model can(y)/cannot(n) used
  // with opset import version.
  ONNX_NAMESPACE::OpSchema schema_ver2;
  schema_ver2.SetName("MySub")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(2)
      .SetDoc("Z = Sub (X, Y)")
      .Input(0, "X", "Input tensor X", "T", OpSchema::Single)
      .Input(1, "Y", "Input tensor Y", "T", OpSchema::Single)
      .Output(0, "Z", "Output tensor Z", "T", OpSchema::Single)
      .TypeConstraint("T", {"tensor(float)", "tensor(double)"}, "Type of the input and output values")
      .FunctionBody(
          R"ONNX(
        {
          Z = Sub (X, Y)
        }
        )ONNX",
          2);

  ONNX_NAMESPACE::OpSchema schema_ver9;
  schema_ver9.SetName("MySub")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(9)
      .SetDoc("Z = Sub (X, Y)")
      .Input(0, "X", "Input tensor X", "T", OpSchema::Single)
      .Input(1, "Y", "Input tensor Y", "T", OpSchema::Single)
      .Output(0, "Z", "Output tensor Z", "T", OpSchema::Single)
      .TypeConstraint("T", {"tensor(float)", "tensor(double)"}, "Type of the input and output values")
      .FunctionBody(
          R"ONNX(
        {
          Z = Sub (X, Y)
        }
        )ONNX",
          9)
      .FunctionBody(
          R"ONNX(
        {
          Z = Sub (X, Y)
        }
        )ONNX",
          16);

  ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce unused2(schema_ver2);
  (void)unused2;
  ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce unused9(schema_ver9);
  (void)unused9;

  const auto* schema2 = OpSchemaRegistry::Schema("MySub", 2, ONNX_DOMAIN);
  EXPECT_TRUE(schema2);
  for (int model_opset_import = 2; model_opset_import < 9; model_opset_import++) {
    try {
      const FunctionProto* function = schema2->GetFunctionWithOpsetVersion(model_opset_import, ONNX_DOMAIN, true);
      ASSERT_TRUE(function);
    } catch (std::runtime_error err) {
      ASSERT_TRUE(model_opset_import == 6 || model_opset_import == 7 || model_opset_import == 8);
    }
  }

  const auto* schema9 = OpSchemaRegistry::Schema("MySub", 9, ONNX_DOMAIN);
  EXPECT_TRUE(schema9);
  for (int model_opset_import = 9; model_opset_import < 10; model_opset_import++) {
    try {
      const FunctionProto* function = schema9->GetFunctionWithOpsetVersion(model_opset_import, ONNX_DOMAIN, true);
      ASSERT_TRUE(function);
    } catch (std::runtime_error err) {
      ASSERT_TRUE(model_opset_import == 13 || model_opset_import == 14 || model_opset_import == 15);
    }
  }
}

TEST(FunctionAPITest, TypeContextTest) {
  RegisterCustomFunctionSchema();

  const auto* schema = OpSchemaRegistry::Schema("CustomFunction", 12, ONNX_DOMAIN);
  EXPECT_TRUE(schema);
  EXPECT_FALSE(schema->HasFunction());
  EXPECT_TRUE(schema->HasContextDependentFunction());

  NodeProto nodeProto;
  nodeProto.set_op_type("CustomFunction");
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
  std::unordered_map<std::string, int> opset_imports({{ONNX_DOMAIN, 12}});
  checkerCtx.set_opset_imports(opset_imports);
  checkerCtx.set_ir_version(7);
  check_function(fnProto, checkerCtx, lexicalScope);
}

} // namespace Test
} // namespace ONNX_NAMESPACE
