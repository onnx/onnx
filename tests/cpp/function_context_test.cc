// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "gtest/gtest.h"
#include "onnx/checker.h"
#include "onnx/common/constants.h"
#include "onnx/defs/function.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/schema.h"
#include "onnx/shape_inference/implementation.h"

namespace ONNX_NAMESPACE::Test {

static TensorProto ToTensor(double value, TensorProto_DataType elem_type) {
  TensorProto t;
  t.set_data_type(elem_type);
  switch (elem_type) {
    case TensorProto_DataType::TensorProto_DataType_FLOAT:
      t.add_float_data(static_cast<float>(value));
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

// A monomorphic context-dependent function test-case.
static bool
BuildFloatFunctionBody(const FunctionBodyBuildContext& /*ctx*/, const OpSchema& schema, FunctionProto& functionProto) {
  FunctionBuilder builder(functionProto);
  builder.Const("Two", ToTensor(2.0, TensorProto_DataType::TensorProto_DataType_FLOAT)).Add("Y = Mul (X, Two)");
  schema.BuildFunction(functionProto);
  return true;
}

static void RegisterCustomFuncFloatSchema() {
  ONNX_NAMESPACE::OpSchema schema;
  schema.SetName("CustomFuncFloat")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(12)
      .SetDoc("This operator returns an output tensor that is twice the input tensor.")
      .Input(0, "X", "Input tensor", "T", OpSchema::Single)
      .Output(0, "Y", "Output tensor", "T", OpSchema::Single)
      .TypeConstraint("T", {"tensor(float)"}, "Type of the input and output values")
      .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
      .SetContextDependentFunctionBodyBuilder(BuildFloatFunctionBody);
  ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce unused(schema);
  (void)unused;
}

// Test for Context dependent function without type context
TEST(FunctionAPITest, ContextDependentFunctionTest) {
  RegisterCustomFuncFloatSchema();

  const auto* const schema = OpSchemaRegistry::Schema("CustomFuncFloat", 12, ONNX_DOMAIN);
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

  checker::LexicalScopeContext lexicalScope;
  checker::CheckerContext checkerCtx;
  std::unordered_map<std::string, int> opset_imports({{ONNX_DOMAIN, 12}});
  checkerCtx.set_opset_imports(opset_imports);
  checkerCtx.set_ir_version(7);
  checker::check_function(fnProto, checkerCtx, lexicalScope);
}

TEST(FunctionAPITest, OmittedTrailingInputDoesNotCaptureCallerValue) {
  const auto* schema = OpSchemaRegistry::Schema("ReduceSumSquare", 18, ONNX_DOMAIN);
  ASSERT_NE(schema, nullptr);
  const auto* function = schema->GetFunction(18);
  ASSERT_NE(function, nullptr);
  for (const int input_mode : {0, 1, 2}) {
    SCOPED_TRACE(input_mode);
    ModelProto model;
    OnnxParser parser(R"ONNX(
      <ir_version: 8, opset_import: ["" : 18]>
      reduction (float[2,3] X, int64[1] axes, int64[1] chosen_axes) => (float[M,N] Y) {
        Y = ReduceSumSquare(X)
      }
    )ONNX");
    ASSERT_TRUE(parser.Parse(model).IsOK());
    for (const auto* name : {"axes", "chosen_axes"}) {
      auto* initializer = model.mutable_graph()->add_initializer();
      initializer->set_name(name);
      initializer->set_data_type(TensorProto::INT64);
      initializer->add_dims(1);
      initializer->add_int64_data(std::string(name) == "axes" ? 1 : 0);
    }
    auto* node = model.mutable_graph()->mutable_node(0);
    if (input_mode != 0) {
      node->add_input(input_mode == 1 ? "" : "chosen_axes");
    }
    checker::check_model(model);
    ModelProto original = model;
    shape_inference::InferShapes(original);
    const auto& expected_shape = original.graph().output(0).type().tensor_type().shape();
    ASSERT_EQ(expected_shape.dim(0).dim_value(), 1);
    ASSERT_EQ(expected_shape.dim(1).dim_value(), input_mode == 2 ? 3 : 1);

    NodeProto call = *node;
    model.mutable_graph()->clear_node();
    FunctionExpandHelper(call, *function, *model.mutable_graph());
    ASSERT_EQ(model.graph().node_size(), 2);
    EXPECT_EQ(model.graph().node(1).input(1), input_mode == 2 ? "chosen_axes" : "");
    checker::check_model(model);
    shape_inference::InferShapes(model);
    EXPECT_EQ(model.graph().output(0).type().tensor_type().shape().SerializeAsString(), expected_shape.SerializeAsString());
  }
}

#ifndef ONNX_NO_EXCEPTIONS
TEST(FunctionAPITest, RejectReferencedGraphCaptureSubstitution) {
  ModelProto model;
  OnnxParser parser(R"ONNX(
    <ir_version: 13, opset_import: ["" : 29]>
    capture (bool A, float[1] X) => (float[1] Y) {
      Y = If(A) <
        then_branch = yes () => (float[1] result) { result = Identity(X) },
        else_branch = no () => (float[1] result) { result = Identity(X) }
      >
    }
  )ONNX");
  ASSERT_TRUE(parser.Parse(model).IsOK());
  checker::check_model(model);
  NodeProto call = model.graph().node(0);
  call.set_name("caller_graph");
  FunctionProto function;
  function.set_name("If");
  function.add_input("X");
  function.add_output("Y");
  function.mutable_opset_import()->CopyFrom(model.opset_import());
  auto* inner = function.add_node();
  inner->set_op_type("If");
  inner->add_input("X");
  inner->add_output("Y");
  for (const auto* name : {"then_branch", "else_branch"}) {
    function.add_attribute(name);
    auto* reference = inner->add_attribute();
    reference->set_name(name);
    reference->set_ref_attr_name(name);
    reference->set_type(AttributeProto::GRAPH);
  }
  for (const bool graph_list : {false, true}) {
    NodeProto actual_call = call;
    if (graph_list) {
      for (auto& attr : *actual_call.mutable_attribute()) {
        GraphProto graph = attr.g();
        attr.clear_g();
        attr.set_type(AttributeProto::GRAPHS);
        attr.add_graphs()->CopyFrom(graph);
      }
    }
    model.mutable_graph()->clear_node();
    try {
      FunctionExpandHelper(actual_call, function, *model.mutable_graph());
      FAIL() << "Caller-owned graph captures must not enter the function renaming scope";
    } catch (const std::runtime_error& error) {
      const std::string message(error.what());
      EXPECT_NE(message.find("caller_graph"), std::string::npos);
      EXPECT_NE(message.find("then_branch"), std::string::npos);
      EXPECT_NE(message.find("Retain the function call"), std::string::npos);
    }
    EXPECT_EQ(model.graph().node_size(), 0);
  }
}
#endif

static void CheckExpandedLinearAttention(bool trajectory, int64_t length) {
  const auto* const schema = OpSchemaRegistry::Schema("LinearAttention", 29, ONNX_DOMAIN);
  ASSERT_NE(schema, nullptr);
  NodeProto node;
  node.set_op_type("LinearAttention");
  for (const auto* name : {"Q", "K", "V", "", "", ""}) {
    node.add_input(name);
  }
  for (const auto* name : {"Y", "S", trajectory ? "States" : ""}) {
    node.add_output(name);
  }
  auto* rule = node.add_attribute();
  rule->set_name("update_rule");
  rule->set_type(AttributeProto::STRING);
  rule->set_s("linear");
  for (const auto* name : {"q_num_heads", "kv_num_heads"}) {
    auto* heads = node.add_attribute();
    heads->set_name(name);
    heads->set_type(AttributeProto::INT);
    heads->set_i(1);
  }
  std::vector<TypeProto> types(6);
  for (size_t i = 0; i < 3; ++i) {
    types[i].mutable_tensor_type()->set_elem_type(TensorProto::FLOAT);
  }
  FunctionBodyBuildContextImpl ctx(node, types);
  FunctionProto function;
  ASSERT_TRUE(schema->BuildContextDependentFunction(ctx, function));
  ASSERT_EQ(function.output_size(), trajectory ? 3 : 2);
  GraphProto expanded;
  FunctionExpandHelper(node, function, expanded);
  EXPECT_GT(expanded.node_size(), 0);
  ModelProto model;
  model.set_ir_version(13);
  model.mutable_graph()->CopyFrom(expanded);
  model.mutable_graph()->set_name("expanded_linear_attention");
  model.mutable_opset_import()->CopyFrom(function.opset_import());
  for (const auto* name : {"Q", "K", "V", "Y", "S", "States"}) {
    const std::string value_name(name);
    if (value_name == "States" && !trajectory) {
      continue;
    }
    const bool output = value_name == "Y" || value_name == "S" || value_name == "States";
    auto* value = output ? model.mutable_graph()->add_output() : model.mutable_graph()->add_input();
    value->set_name(name);
    auto* tensor = value->mutable_type()->mutable_tensor_type();
    tensor->set_elem_type(TensorProto::FLOAT);
    const int rank = value_name == "States" ? 5 : (value_name == "S" ? 4 : 3);
    for (int i = 0; i < rank; ++i) {
      tensor->mutable_shape()->add_dim()->set_dim_value(value_name != "S" && i == 1 ? length : 1);
    }
  }
  checker::check_model(model);
#ifndef ONNX_NO_EXCEPTIONS
  // A named output still has to be bound by the function.
  if (trajectory) {
    node.add_output("Unexpected");
  } else {
    node.set_output(2, "Unexpected");
  }
  EXPECT_THROW(FunctionExpandHelper(node, function, expanded), std::runtime_error);
#endif
}

TEST(FunctionAPITest, ExpandMissingTrailingOptionalOutput) {
  for (const bool trajectory : {false, true}) {
    for (const int64_t length : {0, 2}) {
      CheckExpandedLinearAttention(trajectory, length);
    }
  }
}

TEST(FunctionAPITest, InlinePassthroughSubgraphOutputs) {
  for (const bool initializer_output : {false, true}) {
    ModelProto model;
    OnnxParser parser(R"ONNX(
      <ir_version: 13, opset_import: ["" : 29]>
      passthrough (float[2,1] X) => (float[2,1] Y) {
        Y = Scan (X) <
          num_scan_inputs = 1,
          body = step (float[1] item) => (float[1] item) {}
        >
      }
    )ONNX");
    ASSERT_TRUE(parser.Parse(model).IsOK());
    if (initializer_output) {
      GraphProto* body = nullptr;
      for (auto& attr : *model.mutable_graph()->mutable_node(0)->mutable_attribute()) {
        if (attr.name() == "body") {
          body = attr.mutable_g();
        }
      }
      ASSERT_NE(body, nullptr);
      auto* init = body->add_initializer();
      init->set_name("unchanged");
      init->set_data_type(TensorProto::FLOAT);
      init->add_dims(1);
      init->add_float_data(2.0f);
      body->mutable_output(0)->set_name("unchanged");
    }
    checker::check_model(model);
    FunctionProto function;
    function.set_name("Softsign");
    function.add_input("X");
    function.add_output("Y");
    function.mutable_opset_import()->CopyFrom(model.opset_import());
    FunctionBuilder builder(function);
    builder.AddInlinedCall({"Y"}, model.graph(), {"X"}, "passthrough");
    model.mutable_graph()->mutable_node()->CopyFrom(function.node());
    checker::check_model(model);

    NodeProto call;
    call.set_op_type("Softsign");
    call.add_input("X");
    call.add_output("Y");
    model.mutable_graph()->clear_node();
    FunctionExpandHelper(call, function, *model.mutable_graph());
    checker::check_model(model);
  }
}

TEST(FunctionAPITest, ExpandedCapturesDoNotCollideWithBoundNames) {
  ModelProto model;
  OnnxParser parser(R"ONNX(
    <ir_version: 13, opset_import: ["" : 29]>
    collision (float[1] X, bool condition) => (float[1] Y) {
      c__f = Identity(X)
      Y = If(condition) <
        then_branch = yes () => (float[1] Func_fc) {
          Func_fc = Add(c__f, c__f)
        },
        else_branch = no () => (float[1] other) {
          other = Identity(c__f)
        }
      >
    }
  )ONNX");
  ASSERT_TRUE(parser.Parse(model).IsOK());
  checker::check_model(model);
  FunctionProto function;
  function.set_name("Softsign");
  function.add_input("X");
  function.add_input("condition");
  function.add_output("Y");
  function.mutable_node()->CopyFrom(model.graph().node());
  function.mutable_opset_import()->CopyFrom(model.opset_import());
  NodeProto call;
  call.set_name("f");
  call.set_op_type("Softsign");
  call.add_input("X");
  call.add_input("condition");
  call.add_output("Y");
  model.mutable_graph()->clear_node();
  FunctionExpandHelper(call, function, *model.mutable_graph());
  checker::check_model(model);
  const auto& captured = model.graph().node(0).output(0);
  for (const auto& branch : model.graph().node(1).attribute()) {
    ASSERT_EQ(branch.g().node(0).input(0), captured);
    ASSERT_NE(branch.g().output(0).name(), captured);
  }
}

TEST(FunctionAPITest, NestedValueInfoFollowsRenamedDefinitions) {
  for (const bool expand : {false, true}) {
    ModelProto model;
    OnnxParser parser(R"ONNX(
      <ir_version: 13, opset_import: ["" : 29]>
      annotated (float[2,1] X) => (float[2,1] Y) {
        Y = Scan (X) <
          num_scan_inputs = 1,
          body = step (float[1] item) => (float[1] result) {
            intermediate = Identity(item)
            result = Identity(intermediate)
          }
        >
      }
    )ONNX");
    ASSERT_TRUE(parser.Parse(model).IsOK());
    GraphProto* body = nullptr;
    for (auto& attr : *model.mutable_graph()->mutable_node(0)->mutable_attribute()) {
      if (attr.name() == "body")
        body = attr.mutable_g();
    }
    ASSERT_NE(body, nullptr);
    auto* annotation = body->add_value_info();
    annotation->CopyFrom(body->input(0));
    annotation->set_name("intermediate");
    const auto annotated_type = annotation->type().SerializeAsString();
    auto* quantization = body->add_quantization_annotation();
    quantization->set_tensor_name("intermediate");
    auto* scale = quantization->add_quant_parameter_tensor_names();
    scale->set_key("SCALE_TENSOR");
    scale->set_value("item");
    auto* captured = quantization->add_quant_parameter_tensor_names();
    captured->set_key("ZERO_POINT_TENSOR");
    captured->set_value("X");
    checker::check_model(model);
    FunctionProto function;
    function.set_name("Softsign");
    function.add_input("X");
    function.add_output("Y");
    function.mutable_opset_import()->CopyFrom(model.opset_import());
    if (expand) {
      function.mutable_node()->CopyFrom(model.graph().node());
      NodeProto call;
      call.set_name("annotated_call");
      call.set_op_type("Softsign");
      call.add_input("X");
      call.add_output("Y");
      model.mutable_graph()->clear_node();
      FunctionExpandHelper(call, function, *model.mutable_graph());
    } else {
      FunctionBuilder builder(function);
      builder.AddInlinedCall({"Y"}, model.graph(), {"X"}, "annotated_call");
      model.mutable_graph()->mutable_node()->CopyFrom(function.node());
    }
    checker::check_model(model);
    for (const auto& attr : model.graph().node(0).attribute()) {
      if (attr.name() != "body")
        continue;
      const auto& renamed_body = attr.g();
      ASSERT_EQ(renamed_body.value_info_size(), 1);
      EXPECT_EQ(renamed_body.value_info(0).name(), renamed_body.node(0).output(0));
      EXPECT_EQ(renamed_body.value_info(0).type().SerializeAsString(), annotated_type);
      ASSERT_EQ(renamed_body.quantization_annotation_size(), 1);
      const auto& renamed_quantization = renamed_body.quantization_annotation(0);
      EXPECT_EQ(renamed_quantization.tensor_name(), renamed_body.node(0).output(0));
      ASSERT_EQ(renamed_quantization.quant_parameter_tensor_names_size(), 2);
      EXPECT_EQ(renamed_quantization.quant_parameter_tensor_names(0).key(), "SCALE_TENSOR");
      EXPECT_EQ(renamed_quantization.quant_parameter_tensor_names(0).value(), renamed_body.input(0).name());
      EXPECT_EQ(renamed_quantization.quant_parameter_tensor_names(1).key(), "ZERO_POINT_TENSOR");
      EXPECT_EQ(renamed_quantization.quant_parameter_tensor_names(1).value(), "X");
    }
  }
}

TEST(FunctionAPITest, NestedInitializerRetainsInputBinding) {
  for (const bool sparse : {false, true}) {
    for (const bool expand : {false, true}) {
      ModelProto model;
      OnnxParser parser(R"ONNX(
      <ir_version: 13, opset_import: ["" : 29]>
      initialized (float[2,1] X) => (float[2,1] Y) {
        Y = Scan (X) <
          num_scan_inputs = 1,
          body = step (float[1] item) => (float[1] result) {
            result = Identity(item)
          }
        >
      }
    )ONNX");
      ASSERT_TRUE(parser.Parse(model).IsOK());
      for (auto& attr : *model.mutable_graph()->mutable_node(0)->mutable_attribute()) {
        if (attr.name() == "body") {
          auto* init = attr.mutable_g()->add_initializer();
          init->set_name("item");
          init->set_data_type(TensorProto::FLOAT);
          init->add_dims(1);
          init->add_float_data(99.0f);
          if (sparse) {
            auto* body = attr.mutable_g();
            auto* sparse_init = body->add_sparse_initializer();
            sparse_init->mutable_values()->CopyFrom(*init);
            sparse_init->add_dims(1);
            auto* indices = sparse_init->mutable_indices();
            indices->set_data_type(TensorProto::INT64);
            indices->add_dims(1);
            indices->add_int64_data(0);
            body->clear_initializer();
          }
        }
      }
      checker::check_model(model);
      FunctionProto function;
      function.set_name("Softsign");
      function.add_input("X");
      function.add_output("Y");
      function.mutable_opset_import()->CopyFrom(model.opset_import());
      if (expand) {
        function.mutable_node()->CopyFrom(model.graph().node());
        NodeProto call;
        call.set_name("initialized_call");
        call.set_op_type("Softsign");
        call.add_input("X");
        call.add_output("Y");
        model.mutable_graph()->clear_node();
        FunctionExpandHelper(call, function, *model.mutable_graph());
      } else {
        FunctionBuilder builder(function);
        builder.AddInlinedCall({"Y"}, model.graph(), {"X"}, "initialized_call");
        model.mutable_graph()->mutable_node()->CopyFrom(function.node());
      }
      checker::check_model(model);
      for (const auto& attr : model.graph().node(0).attribute()) {
        if (attr.name() == "body") {
          const auto& body = attr.g();
          const auto& initializer = sparse ? body.sparse_initializer(0).values() : body.initializer(0);
          EXPECT_EQ(body.input(0).name(), initializer.name());
          EXPECT_EQ(body.node(0).input(0), body.input(0).name());
        }
      }
    }
  }
}

// A polymorphic context-dependent function test-case.

static bool
BuildFunctionBody(const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) {
  // Create a scalar-tensor constant 2.0 of input-type:
  const auto* const tp = ctx.getInputType(0);
  if ((tp == nullptr) || (!tp->has_tensor_type()))
    return false;
  auto elem_type = static_cast<TensorProto_DataType>(tp->tensor_type().elem_type());
  FunctionBuilder builder(functionProto);
  builder.Const("Two", ToTensor(2.0, elem_type)).Add("Y = Mul (X, Two)");
  schema.BuildFunction(functionProto);
  return true;
}

static void RegisterCustomFunctionSchema() {
  ONNX_NAMESPACE::OpSchema schema;
  schema.SetName("CustomFunction")
      .SetDomain(ONNX_DOMAIN)
      .SinceVersion(12)
      .SetDoc("This operator returns an output tensor that is twice the input tensor.")
      .Input(0, "X", "Input tensor", "T", OpSchema::Single)
      .Output(0, "Y", "Output tensor", "T", OpSchema::Single)
      .TypeConstraint("T", {"tensor(float)", "tensor(double)"}, "Type of the input and output values")
      .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
      .SetContextDependentFunctionBodyBuilder(BuildFunctionBody);
  ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce unused(schema);
  (void)unused;
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

  const auto* const schema2 = OpSchemaRegistry::Schema("MySub", 2, ONNX_DOMAIN);
  EXPECT_TRUE(schema2);
  for (int model_opset_import = 2; model_opset_import < 9; model_opset_import++) {
    ONNX_TRY {
      bool validate = true;
      const FunctionProto* function = schema2->GetFunction(model_opset_import, validate);
      if (model_opset_import >= 6) { // function body should be updated at opset 6 where Sub is updated
        ASSERT_TRUE(function == nullptr);
      } else {
        ASSERT_TRUE(function);
      }
    }
    ONNX_CATCH(const std::runtime_error&) {
      ONNX_HANDLE_EXCEPTION(
          [&]() { ASSERT_TRUE(model_opset_import == 6 || model_opset_import == 7 || model_opset_import == 8); });
    }
  }

  const auto* const schema9 = OpSchemaRegistry::Schema("MySub", 9, ONNX_DOMAIN);
  EXPECT_TRUE(schema9);
  for (int model_opset_import = 9; model_opset_import < 10; model_opset_import++) {
    ONNX_TRY {
      const FunctionProto* function = schema9->GetFunction(model_opset_import);
      ASSERT_TRUE(function);
    }
    ONNX_CATCH(const std::runtime_error&) {
      ONNX_HANDLE_EXCEPTION(
          [&]() { ASSERT_TRUE(model_opset_import == 13 || model_opset_import == 14 || model_opset_import == 15); });
    }
  }
}

TEST(FunctionAPITest, TypeContextTest) {
  RegisterCustomFunctionSchema();

  const auto* const schema = OpSchemaRegistry::Schema("CustomFunction", 12, ONNX_DOMAIN);
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

  checker::LexicalScopeContext lexicalScope;
  checker::CheckerContext checkerCtx;
  std::unordered_map<std::string, int> opset_imports({{ONNX_DOMAIN, 12}});
  checkerCtx.set_opset_imports(opset_imports);
  checkerCtx.set_ir_version(7);
  checker::check_function(fnProto, checkerCtx, lexicalScope);
}

} // namespace ONNX_NAMESPACE::Test
