/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <iostream>
#include <set>
#include "gtest/gtest.h"
#include "onnx/checker.h"
#include "onnx/common/constants.h"
#include "onnx/defs/parser.h"
#include "onnx/defs/schema.h"
#include "onnx/onnx-operators_pb.h"
#include "onnx/onnx_pb.h"
#include "onnx/shape_inference/implementation.h"

namespace ONNX_NAMESPACE {
namespace Test {
using namespace checker;
using TENSOR_TYPES_MAP = std::unordered_map<std::string, std::vector<std::string>>;

void VerifyTypeConstraint(const OpSchema& function_op, const FunctionProto* function_proto, int& counter) {
  // This is a simple partial type-checker for a function-body.
  // TODO: Revisit to make the type-checker more complete.
  TENSOR_TYPES_MAP tc_map;
  std::set<std::string> primitive_types(OpSchema::all_tensor_types().begin(), OpSchema::all_tensor_types().end());
  for (const auto& input : function_op.inputs()) {
    std::string name = input.GetName();
    auto& tvec = tc_map[name];
    for (const auto& t : input.GetTypes()) {
      tvec.emplace_back(*t);
    }
  }

  for (const auto& output : function_op.outputs()) {
    std::string name = output.GetName();
    auto& tvec = tc_map[name];
    for (const auto& t : output.GetTypes()) {
      tvec.emplace_back(*t);
    }
  }

  for (auto& node : function_proto->node()) {
    std::string op_type = node.op_type();
    const OpSchema* schema = OpSchemaRegistry::Schema(op_type, function_op.since_version(), function_op.domain());

    // Check that the types of actual inputs, if known, are legal as per schema
    // of called op:
    auto num_formal_inputs = static_cast<size_t>(schema->inputs().size());
    auto num_actual_inputs = static_cast<size_t>(node.input_size());

    for (size_t i = 0; i < num_actual_inputs; ++i) {
      auto actual_param_name = node.input(static_cast<int>(i));
      auto iter = tc_map.find(actual_param_name);
      if (iter != tc_map.end()) {
        // if i >= num_formal_inputs, it is a variadic parameter corresponding
        // to the last formal parameter.
        auto formal_i = std::min(i, num_formal_inputs - 1);
        const auto& types = schema->inputs().at(formal_i).GetTypes();
        std::unordered_set<std::string> allowed_types;
        for (auto& s : types) {
          allowed_types.insert(*s);
        }
        for (auto& actual_type : iter->second) {
          if (allowed_types.find(actual_type) == allowed_types.end()) {
            fail_check(
                "Input type " + actual_type + " of parameter " + actual_param_name + " of function " +
                function_op.Name() + " is not allowed by operator " + op_type);
          }
        }
      }
    }

    // No simple check exists for outputs: we need to integrate type inference
    // to identify the possible output types and verify that they are included
    // in the function-schema.
  }

  ++counter;
}

void VerifyFunction(const OpSchema& op, const FunctionProto* function_proto, int& counter) {
  // Verify function proto is valid
  if (!function_proto) {
    fail_check("Cannot get function body for op '", op.Name(), "'");
  }
  CheckerContext ctx;
  std::unordered_map<std::string, int> op_set;
  auto version_range = OpSchemaRegistry::DomainToVersionRange::Instance().Map().at(op.domain());
  if (op.since_version() > version_range.second || op.since_version() < version_range.first) {
    fail_check("Invalid function version in function op '", op.Name(), "'");
  }

  if (function_proto->opset_import_size() > 0) {
    for (const auto& opset_import : function_proto->opset_import()) {
      op_set.insert({opset_import.domain(), opset_import.version()});
    }
  } else {
    op_set.insert({op.domain(), op.since_version()});
  }

  ctx.set_opset_imports(op_set);
  ctx.set_is_main_graph(false);
  LexicalScopeContext lex_ctx;
  ONNX_TRY {
    check_function(*function_proto, ctx, lex_ctx);
  }
  ONNX_CATCH(ValidationError & ex) {
    ONNX_HANDLE_EXCEPTION([&]() { fail_check(ex.what()); });
  }

  // Verify function op has compatible Type constraints defined in
  // op and function body.
  VerifyTypeConstraint(op, function_proto, counter);
}

// Verify registered ops with function body has compatible
// definition on TypeConstraints between ops and function body
TEST(FunctionVerification, VerifyFunctionOps) {
  const std::vector<OpSchema> schemas = OpSchemaRegistry::get_all_schemas();
  int function_counter = 0, verified_counter = 0;
  for (const auto s : schemas) {
    if (!s.HasFunction())
      continue;
    // Skip test for functions with known errors that need to be fixed:
    // Range currently permits int16 parameters, but the operator Sub, called
    // from the body of Range does not yet support int16 parameter.
    if (s.Name() == "Range")
      continue;
    ONNX_TRY {
      ++function_counter;
      auto function_body = s.GetFunction();
      VerifyFunction(s, function_body, verified_counter);
    }
    ONNX_CATCH(ONNX_NAMESPACE::checker::ValidationError e) {
      ONNX_HANDLE_EXCEPTION([&]() { FAIL() << e.what(); });
    }
  }
  std::cerr << "[          ] Verified " << verified_counter << "/" << function_counter << " Functions." << std::endl;
}

// Verify that FunctionExpandHelper obtains missing default attributes
// from schema and adds them to ops in expanded subgraph.
TEST(FunctionVerification, VerifyFunctionExpandHelper) {
  GraphProto graph;
  NodeProto* new_node = graph.add_node();
  new_node->set_op_type("MeanVarianceNormalization");

  const auto* schema = OpSchemaRegistry::Schema("MeanVarianceNormalization", 9, "");
  const FunctionProto* func = schema->GetFunction();
  const auto default_axes_attribute = schema->attributes().at("axes").default_value;

  FunctionExpandHelper(*new_node, *func, graph);

  for (const auto& node : graph.node()) {
    if (node.op_type() == "ReduceMean") {
      auto attr = node.attribute(0);
      EXPECT_EQ(attr.name(), "axes");
      EXPECT_EQ(attr.ints().size(), default_axes_attribute.ints().size());

      for (int i = 0; i < default_axes_attribute.ints().size(); ++i) {
        EXPECT_EQ(attr.ints(i), default_axes_attribute.ints(i));
      }
      return;
    }
  }
  FAIL() << "During expanding MeanVarianceNormalization function, "
         << "the default attribute `axes` has not been assigned to ReduceMean op.";
}

void RegisterFunctionSchema() {
  ONNX_NAMESPACE::OpSchema function_schema;
  function_schema.SetName("DynamicQuantizeLinear_Fake")
      .SetDomain(AI_ONNX_ML_DOMAIN)
      .SinceVersion(2)
      .SetDoc("Test Op")
      .Input(0, "x", "Input tensor", "T1")
      .Output(0, "y", "Quantized output tensor", "T2")
      .Output(
          1, "y_scale", "Output scale. It's a scalar, which means a per-tensor/layer quantization.", "tensor(float)")
      .Output(2, "y_zero_point", "Output zero point. It's a scalar, which means a per-tensor/layer quantization.", "T2")
      .TypeConstraint("T1", {"tensor(float)"}, "Constrain 'x' to float tensor.")
      .TypeConstraint("T2", {"tensor(uint8)"}, "Constrain 'y_zero_point' and 'y' to 8-bit unsigned integer tensor.")
      .FunctionBody(
          FunctionBodyHelper::BuildNodes(
              {// nodes: {outputs, op, inputs, attributes}
               FunctionBodyHelper::Const<float>("Q_Min", 0.f),
               FunctionBodyHelper::Const<float>("Q_Max", 255.f),
               {{"X_Min"}, "ReduceMin", {"x"}, {MakeAttribute("keepdims", int64_t(0))}},
               {{"X_Min_Adjusted"}, "Min", {"X_Min", "Q_Min"}},
               {{"X_Max"}, "ReduceMax", {"x"}, {MakeAttribute("keepdims", int64_t(0))}},
               {{"X_Max_Adjusted"}, "Max", {"X_Max", "Q_Min"}},
               {{"X_Range"}, "Sub", {"X_Max_Adjusted", "X_Min_Adjusted"}},
               {{"Scale"}, "Div", {"X_Range", "Q_Max"}},
               {{"Min_Scaled"}, "Div", {"X_Min_Adjusted", "Scale"}},
               {{"Initial_ZeroPoint_FP"}, "Sub", {"Q_Min", "Min_Scaled"}},
               {{"Clipped_ZeroPoint_FP"}, "Clip", {"Initial_ZeroPoint_FP", "Q_Min", "Q_Max"}},
               {{"Rounded_ZeroPoint_FP"}, "Round", {"Clipped_ZeroPoint_FP"}},
               {{"Zeropoint"}, "Cast", {"Rounded_ZeroPoint_FP"}, {MakeAttribute("to", int64_t(2))}},
               {{"y_scale"}, "Identity", {"Scale"}},
               {{"y_zero_point"}, "Identity", {"Zeropoint"}},
               {{"y"}, "QuantizeLinear", {"x", "Scale", "Zeropoint"}}}),
          []() {
            std::vector<OperatorSetIdProto> operator_sets(2);
            auto& onnx_opset = operator_sets[0];
            onnx_opset.set_domain("");
            onnx_opset.set_version(13);

            auto& test_opset = operator_sets[1];
            test_opset.set_domain(AI_ONNX_ML_DOMAIN);
            test_opset.set_version(2);

            return operator_sets;
          }());
  ONNX_NAMESPACE::OpSchemaRegistry::OpSchemaRegisterOnce unused(function_schema);
  (void)unused;
}

TEST(FunctionVerification, VerifyFunctionBodyWithMultipleDomains) {
  RegisterFunctionSchema();

  const auto* schema = OpSchemaRegistry::Schema("DynamicQuantizeLinear_Fake", 2, AI_ONNX_ML_DOMAIN);
  EXPECT_TRUE(schema);
  EXPECT_TRUE(schema->HasFunction());
  EXPECT_FALSE(schema->HasContextDependentFunction());

  const FunctionProto* fnProto = schema->GetFunction();
  EXPECT_EQ(fnProto->node_size(), 16);

  LexicalScopeContext lexicalScope;
  CheckerContext checkerCtx;
  std::unordered_map<std::string, int> opset_imports({{AI_ONNX_ML_DOMAIN, 2}, {"", 13}});
  checkerCtx.set_opset_imports(opset_imports);
  checkerCtx.set_ir_version(7);
  check_function(*fnProto, checkerCtx, lexicalScope);
}

TEST(FunctionVerification, VerifyModelLocalFunctions) {
  const char* code = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 13, "custom_domain_1" : 1, "custom_domain_2" : 1],
  producer_name: "FunctionProtoTest",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "A test model for model local functions."
>
agraph (float[N] x) => (uint8[N] out)
{
    o1, o2 = custom_domain_1.bar(x)
    o3 = Add(o1, o2)
    o4 = custom_domain_2.foo(o3)
    out = Identity(o4)
}

<
  domain: "custom_domain_1",
  opset_import: [ "" : 13],
  doc_string: "Test function proto"
>
bar (x) => (o1, o2) {
      o1 = Identity (x)
      o2 = Identity (o1)
}

<
  domain: "custom_domain_2",
  opset_import: [ "" : 13],
  doc_string: "Test function proto"
>
foo (x) => (y) {
      Q_Min = Constant <value = float[1] {0.0}> ()
      Q_Max = Constant <value = float[1] {255.0}> ()
      X_Min = ReduceMin <keepdims = 0> (x)
      X_Max = ReduceMax <keepdims = 0> (x)
      X_Range = Sub (X_Max, X_Min)
      Scale = Div (X_Range, Q_Max)
      ZeroPoint_FP = Sub (Q_Min, Scale)
      Zeropoint = Cast <to = 2> (ZeroPoint_FP)
      y = QuantizeLinear (x, Scale, Zeropoint)
}
)ONNX";

  ModelProto model;
  auto status = OnnxParser::Parse(model, code);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  check_model(model);

  ShapeInferenceOptions options{true, 1, true};
  ONNX_NAMESPACE::shape_inference::InferShapes(model, OpSchemaRegistry::Instance(), options);
}

TEST(FunctionVerification, VerifyNestedModelLocalFunctions) {
  const char* code = R"ONNX(
<
  ir_version: 8,
  opset_import: [ "" : 13, "custom_domain_1" : 1, "custom_domain_2" : 1],
  producer_name: "FunctionProtoTest",
  producer_version: "1.0",
  model_version: 1,
  doc_string: "A test model for model local functions."
>
agraph (float[N] x) => (uint8[N] out)
{
    o1, o2 = custom_domain_1.bar(x)
    o3 = Add(o1, o2)
    o4 = custom_domain_2.foo(o3)
    out = Identity(o4)
}

<
  domain: "custom_domain_1",
  opset_import: [ "" : 13],
  doc_string: "Test function proto"
>
bar (x) => (o1, o2) {
      o1 = Identity (x)
      o2 = Identity (o1)
}

<
  domain: "custom_domain_2",
  opset_import: [ "" : 13, "custom_domain_3" : 1],
  doc_string: "Test function proto"
>
foo (x) => (o4) {
      o1 = custom_domain_3.foo (x)
      o4 = Identity (o1)
}

<
  domain: "custom_domain_3",
  opset_import: [ "" : 13],
  doc_string: "Test function proto"
>
foo (x) => (y) {
      Q_Min = Constant <value = float[1] {0.0}> ()
      Q_Max = Constant <value = float[1] {255.0}> ()
      X_Min = ReduceMin <keepdims = 0> (x)
      X_Max = ReduceMax <keepdims = 0> (x)
      X_Range = Sub (X_Max, X_Min)
      Scale = Div (X_Range, Q_Max)
      ZeroPoint_FP = Sub (Q_Min, Scale)
      Zeropoint = Cast <to = 2> (ZeroPoint_FP)
      y = QuantizeLinear (x, Scale, Zeropoint)
}
)ONNX";

  ModelProto model;
  auto status = OnnxParser::Parse(model, code);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();

  check_model(model);

  ShapeInferenceOptions options{true, 1, true};
  ONNX_NAMESPACE::shape_inference::InferShapes(model, OpSchemaRegistry::Instance(), options);
}

} // namespace Test
} // namespace ONNX_NAMESPACE
