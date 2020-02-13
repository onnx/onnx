#include <iostream>
#include <set>
#include "gtest/gtest.h"
#include "onnx/checker.h"
#include "onnx/common/constants.h"
#include "onnx/defs/schema.h"
#include "onnx/onnx-operators_pb.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
namespace Test {
using namespace checker;
using TENSOR_TYPES_MAP =
    std::unordered_map<std::string, std::vector<std::string>>;

void VerifyTypeConstraint(
    const OpSchema& function_op,
    const FunctionProto* function_proto,
    int& counter) {
  // This is a simple partial type-checker for a function-body.
  // TODO: Revisit to make the type-checker more complete.
  TENSOR_TYPES_MAP tc_map;
  std::set<std::string> primitive_types(
      OpSchema::all_tensor_types().begin(), OpSchema::all_tensor_types().end());
  for (const auto& input : function_op.inputs()) {
    std::string name = input.GetName();
    for (const auto& t : input.GetTypes()) {
      tc_map[name].emplace_back(*t);
    }
  }

  for (const auto& output : function_op.outputs()) {
    std::string name = output.GetName();
    for (const auto& t : output.GetTypes()) {
      tc_map[name].emplace_back(*t);
    }
  }

  for (auto& node : function_proto->node()) {
    std::string op_type = node.op_type();
    const OpSchema* schema = OpSchemaRegistry::Schema(
        op_type, function_op.since_version(), function_op.domain());

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
                "Input type " + actual_type + " of parameter " +
                actual_param_name + " of function " + function_op.Name() +
                " is not allowed by operator " + op_type);
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

void VerifyFunction(
    const OpSchema& op,
    const FunctionProto* function_proto,
    int& counter) {
  // Verify function proto is valid
  if (!function_proto) {
    fail_check("Cannot get function body for op '", op.Name(), "'");
  }
  CheckerContext ctx;
  std::unordered_map<std::string, int> op_set;
  if ((int)function_proto->since_version() != op.since_version()) {
    fail_check(
        "Unmatched since_version defined in function op '", op.Name(), "'");
  }
  auto version_range =
      OpSchemaRegistry::DomainToVersionRange::Instance().Map().at(op.domain());
  if (function_proto->since_version() > version_range.second ||
      function_proto->since_version() < version_range.first) {
    fail_check("Invalid function version in function op '", op.Name(), "'");
  }

  op_set.insert({op.domain(), op.since_version()});
  ctx.set_opset_imports(op_set);
  ctx.set_is_main_graph(false);
  LexicalScopeContext lex_ctx;
  try {
    check_function(*function_proto, ctx, lex_ctx);
  } catch (ValidationError& ex) {
    fail_check(ex.what());
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
    try {
      ++function_counter;
      auto function_body = s.GetFunction();
      VerifyFunction(s, function_body, verified_counter);
    } catch (ONNX_NAMESPACE::checker::ValidationError e) {
      FAIL() << e.what();
    }
  }
  std::cerr << "[          ] Verified " << verified_counter << "/"
            << function_counter << " Functions." << std::endl;
}

// Verify that FunctionExpandHelper obtains missing default attributes
// from schema and adds them to ops in expanded subgraph.
TEST(FunctionVerification, VerifyFunctionExpandHelper) {
  GraphProto graph;
  NodeProto* new_node = graph.add_node();
  new_node->set_op_type("MeanVarianceNormalization");

  const auto* schema =
      OpSchemaRegistry::Schema("MeanVarianceNormalization", 9, "");
  const FunctionProto* func = schema->GetFunction();
  const auto default_axes_attribute =
      schema->attributes().at("axes").default_value;

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
  FAIL()
      << "During expanding MeanVarianceNormalization function, "
      << "the default attribute `axes` has not been assigned to ReduceMean op.";
}
} // namespace Test
} // namespace ONNX_NAMESPACE
