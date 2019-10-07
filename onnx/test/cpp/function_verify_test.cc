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

// Unfortunately, the different data-structures use different int types to
// represent a parameter-index.
using parameter_index = unsigned int;

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
      if (!primitive_types.count(*t)) {
        return; // skip variable types check for now
      }
      tc_map[name].emplace_back(*t);
    }
  }

  for (const auto& output : function_op.outputs()) {
    std::string name = output.GetName();
    for (const auto& t : output.GetTypes()) {
      if (!primitive_types.count(*t)) {
        return; // skip variable types check for now
      }
      tc_map[name].emplace_back(*t);
    }
  }

  for (auto& node : function_proto->node()) {
    std::string op_type = node.op_type();
    const OpSchema* schema = OpSchemaRegistry::Schema(
        op_type, function_op.since_version(), function_op.domain());

    // Check inputs of node:
    parameter_index num_formal_inputs = schema->inputs().size();
    parameter_index num_actual_inputs = node.input_size();
    // The above may not be same due to optional inputs
    auto num_checked_inputs = std::min(num_formal_inputs, num_actual_inputs);
    for (parameter_index i = 0; i < num_checked_inputs; ++i) {
      auto actual_param_name = node.input(i);
      auto iter = tc_map.find(actual_param_name);
      if (iter != tc_map.end()) {
        const auto& types = schema->inputs().at(i).GetTypes();
        std::unordered_set<std::string> allowed_types;
        for (auto& s : types) {
          allowed_types.insert(*s);
        }
        for (auto& actual_type : iter.second) {
          if (allowed_types.find(actual_type) == allowed_types.end()) {
            fail_check(
                "Input type " + actual_type + " defined in " + schema->Name() +
                "'s function body is not allowed in node " + op_type);
          }
        }
      }
    }

    // Check outputs of node:
    parameter_index num_formal_outputs = schema->outputs().size();
    parameter_index num_actual_outputs = node.output_size();
    auto num_checked_outputs = std::min(num_formal_outputs, num_actual_outputs);
    for (parameter_index i = 0; i < num_checked_outputs; ++i) {
      auto actual_param_name = node.output(i);
      auto iter = tc_map.find(actual_param_name);
      if (iter != tc_map.end()) {
        const auto& types = schema->outputs().at(i).GetTypes();
        std::unordered_set<std::string> allowed_types;
        for (auto& s : types) {
          allowed_types.insert(*s);
        }
        for (auto& actual_type : iter.second) {
          if (allowed_types.find(actual_type) == allowed_types.end()) {
            fail_check(
                "Output type " + actual_type + " defined in " + schema->Name() +
                "'s function body is not allowed in node " + op_type);
          }
        }
      }
    }
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

} // namespace Test
} // namespace ONNX_NAMESPACE
