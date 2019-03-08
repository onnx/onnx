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
    int& counter
) {
  // TC for function nodes should satisfy the definition defined in the opschema
  // This is designed to be a best-effort test
  // TODO: Revisit to have a more consummate check on it
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

    std::unordered_map<std::string, int> input_tensor_name_idx_map;
    std::unordered_map<std::string, int> output_tensor_name_idx_map;
    // Enforce it on input
    for (unsigned int i = 0; i < schema->inputs().size(); ++i) {
      auto& input = schema->inputs().at(i);
      input_tensor_name_idx_map[input.GetName()] = i;
    }
    for (auto& tensor_name_tc : tc_map) {
      auto iter = input_tensor_name_idx_map.find(tensor_name_tc.first);
      if (iter == input_tensor_name_idx_map.end())
        continue;
      const auto& types = schema->inputs().at(iter->second).GetTypes();
      std::unordered_set<std::string> allowed_types;
      for (auto& s : types) {
        allowed_types.insert(*s);
      }
      for (auto& type : tensor_name_tc.second) {
        if (allowed_types.find(type) == allowed_types.end()) {
          fail_check(
              "Input type " + type + " defined in " + schema->Name() +
              "'s function body is not allowed in node " + op_type);
        }
      }
    }

    // Enforce it on output
    for (unsigned int i = 0; i < schema->outputs().size(); ++i) {
      auto& output = schema->outputs().at(i);
      output_tensor_name_idx_map[output.GetName()] = i;
    }

    for (auto& tensor_name_tc : tc_map) {
      auto iter = output_tensor_name_idx_map.find(tensor_name_tc.first);
      if (iter == output_tensor_name_idx_map.end())
        continue;
      const auto& types = schema->outputs().at(iter->second).GetTypes();
      std::unordered_set<std::string> allowed_types;
      for (auto& s : types) {
        allowed_types.insert(*s);
      }
      for (auto& type : tensor_name_tc.second) {
        if (allowed_types.find(type) == allowed_types.end()) {
          fail_check(
              "Output type " + type + " defined in " + schema->Name() +
              "'s function body is not allowed in node " + op_type);
        }
      }
    }
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
  if ((int)function_proto->since_version() != op.since_version()) {
    fail_check("Unmatched since_version defined in function op '", op.Name(), "'");
  }
  auto version_range =
      OpSchemaRegistry::DomainToVersionRange::Instance().Map().at(
          op.domain());
  if (function_proto->since_version() > version_range.second ||
      function_proto->since_version() < version_range.first) {
    fail_check("Invalid function version in function op '", op.Name(), "'");
  }
  
  op_set.insert(
      {op.domain(), op.since_version()});
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
    if (!s.HasFunction()) continue;
    try{
      ++function_counter;
      auto function_body = s.GetFunction();
      VerifyFunction(s, function_body, verified_counter);
    }catch (ONNX_NAMESPACE::checker::ValidationError e){
      FAIL() << e.what();
    }
  }
  std::cerr << "[          ] Verified " << verified_counter << "/" 
    << function_counter << " Functions." << std::endl;
}

} // namespace Test
} // namespace ONNX_NAMESPACE
