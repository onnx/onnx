#pragma once

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/onnx-operators_pb.h"
#include "onnx/onnx_pb.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {
namespace checker {
class ValidationError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
  const char* what() const noexcept override {
    if (!expanded_message_.empty()) {
      return expanded_message_.c_str();
    }
    return std::runtime_error::what();
  }
  void AppendContext(const std::string& context) {
    expanded_message_ = ONNX_NAMESPACE::MakeString(
        std::runtime_error::what(), "\n\n==> Context: ", context);
  }

 private:
  std::string expanded_message_;
};

#define fail_check(...)                           \
  throw ONNX_NAMESPACE::checker::ValidationError( \
      ONNX_NAMESPACE::MakeString(__VA_ARGS__));

class CheckerContext final {
 public:
  int get_ir_version() const {
    return ir_version_;
  }
  void set_ir_version(int v) {
    ir_version_ = v;
  }
  const std::unordered_map<std::string, int>& get_opset_imports() const {
    return opset_imports_;
  }
  void set_opset_imports(std::unordered_map<std::string, int> imps) {
    opset_imports_ = std::move(imps);
  }
  bool is_main_graph() const {
    return is_main_graph_;
  }
  void set_is_main_graph(bool is_main_graph) {
    is_main_graph_ = is_main_graph;
  }

  void set_schema_registry(const ISchemaRegistry* schema_registry) {
    schema_registry_ = schema_registry;
  }

  const ISchemaRegistry* get_schema_registry() const {
    return schema_registry_;
  }

  void set_func_registry(const IFunctionBuilderRegistry* func_registry) {
    func_registry_ = func_registry;
  }

  const IFunctionBuilderRegistry* get_func_registry() const {
    return func_registry_;
  }

  explicit CheckerContext() : ir_version_(-1) {}

 private:
  int ir_version_;
  std::unordered_map<std::string, int> opset_imports_;
  bool is_main_graph_ = true;
  const ISchemaRegistry* schema_registry_ = OpSchemaRegistry::Instance();
  const IFunctionBuilderRegistry* func_registry_ =
      &FunctionBuilderRegistry::OnnxInstance();
};

struct LexicalScopeContext final {
  std::unordered_set<std::string> output_names;
};

using IR_VERSION_TYPE = decltype(Version::IR_VERSION);
void check_value_info(const ValueInfoProto& value_info, const CheckerContext&);
void check_tensor(const TensorProto& tensor, const CheckerContext&);
void check_attribute(
    const AttributeProto& attr,
    const CheckerContext&,
    const LexicalScopeContext&);
void check_node(
    const NodeProto& node,
    const CheckerContext&,
    const LexicalScopeContext&);
void check_graph(
    const GraphProto& graph,
    const CheckerContext&,
    const LexicalScopeContext&);
void check_function(
    const FunctionProto& function,
    const CheckerContext&,
    const LexicalScopeContext&);

void check_model(const ModelProto& model);

void VerifyFunctionNode(
    const NodeProto&,
    const FunctionProto&,
    const CheckerContext&,
    const LexicalScopeContext&);

} // namespace checker
} // namespace ONNX_NAMESPACE
