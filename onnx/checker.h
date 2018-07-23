#pragma once

#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
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

#define fail_check(ctx, ...) \
  ctx.log_error(ONNX_NAMESPACE::MakeString(__VA_ARGS__));

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

  void log_error(const std::string& error) {
    if (!error.empty()) {
      errors_.append(error);
      errors_.append("\n");
    }
  }

  void raise_error() {
    if (!errors_.empty()) {
      throw ValidationError(errors_);
    }
  }
  explicit CheckerContext() : ir_version_(-1) {}

  CheckerContext(const CheckerContext& src)
      : ir_version_(src.ir_version_),
        opset_imports_(src.opset_imports_),
        is_main_graph_(src.is_main_graph_),
        schema_registry_(src.schema_registry_),
        errors_(src.errors_) {}

 private:
  int ir_version_;
  std::unordered_map<std::string, int> opset_imports_;
  bool is_main_graph_ = true;
  const ISchemaRegistry* schema_registry_ = OpSchemaRegistry::Instance();
  std::string errors_;
};

struct LexicalScopeContext final {
  std::unordered_set<std::string> output_names;
};

using IR_VERSION_TYPE = decltype(Version::IR_VERSION);
void check_value_info(const ValueInfoProto& value_info, CheckerContext& ctx);
void check_tensor(const TensorProto& tensor, CheckerContext& ctx);
void check_attribute(
    const AttributeProto& attr,
    CheckerContext& ctx,
    const LexicalScopeContext&);
void check_node(
    const NodeProto& node,
    CheckerContext& ctx,
    const LexicalScopeContext&);
void check_node(
    const NodeProto& node,
    const GraphProto& graph,
    CheckerContext& ctx,
    const LexicalScopeContext&);
void check_graph(
    const GraphProto& graph,
    CheckerContext& ctx,
    const LexicalScopeContext&);
void check_function(
    const FunctionProto& function,
    CheckerContext& ctx,
    const LexicalScopeContext&);

void check_model(const ModelProto& model, CheckerContext& ctx);
} // namespace checker
} // namespace ONNX_NAMESPACE
