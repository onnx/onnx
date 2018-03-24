#include "onnx/checker.h"
#include "onnx/proto_utils.h"

#include "onnx/defs/schema.h"

#include <unordered_set>

namespace ONNX_NAMESPACE {
namespace checker {

#define enforce_has_field(proto, field)                                     \
  do {                                                                      \
    if (!proto.has_##field()) {                                             \
      fail_check(                                                           \
          "Field '", #field, "' of ", #proto, " is required but missing."); \
    }                                                                       \
  } while (0)

#define enforce_has_repeated_field(proto, field)                            \
  do {                                                                      \
    if (!proto.field##_size()) {                                            \
      fail_check("Repeated Field '", #field, "' is required but missing."); \
    }                                                                       \
  } while (0)

#define enforce_non_empty_field(proto, field) \
  do {                                        \
    if (proto.field().empty()) {              \
      fail_check(                             \
          "Field '",                          \
          #field,                             \
          "' of ",                            \
          #proto,                             \
          " is required to be non-empty.");   \
    }                                         \
  } while (0)

void check_value_info(const ValueInfoProto& value_info, const CheckerContext&) {
  enforce_non_empty_field(value_info, name);
  enforce_has_field(value_info, type);
  const auto value_case = value_info.type().value_case();
  switch (value_case) {
    case TypeProto::kTensorType: {
      const auto& type = value_info.type().tensor_type();
      enforce_has_field(type, elem_type);
      enforce_has_field(type, shape);
    } break;
    default:
      fail_check("Unrecognized type value case: ", value_case);
  }
}

void check_tensor(const TensorProto& tensor, const CheckerContext& ctx) {
  enforce_has_field(tensor, data_type);
  if (tensor.data_type() == TensorProto::UNDEFINED) {
    fail_check("setting data_type field to UNDEFINED is not allowed");
  }

  int num_value_fields = 0;

  const char* value_field = nullptr;

#define check_data_field(field)             \
  bool has_##field = tensor.field().size(); \
  if (has_##field) {                        \
    ++num_value_fields;                     \
    value_field = #field;                   \
  }

  check_data_field(float_data);
  check_data_field(int32_data);
  check_data_field(string_data);
  check_data_field(int64_data);
  check_data_field(raw_data);
  check_data_field(double_data);
  check_data_field(uint64_data);

#undef check_data_field

  if (num_value_fields != 1) {
    fail_check("TensorProto should contain one and only one value field.");
  }
  if (has_raw_data) {
    if (tensor.data_type() == TensorProto::STRING) {
      fail_check("STRING data should not be stored in raw_data field");
    }
    return;
  } else {
#define check_field(field)               \
  if (!has_##field) {                    \
    fail_check(                          \
        "values of data_type '",         \
        tensor.data_type(),              \
        "' should be stored in field '", \
        #field,                          \
        "' instead of '",                \
        value_field,                     \
        "'");                            \
  }

    switch (tensor.data_type()) {
      case TensorProto::FLOAT:
      case TensorProto::COMPLEX64:
      case TensorProto::COMPLEX128:
        check_field(float_data);
        break;

      case TensorProto::DOUBLE:
        check_field(double_data);
        break;

      case TensorProto::INT32:
      case TensorProto::UINT16:
      case TensorProto::BOOL:
        check_field(int32_data);
        break;

      case TensorProto::INT64:
        check_field(int64_data);
        break;

      case TensorProto::UINT32:
      case TensorProto::UINT64:
        check_field(uint64_data);
        break;

      case TensorProto::STRING:
        check_field(string_data);
        break;

      default:
        fail_check("Unrecognized data_type: ", tensor.data_type());
    }
  }

#undef check_field
}

// NB: This is a generic "attribute well-formedness" check, it doesn't
// actually test if an attribute is valid per a schema
void check_attribute(
    const AttributeProto& attr,
    const CheckerContext& ctx,
    const LexicalScopeContext& lex_ctx) {
  enforce_non_empty_field(attr, name);

  if (ctx.get_ir_version() >= 0x00000002) {
    enforce_has_field(attr, type);
  }

  int used_fields = 0;

#define check_type(expected_type)                        \
  if (attr.has_type() && attr.type() != expected_type) { \
    fail_check("type field and data field mismatch.");   \
  }

#define check_singular_field(field, type) \
  if (attr.has_##field()) {               \
    ++used_fields;                        \
    check_type(type);                     \
  }

#define check_repeated_field(field, type) \
  if (attr.field##_size() > 0) {          \
    ++used_fields;                        \
    check_type(type);                     \
  }

  check_singular_field(f, AttributeProto::FLOAT);
  check_singular_field(i, AttributeProto::INT);
  check_singular_field(s, AttributeProto::STRING);
  check_singular_field(t, AttributeProto::TENSOR);
  check_singular_field(g, AttributeProto::GRAPH);
  check_repeated_field(floats, AttributeProto::FLOATS);
  check_repeated_field(ints, AttributeProto::INTS);
  check_repeated_field(strings, AttributeProto::STRINGS);
  check_repeated_field(tensors, AttributeProto::TENSORS);
  check_repeated_field(graphs, AttributeProto::GRAPHS);

#undef check_type
#undef check_singular_field
#undef check_repeated_field

  if (used_fields != 1) {
    fail_check("Attribute should contain one and only one value field.");
  }

  if (attr.has_t()) {
    check_tensor(attr.t(), ctx);
  }

  if (attr.has_g()) {
    check_graph(attr.g(), ctx, lex_ctx);
  }

  for (const auto& tensor : attr.tensors()) {
    check_tensor(tensor, ctx);
  }
  for (const auto& graph : attr.graphs()) {
    check_graph(graph, ctx, lex_ctx);
  }
}

void check_node(
    const NodeProto& node,
    const CheckerContext& ctx,
    const LexicalScopeContext& lex_ctx) {
  enforce_non_empty_field(node, op_type);

  if (node.input().empty() && node.output().empty()) {
    fail_check("NodeProto has zero input and zero output.");
  }

  // Resolve domain for node
  const auto& opset_imports = ctx.get_opset_imports();
  auto dit = opset_imports.find(node.domain());
  if (dit == opset_imports.end()) {
    fail_check("No opset import for domain '" + node.domain() + "'");
  }
  auto domain_version = dit->second;

  for (const auto& attr : node.attribute()) {
    check_attribute(attr, ctx, lex_ctx);
  }

  const auto* schema =
      OpSchemaRegistry::Schema(node.op_type(), domain_version, node.domain());
  if (!schema) {
    fail_check(
        "No Schema registered for " + node.op_type() +
        " with domain_version of " + ONNX_NAMESPACE::to_string(domain_version));
  }
  schema->Verify(node);
}

void check_graph(
    const GraphProto& graph,
    const CheckerContext& ctx,
    const LexicalScopeContext& parent_lex) {
  enforce_non_empty_field(graph, name);

  for (const auto& value_info : graph.input()) {
    check_value_info(value_info, ctx);
  }
  for (const auto& value_info : graph.output()) {
    check_value_info(value_info, ctx);
  }

  std::unordered_set<std::string> output_names{};
  // Inherit values avaiailable in outer scope
  // Note that we do not allow shadowing, so the presence of an already-defined
  // name is always an error.
  for (const auto& value_info : graph.input()) {
    if (output_names.count(value_info.name())) {
      fail_check(
          "Graph must be in SSA form, however '",
          value_info.name(),
          "' has been used as graph input names multiple times.");
    }
    output_names.insert(value_info.name());
  }
  output_names.insert(
      parent_lex.output_names.begin(), parent_lex.output_names.end());
  for (const auto& init : graph.initializer()) {
    if (!output_names.count(init.name())) {
      fail_check(init.name() + " in initializer but not in graph input");
    }
    check_tensor(init, ctx);
  }

  for (const auto& node : graph.node()) {
    // nodes must be in topologically sorted order
    for (const auto& input : node.input()) {
      // explicit optional input
      if (input.empty()) {
        continue;
      }
      if (!output_names.count(input)) {
        fail_check(
            "Nodes in a graph must be topologically sorted, however input '",
            input,
            "' of node: \n",
            ProtoDebugString(node),
            "\n is not output of any previous nodes.");
      }
    }
    // This needs to happen before SSA check since we don't want to recurse and
    // find that outputs from control flow ops are colliding with names in the
    // inner block
    LexicalScopeContext lex_ctx;
    lex_ctx.output_names = output_names;
    try {
      check_node(node, ctx, lex_ctx);
    } catch (ValidationError& ex) {
      ex.AppendContext("Bad node spec: " + ProtoDebugString(node));
      throw ex;
    }
    // check for SSA form
    for (const auto& output : node.output()) {
      // optional output
      if (output.empty()) {
        continue;
      }
      if (output_names.count(output)) {
        fail_check(
            "Graph must be in SSA form, however '",
            output,
            "' has been used as output names multiple times.");
      }
      output_names.insert(output);
    }
  }
}

void check_model(const ModelProto& model) {
  if (!model.ir_version()) {
    fail_check("The model does not have an ir_version set properly.");
  }
  if (model.ir_version() > IR_VERSION) {
    fail_check("Your model ir_version is higher than the checker's.");
  }
  if (model.metadata_props_size() > 1) {
    std::unordered_set<std::string> keys;
    for (const StringStringEntryProto& entry : model.metadata_props()) {
      auto i = keys.insert(entry.key());
      if (!i.second) {
        fail_check("Your model has duplicate keys in metadata_props.");
      }
    }
  }
  std::unordered_map<std::string, int> versions;
  CheckerContext ctx;
  ctx.set_ir_version(static_cast<int>(model.ir_version()));
  std::unordered_map<std::string, int> opset_imports;
  for (const auto& opset_import : model.opset_import()) {
    opset_imports[opset_import.domain()] = static_cast<int>(opset_import.version());
  }
  auto dit = opset_imports.find(ONNX_DOMAIN);
  if (dit == opset_imports.end()) {
    if (model.ir_version() >= 3) {
      fail_check(
          "model with IR version >= 3 must specify opset_import for ONNX");
    } else {
      opset_imports[ONNX_DOMAIN] = 1;
    }
  }
  ctx.set_opset_imports(opset_imports);
  LexicalScopeContext lex_ctx;
  check_graph(model.graph(), ctx, lex_ctx);
}

#undef fail_check
#undef enforce_has_field
#undef enforce_has_repeated_field
#undef enforce_non_empty_field

} // namespace checker
} // namespace ONNX_NAMESPACE
