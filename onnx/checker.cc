#include "onnx/checker.h"
#include "onnx/defs/schema.h"
#include "onnx/proto_utils.h"
#include "onnx/shape_inference/implementation.h"
#include "onnx/string_utils.h"

#include <unordered_set>

namespace ONNX_NAMESPACE {
namespace checker {

#define enforce_has_field(proto, field) \
  do {                                  \
    if (!proto.has_##field()) {         \
      fail_check(                       \
          ctx,                          \
          "Field '",                    \
          #field,                       \
          "' of ",                      \
          #proto,                       \
          " is required but missing."); \
    }                                   \
  } while (0)

#define enforce_has_repeated_field(proto, field)                          \
  do {                                                                    \
    if (!proto.field##_size()) {                                          \
      fail_check(                                                         \
          ctx, "Repeated Field '", #field, "' is required but missing."); \
    }                                                                     \
  } while (0)

#define enforce_non_empty_field(proto, field) \
  do {                                        \
    if (proto.field().empty()) {              \
      fail_check(                             \
          ctx,                                \
          "Field '",                          \
          #field,                             \
          "' of ",                            \
          #proto,                             \
          " is required to be non-empty.");   \
    }                                         \
  } while (0)

bool check_name_syntax(const std::string& name) {
  if (name.empty()) {
    return false;
  }

  bool correct = true;

  // Names should adhere to C identifier syntax.
  auto iter = name.cbegin();

  char c = *iter;
  if (!(isalpha(c) || c == '_')) {
    correct = false;
  }

  ++iter;

  for (; iter < name.cend(); ++iter) {
    c = *iter;
    if (!(isalnum(c) || c == '_')) {
      correct = false;
    }
  }

  return correct;
}

bool check_domain_syntax(const std::string& name) {
  if (name.empty()) {
    return false;
  }

  bool correct = true;

  // Names should adhere to C identifier syntax.
  auto iter = name.cbegin();

  char c = *iter;
  if (!isalpha(c)) {
    correct = false;
  }

  ++iter;

  for (; iter < name.cend(); ++iter) {
    c = *iter;
    if (!(isalnum(c) || c == '.')) {
      correct = false;
    }
  }

  return correct;
}

#define enforce_c_identifier(proto, field)       \
  do {                                           \
    const auto& str = proto.field();             \
    if (!check_name_syntax(str)) {               \
      fail_check(                                \
          ctx,                                   \
          "'",                                   \
          str,                                   \
          "' is invalid for " #proto,            \
          ".",                                   \
          #field,                                \
          ". It must use C identifier syntax."); \
    }                                            \
  } while (0)

#define enforce_domain_name_rules(proto, field)     \
  do {                                              \
    const auto& str = proto.field();                \
    if (!check_domain_syntax(str)) {                \
      fail_check(                                   \
          ctx,                                      \
          "'",                                      \
          str,                                      \
          "' is invalid for " #proto,               \
          ".",                                      \
          #field,                                   \
          ". It must be a valid DNS domain name."); \
    }                                               \
  } while (0)

void check_value_info(CheckerContext& ctx, const ValueInfoProto& value_info) {
  enforce_non_empty_field(value_info, name);
  enforce_has_field(value_info, type);
  const auto value_case = value_info.type().value_case();
  switch (value_case) {
    case TypeProto::kTensorType: {
      const auto& type = value_info.type().tensor_type();
      enforce_has_field(type, elem_type);

      if (type.has_shape()) {

        for (auto& dim : type.shape().dim()) {
          const auto dim_case = dim.value_case();
          switch (dim_case) {
            case 0:
              // Special case: treated as a "" (unknown) dimension.
              break;
            case TensorShapeProto::Dimension::kDimValue: {
              if (dim.dim_value() < 0) {
                fail_check(
                    ctx,
                    "Negative dimension size found in the type of '",
                    value_info.name(),
                    "': ",
                    dim.dim_value());
              }
            } break;
            case TensorShapeProto::Dimension::kDimParam: {
              const auto& dim_param = dim.dim_param();
              if (!(dim_param.empty() || dim_param == "*" ||
                    check_name_syntax(dim_param))) {
                fail_check(
                    ctx,
                    "'",
                    dim.dim_param(),
                    "' is an invalid dimension parameter name. It must use C identifier syntax.");
              }
            } break;
            default:
              fail_check(
                  ctx,
                  "Unrecognized dimension value case (value_info name: ",
                  value_info.name(),
                  "): ",
                  dim_case);
          }
        }
      }
    } break;
#ifdef ONNX_ML
    case TypeProto::kSequenceType: {
      const auto& type = value_info.type().sequence_type();
      enforce_has_field(type, elem_type);
    } break;
    case TypeProto::kMapType: {
      const auto& type = value_info.type().map_type();
      enforce_has_field(type, key_type);
      enforce_has_field(type, value_type);
    } break;
#endif
    default:
      fail_check(
          ctx,
          "Unrecognized type value case (value_info name: ",
          value_info.name(),
          "): ",
          value_case);
  }
}

void check_tensor(CheckerContext& ctx, const TensorProto& tensor) {
  enforce_has_field(tensor, data_type);
  if (tensor.data_type() == TensorProto::UNDEFINED) {
    fail_check(
        ctx,
        "setting data_type field (tensor name: ",
        tensor.name(),
        ") to UNDEFINED is not allowed");
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
    fail_check(
        ctx,
        "TensorProto (tensor name: ",
        tensor.name(),
        ") should contain one and only one value field.");
  }
  if (has_raw_data) {
    if (tensor.data_type() == TensorProto::STRING) {
      fail_check(
          ctx,
          "STRING data (tensor name: ",
          tensor.name(),
          ") should not be stored in raw_data field");
    }
    return;
  } else {
#define check_field(field)               \
  if (!has_##field) {                    \
    fail_check(                          \
        ctx,                             \
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
        fail_check(
            ctx,
            "Unrecognized data_type (tensor name: ",
            tensor.name(),
            "): ",
            tensor.data_type());
    }
  }

#undef check_field
}

// NB: This is a generic "attribute well-formedness" check, it doesn't
// actually test if an attribute is valid per a schema
void check_attribute(
    CheckerContext& ctx,
    const AttributeProto& attr,
    const LexicalScopeContext& lex_ctx) {
  enforce_non_empty_field(attr, name);
  enforce_c_identifier(attr, name);

  if (ctx.get_ir_version() >= 0x00000002) {
    enforce_has_field(attr, type);
  }

  int used_fields = 0;

  if (!attr.has_type()) {
    fail_check(ctx, "type field not set in attribute ", attr.name(), ".");
  } else {
    switch (attr.type()) {
      case AttributeProto::FLOAT:
      case AttributeProto::INT:
      case AttributeProto::STRING:
      case AttributeProto::TENSOR:
      case AttributeProto::GRAPH:
      case AttributeProto::FLOATS:
      case AttributeProto::INTS:
      case AttributeProto::STRINGS:
      case AttributeProto::TENSORS:
      case AttributeProto::GRAPHS: {
      } break;
      default: {
        fail_check(ctx, "invalid type field in attribute ", attr.name(), ".");
      } break;
    }
  }

#define check_type(expected_type)                           \
  if (attr.has_type() && attr.type() != expected_type) {    \
    fail_check(                                             \
        ctx,                                                \
        "type field and data field mismatch in attribute ", \
        attr.name(),                                        \
        ".");                                               \
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

  if (ctx.is_main_graph()) {
    // 0 is a valid number, because some PB writers won't distinguish empty
    // values from absent values.
    if (used_fields > 1) {
      fail_check(
          ctx,
          "Attribute (name: ",
          attr.name(),
          ") should contain one and only one value field.");
    }
  } else {
    // It's an attribute of a node in function body.
    if (used_fields != 1 && (used_fields != 0 || !attr.has_ref_attr_name())) {
      fail_check(
          ctx,
          "Attribute (name: ",
          attr.name(),
          ") should contain one value field or refer to attribute declared in function.");
    }
  }

  if (attr.has_t()) {
    check_tensor(ctx, attr.t());
  }

  if (attr.has_g()) {
    check_graph(ctx, attr.g(), lex_ctx);
  }

  for (const auto& tensor : attr.tensors()) {
    check_tensor(ctx, tensor);
  }
  for (const auto& graph : attr.graphs()) {
    check_graph(ctx, graph, lex_ctx);
  }
}

void check_node(
    CheckerContext& ctx,
    const NodeProto& node,
    const GraphProto& graph,
    const LexicalScopeContext& lex_ctx) {
  check_node(ctx, node, lex_ctx);

  // Resolve domain for node
  auto domain = node.domain();

  const auto& opset_imports = ctx.get_opset_imports();
  auto dit = opset_imports.find(domain);
  if (dit == opset_imports.end()) {
    fail_check(ctx, "No opset import for '" + domain + "'");
  }

  auto domain_version = dit->second;

  const auto* schema = ctx.get_schema_registry()->GetSchema(
      node.op_type(), domain_version, domain);

  schema->Verify(ctx, node, &graph);
}

void check_node(
    CheckerContext& ctx,
    const NodeProto& node,
    const LexicalScopeContext& lex_ctx) {
  enforce_non_empty_field(node, op_type);

  if (node.input().empty() && node.output().empty()) {
    fail_check(
        ctx,
        "NodeProto (name: ",
        node.name(),
        ", type: ",
        node.op_type(),
        ") has zero input and zero output.");
  }

  if (node.has_name() && node.name() != "") {
    enforce_c_identifier(node, name);
  }

  // No need to validate node input syntax -- the error would already have been
  // reported elsewhere.

  for (auto& out : node.output()) {
    if (!(check_name_syntax(out) ||
          out == "")) { // Optional outputs are named ""
      fail_check(
          ctx,
          "'",
          out,
          "' is an invalid node output name. It must use C identifier syntax.");
    }
  }

  // Resolve domain for node
  auto domain = node.domain();

  const auto& opset_imports = ctx.get_opset_imports();
  auto dit = opset_imports.find(domain);
  if (dit == opset_imports.end()) {
    fail_check(ctx, "No opset import for '" + domain + "'");
  }

  auto domain_version = dit->second;

  for (const auto& attr : node.attribute()) {
    check_attribute(ctx, attr, lex_ctx);
  }

  const auto* schema = ctx.get_schema_registry()->GetSchema(
      node.op_type(), domain_version, domain);

  schema->Verify(ctx, node, nullptr);
}

void check_graph(
    CheckerContext& ctx,
    const GraphProto& graph,
    const LexicalScopeContext& parent_lex) {
  enforce_non_empty_field(graph, name);
  enforce_c_identifier(graph, name);

  for (const auto& value_info : graph.input()) {
    check_value_info(ctx, value_info);
    if (!check_name_syntax(value_info.name())) {
      fail_check(
          ctx,
          "'",
          value_info.name(),
          "' is an invalid graph input name. It must use C identifier syntax.");
    }
  }
  for (const auto& value_info : graph.output()) {
    check_value_info(ctx, value_info);
    if (!check_name_syntax(value_info.name())) {
      fail_check(
          ctx,
          "'",
          value_info.name(),
          "' is an invalid graph output name. It must use C identifier syntax.");
    }
  }
  for (const auto& value_info : graph.value_info()) {
    check_value_info(ctx, value_info);
    if (!check_name_syntax(value_info.name())) {
      fail_check(
          ctx,
          "'",
          value_info.name(),
          "' is an invalid value name. It must use C identifier syntax.");
    }
  }

  std::unordered_set<std::string> node_names{};
  for (const auto& node : graph.node()) {
    if (node.has_name() && node.name() != "") {
      auto count = node_names.count(node.name());
      if (count > 0) {
        fail_check(ctx, "Duplicate node name: '", node.name(), "'");
      }
      node_names.insert(node.name());
    }
  }

  std::unordered_set<std::string> output_names{};
  // Inherit values avaiailable in outer scope
  // Note that we do not allow shadowing, so the presence of an already-defined
  // name is always an error.
  for (const auto& value_info : graph.input()) {
    if (output_names.count(value_info.name())) {
      fail_check(
          ctx,
          "Graph must be in single static assignment (SSA) form, however '",
          value_info.name(),
          "' has been used as graph input names multiple times.");
    }
    output_names.insert(value_info.name());
  }
  output_names.insert(
      parent_lex.output_names.begin(), parent_lex.output_names.end());
  for (const auto& init : graph.initializer()) {
    if (!init.has_name() || init.name() == "") {
      fail_check(ctx, "Orphaned (no name) initializer value found.");
    }
    if (!output_names.count(init.name())) {
      fail_check(ctx, init.name() + " in initializer but not in graph input.");
    }
    enforce_c_identifier(init, name);
    check_tensor(ctx, init);
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
            ctx,
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
      check_node(ctx, node, graph, lex_ctx);
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
            ctx,
            "Graph must be in single static assignment (SSA) form, however '",
            output,
            "' has been used as output names multiple times.");
      }
      output_names.insert(output);
    }
  }
}

void check_function(
    CheckerContext& ctx,
    const FunctionProto& function,
    const LexicalScopeContext& parent_lex) {
  enforce_non_empty_field(function, name);
  enforce_has_field(function, since_version);

  std::unordered_set<std::string> output_names;
  for (const auto& input : function.input()) {
    auto result = output_names.insert(input);
    if (!result.second) {
      fail_check(
          ctx,
          "function (",
          function.name(),
          ") should not have duplicate inputs specified.");
    }
  }
  std::unordered_set<std::string> outputs;
  for (const auto& output : function.output()) {
    auto result = outputs.insert(output);
    if (!result.second) {
      fail_check(
          ctx,
          "function (",
          function.name(),
          ") should not have duplicate outputs specified.");
    }
  }
  std::unordered_set<std::string> attrs;
  for (const auto& attr : function.attribute()) {
    auto result = attrs.insert(attr);
    if (!result.second) {
      fail_check(
          ctx,
          "function (",
          function.name(),
          ") should not have duplicate attributes specified.");
    }
  }

  for (const auto& node : function.node()) {
    // nodes must be in topologically sorted order
    for (const auto& input : node.input()) {
      // explicit optional input
      if (input.empty()) {
        continue;
      }
      if (!output_names.count(input)) {
        fail_check(
            ctx,
            "Nodes in a function must be topologically sorted, however input '",
            input,
            "' of node: \n",
            ProtoDebugString(node),
            "\n is neither output of any previous nodes nor input of the function.");
      }
    }

    LexicalScopeContext lex_ctx;
    lex_ctx.output_names = output_names;
    check_node(ctx, node, lex_ctx);
    // check for SSA form
    for (const auto& output : node.output()) {
      // optional output
      if (output.empty()) {
        continue;
      }
      if (output_names.count(output)) {
        fail_check(
            ctx,
            "Function must be in single static assignment (SSA) form, however '",
            output,
            "' has been used as output names multiple times.");
      }
      output_names.insert(output);
    }
  }
}

void check_model(CheckerContext& ctx, const ModelProto& m) {
  ModelProto model(m);

  if (!model.ir_version()) {
    fail_check(ctx, "The model does not have an ir_version set properly.");
  }

  const auto ir_version = model.ir_version();

  if (ir_version < 1) {
    fail_check(
        ctx, "Your model's ir_version is invalid or no longer supported.");
  }
  if (ir_version > IR_VERSION) {
    fail_check(ctx, "Your model's ir_version is higher than the checker's.");
  }
  if (ir_version > 2) {
    if (model.domain().empty()) {
      fail_check(ctx, "The model does not have a domain defined.");
    } else {
      enforce_domain_name_rules(model, domain);
    }
  }

  if (model.metadata_props_size() > 1) {
    std::unordered_set<std::string> keys;
    for (const StringStringEntryProto& entry : model.metadata_props()) {
      auto i = keys.insert(entry.key());
      if (!i.second) {
        fail_check(ctx, "Your model has duplicate keys in metadata_props.");
      }
    }
  }
  std::unordered_map<std::string, int> versions;
  ctx.set_ir_version(static_cast<int>(model.ir_version()));

  auto domain_map = OpSchemaRegistry::DomainToVersionRange::Instance().Map();

  std::unordered_map<std::string, int> opset_imports;
  for (const auto& opset_import : model.opset_import()) {
    auto domain = opset_import.domain();
    auto version = static_cast<int>(opset_import.version());
    auto known_versions = domain_map[domain];

    if (version < known_versions.first || version > known_versions.second) {
      fail_check(
          ctx,
          "The import of '",
          domain == ONNX_DOMAIN ? "ONNX" : domain,
          "' refers to an unknown version of the operator set.");
    }

    opset_imports[domain] = version;
  }

  auto onnx_import = opset_imports.find("");
  if (onnx_import == opset_imports.end()) {
    // Add an import of the latest onnx default operator set.
    auto onnxDomain = domain_map[std::string(ONNX_DOMAIN)];
    opset_imports[ONNX_DOMAIN] = onnxDomain.second;
  }

  if (model.ir_version() < 3) {
    if (opset_imports.empty())
      opset_imports[ONNX_DOMAIN] = 1;
    else
      fail_check(ctx, "model with IR version < 3 cannot import operator sets.");
  }

  if (ctx.get_schema_registry() != nullptr) {
    // Having performed shape and type inference will allow us to do more
    // in-depth checking of the model, specifically whether node arguments match
    // the signatures of the operators that are invoked.
    shape_inference::InferShapes(model);
  }

  ctx.set_opset_imports(opset_imports);
  LexicalScopeContext lex_ctx;
  check_graph(ctx, model.graph(), lex_ctx);
}

#undef fail_check
#undef enforce_has_field
#undef enforce_has_repeated_field
#undef enforce_non_empty_field

} // namespace checker
} // namespace ONNX_NAMESPACE
