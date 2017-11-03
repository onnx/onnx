#include "onnx/checker.h"

#include "onnx/defs/schema.h"
#include "onnx/string_utils.h"

#include <unordered_set>

namespace onnx {
namespace checker {

#define fail(...) throw ValidationError(onnx::MakeString(__VA_ARGS__));

#define enforce_has_field(proto, field)                                      \
  do {                                                                       \
    if (!proto.has_##field()) {                                              \
      fail("Field '", #field, "' of ", #proto, " is required but missing."); \
    }                                                                        \
  } while (0)

#define enforce_has_repeated_field(proto, field)                      \
  do {                                                                \
    if (!proto.field##_size()) {                                      \
      fail("Repeated Field '", #field, "' is required but missing."); \
    }                                                                 \
  } while (0)

#define enforce_non_empty_field(proto, field) \
  do {                                        \
    if (proto.field().empty()) {              \
      fail(                                   \
          "Field '",                          \
          #field,                             \
          "' of ",                            \
          #proto,                             \
          " is required to be non-empty.");   \
    }                                         \
  } while (0)

void check_value_info(const ValueInfoProto& value_info, int ir_version) {
  enforce_non_empty_field(value_info, name);
  enforce_has_field(value_info, type);
  const auto value_case = value_info.type().value_case();
  switch (value_case) {
    case TypeProto::kTensorType: {
      const auto& type = value_info.type().tensor_type();
      enforce_has_field(type, elem_type);
      enforce_has_field(type, shape);
    } break;
    case TypeProto::kSparseTensorType: {
      const auto& type = value_info.type().sparse_tensor_type();
      enforce_has_field(type, elem_type);
      enforce_has_field(type, shape);
    } break;
    default:
      fail("Unrecognized type value case: ", value_case);
  }
}

void check_tensor(const TensorProto& tensor, int ir_version) {
  enforce_has_field(tensor, data_type);
  if (tensor.data_type() == TensorProto::UNDEFINED) {
    fail("setting data_type field to UNDEFINED is not allowed");
  }

  enforce_has_repeated_field(tensor, dims);

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
    fail("TensorProto should contain one and only one value field.");
  }
  if (has_raw_data) {
    if (tensor.data_type() == TensorProto::STRING) {
      fail("STRING data should not be stored in raw_data field");
    }
    return;
  } else {
#define check_field(field)               \
  if (!has_##field) {                    \
    fail(                                \
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
        fail("Unrecognized data_type: ", tensor.data_type());
    }
  }

#undef check_field
}

void check_attribute(const AttributeProto& attr, int ir_version) {
  enforce_non_empty_field(attr, name);

  if (ir_version >= 0x00000002) {
    enforce_has_field(attr, type);
  }

  int used_fields = 0;

#define check_type(expected_type)                        \
  if (attr.has_type() && attr.type() != expected_type) { \
    fail("type field and data field mismatch.");         \
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
    fail("Attribute should contain one and only one value field.");
  }

  for (const auto& tensor : attr.tensors()) {
    check_tensor(tensor, ir_version);
  }
  for (const auto& graph : attr.graphs()) {
    check_graph(graph, ir_version);
  }
}

void check_node(const NodeProto& node, int ir_version) {
  enforce_non_empty_field(node, op_type);

  if (node.input().empty() && node.output().empty()) {
    fail("NodeProto has zero input and zero output.");
  }

  for (const auto& attr : node.attribute()) {
    check_attribute(attr, ir_version);
  }

  const auto* schema = OpSchemaRegistry::Schema(node.op_type());
  if (!schema) {
    fail("No Schema registered for " + node.op_type());
  }
  if (!schema->Verify(node)) {
    fail("NodeProto of type " + node.op_type() + " did not pass schema check.");
  }
}

void check_graph(const GraphProto& graph, int ir_version) {
  enforce_non_empty_field(graph, name);

  for (const auto& value_info : graph.input()) {
    check_value_info(value_info, ir_version);
  }
  for (const auto& value_info : graph.output()) {
    check_value_info(value_info, ir_version);
  }
  for (const auto& node : graph.node()) {
    check_node(node, ir_version);
  }

  std::unordered_set<std::string> input_names{};
  for (const auto& value_info : graph.input()) {
    input_names.insert(value_info.name());
  }
  for (const auto& init : graph.initializer()) {
    if (!input_names.count(init.name())) {
      fail(init.name() + " in initializer but not in graph input");
    }
    check_tensor(init, ir_version);
  }
}

void check_model(const ModelProto& model, int ir_version) {
  if (!model.ir_version()) {
    fail("The model does not have an ir_version set properly.");
  }
  if (model.ir_version() > ir_version) {
    fail("Your model ir_version is higher than the checker's.");
  }
  check_graph(model.graph(), model.ir_version());
}

#undef fail
#undef enforce_has_field
#undef enforce_has_repeated_field
#undef enforce_non_empty_field

} // namespace checker
} // namespace onnx
