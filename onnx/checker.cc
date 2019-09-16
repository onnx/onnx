#include "onnx/checker.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/proto_utils.h"
#include "onnx/string_utils.h"

#include <fstream>
#include <iterator>
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

void check_value_info(
    const ValueInfoProto& value_info,
    const CheckerContext& ctx) {
  enforce_non_empty_field(value_info, name);
  // Relax constraint for subgraph input/output.
  if (!ctx.is_main_graph())
    return;
  enforce_has_field(value_info, type);
  const auto value_case = value_info.type().value_case();
  switch (value_case) {
    case TypeProto::kTensorType: {
      const auto& type = value_info.type().tensor_type();
      enforce_has_field(type, elem_type);
      enforce_has_field(type, shape);
    } break;
    case TypeProto::kSequenceType: {
      const auto& type = value_info.type().sequence_type();
      enforce_has_field(type, elem_type);
    } break;
    case TypeProto::kMapType: {
      const auto& type = value_info.type().map_type();
      enforce_has_field(type, key_type);
      enforce_has_field(type, value_type);
    } break;
#ifdef ONNX_ML
    case TypeProto::kOpaqueType:
      break;
    case TypeProto::kSparseTensorType: {
      const auto& type = value_info.type().sparse_tensor_type();
      enforce_has_field(type, elem_type);
      enforce_has_field(type, shape);
    } break;
#endif

    default:
      fail_check(
          "Unrecognized type value case (value_info name: ",
          value_info.name(),
          "): ",
          value_case);
  }
}

void check_tensor(const TensorProto& tensor, const CheckerContext& ctx) {
  enforce_has_field(tensor, data_type);
  if (tensor.data_type() == TensorProto::UNDEFINED) {
    fail_check(
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

  bool stored_externally = tensor.has_data_location() &&
      tensor.data_location() == TensorProto::EXTERNAL;
  if (stored_externally) {
    if (num_value_fields != 0) {
      fail_check(
          "Data of TensorProto ( tensor name: ",
          tensor.name(),
          ") is stored externally and should not have data field.",
          value_field);
    }

    bool has_location = false;
    for (const StringStringEntryProto& entry : tensor.external_data()) {
      if (entry.has_key() && entry.has_value() && entry.key() == "location") {
        has_location = true;
        if (!std::ifstream(ctx.get_model_dir() + entry.value())) {
          fail_check(
              "Data of TensorProto ( tensor name: ",
              tensor.name(),
              ") should be stored in ",
              ctx.get_model_dir() + entry.value(),
              ", but it doesn't exist or is not accessible.");
        }
      }
    }
    if (!has_location) {
      fail_check(
          "TensorProto ( tensor name: ",
          tensor.name(),
          ") is stored externally but doesn't have a location.");
    }
    return;
  }
  int64_t nelem = 1;
  for (auto x : tensor.dims()) {
    nelem *= x;
  }
  if (nelem == 0 && num_value_fields != 0) {
    fail_check(
        "TensorProto (tensor name: ",
        tensor.name(),
        ") is 0-element but contains data!");
  }
  if (nelem != 0 && num_value_fields != 1) {
    fail_check(
        "TensorProto (tensor name: ",
        tensor.name(),
        ") should contain one and only one value field.");
  }
  if (has_raw_data) {
    if (tensor.data_type() == TensorProto::STRING) {
      fail_check(
          "STRING data (tensor name: ",
          tensor.name(),
          ") should not be stored in raw_data field");
    }
    return;
  } else {
#define check_field(field)               \
  if (nelem != 0 && !has_##field) {      \
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
        check_field(float_data);
        break;

      case TensorProto::DOUBLE:
      case TensorProto::COMPLEX128:
        check_field(double_data);
        break;

      case TensorProto::INT32:
      case TensorProto::UINT8:
      case TensorProto::INT8:
      case TensorProto::UINT16:
      case TensorProto::INT16:
      case TensorProto::BOOL:
      case TensorProto::FLOAT16:
      case TensorProto::BFLOAT16:
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
            "Unrecognized data_type (tensor name: ",
            tensor.name(),
            "): ",
            tensor.data_type());
    }
  }

#undef check_field
}

// Check that the index data stored in a SparseTensorProto is valid.
// indices: a 1-dimensional tensor; indices[i] represents the
// linearized index value for the i-th nonzero value.
void check_sparse_tensor_indices_1(
    const TensorProto& indices,
    const SparseTensorProto& sparse_tensor_proto,
    size_t nnz) {
  int dense_rank = sparse_tensor_proto.dims_size();
  int64_t dense_size = 1;
  for (int i = 0; i < dense_rank; ++i)
    dense_size *= sparse_tensor_proto.dims(i);
  if (static_cast<size_t>(indices.dims(0)) != nnz)
    fail_check(
        "Sparse tensor indices (",
        indices.name(),
        ") has ",
        indices.dims(0),
        " values, but NNZ is ",
        nnz);

  // Check if indices appear in ascending order, and if they have valid
  // values. The i-th value in index_data is the linear index of the i-th
  // non-zero value.
  const std::vector<int64_t> index_data = ParseData<int64_t>(&indices);

  int64_t prev_index = -1;
  for (size_t i = 0; i < nnz; ++i) {
    int64_t curr_index = index_data[i]; // linearized index of i-th value
    if (curr_index < 0 || curr_index >= dense_size)
      fail_check(
          "Sparse tensor (",
          indices.name(),
          ") index value at position [",
          i,
          "] out of range [0, ",
          dense_size - 1,
          "]");
    if (curr_index <= prev_index) {
      fail_check(
          "Sparse tensor (",
          indices.name(),
          ") index value at position [",
          i,
          "] not in sorted order.");
    }
    prev_index = curr_index;
  }
}

// Check that the index data stored in a SparseTensorProto is valid.
// indices: a 2-dimensional tensor; indices[i,j] represents the j-th
// index value for the i-th nonzero value.
void check_sparse_tensor_indices_2(
    const TensorProto& indices,
    const SparseTensorProto& sparse_tensor_proto,
    size_t nnz) {
  int dense_rank = sparse_tensor_proto.dims_size();
  if (static_cast<size_t>(indices.dims(0)) != nnz)
    fail_check(
        "Sparse tensor indices (",
        indices.name(),
        ") first dimension size does not equal NNZ.");
  if (indices.dims(1) != dense_rank)
    fail_check(
        "Sparse tensor indices (",
        indices.name(),
        ") second dimension size does not match rank of tensor.");

  // Check if indices appear in ascending order, and if they have valid
  // values.
  const std::vector<int64_t> index_data = ParseData<int64_t>(&indices);

  int64_t prev_index = -1;
  for (size_t i = 0; i < nnz; ++i) {
    int64_t curr_index = 0; // linearized index of i-th value
    for (int j = 0; j < dense_rank; ++j) {
      auto index_ij = index_data[i * dense_rank + j];
      if ((index_ij < 0) || (index_ij >= sparse_tensor_proto.dims(j)))
        fail_check(
            "Sparse tensor (",
            indices.name(),
            ") index value at position [",
            i,
            ",",
            j,
            "] out of range.");
      curr_index = curr_index * sparse_tensor_proto.dims(j) + index_ij;
    }
    if (curr_index <= prev_index) {
      fail_check(
          "Sparse tensor (",
          indices.name(),
          ") index value at position [",
          i,
          "] not in lexicographic sorted order.");
    }
    prev_index = curr_index;
  }
}

void check_sparse_tensor(
    const SparseTensorProto& sparse_tensor_proto,
    const CheckerContext& ctx) {
  enforce_has_field(sparse_tensor_proto, values);

  const TensorProto& values = sparse_tensor_proto.values();
  check_tensor(values, ctx);

  // values must be a tensor of shape [NNZ]
  // Currently we restrict the value associated with a particular index-tuple
  // to be a single value. In the future, if there is a requirement,
  // we may extend this to permit the value to be a "sub-tensor", in which
  // case values will have dimension > 1.
  if (values.dims_size() != 1)
    fail_check("Sparse tensor values (", values.name(), ") must have rank 1.");
  size_t nnz = static_cast<size_t>(values.dims(0));

  int dense_rank = sparse_tensor_proto.dims_size();
  if (dense_rank == 0) {
    // TODO: Should we add a name field for a sparse-tensor-proto?
    // Currently, values has a name, but message may be a bit confusing.
    fail_check(
        "Sparse tensor (", values.name(), ") must have a dense-rank > 0");
  }
  for (int i = 0; i < dense_rank; ++i) {
    if (sparse_tensor_proto.dims(i) <= 0)
      fail_check(
          "Sparse tensor (", values.name(), ") dimensions are not positive.");
  }

  if (sparse_tensor_proto.has_indices()) {
    const TensorProto& indices = sparse_tensor_proto.indices();
    check_tensor(indices, ctx);
    if (indices.data_type() != TensorProto::INT64)
      fail_check(
          "Sparse tensor indices (", indices.name(), ") must have INT64 type.");
    switch (indices.dims().size()) {
      case 1:
        // Indices in linearized format
        check_sparse_tensor_indices_1(indices, sparse_tensor_proto, nnz);
        return;
      case 2:
        // Check COO-style index. E.g., an index for a 3D tensor is a 3-tuple.
        check_sparse_tensor_indices_2(indices, sparse_tensor_proto, nnz);
        return;
      default:
        fail_check(
            "Sparse tensor indices (",
            indices.name(),
            ") must have rank 1 or 2.");
    }
  } else if (nnz != 0)
    fail_check("Sparse tensor (", values.name(), ") has no index values.");
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

#define check_type(expected_type)                                              \
  if (attr.has_type() && attr.type() != expected_type) {                       \
    fail_check(                                                                \
        "type field and data field mismatch in attribute ", attr.name(), "."); \
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
  check_singular_field(sparse_tensor, AttributeProto::SPARSE_TENSOR);
  check_repeated_field(floats, AttributeProto::FLOATS);
  check_repeated_field(ints, AttributeProto::INTS);
  check_repeated_field(strings, AttributeProto::STRINGS);
  check_repeated_field(tensors, AttributeProto::TENSORS);
  check_repeated_field(graphs, AttributeProto::GRAPHS);
  check_repeated_field(sparse_tensors, AttributeProto::SPARSE_TENSORS);

#undef check_type
#undef check_singular_field
#undef check_repeated_field

  // Normally, used_fields is expected to be 1.
  // In proto3, when the value to be set is type default value (say 0 for
  // int), used_fields may be 0.
  if (used_fields > 1) {
    fail_check(
        "Attribute (name: ",
        attr.name(),
        ") should not contain more than one value field.");
  }

  if (!ctx.is_main_graph()) {
    // It's an attribute of a node in function body.
    if (attr.has_ref_attr_name() && used_fields != 0) {
      // The attribute proto is supposed to refer to data outside and does not
      // have its own value field set.
      fail_check(
          "Attribute (name: ",
          attr.name(),
          ") should refer to attribute in parent node.");
    }
  }

  if (attr.has_t()) {
    check_tensor(attr.t(), ctx);
  }

  if (attr.has_sparse_tensor()) {
    check_sparse_tensor(attr.sparse_tensor(), ctx);
  }

  if (attr.has_g()) {
    CheckerContext subgraph_ctx(ctx);
    subgraph_ctx.set_is_main_graph(false);
    check_graph(attr.g(), subgraph_ctx, lex_ctx);
  }

  for (const auto& tensor : attr.tensors()) {
    check_tensor(tensor, ctx);
  }
  for (const auto& sparse_tensor : attr.sparse_tensors()) {
    check_sparse_tensor(sparse_tensor, ctx);
  }
  if (attr.graphs().size() > 0) {
    CheckerContext subgraph_ctx(ctx);
    subgraph_ctx.set_is_main_graph(false);
    for (const auto& graph : attr.graphs()) {
      check_graph(graph, subgraph_ctx, lex_ctx);
    }
  }
}

void check_node(
    const NodeProto& node,
    const CheckerContext& ctx,
    const LexicalScopeContext& lex_ctx) {
  enforce_non_empty_field(node, op_type);

  if (node.input().empty() && node.output().empty()) {
    fail_check(
        "NodeProto (name: ",
        node.name(),
        ", type: ",
        node.op_type(),
        ") has zero input and zero output.");
  }

  // Put the removed experimental ops here
  static std::set<std::string> experimental_ops = {"ATen",
                                                   "Affine",
                                                   "ConstantFill",
                                                   "Crop",
                                                   "DynamicSlice",
                                                   "GRUUnit",
                                                   "GivenTensorFill",
                                                   "ImageScaler",
                                                   "ParametricSoftplus",
                                                   "Scale",
                                                   "ScaledTanh"};
  if (experimental_ops.count(node.op_type())) {
    std::cerr << "Warning: " << node.op_type() << " was a removed "
              << "experimental ops. In the future, we may directly "
              << "reject this operator. Please update your model as soon "
              << "as possible." << std::endl;
    return;
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

  const auto* schema = ctx.get_schema_registry()->GetSchema(
      node.op_type(), domain_version, node.domain());
  if (!schema) {
    if (node.domain() == ONNX_DOMAIN || node.domain() == AI_ONNX_ML_DOMAIN ||
        node.domain() == "ai.onnx") {
      // fail the checker if op in built-in domains has no schema
      fail_check(
          "No Op registered for " + node.op_type() +
          " with domain_version of " +
          ONNX_NAMESPACE::to_string(domain_version));
    } else {
      // TODO: expose the registration of the op schemas appropriately in
      // python, so we can load and register operators in other domains
      //
      // before we complete the above todo, let's skip the schema check for
      // now
    }
  } else if (schema->Deprecated()) {
    fail_check(
        "Op registered for " + node.op_type() +
        " is deprecated in domain_version of " +
        ONNX_NAMESPACE::to_string(domain_version));
  } else {
    schema->Verify(node);
  }
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

  // Inherit values available in outer scope
  // Note that we do not allow shadowing, so the presence of an already-defined
  // name is always an error.
  LexicalScopeContext lex_ctx{parent_lex};

  for (const auto& value_info : graph.input()) {
    // TODO: If shadowing isn't allowed, this should maybe use
    // this_or_ancestor_graph_has
    if (lex_ctx.this_graph_has(value_info.name())) {
      fail_check(
          "Graph must be in single static assignment (SSA) form, however '",
          value_info.name(),
          "' has been used as graph input names multiple times.");
    }
    lex_ctx.add(value_info.name());
  }

  for (const auto& init : graph.initializer()) {
    if (ctx.get_ir_version() <= 0x00000003) {
      // Initializers are a subset of graph inputs for IR_VERSION <= 3
      if (!lex_ctx.this_graph_has(init.name())) {
        fail_check(init.name() + " in initializer but not in graph input");
      }
    } else {
      // An initializer is allowed to have the same name as an input,
      // but is not required to (for IR_VERSION >= 4)
      lex_ctx.add(init.name());
    }
    check_tensor(init, ctx);
  }

  // TODO: Need to check that sparse-initializers names are distinct from
  // initializer names. It looks like the existing checker does not check for
  // certain duplication of names: e.g., two entries in the initializer list
  // with same name. Will add a new integrated check.
  for (const auto& sparse_init : graph.sparse_initializer()) {
    check_sparse_tensor(sparse_init, ctx);
    lex_ctx.add(sparse_init.values().name());
  }

  for (const auto& node : graph.node()) {
    // nodes must be in topologically sorted order
    for (const auto& input : node.input()) {
      // explicit optional input
      if (input.empty()) {
        continue;
      }
      if (!lex_ctx.this_or_ancestor_graph_has(input)) {
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

      if (lex_ctx.this_or_ancestor_graph_has(output)) {
        fail_check(
            "Graph must be in single static assignment (SSA) form, however '",
            output,
            "' has been used as output names multiple times.");
      }
      lex_ctx.add(output);
    }
  }
}

void check_function(
    const FunctionProto& function,
    const CheckerContext& ctx,
    const LexicalScopeContext& parent_lex) {
  enforce_non_empty_field(function, name);
  enforce_has_field(function, since_version);

  LexicalScopeContext lex_ctx{parent_lex};

  for (const auto& input : function.input()) {
    // TODO: If shadowing isn't allowed, this should maybe use
    // this_or_ancestor_graph_has
    if (lex_ctx.this_graph_has(input)) {
      fail_check(
          "Graph must be in single static assignment (SSA) form, however '",
          input,
          "' has been used multiple times.");
    }
    lex_ctx.add(input);
  }

  std::unordered_set<std::string> outputs;
  for (const auto& output : function.output()) {
    auto result = outputs.insert(output);
    if (!result.second) {
      fail_check(
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
      if (!lex_ctx.this_graph_has(input)) {
        fail_check(
            "Nodes in a function must be topologically sorted, however input '",
            input,
            "' of node: \n",
            ProtoDebugString(node),
            "\n is neither output of any previous nodes nor input of the function.");
      }
    }

    check_node(node, ctx, lex_ctx);

    // check for SSA form
    for (const auto& output : node.output()) {
      // optional output
      if (output.empty()) {
        continue;
      }
      if (lex_ctx.this_or_ancestor_graph_has(output)) {
        fail_check(
            "Function must be in single static assignment (SSA) form, however '",
            output,
            "' has been used as output names multiple times.");
      }
      lex_ctx.add(output);
    }
  }
}

void check_model(const ModelProto& model, CheckerContext& ctx) {
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
  ctx.set_ir_version(static_cast<int>(model.ir_version()));
  std::unordered_map<std::string, int> opset_imports;
  for (const auto& opset_import : model.opset_import()) {
    opset_imports[opset_import.domain()] =
        static_cast<int>(opset_import.version());
  }
  if (model.ir_version() >= 3) {
    if (opset_imports.empty())
      fail_check(
          "model with IR version >= 3 must specify opset_import for ONNX");
  } else {
    if (opset_imports.empty())
      opset_imports[ONNX_DOMAIN] = 1;
    else
      fail_check(
          "model with IR version < 3 cannot have opset_import specified");
  }
  ctx.set_opset_imports(opset_imports);
  LexicalScopeContext lex_ctx;
  check_graph(model.graph(), ctx, lex_ctx);
}

void check_model(const std::string& model_path) {
  ModelProto model;
  std::fstream model_stream(model_path, std::ios::in | std::ios::binary);
  if (!model_stream.good()) {
    fail_check(
        "Unable to open model file:",
        model_path,
        ". Please check if it is a valid file.");
  }
  std::string data{std::istreambuf_iterator<char>{model_stream},
                   std::istreambuf_iterator<char>{}};
  if (!ParseProtoFromBytes(&model, data.c_str(), data.size())) {
    fail_check(
        "Unable to parse model from file:",
        model_path,
        ". Please check if it is a valid protobuf file of model.");
  }

  CheckerContext ctx;
  std::string model_dir;
  size_t pos = model_path.find_last_of("\\/");
  if (pos != std::string::npos) {
    model_dir = model_path.substr(0, pos + 1);
  }
  ctx.set_model_dir(model_dir);
  check_model(model, ctx);
}

void check_model(const ModelProto& model) {
  CheckerContext ctx;
  check_model(model, ctx);
}

#undef fail_check
#undef enforce_has_field
#undef enforce_has_repeated_field
#undef enforce_non_empty_field

} // namespace checker
} // namespace ONNX_NAMESPACE
