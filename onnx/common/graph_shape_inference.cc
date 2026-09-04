// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/common/graph_shape_inference.h"

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "onnx/common/constants.h"
#include "onnx/common/ir_pb_converter_internal.h"
#include "onnx/defs/schema.h"
#include "onnx/shape_inference/implementation.h"

namespace ONNX_NAMESPACE {

namespace {

// Shape-describing inputs (Reshape's `shape`, Slice's starts/ends/axes, ...)
// are always small; a tensor above this size is presumably a weight or
// activation, so it's left out of input_data_by_name rather than copied. A
// node that did need it just sees a null getInputData(), like any other
// statically-unknown value.
constexpr int64_t kMaxInputDataElements = 4096;

// Encodes `v`'s current type: tensor_type from elemType()/sizes(), or a copy
// of v.type() for non-tensor types (Sequence/Optional/Map/...) -- matching
// encodeValueInfo in ir_pb_converter.cc.
void EncodeCurrentType(Value& v, TypeProto& out) {
  if (v.elemType() != 0 || v.has_sizes()) {
    encodeTypeProtoTensorType(*out.mutable_tensor_type(), v);
  } else if (v.type()) {
    out.CopyFrom(*v.type());
  }
}

// Applies a (possibly merged) inferred TypeProto back onto `v`: the tensor
// case updates elemType()/sizes() directly, anything else replaces v.type()
// wholesale.
void ApplyInferredType(const TypeProto& inferred, Value& v) {
  if (inferred.value_case() == TypeProto::VALUE_NOT_SET) {
    return;
  }
  if (inferred.has_tensor_type()) {
    const auto& tensor_type = inferred.tensor_type();
    if (tensor_type.has_elem_type()) {
      v.setElemType(tensor_type.elem_type());
    }
    if (tensor_type.has_shape()) {
      v.setSizes(tensorShapeProtoToDimensions(tensor_type.shape()));
    }
  } else {
    v.type() = std::make_unique<TypeProto>(inferred);
  }
}

bool ElementCountFits(const Tensor& t) {
  int64_t n = 1;
  for (int64_t d : t.sizes()) {
    if (d < 0) {
      return false;
    }
    n *= d;
    if (n > kMaxInputDataElements) {
      return false;
    }
  }
  return true;
}

// Returns the Tensor backing `v`'s statically-known constant value, if any: a
// Constant node's "value" attribute, or a graph initializer. `initializer_by_name`
// is built once per Run() -- a Graph::getInitializer() lookup per input instead
// would be O(rounds * nodes * initializers) on models with many initializers.
const Tensor* ConstantDataFor(Value& v, const std::unordered_map<std::string, const Tensor*>& initializer_by_name) {
  static const Symbol kConstant("Constant");
  static const Symbol kValue("value");

  const Node* producer = v.node();
  if (producer->kind() == kConstant && (!producer->has_domain() || producer->domain().empty()) &&
      producer->kindOf(kValue) == AttributeKind::t) {
    return &producer->t(kValue);
  }
  auto it = initializer_by_name.find(v.uniqueName());
  if (it != initializer_by_name.end()) {
    return it->second;
  }
  return nullptr;
}

// Encodes a Tensor's shape/dtype only, skipping the raw bytes -- for a
// TENSOR/TENSORS attribute too large to be worth copying (same size gate as
// ConstantDataFor).
void EncodeShapeOnly(TensorProto& out, const Tensor& t) {
  out.set_data_type(t.elem_type());
  for (int64_t d : t.sizes()) {
    out.add_dims(d);
  }
}

// Adds one attribute of `node` to `np`, size-gating any TENSOR/TENSORS value
// like ConstantDataFor gates inputs (addAttribute always copies a tensor
// attribute's raw bytes unconditionally). GRAPH/GRAPHS attributes go through
// addAttribute ungated -- ProcessNode needs the real GraphProto to let the
// op's inference function recurse into it.
void AddAttributeForInference(NodeProto& np, Node& node, Symbol name) {
  AttributeKind kind = node.kindOf(name);
  if (kind == AttributeKind::t) {
    const Tensor& t = node.t(name);
    if (ElementCountFits(t)) {
      addAttribute(np, node, name);
      return;
    }
    auto* attr = np.add_attribute();
    attr->set_name(name.toString());
    attr->set_type(AttributeProto_AttributeType_TENSOR);
    EncodeShapeOnly(*attr->mutable_t(), t);
    return;
  }
  if (kind == AttributeKind::ts) {
    bool any_large = false;
    for (const Tensor& t : node.ts(name)) {
      if (!ElementCountFits(t)) {
        any_large = true;
        break;
      }
    }
    if (!any_large) {
      addAttribute(np, node, name);
      return;
    }
    auto* attr = np.add_attribute();
    attr->set_name(name.toString());
    attr->set_type(AttributeProto_AttributeType_TENSORS);
    for (const Tensor& t : node.ts(name)) {
      EncodeShapeOnly(*attr->add_tensors(), t);
    }
    return;
  }
  addAttribute(np, node, name);
}

// A shape_inference::SymbolTable seeded from the Graph IR's own Values
// instead of a GraphProto. Without naming, a bare unknown dim is
// indistinguishable from any other unknown or from 0. GenerateSymbolicShape
// (via MaterializeSymbolicShape in ProcessNode) assigns the names; this class
// only supplies naming authority + collision avoidance, mirroring
// SymbolTableImpl's addFromGraph but scanning Values instead of a GraphProto.
class GraphIrSymbolTable : public SymbolTable {
 public:
  // A nested subgraph's own recursive InferShapesImpl call may call this on
  // the exported protobuf subgraph, so its symbols don't collide with ones
  // generated for the enclosing graph.
  void addFromGraph(const GraphProto& g) override {
    AddExistingDims(g.input());
    AddExistingDims(g.output());
    AddExistingDims(g.value_info());
  }
  void AddExistingSymbol(const std::string& symbol) {
    existing_symbols_.insert(symbol);
  }
  std::string createNew(const std::string& symbol_prefix) override {
    std::string new_symbol;
    do {
      new_symbol = symbol_prefix + std::to_string(index_++);
    } while (existing_symbols_.count(new_symbol) > 0);
    existing_symbols_.insert(new_symbol);
    return new_symbol;
  }

 private:
  template <typename RepeatedValueInfo>
  void AddExistingDims(const RepeatedValueInfo& value_infos) {
    for (const auto& vi : value_infos) {
      if (!vi.type().has_tensor_type() || !vi.type().tensor_type().has_shape()) {
        continue;
      }
      for (const auto& dim : vi.type().tensor_type().shape().dim()) {
        if (dim.has_dim_param()) {
          existing_symbols_.insert(dim.dim_param());
        }
      }
    }
  }

  unsigned int index_ = 0;
  std::unordered_set<std::string> existing_symbols_;
};

class GraphShapeInferenceRunner {
 public:
  explicit GraphShapeInferenceRunner(const ShapeInferenceOptions& options) : options_(options) {}

  // Returns whether anything changed.
  bool Run(Graph& g) {
    std::unordered_map<std::string, int> opset_imports;
    for (const OpSetID& opset : g.opset_versions_mutable()) {
      opset_imports[opset.domain()] = static_cast<int>(opset.version());
    }
    const ISchemaRegistry* registry = OpSchemaRegistry::Instance();

    // Built once per Run(), not once per node visit -- see ConstantDataFor.
    std::unordered_map<std::string, const Tensor*> initializer_by_name;
    const auto& initializers = g.initializers();
    const auto& initializer_names = g.initializer_names();
    initializer_by_name.reserve(initializers.size());
    for (size_t i = 0; i < initializers.size(); ++i) {
      initializer_by_name[initializer_names[i]] = &initializers[i];
    }

    // Whole-graph type map (mirroring InferShapesImpl's
    // "value_types_by_name"), used only by GraphInferenceContext to resolve
    // an If/Loop/Scan subgraph's captured references against the enclosing
    // scope. outer_scope_storage owns the TypeProtos (must outlive Run());
    // outer_scope_types is the name->pointer view GraphInferenceContext takes;
    // unordered_map references stay valid across insertion, so earlier
    // pointers stay good as more entries are added.
    std::unordered_map<std::string, TypeProto> outer_scope_storage;
    std::unordered_map<std::string, TypeProto*> outer_scope_types;
    auto RecordOuterScopeType = [&](Value* v) {
      TypeProto& t = outer_scope_storage[v->uniqueName()];
      EncodeCurrentType(*v, t);
      if (t.value_case() != TypeProto::VALUE_NOT_SET) {
        outer_scope_types[v->uniqueName()] = &t;
      }
    };
    for (Value* input : g.inputs()) {
      RecordOuterScopeType(input);
    }
    for (size_t i = 0; i < initializers.size(); ++i) {
      // Input has priority over initializer of the same name, matching
      // onnx's own ProcessInitializer.
      if (outer_scope_types.count(initializer_names[i]) > 0) {
        continue;
      }
      TypeProto& t = outer_scope_storage[initializer_names[i]];
      auto* tensor_type = t.mutable_tensor_type();
      tensor_type->set_elem_type(initializers[i].elem_type());
      auto* shape = tensor_type->mutable_shape();
      for (int64_t d : initializers[i].sizes()) {
        shape->add_dim()->set_dim_value(d);
      }
      outer_scope_types[initializer_names[i]] = &t;
    }

    // Seed the symbol table with every dim_param already in the graph, so a
    // freshly-generated name (see GraphIrSymbolTable) never collides with an
    // unrelated existing one.
    auto SeedSymbolTable = [&](Value* v) {
      if (!v->has_sizes()) {
        return;
      }
      for (const Dimension& d : v->sizes()) {
        if (!d.is_unknown && !d.is_int && !d.param.empty()) {
          symbol_table_.AddExistingSymbol(d.param);
        }
      }
    };
    for (Value* input : g.inputs()) {
      SeedSymbolTable(input);
    }
    for (Node* node : g.nodes()) {
      for (Value* output : node->outputs()) {
        SeedSymbolTable(output);
      }
    }

    bool changed = false;
    for (Node* node : g.nodes()) {
      if (node->kind() == kUndefined || node->kind() == kCaptured) {
        continue;
      }
      changed |= ProcessNode(*node, opset_imports, registry, initializer_by_name, outer_scope_types);
      for (Value* output : node->outputs()) {
        RecordOuterScopeType(output);
      }
    }
    return changed;
  }

 private:
  bool ProcessNode(
      Node& node,
      const std::unordered_map<std::string, int>& opset_imports,
      const ISchemaRegistry* registry,
      const std::unordered_map<std::string, const Tensor*>& initializer_by_name,
      const std::unordered_map<std::string, TypeProto*>& outer_scope_types) {
    const std::vector<Symbol> attr_names = node.attributeNames();

    bool has_subgraph_attr = false;
    for (Symbol name : attr_names) {
      AttributeKind kind = node.kindOf(name);
      if (kind == AttributeKind::g || kind == AttributeKind::gs) {
        has_subgraph_attr = true;
        break;
      }
    }

    const std::string domain = node.has_domain() ? node.domain() : std::string(ONNX_DOMAIN);
    const int* domain_version = shape_inference::LookupOpsetImport(domain, opset_imports);
    if (domain_version == nullptr) {
      return false; // no opset import for this domain -- leave outputs as-is.
    }
    const std::string op_type = node.kind().toString();
    const OpSchema* schema = registry->GetSchema(op_type, *domain_version, domain);
    if (schema == nullptr || !schema->has_type_and_shape_inference_function()) {
      // Unsupported op (no schema, or a function-body-only op -- see this
      // file's v1-scope doc comment): leave outputs as-is, same as onnx's
      // own protobuf-based InferShapes does for a genuinely unknown op.
      return false;
    }

    // A lightweight NodeProto shell; only attribute conversion (addAttribute)
    // is shared with the Export path, everything else is small metadata.
    NodeProto np;
    np.set_op_type(op_type);
    if (node.has_domain()) {
      np.set_domain(node.domain());
    }
    if (node.has_name()) {
      np.set_name(node.name());
    }
    const auto& inputs = node.inputs();
    const auto& outputs = node.outputs();
    for (Value* input : inputs) {
      np.add_input(input->node()->kind() == kUndefined ? "" : input->uniqueName());
    }
    for (Value* output : outputs) {
      np.add_output(output->uniqueName());
    }
    for (Symbol attr_name : attr_names) {
      AddAttributeForInference(np, node, attr_name);
    }

    // Per-input TypeProto/TensorProto adapters, built fresh for this node
    // visit; the backing vectors only need to outlive this function.
    std::vector<TypeProto> input_types(inputs.size());
    std::unordered_map<std::string, TypeProto*> value_types_by_name;
    std::vector<TensorProto> input_data_storage;
    input_data_storage.reserve(inputs.size());
    std::unordered_map<std::string, const TensorProto*> input_data_by_name;
    const std::unordered_map<std::string, const SparseTensorProto*> input_sparse_data_by_name; // always empty (v1)

    for (size_t i = 0; i < inputs.size(); ++i) {
      Value* input = inputs[i];
      if (input->node()->kind() == kUndefined) {
        continue; // absent optional input
      }
      EncodeCurrentType(*input, input_types[i]);
      value_types_by_name[input->uniqueName()] = &input_types[i];

      if (const Tensor* data = ConstantDataFor(*input, initializer_by_name)) {
        if (ElementCountFits(*data)) {
          input_data_storage.emplace_back();
          encodeTensor(input_data_storage.back(), *data);
          input_data_by_name[input->uniqueName()] = &input_data_storage.back();
        }
      }
    }

    // Lets If/Loop/Scan's own inference function recurse into its exported
    // body subgraph via ctx.getGraphAttributeInferencer().
    std::unique_ptr<shape_inference::GraphInferenceContext> graph_inference_context;
    if (has_subgraph_attr) {
      graph_inference_context =
          std::make_unique<shape_inference::GraphInferenceContext>(outer_scope_types, opset_imports, &symbol_table_);
    }

    shape_inference::InferenceContextImpl ctx(
        np,
        value_types_by_name,
        input_data_by_name,
        input_sparse_data_by_name,
        options_,
        /*generatedShapeData=*/nullptr,
        graph_inference_context.get());

    // Mirror InferShapesImpl: a node-level error (e.g. a shape conflict)
    // doesn't abort the whole pass, it just leaves that node's outputs unchanged.
    bool changed = false;
    ONNX_TRY {
      schema->GetTypeAndShapeInferenceFunction()(ctx);
      for (size_t i = 0; i < outputs.size(); ++i) {
        TypeProto* inferred = ctx.getOutputType(i);
        if (inferred == nullptr) {
          continue;
        }
        // Name any bare unknown dim before merging -- see GraphIrSymbolTable.
        shape_inference::MaterializeSymbolicShape(inferred, symbol_table_);
        Value* out = outputs[i];
        TypeProto existing;
        EncodeCurrentType(*out, existing);
        // mergeShapesAndTypes mutates `existing` in place with no "changed"
        // signal of its own; detect it by comparing serialized form.
        const std::string before = existing.SerializeAsString();
        shape_inference::mergeShapesAndTypes(*inferred, &existing);
        if (existing.SerializeAsString() != before) {
          changed = true;
          ApplyInferredType(existing, *out);
        }
      }
    }
    ONNX_CATCH(const std::exception&) {
      return false;
    }
    return changed;
  }

  const ShapeInferenceOptions& options_;
  // Reset (default-constructed) for every Run() call and re-seeded from the
  // graph's current state each time -- see the seeding loop in Run().
  GraphIrSymbolTable symbol_table_;
};

} // namespace

bool InferShapesOnGraph(Graph& g, const ShapeInferenceOptions& options) {
  GraphShapeInferenceRunner runner(options);
  return runner.Run(g);
}

} // namespace ONNX_NAMESPACE
