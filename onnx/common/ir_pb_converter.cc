// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#include "onnx/common/ir_pb_converter.h"

namespace ONNX_NAMESPACE {

// Part 1: convert ONNX Protobuf to IR
std::unique_ptr<Graph> graphProtoToGraph(const ONNX_NAMESPACE::GraphProto& gp);

Tensor tensorProtoToTensor(const ONNX_NAMESPACE::TensorProto & tp) {
  Tensor ret;

  ret.sizes().reserve(tp.dims_size());
  for (int i = 0; i < tp.dims_size(); i++) {
    ret.sizes().push_back(tp.dims(i));
  }

  ret.elem_type() = tp.data_type();
  switch(tp.data_type()) {
  case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
  case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
    ret.floats().reserve(tp.float_data_size());
    for (int i = 0; i < tp.float_data_size(); i++) {
      ret.floats().push_back(tp.float_data(i));
    }
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
  case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
  case ONNX_NAMESPACE::TensorProto_DataType_INT8:
  case ONNX_NAMESPACE::TensorProto_DataType_INT16:
  case ONNX_NAMESPACE::TensorProto_DataType_INT32:
  case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
  case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
    ret.int32s().reserve(tp.int32_data_size());
    for (int i = 0; i < tp.int32_data_size(); i++) {
      ret.int32s().push_back(tp.int32_data(i));
    }
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
    ret.int64s().reserve(tp.int64_data_size());
    for (int i = 0; i < tp.int64_data_size(); i++) {
      ret.int64s().push_back(tp.int64_data(i));
    }
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
  case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
    ret.uint64s().reserve(tp.uint64_data_size());
    for (int i = 0; i < tp.uint64_data_size(); i++) {
      ret.uint64s().push_back(tp.uint64_data(i));
    }
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
  case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: {
    ret.doubles().reserve(tp.double_data_size());
    for (int i = 0; i < tp.double_data_size(); i++) {
      ret.doubles().push_back(tp.double_data(i));
    }
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_STRING: {
    ret.strings().reserve(tp.string_data_size());
    for (int i = 0; i < tp.string_data_size(); i++) {
      ret.strings().push_back(tp.string_data(i));
    }
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
    abort();
  }

  // The only way to know if we should be using raw_data or
  // <type>_data is to look at which of them is size zero.
  if (tp.has_raw_data()) {
    ret.set_raw_data(tp.raw_data());
  }

  if (tp.has_name()) {
    ret.setName(tp.name());
  }
  if (tp.has_segment()) {
    ret.set_segment_begin_and_end(tp.segment().begin(), tp.segment().end());
  }
  return ret;
}

void convertAttribute(const ONNX_NAMESPACE::AttributeProto & ap, Node * n) {
  Symbol sym = Symbol(ap.name());
  switch(ap.type()) {
  case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT:
    n->f_(sym, ap.f());
    break;
  case ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS: {
    std::vector<double> floats;
    floats.reserve(ap.floats_size());
    for (int i = 0; i < ap.floats_size(); i++) {
      floats.push_back(ap.floats(i));
    }
    n->fs_(sym, std::move(floats));
    break;
  }
  case ONNX_NAMESPACE::AttributeProto_AttributeType_INT:
    n->i_(sym, ap.i());
    break;
  case ONNX_NAMESPACE::AttributeProto_AttributeType_INTS: {
    std::vector<int64_t> ints;
    ints.reserve(ap.ints_size());
    for (int i = 0; i < ap.ints_size(); i++) {
      ints.push_back(ap.ints(i));
    }
    n->is_(sym, std::move(ints));
    break;
  }
  case ONNX_NAMESPACE::AttributeProto_AttributeType_STRING:
    n->s_(sym, ap.s());
    break;
  case ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS: {
    std::vector<std::string> strings;
    strings.reserve(ap.strings_size());
    for (int i = 0; i < ap.strings_size(); i++) {
      strings.push_back(ap.strings(i));
    }
    n->ss_(sym, std::move(strings));
    break;
  }
  case ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR:
    n->t_(sym, tensorProtoToTensor(ap.t()));
    break;
  case ONNX_NAMESPACE::AttributeProto_AttributeType_TENSORS: {
    std::vector<Tensor> tensors;
    tensors.reserve(ap.tensors_size());
    for (int i = 0; i < ap.tensors_size(); i++) {
      tensors.push_back(tensorProtoToTensor(ap.tensors(i)));
    }
    n->ts_(sym, std::move(tensors));
    break;
  }
  case ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH:
    n->g_(sym, graphProtoToGraph(ap.g()));
    break;
  case ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS: {
    std::vector<std::shared_ptr<Graph>> graphs;
    graphs.reserve(ap.graphs_size());
    for (int i = 0; i < ap.graphs_size(); i++) {
      graphs.push_back(graphProtoToGraph(ap.graphs(i)));
    }
    n->gs_(sym, std::move(graphs));
    break;
  }
  case ONNX_NAMESPACE::AttributeProto_AttributeType_UNDEFINED:
    abort();
    break;
  }
}

void convertAttributes(ONNX_NAMESPACE::NodeProto & np, Node * n) {
  for (int i = 0; i < np.attribute_size(); i++) {
    convertAttribute(np.attribute(i), n);
  }
}

std::vector<Dimension> tensorShapeProtoToDimensions(const ONNX_NAMESPACE::TensorShapeProto & tsp) {
  std::vector<Dimension> dims;
  dims.reserve(tsp.dim_size());
  for (int i = 0; i < tsp.dim_size(); i++) {
    if (tsp.dim(i).has_dim_value()) {
      dims.push_back(Dimension(static_cast<int>(tsp.dim(i).dim_value())));
    } else {
      dims.push_back(Dimension(tsp.dim(i).dim_param()));
    }
  }
  return dims;
}

std::unique_ptr<Graph> graphProtoToGraph(const ONNX_NAMESPACE::GraphProto& gp) {
  std::unique_ptr<Graph> g(new Graph());

  if (gp.has_name()) {
    g->setName(gp.name());
  }
  if (gp.has_doc_string()) {
    g->setDocString(gp.doc_string());
  }

  // Values are created (as in `new Value(..)`) by the Node that
  // outputs them. Therefore we initialize the Nodes and Values in
  // several stages.
  //
  // 1) add all input (to the graph) Values, owned by the sentinel Param node
  // 2) add all Nodes and their output Values, but don't intialize inputs
  // 3) initialize inputs of all Nodes
  // 4) initialize inputs of the Return sentinel node
  // 5) fill in type info for graph outputs, and register them as outputs
  // 5) fill in type info for Values from the value_info list in the graph

  // In ONNX proto land, Values are just strings. We are going to make
  // objects out of them, and equal strings must be mapped to the same
  // Value object.
  std::unordered_map<std::string, Value*> value_by_name_of;

  // We initialize Node inputs in a separate pass from the Nodes
  // themselves. To do so, we need to have access to the names of the
  // inputs.
  std::unordered_map<Node*, std::vector<std::string>> inputs_by_node;

  {
    // ONNX represents optional arguments in two ways
    //  - they are simply not privided
    //  - OR the empty string is passed as the input name
    // This is to handle that second case, which needs a dummy node to
    // be representable in the graph IR.
    auto * n = g->create(kUndefined, 1);
    g->appendNode(n);
    n->outputs()[0]->setUniqueName("");
    value_by_name_of[""] = n->outputs()[0];
  }

  for (int i = 0; i < gp.input_size(); i++) {
    auto vip = gp.input(i);
    auto v = g->addInput();
    v->setElemType(vip.type().tensor_type().elem_type());
    v->setSizes(tensorShapeProtoToDimensions(vip.type().tensor_type().shape()));
    v->setUniqueName(vip.name());
    value_by_name_of[vip.name()] = v;
  }

  for (int i = 0; i < gp.node_size(); i++) {
    auto np = gp.node(i);
    auto * n = g->create(Symbol(np.op_type()), /* num_outputs = */ np.output_size());
    g->appendNode(n);
    for (int j = 0; j < np.output_size(); j++) {
      auto out = n->outputs()[j];
      // we don't know the real type here, so that's done in a later pass
      out->setElemType(ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED);
      out->setUniqueName(np.output(j));
      value_by_name_of[np.output(j)] = out;
    }
    convertAttributes(np, n);
    std::vector<std::string> inputs;
    inputs.reserve(np.input_size());
    for (int j = 0; j < np.input_size(); j++) {
      inputs.push_back(np.input(j));
    }
    inputs_by_node[n] = inputs;
    if (np.has_doc_string()) {
      n->setDocString(np.doc_string());
    }
    if (np.has_name()) {
      n->setName(np.name());
    }
  }

  for (auto n : g->nodes()) {
    auto search = inputs_by_node.find(n);
    if (search == inputs_by_node.end()) {
      continue;
    }
    for (auto input : search->second) {
      n->addInput(value_by_name_of[input]);
    }
  }

  for (int i = 0; i < gp.output_size(); i++) {
    value_by_name_of[gp.output(i).name()]->setElemType(gp.output(i).type().tensor_type().elem_type());
    value_by_name_of[gp.output(i).name()]->setSizes(tensorShapeProtoToDimensions(gp.output(i).type().tensor_type().shape()));
    g->registerOutput(value_by_name_of[gp.output(i).name()]);
  }

  for (int i = 0; i < gp.value_info_size(); i++) {
    value_by_name_of[gp.value_info(i).name()]->setElemType(gp.value_info(i).type().tensor_type().elem_type());
    value_by_name_of[gp.value_info(i).name()]->setSizes(tensorShapeProtoToDimensions(gp.value_info(i).type().tensor_type().shape()));
  }

  for (int i = 0; i < gp.initializer_size(); i++) {
    auto init = tensorProtoToTensor(gp.initializer(i));
    g->addInitializer(init, init.name());
  }

  return g;
}

std::unique_ptr<Graph> ImportModelProto(const ONNX_NAMESPACE::ModelProto& mp) {
  if (!mp.has_ir_version()) {
    return nullptr;
  } else if (mp.ir_version() == 1) {
    return nullptr;
  }

  return graphProtoToGraph(mp.graph());
}


// Part 2: convert IR to ONNX Protobuf
std::string value_name(Value* n) {
  return n->uniqueName();
}

void encodeGraph(ONNX_NAMESPACE::GraphProto * p_g, const std::shared_ptr<Graph> & g);

void encodeTensor(ONNX_NAMESPACE::TensorProto * p, const Tensor & tensor) {
  if (tensor.hasName()) {
    p->set_name(tensor.name());
  }
  if (tensor.is_segment()) {
    ONNX_NAMESPACE::TensorProto_Segment segment;
    segment.set_begin(tensor.segment_begin());
    segment.set_end(tensor.segment_end());
    p->mutable_segment()->CopyFrom(segment);
  }
  for(auto d : tensor.sizes()) {
    p->add_dims(d);
  }
  p->set_data_type(tensor.elem_type());
  switch(tensor.elem_type()) {
  case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
  case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX64: {
    for (float x : tensor.floats()) {
      p->add_float_data(x);
    }
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
  case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
  case ONNX_NAMESPACE::TensorProto_DataType_INT8:
  case ONNX_NAMESPACE::TensorProto_DataType_INT16:
  case ONNX_NAMESPACE::TensorProto_DataType_INT32:
  case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
  case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
    for (int32_t x : tensor.int32s()) {
      p->add_int32_data(x);
    }
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_INT64: {
    for (int64_t x : tensor.int64s()) {
      p->add_int64_data(x);
    }
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
  case ONNX_NAMESPACE::TensorProto_DataType_UINT64: {
    for (uint64_t x : tensor.uint64s()) {
      p->add_uint64_data(x);
    }
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
  case ONNX_NAMESPACE::TensorProto_DataType_COMPLEX128: {
    for (double x : tensor.doubles()) {
      p->add_double_data(x);
    }
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_STRING: {
    for (const std::string& x : tensor.strings()) {
      p->add_string_data(x);
    }
    break;
  }
  case ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED:
    abort();
  }
  if (!tensor.raw().empty()) {
    p->set_raw_data(tensor.raw());
  }
}

void addAttribute(ONNX_NAMESPACE::NodeProto * n_p, Node * n, Symbol name) {
  auto attr = n_p->add_attribute();
  attr->set_name(name.toString());
  switch(n->kindOf(name)) {
    case AttributeKind::f:
      attr->set_f(static_cast<float>(n->f(name)));
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
      break;
    case AttributeKind::fs:
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS);
      for(auto & v : n->fs(name))
        attr->add_floats(static_cast<float>(v));
      break;
    case AttributeKind::i:
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
      attr->set_i(n->i(name));
      break;
    case AttributeKind::is:
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
      for(auto & v : n->is(name))
        attr->add_ints(v);
      break;
    case AttributeKind::s:
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
      attr->set_s(n->s(name));
      break;
    case AttributeKind::ss:
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS);
      for(auto & v : n->ss(name))
        attr->add_strings(v);
      break;
    case AttributeKind::t: {
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
      auto t = attr->mutable_t();
      encodeTensor(t, n->t(name));
    } break;
    case AttributeKind::ts:
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSORS);
      for(auto & v : n->ts(name)) {
        auto t = attr->add_tensors();
        encodeTensor(t, v);
      }
      break;
    case AttributeKind::g: {
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPH);
      auto g = attr->mutable_g();
      encodeGraph(g, n->g(name));
    } break;
    case AttributeKind::gs:
      attr->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_GRAPHS);
      for(auto & v : n->gs(name)) {
        auto g = attr->add_graphs();
        encodeGraph(g, v);
      }
      break;
  }
}

void encodeTypeProtoTensorType(ONNX_NAMESPACE::TypeProto_Tensor* tensor_type, Value* n) {
  tensor_type->set_elem_type(n->elemType());
  ONNX_NAMESPACE::TensorShapeProto* shape = tensor_type->mutable_shape();
  for (const Dimension& d : n->sizes()) {
    auto dim = shape->add_dim();
    if (d.is_int) {
      dim->set_dim_value(d.dim);
    } else {
      dim->set_dim_param(d.param);
    }
  }
}

void encodeValueInfo(ONNX_NAMESPACE::ValueInfoProto* v, Value* n) {
  if (n->has_unique_name()) {
    v->set_name(value_name(n));
  }
  ONNX_NAMESPACE::TypeProto* t = v->mutable_type();
  ONNX_NAMESPACE::TypeProto_Tensor* tensor_type = t->mutable_tensor_type();
  encodeTypeProtoTensorType(tensor_type, n);
}

void encodeGraph(ONNX_NAMESPACE::GraphProto * p_g, const std::shared_ptr<Graph> & g) {
  ONNX_ASSERT(p_g != nullptr);

  if (g->has_name()) {
    p_g->set_name(g->name());
  }

  if (g->has_doc_string()) {
    p_g->set_doc_string(g->docString());
  }

  for (auto input : g->inputs()) {
    ONNX_NAMESPACE::ValueInfoProto* v = p_g->add_input();
    encodeValueInfo(v, input);
  }
  for (auto output : g->outputs()) {
    ONNX_NAMESPACE::ValueInfoProto* v = p_g->add_output();
    encodeValueInfo(v, output);
  }

  std::unordered_set<Value*> graph_outputs(g->outputs().begin(), g->outputs().end());

  for (auto node : g->nodes()) {
    if (node->kind() == kUndefined) {
      // Undefined nodes are used to represent optional inputs that are not provided.
      continue;
    }
    auto p_n = p_g->add_node();
    for(auto input : node->inputs()) {
      if (input->node()->kind() == kUndefined) {
        p_n->add_input("");
      } else {
        p_n->add_input(value_name(input));
      }
    }
    for(auto output : node->outputs()) {
      p_n->add_output(value_name(output));

      // only save it if
      //  - it has actual information worth saving
      //  - it's not already saved in the graph outputs value info
      if (graph_outputs.find(output) != graph_outputs.end()) {
        continue;
      }
      if (output->elemType() == ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED &&
          output->sizes().empty()) {
        continue;
      }
      ONNX_NAMESPACE::ValueInfoProto* v = p_g->add_value_info();
      encodeValueInfo(v, output);
    }
    p_n->set_op_type(node->kind().toString());
    for(auto attr_name : node->attributeNames()) {
      addAttribute(p_n, node, attr_name);
    }
    if (node->has_doc_string()) {
      p_n->set_doc_string(node->docString());
    }
    if (node->has_name()) {
      p_n->set_name(node->name());
    }
  }

  auto num_initializers = g->initializers().size();
  for (unsigned int i = 0; i < num_initializers; i++) {
    auto p = p_g->add_initializer();
    p->set_name(g->initializer_names()[i]);
    encodeTensor(p, g->initializers()[i]);
  }
}

void ExportModelProto(ONNX_NAMESPACE::ModelProto* p_m, const std::shared_ptr<Graph>& g) {
  ONNX_NAMESPACE::GraphProto* p_g = p_m->mutable_graph();
  encodeGraph(p_g, g);
}

} // namespace ONNX_NAMESPACE
