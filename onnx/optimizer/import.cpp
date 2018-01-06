#include "onnx/optimizer/import.h"

namespace onnx { namespace optimization {

std::unique_ptr<Graph> graphProtoToGraph(const onnx::GraphProto& gp);

Tensor tensorProtoToTensor(const onnx::TensorProto & tp) {
  Tensor ret;

  ret.sizes().reserve(tp.dims_size());
  for (int i = 0; i < tp.dims_size(); i++) {
    ret.sizes().push_back(tp.dims(i));
  }

  ret.elem_type() = tp.data_type();
  switch(tp.data_type()) {
  case onnx::TensorProto_DataType_FLOAT:
  case onnx::TensorProto_DataType_COMPLEX64: {
    ret.floats().reserve(tp.float_data_size());
    for (int i = 0; i < tp.float_data_size(); i++) {
      ret.floats().push_back(tp.float_data(i));
    }
    break;
  }
  case onnx::TensorProto_DataType_FLOAT16:
  case onnx::TensorProto_DataType_BOOL:
  case onnx::TensorProto_DataType_INT8:
  case onnx::TensorProto_DataType_INT16:
  case onnx::TensorProto_DataType_INT32:
  case onnx::TensorProto_DataType_UINT8:
  case onnx::TensorProto_DataType_UINT16: {
    ret.int32s().reserve(tp.int32_data_size());
    for (int i = 0; i < tp.int32_data_size(); i++) {
      ret.int32s().push_back(tp.int32_data(i));
    }
    break;
  }
  case onnx::TensorProto_DataType_INT64: {
    ret.int64s().reserve(tp.int64_data_size());
    for (int i = 0; i < tp.int64_data_size(); i++) {
      ret.int64s().push_back(tp.int64_data(i));
    }
    break;
  }
  case onnx::TensorProto_DataType_UINT32:
  case onnx::TensorProto_DataType_UINT64: {
    ret.uint64s().reserve(tp.uint64_data_size());
    for (uint i = 0; i < tp.uint64_data_size(); i++) {
      ret.uint64s().push_back(tp.uint64_data(i));
    }
    break;
  }
  case onnx::TensorProto_DataType_DOUBLE:
  case onnx::TensorProto_DataType_COMPLEX128: {
    ret.doubles().reserve(tp.double_data_size());
    for (int i = 0; i < tp.double_data_size(); i++) {
      ret.doubles().push_back(tp.double_data(i));
    }
    break;
  }
  case onnx::TensorProto_DataType_STRING: {
    ret.strings().reserve(tp.string_data_size());
    for (int i = 0; i < tp.string_data_size(); i++) {
      ret.strings().push_back(tp.string_data(i));
    }
    break;
  }
  case onnx::TensorProto_DataType_UNDEFINED:
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

void convertAttribute(const onnx::AttributeProto & ap, optimization::Node * n) {
  Symbol sym = stringToSymbol(ap.name());
  switch(ap.type()) {
  case onnx::AttributeProto_AttributeType_FLOAT:
    n->f_(sym, ap.f());
    break;
  case onnx::AttributeProto_AttributeType_FLOATS: {
    std::vector<double> floats;
    floats.reserve(ap.floats_size());
    for (int i = 0; i < ap.floats_size(); i++) {
      floats.push_back(ap.floats(i));
    }
    n->fs_(sym, std::move(floats));
    break;
  }
  case onnx::AttributeProto_AttributeType_INT:
    n->i_(sym, ap.i());
    break;
  case onnx::AttributeProto_AttributeType_INTS: {
    std::vector<int64_t> ints;
    ints.reserve(ap.ints_size());
    for (int i = 0; i < ap.ints_size(); i++) {
      ints.push_back(ap.ints(i));
    }
    n->is_(sym, std::move(ints));
    break;
  }
  case onnx::AttributeProto_AttributeType_STRING:
    n->s_(sym, ap.s());
    break;
  case onnx::AttributeProto_AttributeType_STRINGS: {
    std::vector<std::string> strings;
    strings.reserve(ap.strings_size());
    for (int i = 0; i < ap.strings_size(); i++) {
      strings.push_back(ap.strings(i));
    }
    n->ss_(sym, std::move(strings));
    break;
  }
  case onnx::AttributeProto_AttributeType_TENSOR:
    n->t_(sym, tensorProtoToTensor(ap.t()));
    break;
  case onnx::AttributeProto_AttributeType_TENSORS: {
    std::vector<Tensor> tensors;
    tensors.reserve(ap.tensors_size());
    for (int i = 0; i < ap.tensors_size(); i++) {
      tensors.push_back(tensorProtoToTensor(ap.tensors(i)));
    }
    n->ts_(sym, std::move(tensors));
    break;
  }
  case onnx::AttributeProto_AttributeType_GRAPH:
    n->g_(sym, graphProtoToGraph(ap.g()));
    break;
  case onnx::AttributeProto_AttributeType_GRAPHS: {
    std::vector<std::shared_ptr<Graph>> graphs;
    graphs.reserve(ap.graphs_size());
    for (int i = 0; i < ap.graphs_size(); i++) {
      graphs.push_back(graphProtoToGraph(ap.graphs(i)));
    }
    n->gs_(sym, std::move(graphs));
    break;
  }
  case onnx::AttributeProto_AttributeType_UNDEFINED:
    abort();
    break;
  }
}

void convertAttributes(onnx::NodeProto & np, optimization::Node * n) {
  for (int i = 0; i < np.attribute_size(); i++) {
    convertAttribute(np.attribute(i), n);
  }
}

std::vector<optimization::Dimension> tensorShapeProtoToDimensions(const onnx::TensorShapeProto & tsp) {
  std::vector<optimization::Dimension> dims;
  dims.reserve(tsp.dim_size());
  for (int i = 0; i < tsp.dim_size(); i++) {
    if (tsp.dim(i).has_dim_value()) {
      dims.push_back(optimization::Dimension(true, tsp.dim(i).dim_value(), ""));
    } else {
      dims.push_back(optimization::Dimension(false, -1, tsp.dim(i).dim_param()));
    }
  }
  return dims;
}

std::unique_ptr<Graph> graphProtoToGraph(const onnx::GraphProto& gp) {
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
    auto * n = g->create(stringToSymbol(np.op_type()), /* num_outputs = */ np.output_size());
    g->appendNode(n);
    for (int j = 0; j < np.output_size(); j++) {
      auto out = n->outputs()[j];
      // we don't know the real type here, so that's done in a later pass
      out->setElemType(onnx::TensorProto_DataType_UNDEFINED);
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

std::unique_ptr<Graph> ImportModel(const onnx::ModelProto& mp) {
  if (!mp.has_ir_version()) {
    return nullptr;
  } else if (mp.ir_version() == 1) {
    return nullptr;
  }

  return graphProtoToGraph(mp.graph());
}

}}
