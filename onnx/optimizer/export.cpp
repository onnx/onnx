#include "onnx/optimizer/export.h"

namespace onnx { namespace optimization {

namespace {

std::string value_name(Value* n) {
  return n->uniqueName();
}


void encodeGraph(onnx::GraphProto * p_g, const std::shared_ptr<Graph> & g);

void encodeTensor(onnx::TensorProto * p, const Tensor & tensor) {
  if (tensor.hasName()) {
    p->set_name(tensor.name());
  }
  if (tensor.is_segment()) {
    onnx::TensorProto_Segment segment;
    segment.set_begin(tensor.segment_begin());
    segment.set_end(tensor.segment_end());
    p->mutable_segment()->CopyFrom(segment);
  }
  for(auto d : tensor.sizes()) {
    p->add_dims(d);
  }
  p->set_data_type(tensor.elem_type());
  switch(tensor.elem_type()) {
  case onnx::TensorProto_DataType_FLOAT:
  case onnx::TensorProto_DataType_COMPLEX64: {
    for (float x : tensor.floats()) {
      p->add_float_data(x);
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
    for (int32_t x : tensor.int32s()) {
      p->add_int32_data(x);
    }
    break;
  }
  case onnx::TensorProto_DataType_INT64: {
    for (int64_t x : tensor.int64s()) {
      p->add_int64_data(x);
    }
    break;
  }
  case onnx::TensorProto_DataType_UINT32:
  case onnx::TensorProto_DataType_UINT64: {
    for (uint64_t x : tensor.uint64s()) {
      p->add_uint64_data(x);
    }
    break;
  }
  case onnx::TensorProto_DataType_DOUBLE:
  case onnx::TensorProto_DataType_COMPLEX128: {
    for (double x : tensor.doubles()) {
      p->add_double_data(x);
    }
    break;
  }
  case onnx::TensorProto_DataType_STRING: {
    for (const std::string& x : tensor.strings()) {
      p->add_string_data(x);
    }
    break;
  }
  case onnx::TensorProto_DataType_UNDEFINED:
    abort();
  }
  if (!tensor.raw().empty()) {
    p->set_raw_data(tensor.raw());
  }
}

void addAttribute(onnx::NodeProto * n_p, optimization::Node * n, optimization::Symbol name) {
  auto attr = n_p->add_attribute();
  attr->set_name(optimization::symbolToString(name));
  switch(n->kindOf(name)) {
    case AttributeKind::f:
      attr->set_f(n->f(name));
      attr->set_type(onnx::AttributeProto_AttributeType_FLOAT);
      break;
    case AttributeKind::fs:
      attr->set_type(onnx::AttributeProto_AttributeType_FLOATS);
      for(auto & v : n->fs(name))
        attr->add_floats(v);
      break;
    case AttributeKind::i:
      attr->set_type(onnx::AttributeProto_AttributeType_INT);
      attr->set_i(n->i(name));
      break;
    case AttributeKind::is:
      attr->set_type(onnx::AttributeProto_AttributeType_INTS);
      for(auto & v : n->is(name))
        attr->add_ints(v);
      break;
    case AttributeKind::s:
      attr->set_type(onnx::AttributeProto_AttributeType_STRING);
      attr->set_s(n->s(name));
      break;
    case AttributeKind::ss:
      attr->set_type(onnx::AttributeProto_AttributeType_STRINGS);
      for(auto & v : n->ss(name))
        attr->add_strings(v);
      break;
    case AttributeKind::t: {
      attr->set_type(onnx::AttributeProto_AttributeType_TENSOR);
      auto t = attr->mutable_t();
      encodeTensor(t, n->t(name));
    } break;
    case AttributeKind::ts:
      attr->set_type(onnx::AttributeProto_AttributeType_TENSORS);
      for(auto & v : n->ts(name)) {
        auto t = attr->add_tensors();
        encodeTensor(t, v);
      }
      break;
    case AttributeKind::g: {
      attr->set_type(onnx::AttributeProto_AttributeType_GRAPH);
      auto g = attr->mutable_g();
      encodeGraph(g, n->g(name));
    } break;
    case AttributeKind::gs:
      attr->set_type(onnx::AttributeProto_AttributeType_GRAPHS);
      for(auto & v : n->gs(name)) {
        auto g = attr->add_graphs();
        encodeGraph(g, v);
      }
      break;
  }
}

void encodeTypeProtoTensorType(onnx::TypeProto_Tensor* tensor_type, Value* n) {
  tensor_type->set_elem_type(n->elemType());
  onnx::TensorShapeProto* shape = tensor_type->mutable_shape();
  for (const Dimension& d : n->sizes()) {
    auto dim = shape->add_dim();
    if (d.is_int) {
      dim->set_dim_value(d.dim);
    } else {
      dim->set_dim_param(d.param);
    }
  }
}

void encodeValueInfo(onnx::ValueInfoProto* v, Value* n) {
  if (n->has_name()) {
    v->set_name(value_name(n));
  }
  onnx::TypeProto* t = v->mutable_type();
  onnx::TypeProto_Tensor* tensor_type = t->mutable_tensor_type();
  encodeTypeProtoTensorType(tensor_type, n);
}

void encodeGraph(onnx::GraphProto * p_g, const std::shared_ptr<Graph> & g) {
  JIT_ASSERT(p_g != nullptr);

  if (g->has_name()) {
    p_g->set_name(g->name());
  }

  if (g->has_doc_string()) {
    p_g->set_doc_string(g->docString());
  }

  for (auto input : g->inputs()) {
    onnx::ValueInfoProto* v = p_g->add_input();
    encodeValueInfo(v, input);
  }
  for (auto output : g->outputs()) {
    onnx::ValueInfoProto* v = p_g->add_output();
    encodeValueInfo(v, output);
  }
  for (auto node : g->nodes()) {
    if (node->kind() == kUndefined && !node->hasUses()) {
      // Undefined nodes never show up in ONNX; they're just a tool
      // to help symbolics do the right thing.
      continue;
    }
    auto p_n = p_g->add_node();
    for(auto input : node->inputs()) {
      p_n->add_input(value_name(input));
    }
    for(auto output : node->outputs()) {
      p_n->add_output(value_name(output));
    }
    p_n->set_op_type(symbolToString(node->kind()));
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

  for (int i = 0; i < g->initializers().size(); i++) {
    auto p = p_g->add_initializer();
    p->set_name(g->initializer_names()[i]);
    encodeTensor(p, g->initializers()[i]);
  }
}

}

void encodeGraph(onnx::ModelProto* p_m, const std::shared_ptr<Graph>& g) {
  onnx::GraphProto* p_g = p_m->mutable_graph();
  encodeGraph(p_g, g);
}

}}
