#include <benchmark/benchmark.h>

#include <onnx/onnx.pb.h>

using namespace ONNX_NAMESPACE;


inline void createValueInfo4D(
    ValueInfoProto& value_info,
    const std::string& name,
    int64_t n,
    int64_t c,
    int64_t h,
    int64_t w) {
  value_info.set_name(name);

  TypeProto_Tensor* tensor_type =
      value_info.mutable_type()->mutable_tensor_type();
  tensor_type->set_elem_type(TensorProto_DataType_FLOAT);

  TensorShapeProto* shape = tensor_type->mutable_shape();
  shape->add_dim()->set_dim_value(n);
  shape->add_dim()->set_dim_value(c);
  shape->add_dim()->set_dim_value(h);
  shape->add_dim()->set_dim_value(w);
}

inline void createValueInfo2D(
    ValueInfoProto& value_info,
    const std::string& name,
    int64_t h,
    int64_t w) {
  value_info.set_name(name);

  TypeProto* type = value_info.mutable_type();

  TypeProto_Tensor* tensor_type = type->mutable_tensor_type();
  tensor_type->set_elem_type(TensorProto_DataType_FLOAT);
  TensorShapeProto* shape = tensor_type->mutable_shape();
  shape->add_dim()->set_dim_value(h);
  shape->add_dim()->set_dim_value(w);
}

inline void createConv2D(
    NodeProto& node,
    const std::string& input,
    const std::string& weights,
    const std::string& bias,
    const std::string& output,
    uint32_t kernel_size) {
  node.set_op_type("Conv");
  node.add_input(input);
  node.add_input(weights);
  node.add_input(bias);
  node.add_output(output);

  {
    AttributeProto* kernel = node.add_attribute();
    kernel->set_name("kernel_shape");
    kernel->set_type(AttributeProto::INTS);
    kernel->add_ints(kernel_size);
    kernel->add_ints(kernel_size);
  }
  {
    AttributeProto* dilation = node.add_attribute();
    dilation->set_name("dilations");
    dilation->set_type(AttributeProto::INTS);
    dilation->add_ints(1);
    dilation->add_ints(1);
  }
  {
    AttributeProto* stride = node.add_attribute();
    stride->set_name("strides");
    stride->set_type(AttributeProto::INTS);
    stride->add_ints(1);
    stride->add_ints(1);
  }
  {
    AttributeProto* group = node.add_attribute();
    group->set_name("group");
    group->set_type(AttributeProto::INTS);
    group->set_i(1);
  }
  {
    AttributeProto* padding = node.add_attribute();
    padding->set_name("pads");
    padding->set_type(AttributeProto::INTS);
    /* Use "same" padding */
    padding->add_ints(kernel_size / 2);
    padding->add_ints(kernel_size / 2);
    padding->add_ints(kernel_size - 1 - kernel_size / 2);
    padding->add_ints(kernel_size - 1 - kernel_size / 2);
  }
}

static void ConvGraph(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::string data;
    GraphProto graph;

    createConv2D(*graph.add_node(), "input", "weights", "bias", "output", 3);

    createValueInfo4D(*graph.add_input(), "input", 1, 3, 224, 224);
    createValueInfo4D(*graph.add_input(), "weights", 16, 16, 3, 3);
    createValueInfo2D(*graph.add_input(), "bias", 1, 16);
    createValueInfo4D(*graph.add_output(), "output", 16, 3, 224, 224);

    graph.SerializeToString(&data);

    GraphProto decodedGraph;
    decodedGraph.ParseFromString(data);
  }

  state.SetItemsProcessed(int64_t(state.iterations()));
}
BENCHMARK(ConvGraph)->Unit(benchmark::kMicrosecond);

static void ConvModel(benchmark::State& state) {
  while (state.KeepRunning()) {
    std::string data;
    ModelProto model;
    model.set_ir_version(IR_VERSION);
    OperatorSetIdProto* op_set_id = model.add_opset_import();
    op_set_id->set_domain("");
    op_set_id->set_version(4);

    GraphProto* graph = model.mutable_graph();

    createConv2D(*graph->add_node(), "input", "weights", "bias", "output", 3);

    createValueInfo4D(*graph->add_input(), "input", 1, 3, 224, 224);
    createValueInfo4D(*graph->add_input(), "weights", 16, 16, 3, 3);
    createValueInfo2D(*graph->add_input(), "bias", 1, 16);
    createValueInfo4D(*graph->add_output(), "output", 16, 3, 224, 224);

    model.SerializeToString(&data);

    ModelProto decodedModel;
    decodedModel.ParseFromString(data);
  }

  state.SetItemsProcessed(int64_t(state.iterations()));
}
BENCHMARK(ConvModel)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
