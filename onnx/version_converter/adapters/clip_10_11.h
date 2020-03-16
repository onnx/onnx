// Adapter for Clip in default domain from version 10 to 11

#pragma once
#include <limits>

namespace ONNX_NAMESPACE { namespace version_conversion {

class Clip_10_11 final : public Adapter {
  public:
    explicit Clip_10_11()
      : Adapter("Clip", OpSetID(10), OpSetID(11)) {}

    void adapt_clip_10_11(std::shared_ptr<Graph> graph, Node* node) const {
      bool has_min = node->hasAttribute(kmin);
      bool has_max = node->hasAttribute(kmax);

      // Turn min/max attributes into tensor (if present) and add value as input
      if(has_min) {
        addMinInit(graph, node);
      }
      if(has_max) {
        if(!has_min) {addMinDefault(graph, node);}
        addMaxInit(graph, node);
      }
    }

    void addMinInit(std::shared_ptr<Graph> graph, Node *node) const {
      Tensor t_min;
      t_min.elem_type() = TensorProto_DataType_FLOAT;
      auto& data_min = t_min.floats();
      data_min.emplace_back(node->f(kmin));
      Value* v_min;
      v_min = graph->addInitializerAndInput(t_min, "min");
      node->addInput(v_min);
      node->removeAttribute(kmin);
    }

    void addMaxInit(std::shared_ptr<Graph> graph, Node *node) const {
      Tensor t_max;
      t_max.elem_type() = TensorProto_DataType_FLOAT;
      auto& data_max = t_max.floats();
      data_max.emplace_back(node->f(kmax));
      Value* v_max;
      v_max = graph->addInitializerAndInput(t_max, "max");
      node->addInput(v_max);
      node->removeAttribute(kmax);
    }

    void addMinDefault(std::shared_ptr<Graph> graph, Node *node) const {
      Tensor t_min;
      t_min.elem_type() = TensorProto_DataType_FLOAT;
      auto& data_min = t_min.floats();
      data_min.emplace_back(std::numeric_limits<float>::lowest());
      Value* v_min;
      v_min = graph->addInitializerAndInput(t_min, "min");
      node->addInput(v_min);
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
	    adapt_clip_10_11(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
