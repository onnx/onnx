// Adapter for Upsample in default domain from version 9 to 8

#pragma once

#include "onnx/version_converter/adapters/adapter.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

struct Upsample_9_8 final : public Adapter {
  explicit Upsample_9_8()
    : Adapter("Upsample", OpSetID(9), OpSetID(8)) {
    }

  void adapt_upsample_9_8(std::shared_ptr<Graph> graph, Node* node) const {

      const ArrayRef<Value*>& inputs = node->inputs();
      const std::vector<Tensor>& initializers = graph->initializers();
      std::string scale_input_name = node->inputs()[1]->uniqueName();
    
      if (initializers.size() > 0)
      {
        for(size_t i = 0; i < initializers.size(); i++)
        {
            if(initializers[i].name() == inputs[1]->uniqueName())
            {
              const std::vector<float>& value = initializers[i].floats();
              std::vector<double> d_values;
              for (size_t j = 0; j < value.size(); j++)
              {
                d_values.push_back(static_cast<double>(value[j]));
              }
              node->fs_(kscales, const_cast<std::vector<double>&&>(d_values));
              
              node->removeInput(1);
              graph->eraseInitializer(initializers[i].name());            
              for(size_t j = 0; j < graph->inputs().size(); j++)
              {
                if(graph->inputs()[j]->uniqueName() == scale_input_name)
                {
                  graph->eraseInput(j);
                  break;
                }
              }
              break;
            }
        }
      }
      else
      {
        for(Node *op : graph->nodes())
        {
          if (op->kind() == kConstant && op->outputs()[0]->uniqueName() == scale_input_name)
          {
            const std::vector<float>& value = op->t(kvalue).floats();
            std::vector<double> d_values;
            for (size_t j = 0; j < value.size(); j++)
            {
              d_values.push_back(static_cast<double>(value[j]));
            }            
            node->fs_(kscales, const_cast<std::vector<double>&&>(d_values));
            node->removeInput(1);
            return;
          }
        }
      }
      ONNX_ASSERT("Unsuppported conversion due to unavailable input: scale");
  }

  void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
    adapt_upsample_9_8(graph, node);
  }
};

}} // namespace ONNX_NAMESPACE::version_conversion
