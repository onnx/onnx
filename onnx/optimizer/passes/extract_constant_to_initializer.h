// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/optimizer/passes/optimize_pass.h"

namespace ONNX_NAMESPACE { namespace optimization {

    struct ExtractConstantToInitializer final : public OptimizePass{

        explicit ExtractConstantToInitializer()
            : OptimizePass("extract_constant_to_initializer", API_TYPE::IR){
            }

        void extract_constant_to_initializer(Graph& graph){
            for (auto it = graph.begin(); it != graph.end(); ++it){
                auto* n = *it;
                DescendOnGraphAttributes(n, [this](Graph& g){extract_constant_to_initializer(g);});
                if (n->kind() == kConstant){
                    
                    Symbol sym = Symbol("value");
                    Tensor t = n->t(sym);
                    auto name = n->output()->uniqueName();
               
                    graph.addInitializer(t, name);
                     
                    Node* param = graph.create(kParam, 1);
                    std::vector<Dimension> s = {1};
                    param->output()->setUniqueName(name);
                    param->output()->setSizes(s);
                    param->output()->setElemType(TensorProto_DataType_INT64);
                    
                    graph.addInput()->copyMetadata(param->output());
                    n->replaceAllUsesWith(param);
                    it.destroyCurrent();
                }
            }
        }

        void optimize(Graph& graph) override {
            extract_constant_to_initializer(graph);
        }
    };


}} // namespace ONNX_NAMESPACE::optimization
