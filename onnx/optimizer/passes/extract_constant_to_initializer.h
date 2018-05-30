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
                    
                    const auto name = n->output()->uniqueName();
                    
                    Symbol sym = Symbol("value");
                    Tensor t = n->t(sym);
                    t.setName(name); 
                    graph.addInitializer(t, name);

                    std::vector<Dimension> tsizes;
                    for (auto v : t.sizes())
                        tsizes.push_back(v);

                    Node* param = graph.create(kParam, 1);
                    param->output()->setUniqueName(name);
                    param->output()->setSizes(tsizes);
                    param->output()->setElemType(t.elem_type());
                    
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
