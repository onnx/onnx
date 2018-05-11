// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/common/stl_backports.h"
#include "onnx/optimizer/passes/eliminate_identity.h"
#include "onnx/optimizer/passes/eliminate_nop_transpose.h"
#include "onnx/optimizer/passes/fuse_consecutive_transposes.h"
#include "onnx/optimizer/passes/fuse_add_bias_into_conv.h"
#include "onnx/optimizer/passes/fuse_transpose_into_gemm.h"
#include "onnx/optimizer/passes/lift_lexical_references.h"
#include "onnx/optimizer/passes/nop.h"
#include "onnx/optimizer/passes/split.h"
#include "onnx/proto_utils.h"

namespace ONNX_NAMESPACE { namespace optimization {

ONNX_NAMESPACE::ModelProto PrepareOutput(const ONNX_NAMESPACE::ModelProto& mp_in);

struct Optimizer {
  std::map<std::string, std::unique_ptr<OptimizePass>> passes;

  Optimizer() {
    // Register the optimization passes to the optimizer.
    _registerOptimizer<EliminateIdentity>();
    _registerOptimizer<EliminateNopTranspose>();
    _registerOptimizer<FuseConsecutiveTransposes>();
    _registerOptimizer<FuseTransposeIntoGemm>();
    _registerOptimizer<FuseAddBiasIntoConv>();
    _registerOptimizer<Nop>();
    _registerOptimizer<SplitInit>();
    _registerOptimizer<SplitPredict>();
    _registerOptimizer<LiftLexicalReferences>();
  }

  virtual ~Optimizer() = default;

  ONNX_NAMESPACE::ModelProto optimize(
      const ONNX_NAMESPACE::ModelProto& mp_in,
      const std::vector<std::string>& names) {
    std::shared_ptr<ONNX_NAMESPACE::Graph> g(ONNX_NAMESPACE::ImportModelProto(mp_in));

    if (g.get() == nullptr) {
      std::cerr << "Warning: onnx optimizer is unable to parse input model. "
        << "(The IR version of the ONNX model may be too old.)" << std::endl;
      // If we can't parse the file, just return the input.
      return mp_in;
    }

    ONNX_NAMESPACE::ModelProto mp_out = PrepareOutput(mp_in);

    for (const auto& name : names) {
      auto it = passes.find(name);
      ONNX_ASSERTM(it != passes.end(), "pass %s is unknown.", name.c_str());
      if (it != passes.end()) {
        const auto& pass = it->second;
        if (pass->type == API_TYPE::PROTO) {
          // Operate on ModelProto.
          ExportModelProto(&mp_out, g);
          pass->optimize(mp_out);
          g = ONNX_NAMESPACE::ImportModelProto(mp_out);

        } else {
          // Operate on Graph (IR).
          pass->optimize(*g);
        }
      }
    }

    ExportModelProto(&mp_out, g);
    return mp_out;
  }

private:
  template<class Optimizer, class... Args> void _registerOptimizer(Args&& ...args) {
    auto optimizer = make_unique<Optimizer>(std::forward<Args>(args)...);
    passes[optimizer->name] = std::move(optimizer);
  }
};

ONNX_NAMESPACE::ModelProto Optimize(
    const ONNX_NAMESPACE::ModelProto& mp_in,
    const std::vector<std::string>& names);
}}
