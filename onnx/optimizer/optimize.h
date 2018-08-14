// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/common/stl_backports.h"
#include "onnx/optimizer/passes/eliminate_identity.h"
#include "onnx/optimizer/passes/eliminate_nop_transpose.h"
#include "onnx/optimizer/passes/eliminate_unused_initializer.h"
#include "onnx/optimizer/passes/extract_constant_to_initializer.h"
#include "onnx/optimizer/passes/fuse_add_bias_into_conv.h"
#include "onnx/optimizer/passes/fuse_consecutive_squeezes.h"
#include "onnx/optimizer/passes/fuse_consecutive_transposes.h"
#include "onnx/optimizer/passes/fuse_transpose_into_gemm.h"
#include "onnx/optimizer/passes/lift_lexical_references.h"
#include "onnx/optimizer/passes/nop.h"
#include "onnx/optimizer/passes/split.h"
#include "onnx/optimizer/passes/fuse_bn_into_conv.h"
#include "onnx/proto_utils.h"

namespace ONNX_NAMESPACE { namespace optimization {

struct Optimizer {
  std::map<std::string, std::unique_ptr<OptimizePass>> passes;

  Optimizer() {
    // Register the optimization passes to the optimizer.
    registerOptimizer<EliminateIdentity>();
    registerOptimizer<EliminateNopTranspose>();
    registerOptimizer<EliminateUnusedInitializer>();
    registerOptimizer<ExtractConstantToInitializer>();
    registerOptimizer<FuseConsecutiveSqueezes>();
    registerOptimizer<FuseConsecutiveTransposes>();
    registerOptimizer<FuseTransposeIntoGemm>();
    registerOptimizer<FuseAddBiasIntoConv>();
    registerOptimizer<Nop>();
    registerOptimizer<SplitInit>();
    registerOptimizer<SplitPredict>();
    registerOptimizer<LiftLexicalReferences>();
    registerOptimizer<FuseBNIntoConv>();
  }

  virtual ~Optimizer() = default;

  ModelProto optimize(
      const ModelProto& mp_in,
      const std::vector<std::string>& names) {
    std::shared_ptr<Graph> g(ImportModelProto(mp_in));

    if (g.get() == nullptr) {
      std::cerr << "Warning: onnx optimizer is unable to parse input model. "
        << "(The IR version of the ONNX model may be too old.)" << std::endl;
      // If we can't parse the file, just return the input.
      return mp_in;
    }

    ModelProto mp_out = PrepareOutput(mp_in);

    for (const auto& name : names) {
      auto it = passes.find(name);
      ONNX_ASSERTM(it != passes.end(), "pass %s is unknown.", name.c_str());
      if (it != passes.end()) {
        const auto& pass = it->second;
        if (pass->type == API_TYPE::PROTO) {
          // Operate on ModelProto.
          ExportModelProto(&mp_out, g);
          pass->optimize(mp_out);
          g = ImportModelProto(mp_out);

        } else {
          // Operate on Graph (IR).
          pass->optimize(*g);
        }
      }
    }

    ExportModelProto(&mp_out, g);
    return mp_out;
  }

  template<class Optimizer, class... Args> void registerOptimizer(Args&& ...args) {
    auto optimizer = make_unique<Optimizer>(std::forward<Args>(args)...);
    passes[optimizer->name] = std::move(optimizer);
  }
};

const std::vector<std::string> GetAvailablePasses();

ModelProto Optimize(
    const ModelProto& mp_in,
    const std::vector<std::string>& names);

} // namespace optimization
} // namespace ONNX_NAMESPACE
