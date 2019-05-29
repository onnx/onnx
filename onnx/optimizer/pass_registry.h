// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/common/stl_backports.h"
#include "onnx/optimizer/passes/eliminate_deadend.h"
#include "onnx/optimizer/passes/eliminate_identity.h"
#include "onnx/optimizer/passes/eliminate_nop_dropout.h"
#include "onnx/optimizer/passes/eliminate_nop_monotone_argmax.h"
#include "onnx/optimizer/passes/eliminate_nop_pad.h"
#include "onnx/optimizer/passes/eliminate_nop_transpose.h"
#include "onnx/optimizer/passes/eliminate_unused_initializer.h"
#include "onnx/optimizer/passes/extract_constant_to_initializer.h"
#include "onnx/optimizer/passes/fuse_add_bias_into_conv.h"
#include "onnx/optimizer/passes/fuse_bn_into_conv.h"
#include "onnx/optimizer/passes/fuse_consecutive_concats.h"
#include "onnx/optimizer/passes/fuse_consecutive_log_softmax.h"
#include "onnx/optimizer/passes/fuse_consecutive_reduce_unsqueeze.h"
#include "onnx/optimizer/passes/fuse_consecutive_squeezes.h"
#include "onnx/optimizer/passes/fuse_consecutive_transposes.h"
#include "onnx/optimizer/passes/fuse_matmul_add_bias_into_gemm.h"
#include "onnx/optimizer/passes/fuse_pad_into_conv.h"
#include "onnx/optimizer/passes/fuse_transpose_into_gemm.h"
#include "onnx/optimizer/passes/lift_lexical_references.h"
#include "onnx/optimizer/passes/nop.h"
#include "onnx/optimizer/passes/split.h"
#include "onnx/proto_utils.h"

#include <unordered_set>
#include <vector>

namespace ONNX_NAMESPACE {
namespace optimization {

// Registry containing all passes available in ONNX.
struct GlobalPassRegistry {
  std::map<std::string, std::shared_ptr<Pass>> passes;

  GlobalPassRegistry() {
    // Register the optimization passes to the optimizer.
    registerPass<NopEmptyPass>();
    registerPass<EliminateDeadEnd>();
    registerPass<EliminateNopDropout>();
    registerPass<EliminateIdentity>();
    registerPass<EliminateNopMonotoneArgmax>();
    registerPass<EliminateNopPad>();
    registerPass<EliminateNopTranspose>();
    registerPass<EliminateUnusedInitializer>();
    registerPass<ExtractConstantToInitializer>();
    registerPass<FuseAddBiasIntoConv>();
    registerPass<FuseBNIntoConv>();
    registerPass<FuseConsecutiveConcats>();
    registerPass<FuseConsecutiveLogSoftmax>();
    registerPass<FuseConsecutiveReduceUnsqueeze>();
    registerPass<FuseConsecutiveSqueezes>();
    registerPass<FuseConsecutiveTransposes>();
    registerPass<FuseMatMulAddBiasIntoGemm>();
    registerPass<FusePadIntoConv>();
    registerPass<FuseTransposeIntoGemm>();
    registerPass<LiftLexicalReferences>();
    registerPass<SplitInit>();
    registerPass<SplitPredict>();
  }

  ~GlobalPassRegistry() {
    this->passes.clear();
  }

  std::shared_ptr<Pass> find(std::string pass_name) {
    auto it = this->passes.find(pass_name);
    ONNX_ASSERTM(
        it != this->passes.end(), "pass %s is unknown.", pass_name.c_str());
    return it->second;
  }
  const std::vector<std::string> GetAvailablePasses();

  template <typename T>
  void registerPass() {
    static_assert(std::is_base_of<Pass, T>::value, "T must inherit from Pass");
    std::shared_ptr<Pass> pass(new T());
    passes[pass->getPassName()] = pass;
  }
};
} // namespace optimization
} // namespace ONNX_NAMESPACE
