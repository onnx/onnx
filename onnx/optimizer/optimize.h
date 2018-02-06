// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include "onnx/common/ir.h"
#include "onnx/common/ir_pb_converter.h"
#include "onnx/optimizer/passes/eliminate_nop_transpose.h"
#include "onnx/optimizer/passes/fuse_consecutive_transposes.h"
#include "onnx/optimizer/passes/fuse_transpose_into_gemm.h"
#include "onnx/optimizer/passes/split.h"
#include "onnx/proto_utils.h"

namespace ONNX_NAMESPACE { namespace optimization {

void PrepareOutput(const ONNX_NAMESPACE::ModelProto& mp_in, ONNX_NAMESPACE::ModelProto& mp_out);

struct Optimizer {
  std::map<std::string, std::unique_ptr<OptimizePass>> passes;

  Optimizer() {
    // Register the optimization passes to the optimizer.
    std::unique_ptr<FuseConsecutiveTransposes> fct(new FuseConsecutiveTransposes());
    passes[fct->name] = std::move(fct);
    std::unique_ptr<EliminateNopTranspose> ent(new EliminateNopTranspose());
    passes[ent->name] = std::move(ent);
    std::unique_ptr<FuseTransposeIntoGemm> ftg(new FuseTransposeIntoGemm());
    passes[ftg->name] = std::move(ftg);
    std::unique_ptr<SplitInit> si(new SplitInit());
    passes[si->name] = std::move(si);
    std::unique_ptr<SplitPredict> sp(new SplitPredict());
    passes[sp->name] = std::move(sp);
  }

  std::unique_ptr<ONNX_NAMESPACE::ModelProto> optimize(std::unique_ptr<ONNX_NAMESPACE::ModelProto> mp_in, std::vector<std::string>& names) {

    std::shared_ptr<ONNX_NAMESPACE::Graph> g(ONNX_NAMESPACE::ImportModelProto(*mp_in));

    if (g.get() == nullptr) {
      std::cerr << "Warning: onnx optimizer is unable to parse input model. "
        << "(The IR version of the ONNX model may be too old.)" << std::endl;
      // If we can't parse the file, just return the input.
      return mp_in;
    }

    for (auto & name : names) {
      auto it = passes.find(name);
      ONNX_ASSERTM(it != passes.end(), "pass %s is unknown.", name.c_str());
      if (it != passes.end()) {
        auto& pass = it->second;
        if (pass->type == API_TYPE::PROTO) {
          // Operate on ModelProto.
          std::unique_ptr<ONNX_NAMESPACE::ModelProto> temp_out(new ModelProto());
          PrepareOutput(*mp_in, *temp_out);
          ExportModelProto(temp_out.get(), g);
          pass->optimize(*temp_out);
          g = ONNX_NAMESPACE::ImportModelProto(*temp_out);
          mp_in = std::move(temp_out);
        } else {
          // Operate on Graph (IR).
          pass->optimize(*g);
        }
      }
    }
    std::unique_ptr<ONNX_NAMESPACE::ModelProto> mp_out(new ModelProto());
    PrepareOutput(*mp_in, *mp_out);
    ExportModelProto(mp_out.get(), g);
    return mp_out;
  }
};

std::unique_ptr<ONNX_NAMESPACE::ModelProto> Optimize(std::unique_ptr<ONNX_NAMESPACE::ModelProto> mp_in, std::vector<std::string>& names);

}}
