#pragma once

#include <list>

#include "onnx/ir.h"
#include "onnx/ir_pb_converter.h"
#include "onnx/optimizer/passes/eliminate_nop_transpose.h"
#include "onnx/optimizer/passes/fuse_consecutive_transposes.h"
#include "onnx/optimizer/passes/fuse_transpose_into_gemm.h"
#include "onnx/optimizer/passes/split.h"
#include "onnx/proto_utils.h"

namespace onnx { namespace optimization {

void PrepareOutput(const onnx::ModelProto& mp_in, onnx::ModelProto& mp_out);

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

  std::string optimize(const std::string& content, std::list<std::string>& names) {

    onnx::ModelProto mp_in;
    ParseProtoFromBytes(&mp_in, content.c_str(), content.size());
    std::shared_ptr<onnx::Graph> g = onnx::ImportModelProto(mp_in);
    std::string out;

    if (g.get() == nullptr) {
      std::cerr << "Warning: onnx optimizer is unable to parse input model" << std::endl;
      // If we can't parse the file, just return the original content.
      out = content;
    } else {
      for (auto & name : names) {
        auto it = passes.find(name);
        if (it != passes.end()) {
          auto& pass = it->second;
          if (pass->type == API_TYPE::proto) {
            // Operate on ModelProto.
            onnx::ModelProto temp_out;
            PrepareOutput(mp_in, temp_out);
            ExportModelProto(&temp_out, g);
            pass->optimize(temp_out);
            g = onnx::ImportModelProto(mp_in);
          } else {
            // Operate on Graph (IR).
            pass->optimize(g);
          }
        }
      }
      onnx::ModelProto mp_out;
      PrepareOutput(mp_in, mp_out);
      ExportModelProto(&mp_out, g);
      mp_out.SerializeToString(&out);
    }

    return out;
  }
};

std::string Optimize(const std::string& content, std::list<std::string>& names);

}}
