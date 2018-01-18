#pragma once

#include "onnx/ir.h"

namespace onnx { namespace optimization {

void optimize(std::shared_ptr<Graph> graph);
void fuse_consecutive_transposes(std::shared_ptr<Graph>& graph);
void eliminate_nop_transpose(std::shared_ptr<Graph>& graph);
void fuse_transpose_into_gemm(std::shared_ptr<Graph>& graph);
void split_init_and_predict(std::shared_ptr<Graph> graph, bool init, bool predict);

}}
