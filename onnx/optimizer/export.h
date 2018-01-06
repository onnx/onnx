#pragma once

#include "onnx/optimizer/ir.h"
#include "onnx/onnx_pb.h"

namespace onnx { namespace optimization {

void encodeGraph(onnx::ModelProto* p_m, const std::shared_ptr<Graph>& g);

}}
