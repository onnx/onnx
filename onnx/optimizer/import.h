#pragma once

#include "onnx/optimizer/ir.h"
#include "onnx/onnx_pb.h"

namespace onnx { namespace optimization {

std::unique_ptr<Graph> ImportModel(const onnx::ModelProto& mp);

}}
