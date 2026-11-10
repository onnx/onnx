// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "onnx/defs/image/utils.h"

namespace ONNX_NAMESPACE::defs::image::utils {

void ImageDecoderShapeInference_opset20(InferenceContext& ctx) {
  if (hasInputShape(ctx, 0)) {
    auto& input_shape = getInputShape(ctx, 0);
    if (input_shape.dim_size() != 1) {
      fail_shape_inference("Input tensor must be 1-dimensional");
    }
  }
  propagateElemTypeFromDtypeToOutput(ctx, TensorProto::UINT8, 0);
  auto output_type = ctx.getOutputType(0);
  auto sh = output_type->mutable_tensor_type()->mutable_shape();
  sh->clear_dim();
  sh->add_dim();
  sh->add_dim();
  sh->add_dim();
}

} // namespace ONNX_NAMESPACE::defs::image::utils
