// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "onnx/defs/doc_strings.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    ImageDecoder,
    20,
    OpSchema()
        .SetDoc(kDoc_ImageDecoder_ver20)
        .Attr(
            "pixel_format",
            "Pixel format. Can be one of \"RGB\", \"BGR\", or \"Grayscale\".",
            AttributeProto::STRING,
            std::string("RGB"))
        .Input(0, "encoded_stream", "Encoded stream", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "image", "Decoded image", "T2", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint("T1", {types::UInt8}, "Constrain input types to 8-bit unsigned integer tensor.")
        .TypeConstraint("T2", {types::UInt8}, "Constrain output types to 8-bit unsigned integer tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
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
        }));

} // namespace ONNX_NAMESPACE
