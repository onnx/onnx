// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "onnx/common/safe_math.h"
#include "onnx/defs/data_propagators.h"
#include "onnx/defs/doc_strings.h"
#include "onnx/defs/function.h"
#include "onnx/defs/tensor/utils.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    25,
    OpSchema()
        .SetDoc(kDoc_Cast_ver24)
        .Attr(
            "to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
        .Attr(
            "saturate",
            "The parameter defines how the conversion behaves if an input value is out of "
            "range of the destination type. It only applies for float 8 conversion "
            "(float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz, float8e8m0). It is true by default. "
            "All cases are fully described in the tables inserted in the operator description.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "round_mode",
            "Rounding mode for conversion to float8e8m0. It only applies to casting to float8e8m0 and is `up` by default. "
            "`up`: round to nearest value away from zero, "
            "`down`: round to nearest value towards zero, "
            "`nearest`: round to nearest value and ties round up.",
            AttributeProto::STRING,
            std::string("up"))
        .Input(0, "input", "Input tensor to be cast.", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "Output tensor with the same shape as input with type "
            "specified by the 'to' argument",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            OpSchema::all_non_complex_tensor_types_ir13(),
            "Constrain input types. Casting from complex is not supported.")
        .TypeConstraint(
            "T2",
            OpSchema::all_non_complex_tensor_types_ir13(),
            "Constrain output types. Casting to complex is not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

static void checked_mul_into(int64_t& accumulator, int64_t value) {
  if (checked_mul_overflow(accumulator, value, &accumulator)) {
    fail_shape_inference("Dimension product overflow in Reshape");
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    GridSample,
    20,
    OpSchema()
        .Attr(
            "mode",
            "Three interpolation modes: linear (default), nearest and cubic. "
            "The \"linear\" mode includes linear and N-linear interpolation modes depending on the number of spatial dimensions "
            "of the input tensor (i.e. linear for 1 spatial dimension, bilinear for 2 spatial dimensions, etc.). "
            "The \"cubic\" mode also includes N-cubic interpolation modes following the same rules. The \"nearest\" mode rounds "
            "to the nearest even index when the sampling point falls halfway between two indices.",
            AttributeProto::STRING,
            std::string("linear"))
        .Attr(
            "padding_mode",
            "Support padding modes for outside grid values: `zeros`(default), `border`, `reflection`. "
            "zeros: use 0 for out-of-bound grid locations, "
            "border: use border values for out-of-bound grid locations, "
            "reflection: use values at locations reflected by the border for out-of-bound grid locations. "
            "If index 0 represents the margin pixel, the reflected value at index -1 will be the same as the value at index 1. "
            "For location far away from the border, it will keep being reflected until becoming in bound. "
            "If pixel location x = -3.5 reflects by border -1 and becomes x' = 1.5, then reflects by border 1 and becomes x'' = 0.5.",
            AttributeProto::STRING,
            std::string("zeros"))
        .Attr(
            "align_corners",
            "If align_corners=1, the extrema (-1 and 1) are considered as referring to the center points of the input's corner pixels (voxels, etc.). "
            "If align_corners=0, they are instead considered as referring to the corner points of the input's corner pixels (voxels, etc.), "
            "making the sampling more resolution agnostic.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(
            0,
            "X",
            "Input tensor of rank r+2 that has shape (N, C, D1, D2, ..., Dr), where N is the batch size, "
            "C is the number of channels, D1, D2, ..., Dr are the spatial dimensions.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "grid",
            "Input offset of shape (N, D1_out, D2_out, ..., Dr_out, r), where D1_out, D2_out, ..., "
            "Dr_out are the spatial dimensions of the grid and output, and r is the number of spatial dimensions. "
            "Grid specifies the sampling locations normalized by the input spatial dimensions. "
            "Therefore, it should have most values in the range of [-1, 1]. If the grid has values outside the range of [-1, 1], "
            "the corresponding outputs will be handled as defined by padding_mode. Following computer vision convention, "
            "the coordinates in the length-r location vector are listed from the innermost tensor dimension to the outermost, "
            "the opposite of regular tensor indexing.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "Y",
            "Output tensor of rank r+2 that has shape (N, C, D1_out, D2_out, ..., Dr_out) of the sampled values. "
            "For integer input types, intermediate values are computed as floating point and cast to integer at the end.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            OpSchema::all_tensor_types(),
            "Constrain input `X` and output `Y` types to all tensor types.")
        .TypeConstraint("T2", {types::Float16, types::Float, types::Double}, "Constrain grid types to float tensors.")
        .SetDoc(kDoc_GridSample_ver20)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { gridSampleShapeInference(ctx); }));

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    24,
    OpSchema()
        .SetDoc(kDoc_Cast_ver24)
        .Attr(
            "to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
        .Attr(
            "saturate",
            "The parameter defines how the conversion behaves if an input value is out of "
            "range of the destination type. It only applies for float 8 conversion "
            "(float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz, float8e8m0). It is true by default. "
            "All cases are fully described in the tables inserted in the operator description.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "round_mode",
            "Rounding mode for conversion to float8e8m0. It only applies to casting to float8e8m0 and is `up` by default. "
            "`up`: round to nearest value away from zero, "
            "`down`: round to nearest value towards zero, "
            "`nearest`: round to nearest value and ties round up.",
            AttributeProto::STRING,
            std::string("up"))
        .Input(0, "input", "Input tensor to be cast.", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "Output tensor with the same shape as input with type "
            "specified by the 'to' argument",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            OpSchema::all_non_complex_tensor_types_ir12(),
            "Constrain input types. Casting from complex is not supported.")
        .TypeConstraint(
            "T2",
            OpSchema::all_non_complex_tensor_types_ir12(),
            "Constrain output types. Casting to complex is not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    23,
    OpSchema()
        .SetDoc(kDoc_Cast_ver23)
        .Attr(
            "to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
        .Attr(
            "saturate",
            "The parameter defines how the conversion behaves if an input value is out of "
            "range of the destination type. It only applies for float 8 conversion "
            "(float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz). It is true by default. "
            "All cases are fully described in two tables inserted in the operator description.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Input(0, "input", "Input tensor to be cast.", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "Output tensor with the same shape as input with type "
            "specified by the 'to' argument",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            OpSchema::all_non_complex_tensor_types_ir11(),
            "Constrain input types. Casting from complex is not supported.")
        .TypeConstraint(
            "T2",
            OpSchema::all_non_complex_tensor_types_ir11(),
            "Constrain output types. Casting to complex is not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    21,
    OpSchema()
        .SetDoc(kDoc_Cast_ver23)
        .Attr(
            "to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
        .Attr(
            "saturate",
            "The parameter defines how the conversion behaves if an input value is out of "
            "range of the destination type. It only applies for float 8 conversion "
            "(float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz). It is true by default. "
            "All cases are fully described in two tables inserted in the operator description.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Input(0, "input", "Input tensor to be cast.", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "Output tensor with the same shape as input with type "
            "specified by the 'to' argument",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            {types::Float16,        types::Float,      types::Double,         types::Int8,     types::Int16,
             types::Int32,          types::Int64,      types::UInt8,          types::UInt16,   types::UInt32,
             types::UInt64,         types::Bool,       types::String,         types::BFloat16, types::Float8E4M3FN,
             types::Float8E4M3FNUZ, types::Float8E5M2, types::Float8E5M2FNUZ, types::UInt4,    types::Int4},
            "Constrain input types. Casting from complex is not supported.")
        .TypeConstraint(
            "T2",
            {types::Float16,        types::Float,      types::Double,         types::Int8,     types::Int16,
             types::Int32,          types::Int64,      types::UInt8,          types::UInt16,   types::UInt32,
             types::UInt64,         types::Bool,       types::String,         types::BFloat16, types::Float8E4M3FN,
             types::Float8E4M3FNUZ, types::Float8E5M2, types::Float8E5M2FNUZ, types::UInt4,    types::Int4},
            "Constrain output types. Casting to complex is not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    19,
    OpSchema()
        .SetDoc(kDoc_Cast_ver23)
        .Attr(
            "to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
        .Attr(
            "saturate",
            "The parameter defines how the conversion behaves if an input value is out of "
            "range of the destination type. It only applies for float 8 conversion "
            "(float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz). It is true by default. "
            "All cases are fully described in two tables inserted in the operator description.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Input(0, "input", "Input tensor to be cast.", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "Output tensor with the same shape as input with type "
            "specified by the 'to' argument",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            {types::Float16,
             types::Float,
             types::Double,
             types::Int8,
             types::Int16,
             types::Int32,
             types::Int64,
             types::UInt8,
             types::UInt16,
             types::UInt32,
             types::UInt64,
             types::Bool,
             types::String,
             types::BFloat16,
             types::Float8E4M3FN,
             types::Float8E4M3FNUZ,
             types::Float8E5M2,
             types::Float8E5M2FNUZ},
            "Constrain input types. Casting from complex is not supported.")
        .TypeConstraint(
            "T2",
            {types::Float16,
             types::Float,
             types::Double,
             types::Int8,
             types::Int16,
             types::Int32,
             types::Int64,
             types::UInt8,
             types::UInt16,
             types::UInt32,
             types::UInt64,
             types::Bool,
             types::String,
             types::BFloat16,
             types::Float8E4M3FN,
             types::Float8E4M3FNUZ,
             types::Float8E5M2,
             types::Float8E5M2FNUZ},
            "Constrain output types. Casting to complex is not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    13,
    OpSchema()
        .SetDoc(kDoc_Cast_ver13)
        .Attr(
            "to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
        .Input(0, "input", "Input tensor to be cast.", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "Output tensor with the same shape as input with type "
            "specified by the 'to' argument",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            {types::Float16,
             types::Float,
             types::Double,
             types::Int8,
             types::Int16,
             types::Int32,
             types::Int64,
             types::UInt8,
             types::UInt16,
             types::UInt32,
             types::UInt64,
             types::Bool,
             types::String,
             types::BFloat16},
            "Constrain input types. Casting from complex is not supported.")
        .TypeConstraint(
            "T2",
            {types::Float16,
             types::Float,
             types::Double,
             types::Int8,
             types::Int16,
             types::Int32,
             types::Int64,
             types::UInt8,
             types::UInt16,
             types::UInt32,
             types::UInt64,
             types::Bool,
             types::String,
             types::BFloat16},
            "Constrain output types. Casting to complex is not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    24,
    OpSchema()
        .SetDoc(kDoc_CastLike_ver24)
        .Attr(
            "saturate",
            "The parameter defines how the conversion behaves if an input value is out of "
            "range of the destination type. It only applies for float 8 conversion "
            "(float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz, float8e8m0). It is true by default. "
            "Please refer to operator Cast description for further details.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "round_mode",
            "Rounding mode for conversion to float8e8m0. It only applies to casting to float8e8m0 and is `up` by default. "
            "`up`: round to nearest value away from zero, "
            "`down`: round to nearest value towards zero, "
            "`nearest`: round to nearest value and ties round up. "
            "Please refer to operator Cast description for further details.",
            AttributeProto::STRING,
            std::string("up"))
        .Input(0, "input", "Input tensor to be cast.", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "target_type",
            "The (first) input tensor will be cast to produce a tensor of the same type as this (second input) tensor.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Output tensor produced by casting the first input tensor to have the same type as the second input tensor.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            OpSchema::all_non_complex_tensor_types_ir12(),
            "Constrain input types. Casting from complex is not supported.")
        .TypeConstraint(
            "T2",
            OpSchema::all_non_complex_tensor_types_ir12(),
            "Constrain output types. Casting to complex is not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        })
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              auto target_type = ctx.getInputType(1);
              if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
                // we cannot create a correct function body without knowing the target element type
                return false;
              }
              auto target_elt_type = target_type->tensor_type().elem_type();
              FunctionBuilder builder(functionProto);
              builder.Add(MakeString(
                              "output = Cast <to= ",
                              static_cast<int64_t>(target_elt_type),
                              ", saturate: int = @saturate> (input)")
                              .c_str());
              schema.BuildFunction(functionProto);
              return true;
            }));

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    23,
    OpSchema()
        .SetDoc(kDoc_CastLike_ver24)
        .Attr(
            "saturate",
            "The parameter defines how the conversion behaves if an input value is out of "
            "range of the destination type. It only applies for float 8 conversion "
            "(float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz). It is true by default. "
            "Please refer to operator Cast description for further details.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Input(0, "input", "Input tensor to be cast.", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "target_type",
            "The (first) input tensor will be cast to produce a tensor of the same type as this (second input) tensor.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Output tensor produced by casting the first input tensor to have the same type as the second input tensor.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            OpSchema::all_non_complex_tensor_types_ir11(),
            "Constrain input types. Casting from complex is not supported.")
        .TypeConstraint(
            "T2",
            OpSchema::all_non_complex_tensor_types_ir11(),
            "Constrain output types. Casting to complex is not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        })
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              auto target_type = ctx.getInputType(1);
              if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
                // we cannot create a correct function body without knowing the target element type
                return false;
              }
              auto target_elt_type = target_type->tensor_type().elem_type();
              FunctionBuilder builder(functionProto);
              builder.Add(MakeString(
                              "output = Cast <to= ",
                              static_cast<int64_t>(target_elt_type),
                              ", saturate: int = @saturate> (input)")
                              .c_str());
              schema.BuildFunction(functionProto);
              return true;
            }));

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    21,
    OpSchema()
        .SetDoc(kDoc_CastLike_ver24)
        .Attr(
            "saturate",
            "The parameter defines how the conversion behaves if an input value is out of "
            "range of the destination type. It only applies for float 8 conversion "
            "(float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz). It is true by default. "
            "Please refer to operator Cast description for further details.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Input(0, "input", "Input tensor to be cast.", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "target_type",
            "The (first) input tensor will be cast to produce a tensor of the same type as this (second input) tensor.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Output tensor produced by casting the first input tensor to have the same type as the second input tensor.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            OpSchema::all_non_complex_tensor_types_ir10(),
            "Constrain input types. Casting from complex is not supported.")
        .TypeConstraint(
            "T2",
            OpSchema::all_non_complex_tensor_types_ir10(),
            "Constrain output types. Casting to complex is not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        })
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              auto target_type = ctx.getInputType(1);
              if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
                // we cannot create a correct function body without knowing the target element type
                return false;
              }
              auto target_elt_type = target_type->tensor_type().elem_type();
              FunctionBuilder builder(functionProto);
              builder.Add(MakeString(
                              "output = Cast <to= ",
                              static_cast<int64_t>(target_elt_type),
                              ", saturate: int = @saturate> (input)")
                              .c_str());
              schema.BuildFunction(functionProto);
              return true;
            }));

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    19,
    OpSchema()
        .SetDoc(kDoc_CastLike_ver24)
        .Attr(
            "saturate",
            "The parameter defines how the conversion behaves if an input value is out of "
            "range of the destination type. It only applies for float 8 conversion "
            "(float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz). It is true by default. "
            "Please refer to operator Cast description for further details.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Input(0, "input", "Input tensor to be cast.", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "target_type",
            "The (first) input tensor will be cast to produce a tensor of the same type as this (second input) tensor.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Output tensor produced by casting the first input tensor to have the same type as the second input tensor.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            {types::Float16,
             types::Float,
             types::Double,
             types::Int8,
             types::Int16,
             types::Int32,
             types::Int64,
             types::UInt8,
             types::UInt16,
             types::UInt32,
             types::UInt64,
             types::Bool,
             types::String,
             types::BFloat16,
             types::Float8E4M3FN,
             types::Float8E4M3FNUZ,
             types::Float8E5M2,
             types::Float8E5M2FNUZ},
            "Constrain input types. Casting from complex is not supported.")
        .TypeConstraint(
            "T2",
            {types::Float16,
             types::Float,
             types::Double,
             types::Int8,
             types::Int16,
             types::Int32,
             types::Int64,
             types::UInt8,
             types::UInt16,
             types::UInt32,
             types::UInt64,
             types::Bool,
             types::String,
             types::BFloat16,
             types::Float8E4M3FN,
             types::Float8E4M3FNUZ,
             types::Float8E5M2,
             types::Float8E5M2FNUZ},
            "Constrain output types. Casting to complex is not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        })
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              auto target_type = ctx.getInputType(1);
              if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
                // we cannot create a correct function body without knowing the target element type
                return false;
              }
              auto target_elt_type = target_type->tensor_type().elem_type();
              FunctionBuilder builder(functionProto);
              builder.Add(MakeString(
                              "output = Cast <to= ",
                              static_cast<int64_t>(target_elt_type),
                              ", saturate: int = @saturate> (input)")
                              .c_str());
              schema.BuildFunction(functionProto);
              return true;
            }));

ONNX_OPERATOR_SET_SCHEMA(
    CastLike,
    15,
    OpSchema()
        .SetDoc(kDoc_CastLike_ver24)
        .Input(0, "input", "Input tensor to be cast.", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "target_type",
            "The (first) input tensor will be cast to produce a tensor of the same type as this (second input) tensor.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Output tensor produced by casting the first input tensor to have the same type as the second input tensor.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            {types::Float16,
             types::Float,
             types::Double,
             types::Int8,
             types::Int16,
             types::Int32,
             types::Int64,
             types::UInt8,
             types::UInt16,
             types::UInt32,
             types::UInt64,
             types::Bool,
             types::String,
             types::BFloat16},
            "Constrain input types. Casting from complex is not supported.")
        .TypeConstraint(
            "T2",
            {types::Float16,
             types::Float,
             types::Double,
             types::Int8,
             types::Int16,
             types::Int32,
             types::Int64,
             types::UInt8,
             types::UInt16,
             types::UInt32,
             types::UInt64,
             types::Bool,
             types::String,
             types::BFloat16},
            "Constrain output types. Casting to complex is not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        })
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
        .SetContextDependentFunctionBodyBuilder(
            [](const FunctionBodyBuildContext& ctx, const OpSchema& schema, FunctionProto& functionProto) -> bool {
              auto target_type = ctx.getInputType(1);
              if ((target_type == nullptr) || (!target_type->has_tensor_type())) {
                // we cannot create a correct function body without knowing the target element type
                return false;
              }
              auto target_elt_type = target_type->tensor_type().elem_type();
              FunctionBuilder builder(functionProto);
              builder.Add("output = Cast (input)", "to", static_cast<int64_t>(target_elt_type));
              schema.BuildFunction(functionProto);
              return true;
            }));

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    9,
    OpSchema()
        .SetDoc(kDoc_Cast_ver9)
        .Attr(
            "to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
        .Input(0, "input", "Input tensor to be cast.", "T1")
        .Output(
            0,
            "output",
            "Output tensor with the same shape as input with type "
            "specified by the 'to' argument",
            "T2")
        .TypeConstraint(
            "T1",
            {types::Float16,
             types::Float,
             types::Double,
             types::Int8,
             types::Int16,
             types::Int32,
             types::Int64,
             types::UInt8,
             types::UInt16,
             types::UInt32,
             types::UInt64,
             types::Bool,
             types::String},
            "Constrain input types. Casting from complex is not supported.")
        .TypeConstraint(
            "T2",
            {types::Float16,
             types::Float,
             types::Double,
             types::Int8,
             types::Int16,
             types::Int32,
             types::Int64,
             types::UInt8,
             types::UInt16,
             types::UInt32,
             types::UInt64,
             types::Bool,
             types::String},
            "Constrain output types. Casting to complex is not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    GridSample,
    16,
    OpSchema()
        .Attr(
            "mode",
            "Three interpolation modes: bilinear (default), nearest and bicubic.",
            AttributeProto::STRING,
            std::string("bilinear"))
        .Attr(
            "padding_mode",
            "Support padding modes for outside grid values: `zeros`(default), `border`, `reflection`. "
            "zeros: use 0 for out-of-bound grid locations, "
            "border: use border values for out-of-bound grid locations, "
            "reflection: use values at locations reflected by the border for out-of-bound grid locations. "
            "If index 0 represents the margin pixel, the reflected value at index -1 will be the same as the value at index 1. "
            "For location far away from the border, it will keep being reflected until becoming in bound. "
            "If pixel location x = -3.5 reflects by border -1 and becomes x' = 1.5, then reflects by border 1 and becomes x'' = 0.5.",
            AttributeProto::STRING,
            std::string("zeros"))
        .Attr(
            "align_corners",
            "If align_corners=1, the extrema (-1 and 1) are considered as referring to the center points of the input's corner pixels. "
            "If align_corners=0, they are instead considered as referring to the corner points of the input's corner pixels, making the sampling more resolution agnostic.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(
            0,
            "X",
            "4-D tensor of shape (N, C, H, W), "
            "where N is the batch size, C is the numbers of channels, "
            "H and W are the height and width of the input data.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "grid",
            "Input offset, 4-D tensor of shape (N, H_out, W_out, 2), "
            "where H_out and W_out are the height and width of grid and output, "
            "Grid specifies the sampling pixel locations normalized by the input spatial dimensions. "
            "Therefore, it should have most values in the range of [-1, 1]. "
            "If grid has values outside the range of [-1, 1], the corresponding outputs will be handled as defined by padding_mode.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "Y",
            "4-D tensor of shape (N, C, H_out, W_out) of sampled values. "
            "For integer input types, intermediate values are computed as floating point and cast to integer at the end.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            OpSchema::all_tensor_types(),
            "Constrain input `X` and output `Y` types to all tensor types.")
        .TypeConstraint("T2", {types::Float16, types::Float, types::Double}, "Constrain grid types to float tensors.")
        .SetDoc(kDoc_GridSample_ver16)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { gridSampleShapeInference(ctx); }));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    24,
    OpSchema()
        .SetDoc(kDoc_Reshape_ver24)
        .Attr(
            "allowzero",
            "(Optional) By default, when any value in the 'shape' input is equal to zero "
            "the corresponding dimension value is copied from the input tensor dynamically. "
            "allowzero=1 indicates that if any value in the 'shape' input is set to zero, "
            "the zero value is honored, similar to NumPy.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "shape",
            "Specified shape for output.",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "reshaped", "Reshaped data.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir12(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          bool found;
          TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
          if (!found) {
            return;
          }

          int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

          // Iterate through targetShape, adding dimensions in the outputShape
          // TensorProto. If the targetShape dimension is -1, we do not set the
          // dimension value in this iteration, but we record the Dimension. If
          // targetShape dimension is 0, we attempt to propagate the dimension
          // value/param. If the value cannot be inferred, we set the flag in
          // the unresolveZeros vector. If targetShape dimension is positive, we
          // set the dimension value in the outputShape. We track the product of
          // the dimensions we are setting outputShape in the outputProduct
          // variable. The outputProduct will potentially be used for inferring
          // a dimension marked -1.
          auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          TensorShapeProto::Dimension* negativeOneDim = nullptr;
          const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
          std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
          int64_t outputProduct = 1;
          bool outputProductValid = true;
          for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
            // Add a new dimension to outputShape
            auto new_dim = outputShape->add_dim();
            if (targetShapeProto.dim(i).has_dim_param()) {
              // There is a tricky edge case here. It is possible that the value of
              // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
              // symbol can be erroneous. This should be a very rare scenario and in such a
              // case an option is to turn off data propagation during shape inference.
              new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
              outputProductValid = false;
            } else {
              if (!targetShapeProto.dim(i).has_dim_value()) {
                outputProductValid = false;
                // treat this dim as unknown dim
                continue;
              }

              const auto dim_value = targetShapeProto.dim(i).dim_value();

              if (dim_value == -1) {
                // Check if multiple -1's. If not, set negativeOneDim, marking
                // this dimension to potentially be filled in later.
                if (negativeOneDim) {
                  fail_shape_inference("Target shape may not have multiple -1 dimensions.");
                }
                negativeOneDim = new_dim;
              } else if (dim_value == 0) {
                // Check if data input has a shape and if the index i is within
                // its bounds. If these conditions are satisfied, any dimension
                // value/param should be propagated. If dimension value cannot be
                // inferred, set the corresponding  unresolvedZeros flag to true.
                // If allowzero is set however, do not propagate values, since output
                // dimension is explicitly zero.
                if (allowzero == 0) {
                  unresolvedZeros[i] = true;
                  if (dataInputTensorType.has_shape()) {
                    if (i >= dataInputTensorType.shape().dim_size()) {
                      fail_shape_inference("Invalid position of 0.");
                    }
                    if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                      const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
                      new_dim->set_dim_value(input_dim_value);
                      checked_mul_into(outputProduct, input_dim_value);
                      unresolvedZeros[i] = false;
                    } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
                      new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
                    }
                  }
                } else {
                  new_dim->set_dim_value(dim_value);
                  checked_mul_into(outputProduct, dim_value);
                }
              } else if (dim_value > 0) {
                // Set the dimension value to dim_value
                new_dim->set_dim_value(dim_value);
                checked_mul_into(outputProduct, dim_value);
              } else {
                // Check if value is less than -1; fail if so
                fail_shape_inference("Invalid dimension value: ", dim_value);
              }
            }
          }
          // If negativeOneDim has been set, we attempt to infer its value. This
          // can be done if all dimension values for the data input tensor shape
          // are known other than the ones corresponding to unresolvedZeros
          // flags.
          if (negativeOneDim && outputProductValid) {
            // First, attempt to compute product of data input shape dimensions
            // that are not marked by unresolvedZeros. If not possible, set the
            // inputProductValid flag to false.
            if (!outputProduct) {
              fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
            }
            int64_t inputProduct = 1;
            bool inputProductValid = true;
            if (!dataInputTensorType.has_shape()) {
              inputProductValid = false;
            } else {
              for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
                if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                  checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
                } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
                  inputProductValid = false;
                  break;
                }
              }
            }
            if (inputProductValid) {
              if (inputProduct % outputProduct != 0) {
                fail_shape_inference("Dimension could not be inferred: incompatible shapes");
              }
              negativeOneDim->set_dim_value(inputProduct / outputProduct);
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    23,
    OpSchema()
        .SetDoc(kDoc_Reshape_ver24)
        .Attr(
            "allowzero",
            "(Optional) By default, when any value in the 'shape' input is equal to zero "
            "the corresponding dimension value is copied from the input tensor dynamically. "
            "allowzero=1 indicates that if any value in the 'shape' input is set to zero, "
            "the zero value is honored, similar to NumPy.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "shape",
            "Specified shape for output.",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "reshaped", "Reshaped data.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir11(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          bool found;
          TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
          if (!found) {
            return;
          }

          int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

          // Iterate through targetShape, adding dimensions in the outputShape
          // TensorProto. If the targetShape dimension is -1, we do not set the
          // dimension value in this iteration, but we record the Dimension. If
          // targetShape dimension is 0, we attempt to propagate the dimension
          // value/param. If the value cannot be inferred, we set the flag in
          // the unresolveZeros vector. If targetShape dimension is positive, we
          // set the dimension value in the outputShape. We track the product of
          // the dimensions we are setting outputShape in the outputProduct
          // variable. The outputProduct will potentially be used for inferring
          // a dimension marked -1.
          auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          TensorShapeProto::Dimension* negativeOneDim = nullptr;
          const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
          std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
          int64_t outputProduct = 1;
          bool outputProductValid = true;
          for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
            // Add a new dimension to outputShape
            auto new_dim = outputShape->add_dim();
            if (targetShapeProto.dim(i).has_dim_param()) {
              // There is a tricky edge case here. It is possible that the value of
              // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
              // symbol can be erroneous. This should be a very rare scenario and in such a
              // case an option is to turn off data propagation during shape inference.
              new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
              outputProductValid = false;
            } else {
              if (!targetShapeProto.dim(i).has_dim_value()) {
                outputProductValid = false;
                // treat this dim as unknown dim
                continue;
              }

              const auto dim_value = targetShapeProto.dim(i).dim_value();

              if (dim_value == -1) {
                // Check if multiple -1's. If not, set negativeOneDim, marking
                // this dimension to potentially be filled in later.
                if (negativeOneDim) {
                  fail_shape_inference("Target shape may not have multiple -1 dimensions.");
                }
                negativeOneDim = new_dim;
              } else if (dim_value == 0) {
                // Check if data input has a shape and if the index i is within
                // its bounds. If these conditions are satisfied, any dimension
                // value/param should be propagated. If dimension value cannot be
                // inferred, set the corresponding  unresolvedZeros flag to true.
                // If allowzero is set however, do not propagate values, since output
                // dimension is explicitly zero.
                if (allowzero == 0) {
                  unresolvedZeros[i] = true;
                  if (dataInputTensorType.has_shape()) {
                    if (i >= dataInputTensorType.shape().dim_size()) {
                      fail_shape_inference("Invalid position of 0.");
                    }
                    if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                      const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
                      new_dim->set_dim_value(input_dim_value);
                      checked_mul_into(outputProduct, input_dim_value);
                      unresolvedZeros[i] = false;
                    } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
                      new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
                    }
                  }
                } else {
                  new_dim->set_dim_value(dim_value);
                  checked_mul_into(outputProduct, dim_value);
                }
              } else if (dim_value > 0) {
                // Set the dimension value to dim_value
                new_dim->set_dim_value(dim_value);
                checked_mul_into(outputProduct, dim_value);
              } else {
                // Check if value is less than -1; fail if so
                fail_shape_inference("Invalid dimension value: ", dim_value);
              }
            }
          }
          // If negativeOneDim has been set, we attempt to infer its value. This
          // can be done if all dimension values for the data input tensor shape
          // are known other than the ones corresponding to unresolvedZeros
          // flags.
          if (negativeOneDim && outputProductValid) {
            // First, attempt to compute product of data input shape dimensions
            // that are not marked by unresolvedZeros. If not possible, set the
            // inputProductValid flag to false.
            if (!outputProduct) {
              fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
            }
            int64_t inputProduct = 1;
            bool inputProductValid = true;
            if (!dataInputTensorType.has_shape()) {
              inputProductValid = false;
            } else {
              for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
                if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                  checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
                } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
                  inputProductValid = false;
                  break;
                }
              }
            }
            if (inputProductValid) {
              if (inputProduct % outputProduct != 0) {
                fail_shape_inference("Dimension could not be inferred: incompatible shapes");
              }
              negativeOneDim->set_dim_value(inputProduct / outputProduct);
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    21,
    OpSchema()
        .SetDoc(kDoc_Reshape_ver24)
        .Attr(
            "allowzero",
            "(Optional) By default, when any value in the 'shape' input is equal to zero "
            "the corresponding dimension value is copied from the input tensor dynamically. "
            "allowzero=1 indicates that if any value in the 'shape' input is set to zero, "
            "the zero value is honored, similar to NumPy.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "shape",
            "Specified shape for output.",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "reshaped", "Reshaped data.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir10(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          bool found;
          TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
          if (!found) {
            return;
          }

          int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

          // Iterate through targetShape, adding dimensions in the outputShape
          // TensorProto. If the targetShape dimension is -1, we do not set the
          // dimension value in this iteration, but we record the Dimension. If
          // targetShape dimension is 0, we attempt to propagate the dimension
          // value/param. If the value cannot be inferred, we set the flag in
          // the unresolveZeros vector. If targetShape dimension is positive, we
          // set the dimension value in the outputShape. We track the product of
          // the dimensions we are setting outputShape in the outputProduct
          // variable. The outputProduct will potentially be used for inferring
          // a dimension marked -1.
          auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          TensorShapeProto::Dimension* negativeOneDim = nullptr;
          const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
          std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
          int64_t outputProduct = 1;
          bool outputProductValid = true;
          for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
            // Add a new dimension to outputShape
            auto new_dim = outputShape->add_dim();
            if (targetShapeProto.dim(i).has_dim_param()) {
              // There is a tricky edge case here. It is possible that the value of
              // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
              // symbol can be erroneous. This should be a very rare scenario and in such a
              // case an option is to turn off data propagation during shape inference.
              new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
              outputProductValid = false;
            } else {
              if (!targetShapeProto.dim(i).has_dim_value()) {
                outputProductValid = false;
                // treat this dim as unknown dim
                continue;
              }

              const auto dim_value = targetShapeProto.dim(i).dim_value();

              if (dim_value == -1) {
                // Check if multiple -1's. If not, set negativeOneDim, marking
                // this dimension to potentially be filled in later.
                if (negativeOneDim) {
                  fail_shape_inference("Target shape may not have multiple -1 dimensions.");
                }
                negativeOneDim = new_dim;
              } else if (dim_value == 0) {
                // Check if data input has a shape and if the index i is within
                // its bounds. If these conditions are satisfied, any dimension
                // value/param should be propagated. If dimension value cannot be
                // inferred, set the corresponding  unresolvedZeros flag to true.
                // If allowzero is set however, do not propagate values, since output
                // dimension is explicitly zero.
                if (allowzero == 0) {
                  unresolvedZeros[i] = true;
                  if (dataInputTensorType.has_shape()) {
                    if (i >= dataInputTensorType.shape().dim_size()) {
                      fail_shape_inference("Invalid position of 0.");
                    }
                    if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                      const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
                      new_dim->set_dim_value(input_dim_value);
                      checked_mul_into(outputProduct, input_dim_value);
                      unresolvedZeros[i] = false;
                    } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
                      new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
                    }
                  }
                } else {
                  new_dim->set_dim_value(dim_value);
                  checked_mul_into(outputProduct, dim_value);
                }
              } else if (dim_value > 0) {
                // Set the dimension value to dim_value
                new_dim->set_dim_value(dim_value);
                checked_mul_into(outputProduct, dim_value);
              } else {
                // Check if value is less than -1; fail if so
                fail_shape_inference("Invalid dimension value: ", dim_value);
              }
            }
          }
          // If negativeOneDim has been set, we attempt to infer its value. This
          // can be done if all dimension values for the data input tensor shape
          // are known other than the ones corresponding to unresolvedZeros
          // flags.
          if (negativeOneDim && outputProductValid) {
            // First, attempt to compute product of data input shape dimensions
            // that are not marked by unresolvedZeros. If not possible, set the
            // inputProductValid flag to false.
            if (!outputProduct) {
              fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
            }
            int64_t inputProduct = 1;
            bool inputProductValid = true;
            if (!dataInputTensorType.has_shape()) {
              inputProductValid = false;
            } else {
              for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
                if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                  checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
                } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
                  inputProductValid = false;
                  break;
                }
              }
            }
            if (inputProductValid) {
              if (inputProduct % outputProduct != 0) {
                fail_shape_inference("Dimension could not be inferred: incompatible shapes");
              }
              negativeOneDim->set_dim_value(inputProduct / outputProduct);
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    19,
    OpSchema()
        .SetDoc(kDoc_Reshape_ver24)
        .Attr(
            "allowzero",
            "(Optional) By default, when any value in the 'shape' input is equal to zero "
            "the corresponding dimension value is copied from the input tensor dynamically. "
            "allowzero=1 indicates that if any value in the 'shape' input is set to zero, "
            "the zero value is honored, similar to NumPy.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "shape",
            "Specified shape for output.",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "reshaped", "Reshaped data.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir9(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          bool found;
          TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
          if (!found) {
            return;
          }

          int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

          // Iterate through targetShape, adding dimensions in the outputShape
          // TensorProto. If the targetShape dimension is -1, we do not set the
          // dimension value in this iteration, but we record the Dimension. If
          // targetShape dimension is 0, we attempt to propagate the dimension
          // value/param. If the value cannot be inferred, we set the flag in
          // the unresolveZeros vector. If targetShape dimension is positive, we
          // set the dimension value in the outputShape. We track the product of
          // the dimensions we are setting outputShape in the outputProduct
          // variable. The outputProduct will potentially be used for inferring
          // a dimension marked -1.
          auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          TensorShapeProto::Dimension* negativeOneDim = nullptr;
          const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
          std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
          int64_t outputProduct = 1;
          bool outputProductValid = true;
          for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
            // Add a new dimension to outputShape
            auto new_dim = outputShape->add_dim();
            if (targetShapeProto.dim(i).has_dim_param()) {
              // There is a tricky edge case here. It is possible that the value of
              // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
              // symbol can be erroneous. This should be a very rare scenario and in such a
              // case an option is to turn off data propagation during shape inference.
              new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
              outputProductValid = false;
            } else {
              if (!targetShapeProto.dim(i).has_dim_value()) {
                outputProductValid = false;
                // treat this dim as unknown dim
                continue;
              }

              const auto dim_value = targetShapeProto.dim(i).dim_value();

              if (dim_value == -1) {
                // Check if multiple -1's. If not, set negativeOneDim, marking
                // this dimension to potentially be filled in later.
                if (negativeOneDim) {
                  fail_shape_inference("Target shape may not have multiple -1 dimensions.");
                }
                negativeOneDim = new_dim;
              } else if (dim_value == 0) {
                // Check if data input has a shape and if the index i is within
                // its bounds. If these conditions are satisfied, any dimension
                // value/param should be propagated. If dimension value cannot be
                // inferred, set the corresponding  unresolvedZeros flag to true.
                // If allowzero is set however, do not propagate values, since output
                // dimension is explicitly zero.
                if (allowzero == 0) {
                  unresolvedZeros[i] = true;
                  if (dataInputTensorType.has_shape()) {
                    if (i >= dataInputTensorType.shape().dim_size()) {
                      fail_shape_inference("Invalid position of 0.");
                    }
                    if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                      const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
                      new_dim->set_dim_value(input_dim_value);
                      checked_mul_into(outputProduct, input_dim_value);
                      unresolvedZeros[i] = false;
                    } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
                      new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
                    }
                  }
                } else {
                  new_dim->set_dim_value(dim_value);
                  checked_mul_into(outputProduct, dim_value);
                }
              } else if (dim_value > 0) {
                // Set the dimension value to dim_value
                new_dim->set_dim_value(dim_value);
                checked_mul_into(outputProduct, dim_value);
              } else {
                // Check if value is less than -1; fail if so
                fail_shape_inference("Invalid dimension value: ", dim_value);
              }
            }
          }
          // If negativeOneDim has been set, we attempt to infer its value. This
          // can be done if all dimension values for the data input tensor shape
          // are known other than the ones corresponding to unresolvedZeros
          // flags.
          if (negativeOneDim && outputProductValid) {
            // First, attempt to compute product of data input shape dimensions
            // that are not marked by unresolvedZeros. If not possible, set the
            // inputProductValid flag to false.
            if (!outputProduct) {
              fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
            }
            int64_t inputProduct = 1;
            bool inputProductValid = true;
            if (!dataInputTensorType.has_shape()) {
              inputProductValid = false;
            } else {
              for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
                if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                  checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
                } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
                  inputProductValid = false;
                  break;
                }
              }
            }
            if (inputProductValid) {
              if (inputProduct % outputProduct != 0) {
                fail_shape_inference("Dimension could not be inferred: incompatible shapes");
              }
              negativeOneDim->set_dim_value(inputProduct / outputProduct);
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    13,
    OpSchema()
        .SetDoc(kDoc_Reshape_ver13)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "shape",
            "Specified shape for output.",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "reshaped", "Reshaped data.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          // Shape Inference if 2nd input data (the target shape) is available
          const TensorProto* targetShapeInitializer = ctx.getInputData(1);
          if (!targetShapeInitializer) {
            return;
          }
          // Make targetShape (0 -> same as originalShape, -1 -> inferred).
          // The targetShape vector represents the specified shape for output.
          std::vector<int64_t> targetShape = ParseData<int64_t>(targetShapeInitializer);
          // Iterate through targetShape, adding dimensions in the outputShape
          // TensorProto. If the targetShape dimension is -1, we do not set the
          // dimension value in this iteration, but we record the Dimension. If
          // targetShape dimension is 0, we attempt to propagate the dimension
          // value/param. If the value cannot be inferred, we set the flag in
          // the unresolveZeros vector. If targetShape dimension is positive, we
          // set the dimension value in the outputShape. We track the product of
          // the dimensions we are setting outputShape in the outputProduct
          // variable. The outputProduct will potentially be used for inferring
          // a dimension marked -1.
          auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          TensorShapeProto::Dimension* negativeOneDim = nullptr;
          const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
          std::vector<bool> unresolvedZeros(targetShape.size(), false);
          int64_t outputProduct = 1;
          for (size_t i = 0; i < targetShape.size(); ++i) {
            // Add a new dimension to outputShape
            auto new_dim = outputShape->add_dim();
            if (targetShape[i] == -1) {
              // Check if multiple -1's. If not, set negativeOneDim, marking
              // this dimension to potentially be filled in later.
              if (negativeOneDim) {
                fail_shape_inference("Target shape may not have multiple -1 dimensions");
              }
              negativeOneDim = new_dim;
            } else if (targetShape[i] == 0) {
              // Check if data input has a shape and if the index i is within
              // its bounds. If these conditions are satisfied, any dimension
              // value/param should be propagated. If dimension value cannot be
              // inferred, set the corresponding  unresolvedZeros flag to true.
              unresolvedZeros[i] = true;
              if (dataInputTensorType.has_shape()) {
                if (i >= static_cast<size_t>(dataInputTensorType.shape().dim_size())) {
                  fail_shape_inference("Invalid position of 0");
                }
                if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                  const auto& dim_value = dataInputTensorType.shape().dim(i).dim_value();
                  new_dim->set_dim_value(dim_value);
                  checked_mul_into(outputProduct, dim_value);
                  unresolvedZeros[i] = false;
                } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
                  const auto& dim_param = dataInputTensorType.shape().dim(i).dim_param();
                  new_dim->set_dim_param(dim_param);
                }
              }
            } else if (targetShape[i] > 0) {
              // Set the dimension value to targetShape[i]
              new_dim->set_dim_value(targetShape[i]);
              checked_mul_into(outputProduct, targetShape[i]);
            } else {
              // Check if value is less than -1; fail if so
              fail_shape_inference("Invalid dimension value: ", targetShape[i]);
            }
          }

          // If negativeOneDim has been set, we attempt to infer its value. This
          // can be done if all dimension values for the data input tensor shape
          // are known other than the ones corresponding to unresolvedZeros
          // flags.
          if (negativeOneDim) {
            // First, attempt to compute product of data input shape dimensions
            // that are not marked by unresolvedZeros. If not possible, set the
            // inputProductValid flag to false.
            if (!outputProduct) {
              fail_shape_inference("Invalid Target shape product of 0");
            }
            int64_t inputProduct = 1;
            bool inputProductValid = true;
            if (!dataInputTensorType.has_shape()) {
              inputProductValid = false;
            } else {
              for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
                if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                  checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
                } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
                  inputProductValid = false;
                  break;
                }
              }
            }
            if (inputProductValid) {
              if (inputProduct % outputProduct != 0) {
                fail_shape_inference("Dimension could not be inferred: incompatible shapes");
              }
              negativeOneDim->set_dim_value(inputProduct / outputProduct);
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    5,
    OpSchema()
        .SetDoc(kDoc_Reshape_ver13)
        .Input(0, "data", "An input tensor.", "T")
        .Input(1, "shape", "Specified shape for output.", "tensor(int64)")
        .Output(0, "reshaped", "Reshaped data.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          // Shape Inference if 2nd input data (the target shape) is available
          const TensorProto* targetShapeInitializer = ctx.getInputData(1);
          if (!targetShapeInitializer) {
            return;
          }
          // Make targetShape (0 -> same as originalShape, -1 -> inferred).
          // The targetShape vector represents the specified shape for output.
          std::vector<int64_t> targetShape = ParseData<int64_t>(targetShapeInitializer);

          // Iterate through targetShape, adding dimensions in the outputShape
          // TensorProto. If the targetShape dimension is -1, we do not set the
          // dimension value in this iteration, but we record the Dimension. If
          // targetShape dimension is 0, we attempt to propagate the dimension
          // value/param. If the value cannot be inferred, we set the flag in
          // the unresolveZeros vector. If targetShape dimension is positive, we
          // set the dimension value in the outputShape. We track the product of
          // the dimensions we are setting outputShape in the outputProduct
          // variable. The outputProduct will potentially be used for inferring
          // a dimension marked -1.
          auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          TensorShapeProto::Dimension* negativeOneDim = nullptr;
          const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
          std::vector<bool> unresolvedZeros(targetShape.size(), false);
          int64_t outputProduct = 1;
          for (int i = 0; i < static_cast<int>(targetShape.size()); ++i) {
            // Add a new dimension to outputShape
            auto new_dim = outputShape->add_dim();
            if (targetShape[i] == -1) {
              // Check if multiple -1's. If not, set negativeOneDim, marking
              // this dimension to potentially be filled in later.
              if (negativeOneDim) {
                fail_shape_inference("Target shape may not have multiple -1 dimensions");
              }
              negativeOneDim = new_dim;
            } else if (targetShape[i] == 0) {
              // Check if data input has a shape and if the index i is within
              // its bounds. If these conditions are satisfied, any dimension
              // value/param should be propagated. If dimension value cannot be
              // inferred, set the corresponding  unresolvedZeros flag to true.
              unresolvedZeros[i] = true;
              if (dataInputTensorType.has_shape()) {
                if (i >= dataInputTensorType.shape().dim_size()) {
                  fail_shape_inference("Invalid position of 0");
                }
                if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                  const auto& dim_value = dataInputTensorType.shape().dim(i).dim_value();
                  new_dim->set_dim_value(dim_value);
                  checked_mul_into(outputProduct, dim_value);
                  unresolvedZeros[i] = false;
                } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
                  const auto& dim_param = dataInputTensorType.shape().dim(i).dim_param();
                  new_dim->set_dim_param(dim_param);
                }
              }
            } else if (targetShape[i] > 0) {
              // Set the dimension value to targetShape[i]
              new_dim->set_dim_value(targetShape[i]);
              checked_mul_into(outputProduct, targetShape[i]);
            } else {
              // Check if value is less than -1; fail if so
              fail_shape_inference("Invalid dimension value: ", targetShape[i]);
            }
          }

          // If negativeOneDim has been set, we attempt to infer its value. This
          // can be done if all dimension values for the data input tensor shape
          // are known other than the ones corresponding to unresolvedZeros
          // flags.
          if (negativeOneDim) {
            // First, attempt to compute product of data input shape dimensions
            // that are not marked by unresolvedZeros. If not possible, set the
            // inputProductValid flag to false.
            if (!outputProduct) {
              fail_shape_inference("Invalid Target shape product of 0");
            }
            int64_t inputProduct = 1;
            bool inputProductValid = true;
            if (!dataInputTensorType.has_shape()) {
              inputProductValid = false;
            } else {
              for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
                if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                  checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
                } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
                  inputProductValid = false;
                  break;
                }
              }
            }
            if (inputProductValid) {
              if (inputProduct % outputProduct != 0) {
                fail_shape_inference("Dimension could not be inferred: incompatible shapes");
              }
              negativeOneDim->set_dim_value(inputProduct / outputProduct);
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    24,
    OpSchema()
        .SetDoc(kDoc_Shape_ver24)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "shape", "Shape of the input tensor", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Attr(
            "start",
            "(Optional) Starting axis for slicing the shape. Default value is 0."
            "Negative value means counting dimensions from the back.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "end",
            "(Optional) Ending axis for slicing the shape. "
            "Negative value means counting dimensions from the back. "
            "If omitted, sizes of all axes upto (including) the last one will be included.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir12(), "Input tensor can be of arbitrary type.")
        .TypeConstraint("T1", {types::Int64}, "Constrain output to int64 tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          auto output_length = output_shape->add_dim();

          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
          int64_t start = getAttribute(ctx, "start", 0);
          if (start < 0)
            start += rank;
          start = (start < 0) ? 0 : (start > rank) ? rank : start;
          int64_t end = getAttribute(ctx, "end", rank);
          if (end < 0)
            end += rank;
          end = (end < 0) ? 0 : (end > rank) ? rank : end;
          output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            int64_t rank = static_cast<int64_t>(input_shape.dim_size());
            int64_t start = getAttribute(ctx, "start", 0);
            if (start < 0)
              start += rank;
            start = (start < 0) ? 0 : (start > rank) ? rank : start;
            int64_t end = getAttribute(ctx, "end", rank);
            if (end < 0)
              end += rank;
            end = (end < 0) ? 0 : (end > rank) ? rank : end;
            TensorShapeProto output_shape;
            for (int64_t d = start; d < end; ++d) {
              *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
            }
            ctx.addOutputData(0, std::move(output_shape));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    23,
    OpSchema()
        .SetDoc(kDoc_Shape_ver24)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "shape", "Shape of the input tensor", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Attr(
            "start",
            "(Optional) Starting axis for slicing the shape. Default value is 0."
            "Negative value means counting dimensions from the back.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "end",
            "(Optional) Ending axis for slicing the shape. "
            "Negative value means counting dimensions from the back. "
            "If omitted, sizes of all axes upto (including) the last one will be included.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir11(), "Input tensor can be of arbitrary type.")
        .TypeConstraint("T1", {types::Int64}, "Constrain output to int64 tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          auto output_length = output_shape->add_dim();

          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
          int64_t start = getAttribute(ctx, "start", 0);
          if (start < 0)
            start += rank;
          start = (start < 0) ? 0 : (start > rank) ? rank : start;
          int64_t end = getAttribute(ctx, "end", rank);
          if (end < 0)
            end += rank;
          end = (end < 0) ? 0 : (end > rank) ? rank : end;
          output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            int64_t rank = static_cast<int64_t>(input_shape.dim_size());
            int64_t start = getAttribute(ctx, "start", 0);
            if (start < 0)
              start += rank;
            start = (start < 0) ? 0 : (start > rank) ? rank : start;
            int64_t end = getAttribute(ctx, "end", rank);
            if (end < 0)
              end += rank;
            end = (end < 0) ? 0 : (end > rank) ? rank : end;
            TensorShapeProto output_shape;
            for (int64_t d = start; d < end; ++d) {
              *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
            }
            ctx.addOutputData(0, std::move(output_shape));
          }
        }));

// Data propagation function for Shape op
// Propagates input shape to output shape
static void ShapeOp13DataPropagator(DataPropagationContext& ctx) {
  if (!hasNInputShapes(ctx, 1)) {
    return;
  }
  if (ctx.getInputType(0)->tensor_type().has_shape()) {
    auto input_shape = ctx.getInputType(0)->tensor_type().shape();
    TensorShapeProto tsp;
    tsp.CopyFrom(input_shape);
    ctx.addOutputData(0, std::move(tsp));
  }
}

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    13,
    OpSchema()
        .SetDoc(kDoc_Shape_ver13)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "shape", "Shape of the input tensor", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Input tensor can be of arbitrary type.")
        .TypeConstraint("T1", {types::Int64}, "Constrain output to int64 tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          auto output_length = output_shape->add_dim();

          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          if (ctx.getInputType(0)->tensor_type().has_shape()) {
            output_length->set_dim_value(ctx.getInputType(0)->tensor_type().shape().dim_size());
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) { ShapeOp13DataPropagator(ctx); }));

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    1,
    OpSchema()
        .SetDoc(kDoc_Shape_ver13)
        .Input(0, "data", "An input tensor.", "T")
        .Output(0, "shape", "Shape of the input tensor", "T1")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Input tensor can be of arbitrary type.")
        .TypeConstraint("T1", {types::Int64}, "Constrain output to int64 tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          auto output_length = output_shape->add_dim();

          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          if (ctx.getInputType(0)->tensor_type().has_shape()) {
            output_length->set_dim_value(ctx.getInputType(0)->tensor_type().shape().dim_size());
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) { ShapeOp13DataPropagator(ctx); }));

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    24,
    OpSchema()
        .SetDoc(kDoc_Size_ver24)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(
            0,
            "size",
            "Total number of elements of the input tensor",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir12(), "Input tensor can be of arbitrary type.")
        .TypeConstraint("T1", {types::Int64}, "Constrain output to int64 tensor, which should be a scalar though.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          const auto input_data = ctx.getInputData(0);
          if (input_data != nullptr) {
            TensorShapeProto tsp;
            tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
            ctx.addOutputData(0, std::move(tsp));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    23,
    OpSchema()
        .SetDoc(kDoc_Size_ver24)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(
            0,
            "size",
            "Total number of elements of the input tensor",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir11(), "Input tensor can be of arbitrary type.")
        .TypeConstraint("T1", {types::Int64}, "Constrain output to int64 tensor, which should be a scalar though.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          const auto input_data = ctx.getInputData(0);
          if (input_data != nullptr) {
            TensorShapeProto tsp;
            tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
            ctx.addOutputData(0, std::move(tsp));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    1,
    OpSchema()
        .SetDoc(kDoc_Size_ver24)
        .Input(0, "data", "An input tensor.", "T")
        .Output(0, "size", "Total number of elements of the input tensor", "T1")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Input tensor can be of arbitrary type.")
        .TypeConstraint("T1", {types::Int64}, "Constrain output to int64 tensor, which should be a scalar though.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Concat,
    11,
    OpSchema()
        .Attr(
            "axis",
            "Which axis to concat on. A negative value means counting dimensions from the back. "
            "Accepted range is [-r, r-1] where r = rank(inputs)..",
            AttributeProto::INT)
        .SetDoc(kDoc_Concat_ver13)
        .Input(0, "inputs", "List of tensors for concatenation", "T", OpSchema::Variadic)
        .Output(0, "concat_result", "Concatenated tensor", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain output types to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto numInputs = ctx.getNumInputs();
          if (numInputs < 1 || !hasNInputShapes(ctx, numInputs)) {
            return;
          }

          auto rank = ctx.getInputType(0)->tensor_type().shape().dim_size();

          auto axisAttr = ctx.getAttribute("axis");
          if (!axisAttr) {
            fail_shape_inference("Required attribute axis is missing");
          }
          int axis = static_cast<int>(axisAttr->i());
          if (axis < -rank || axis >= rank) {
            fail_shape_inference("axis must be in [-rank, rank-1].");
          }
          if (axis < 0) {
            axis += rank;
          }

          if (numInputs == 1) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
            return;
          }

          bool all_lengths_known = true;
          int64_t total_length = 0;

          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          for (int64_t i = 0; i < rank; ++i) {
            output_shape->add_dim();
          }

          for (size_t i = 0; i < numInputs; i++) {
            const auto& shape = ctx.getInputType(i)->tensor_type().shape();
            if (shape.dim_size() != rank) {
              fail_shape_inference(
                  "All inputs to Concat must have same rank. Input ", i, " has rank ", shape.dim_size(), " != ", rank);
            }
            for (int j = 0; j < rank; j++) {
              if (j == axis) {
                if (shape.dim(j).has_dim_value()) {
                  const int64_t dim_val = shape.dim(j).dim_value();
                  if (dim_val < 0) {
                    fail_shape_inference("Negative dimension value on Concat axis");
                  }
                  if (checked_add_overflow(total_length, dim_val, &total_length)) {
                    fail_shape_inference("Integer overflow computing Concat output length");
                  }
                } else {
                  all_lengths_known = false;
                }
              } else {
                auto& output_dim = *output_shape->mutable_dim(j);
                const auto& input_dim = shape.dim(j);
                mergeInDimensionInfo(input_dim, output_dim, j);
              }
            }
          }

          if (all_lengths_known) {
            output_shape->mutable_dim(axis)->set_dim_value(total_length);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    11,
    OpSchema()
        .Input(0, "input", "The tensor to split", "T")
        .Output(0, "outputs", "One or more outputs forming list of tensors after splitting", "T", OpSchema::Variadic)
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .Attr(
            "axis",
            "Which axis to split on. "
            "A negative value means counting dimensions from the back. Accepted range is [-rank, rank-1] "
            "where r = rank(input).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr("split", "length of each output. Values should be >= 0.", AttributeProto::INTS, OPTIONAL_VALUE)
        .SetDoc(kDoc_Split_ver11)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getNumOutputs() == 0) {
            fail_shape_inference("Split must have at least one output");
          }
          for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); ++i) {
            propagateElemTypeFromInputToOutput(ctx, 0, i);
          }
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          const auto& shape = ctx.getInputType(0)->tensor_type().shape();
          int rank = shape.dim_size();
          int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
          if (axis < -rank || axis >= rank) {
            fail_type_inference("Invalid value of attribute 'axis'. Rank=", rank, " Value=", axis);
          }
          if (axis < 0) {
            axis += rank;
          }
          const auto& split_dim = shape.dim(axis);
          if (!split_dim.has_dim_value()) {
            for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
              *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
              ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->Clear();
            }
            return;
          }
          int split_dim_value = static_cast<int>(split_dim.dim_value());

          std::vector<int64_t> split;
          if (getRepeatedAttribute(ctx, "split", split)) {
            if (split.size() != ctx.getNumOutputs()) {
              fail_shape_inference(
                  "Mismatch between number of splits (", split.size(), ") and outputs (", ctx.getNumOutputs(), ")");
            }
            int64_t total_dim = 0;
            for (int64_t d : split) {
              total_dim += d;
            }
            if (total_dim != split_dim_value) {
              fail_shape_inference(
                  "Mismatch between the sum of 'split' (",
                  total_dim,
                  ") and the split dimension of the input (",
                  split_dim_value,
                  ")");
            }
          } else {
            int num_outputs = static_cast<int>(ctx.getNumOutputs());
            if (split_dim_value % num_outputs != 0) {
              fail_shape_inference("The input is not evenly splittable");
            }
            int chunk_size = split_dim_value / num_outputs;
            for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); i++) {
              split.push_back(chunk_size);
            }
          }
          for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
            *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
            ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(split[i]);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    13,
    OpSchema()
        .Input(0, "input", "The tensor to split", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "split",
            "Optional length of each output. Values should be >= 0."
            "Sum of the values must be equal to the dim value at 'axis' specified.",
            "tensor(int64)",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "outputs",
            "One or more outputs forming list of tensors after splitting",
            "T",
            OpSchema::Variadic,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
        .Attr(
            "axis",
            "Which axis to split on. "
            "A negative value means counting dimensions from the back. Accepted range is [-rank, rank-1] "
            "where r = rank(input).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .SetDoc(kDoc_Split_ver13)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getNumOutputs() == 0) {
            fail_shape_inference("Split must have at least one output");
          }
          for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); ++i) {
            propagateElemTypeFromInputToOutput(ctx, 0, i);
          }
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          const auto& shape = ctx.getInputType(0)->tensor_type().shape();
          int rank = shape.dim_size();
          int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
          if (axis < -rank || axis >= rank) {
            fail_type_inference("Invalid value of attribute 'axis'. Rank=", rank, " Value=", axis);
          }
          if (axis < 0) {
            axis += rank;
          }
          const auto& split_dim = shape.dim(axis);
          if (!split_dim.has_dim_value()) {
            for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
              *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
              ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->Clear();
            }
            return;
          }
          int split_dim_value = static_cast<int>(split_dim.dim_value());

          std::vector<int64_t> split;
          size_t num_inputs = ctx.getNumInputs();
          if ((num_inputs == 2) && ctx.getInputType(1)) { //'split' is input
            auto split_proto = ctx.getInputData(1);
            if (split_proto == nullptr) {
              // skip if split is not an initializer
              return;
            }
            split = ParseData<int64_t>(split_proto);
            if (split.size() != ctx.getNumOutputs()) {
              fail_shape_inference(
                  "Mismatch between number of splits (", split.size(), ") and outputs (", ctx.getNumOutputs(), ")");
            }
            int64_t total_dim = 0;
            for (int64_t d : split) {
              total_dim += d;
            }
            if (total_dim != split_dim_value) {
              fail_shape_inference(
                  "Mismatch between the sum of 'split' (",
                  total_dim,
                  ") and the split dimension of the input (",
                  split_dim_value,
                  ")");
            }
          } else { // no value available for 'split'
            int num_outputs = static_cast<int>(ctx.getNumOutputs());
            if (split_dim_value % num_outputs != 0) {
              fail_shape_inference("The input is not evenly splittable");
            }
            int chunk_size = split_dim_value / num_outputs;
            split.reserve(ctx.getNumOutputs());
            for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); i++) {
              split.push_back(chunk_size);
            }
          }
          for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
            *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
            ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(split[i]);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Slice,
    11,
    OpSchema()
        .SetDoc(kDoc_Slice_ver11)
        .Input(0, "data", "Tensor of data to extract slices from.", "T")
        .Input(1, "starts", "1-D tensor of starting indices of corresponding axis in `axes`", "Tind")
        .Input(2, "ends", "1-D tensor of ending indices (exclusive) of corresponding axis in `axes`", "Tind")
        .Input(
            3,
            "axes",
            "1-D tensor of axes that `starts` and `ends` apply to. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(data).",
            "Tind",
            OpSchema::Optional)
        .Input(
            4,
            "steps",
            "1-D tensor of slice step of corresponding axis in `axes`. "
            "Negative value means slicing backward. 'steps' cannot be 0. "
            "Defaults to 1.",
            "Tind",
            OpSchema::Optional)
        .Output(0, "output", "Sliced data tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeConstraint("Tind", {types::Int32, types::Int64}, "Constrain indices to integer types")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          size_t num_inputs = ctx.getNumInputs();
          if (num_inputs != 3 && num_inputs != 4 && num_inputs != 5) {
            fail_type_inference("Slice op must have either three, four or five inputs.");
          }
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          // Shape Inference if
          //     1. 2nd and 3rd input data (starts, ends) are available.
          // and 2. 4th and 5th optional input (axes, steps) are either not set,
          // or set and is initializer.
          const TensorProto* startsInitializer = ctx.getInputData(1);
          const TensorProto* endsInitializer = ctx.getInputData(2);
          const TensorProto* axesInitializer = hasInputShape(ctx, 3) ? ctx.getInputData(3) : nullptr;
          const TensorProto* stepsInitializer = hasInputShape(ctx, 4) ? ctx.getInputData(4) : nullptr;

          if (!startsInitializer || !endsInitializer || (hasInputShape(ctx, 3) && !ctx.getInputData(3)) ||
              (hasInputShape(ctx, 4) && !ctx.getInputData(4))) {
            const auto input_rank = ctx.getInputType(0)->tensor_type().shape().dim_size();
            // we can infer the output rank - it never changes
            for (int i = 0; i < input_rank; ++i) {
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
            }
            return;
          }

          // don't know data_type- can't proceed
          if (!startsInitializer->has_data_type())
            return;

          auto get_initializer_data = [](const TensorProto* initializer) -> std::vector<int64_t> {
            std::vector<int64_t> vec;
            if (initializer->data_type() == TensorProto::INT64) {
              const auto data = ParseData<int64_t>(initializer);
              vec.insert(vec.end(), data.begin(), data.end());
            } else if (initializer->data_type() == TensorProto::INT32) {
              const auto data = ParseData<int32_t>(initializer);
              vec.insert(vec.end(), data.begin(), data.end());
            } else {
              // unaccepted data type
              fail_shape_inference("Only supports `int32_t` or `int64_t` inputs for starts/ends/axes/steps");
            }
            return vec;
          };

          std::vector<int64_t> starts = get_initializer_data(startsInitializer);
          std::vector<int64_t> ends = get_initializer_data(endsInitializer);

          if (starts.size() != ends.size()) {
            fail_shape_inference("Incorrect or missing input value for starts and ends");
          }

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_rank = input_shape.dim_size();
          std::vector<int64_t> axes(starts.size());
          if (!axesInitializer) {
            std::iota(axes.begin(), axes.end(), 0);
          } else {
            axes = get_initializer_data(axesInitializer);
            if (axes.size() != starts.size()) {
              fail_shape_inference("Input axes has incorrect length");
            }
          }

          std::vector<int64_t> steps;
          if (!stepsInitializer) {
            steps = std::vector<int64_t>(starts.size(), 1);
          } else {
            steps = get_initializer_data(stepsInitializer);
            if (steps.size() != axes.size()) {
              fail_shape_inference("Input steps has incorrect length");
            }
          }

          for (int i = 0; i < input_rank; ++i) {
            // first update rank of output dim
            auto output_dim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
            const auto& input_dim = input_shape.dim(i);
            if (input_dim.has_dim_value()) {
              output_dim->set_dim_value(input_dim.dim_value());
            } else if (input_dim.has_dim_param()) {
              output_dim->set_dim_param(input_dim.dim_param());
            }
          }

          std::unordered_set<int64_t> unique_axes;
          size_t axes_size = axes.size();
          for (size_t axis_index = 0; axis_index < axes_size; ++axis_index) {
            auto axis = axes[axis_index] < 0 ? axes[axis_index] + static_cast<int64_t>(input_rank) : axes[axis_index];

            if (axis >= static_cast<int64_t>(input_rank) || axis < 0) {
              fail_shape_inference("Input axes has invalid data");
            }

            if (unique_axes.find(axis) != unique_axes.end()) {
              fail_shape_inference("'axes' has duplicates");
            }

            unique_axes.insert(axis);

            auto input_dim = ctx.getInputType(0)->tensor_type().shape().dim(static_cast<int>(axis));

            // input dim value is missing - cannot perform shape inference for
            // this axis
            if (!input_dim.has_dim_value()) {
              // Clear any previously propagated dim_param and leave this
              // dimension "empty", before moving on to the next dimension
              ctx.getOutputType(0)
                  ->mutable_tensor_type()
                  ->mutable_shape()
                  ->mutable_dim(static_cast<int>(axis))
                  ->clear_dim_param();
              continue;
            }

            const auto input_dim_value = input_dim.dim_value();

            // Empty dimension: clamp bounds are invalid when dimension size is 0,
            // so short-circuit to produce a zero-length output.
            if (input_dim_value == 0) {
              ctx.getOutputType(0)
                  ->mutable_tensor_type()
                  ->mutable_shape()
                  ->mutable_dim(static_cast<int>(axis))
                  ->set_dim_value(0);
              continue;
            }

            // process step
            auto step = steps[axis_index];
            if (step == 0) {
              fail_shape_inference("'step' cannot be 0");
            }

            // process start
            auto start = starts[axis_index];
            if (start < 0)
              start += input_dim_value;
            if (step < 0)
              start = std::clamp(start, static_cast<int64_t>(0), input_dim_value - 1);
            else
              start = std::clamp(start, static_cast<int64_t>(0), input_dim_value);

            // process end
            auto end = ends[axis_index];
            if (end < static_cast<int64_t>(0))
              end += input_dim_value;
            if (step < static_cast<int64_t>(0))
              end = std::clamp(end, static_cast<int64_t>(-1), input_dim_value);
            else
              end = std::clamp(end, static_cast<int64_t>(0), input_dim_value);

            // find output dim value for this axis
            auto temp = static_cast<int64_t>(ceil(1.0 * (end - start) / step));
            if (temp < 0)
              temp = 0;

            // assign output value
            ctx.getOutputType(0)
                ->mutable_tensor_type()
                ->mutable_shape()
                ->mutable_dim(static_cast<int>(axis))
                ->set_dim_value(temp);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    24,
    OpSchema()
        .SetDoc(kDoc_Transpose_ver13)
        .Attr(
            "perm",
            "A list of integers. By default, reverse the dimensions, "
            "otherwise permute the axes according to the values given. "
            "Its length must be equal to the rank of the input.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "transposed", "Transposed output.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir12(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          auto input_type = ctx.getInputType(0);
          const TensorShapeProto& shape = input_type->tensor_type().shape();
          std::vector<int64_t> perm;
          bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
          if (!has_perm_attr) {
            perm.reserve(shape.dim_size());
            for (int i = shape.dim_size() - 1; i >= 0; --i)
              perm.push_back(i);
          } else if (!perm.empty()) {
            // check if every index is valid
            std::vector<bool> seen(shape.dim_size(), false);
            for (int64_t fromDimIndex : perm) {
              if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
                std::ostringstream oss;
                oss << "Invalid attribute perm {" << perm[0];
                for (size_t i = 1; i != perm.size(); ++i) {
                  oss << ", " << perm[i];
                }
                oss << "}, input shape = {";
                if (shape.dim_size() > 0) {
                  oss << shape.dim(0).dim_value();
                  for (int i = 1; i != shape.dim_size(); ++i) {
                    oss << ", " << shape.dim(i).dim_value();
                  }
                  oss << "}";
                }
                fail_type_inference(oss.str());
              } else {
                // check if any perm is repeated
                if (seen[fromDimIndex]) {
                  fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
                }
                seen[fromDimIndex] = true;
              }
            }
          }

          getOutputShape(ctx, 0);

          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          for (int64_t i : perm) {
            appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    23,
    OpSchema()
        .SetDoc(kDoc_Transpose_ver13)
        .Attr(
            "perm",
            "A list of integers. By default, reverse the dimensions, "
            "otherwise permute the axes according to the values given. "
            "Its length must be equal to the rank of the input.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "transposed", "Transposed output.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir11(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          auto input_type = ctx.getInputType(0);
          const TensorShapeProto& shape = input_type->tensor_type().shape();
          std::vector<int64_t> perm;
          bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
          if (!has_perm_attr) {
            perm.reserve(shape.dim_size());
            for (int i = shape.dim_size() - 1; i >= 0; --i)
              perm.push_back(i);
          } else if (!perm.empty()) {
            // check if every index is valid
            std::vector<bool> seen(shape.dim_size(), false);
            for (int64_t fromDimIndex : perm) {
              if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
                std::ostringstream oss;
                oss << "Invalid attribute perm {" << perm[0];
                for (size_t i = 1; i != perm.size(); ++i) {
                  oss << ", " << perm[i];
                }
                oss << "}, input shape = {";
                if (shape.dim_size() > 0) {
                  oss << shape.dim(0).dim_value();
                  for (int i = 1; i != shape.dim_size(); ++i) {
                    oss << ", " << shape.dim(i).dim_value();
                  }
                  oss << "}";
                }
                fail_type_inference(oss.str());
              } else {
                // check if any perm is repeated
                if (seen[fromDimIndex]) {
                  fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
                }
                seen[fromDimIndex] = true;
              }
            }
          }

          getOutputShape(ctx, 0);

          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          for (int64_t i : perm) {
            appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    21,
    OpSchema()
        .SetDoc(kDoc_Transpose_ver13)
        .Attr(
            "perm",
            "A list of integers. By default, reverse the dimensions, "
            "otherwise permute the axes according to the values given. "
            "Its length must be equal to the rank of the input.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "transposed", "Transposed output.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir10(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          auto input_type = ctx.getInputType(0);
          const TensorShapeProto& shape = input_type->tensor_type().shape();
          std::vector<int64_t> perm;
          bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
          if (!has_perm_attr) {
            perm.reserve(shape.dim_size());
            for (int i = shape.dim_size() - 1; i >= 0; --i)
              perm.push_back(i);
          } else if (!perm.empty()) {
            // check if every index is valid
            std::vector<bool> seen(shape.dim_size(), false);
            for (int64_t fromDimIndex : perm) {
              if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
                std::ostringstream oss;
                oss << "Invalid attribute perm {" << perm[0];
                for (size_t i = 1; i != perm.size(); ++i) {
                  oss << ", " << perm[i];
                }
                oss << "}, input shape = {";
                if (shape.dim_size() > 0) {
                  oss << shape.dim(0).dim_value();
                  for (int i = 1; i != shape.dim_size(); ++i) {
                    oss << ", " << shape.dim(i).dim_value();
                  }
                  oss << "}";
                }
                fail_type_inference(oss.str());
              } else {
                // check if any perm is repeated
                if (seen[fromDimIndex]) {
                  fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
                }
                seen[fromDimIndex] = true;
              }
            }
          }

          getOutputShape(ctx, 0);

          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          for (int64_t i : perm) {
            appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    13,
    OpSchema()
        .SetDoc(kDoc_Transpose_ver13)
        .Attr(
            "perm",
            "A list of integers. By default, reverse the dimensions, "
            "otherwise permute the axes according to the values given.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "transposed", "Transposed output.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          auto input_type = ctx.getInputType(0);
          const TensorShapeProto& shape = input_type->tensor_type().shape();
          std::vector<int64_t> perm;
          bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
          if (!has_perm_attr) {
            perm.reserve(shape.dim_size());
            for (int i = shape.dim_size() - 1; i >= 0; --i)
              perm.push_back(i);
          } else if (!perm.empty()) {
            // check if every index is valid
            std::vector<bool> seen(shape.dim_size(), false);
            for (int64_t fromDimIndex : perm) {
              if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
                std::ostringstream oss;
                oss << "Invalid attribute perm {" << perm[0];
                for (size_t i = 1; i != perm.size(); ++i) {
                  oss << ", " << perm[i];
                }
                oss << "}, input shape = {";
                if (shape.dim_size() > 0) {
                  oss << shape.dim(0).dim_value();
                  for (int i = 1; i != shape.dim_size(); ++i) {
                    oss << ", " << shape.dim(i).dim_value();
                  }
                  oss << "}";
                }
                fail_type_inference(oss.str());
              } else {
                // check if any perm is repeated
                if (seen[fromDimIndex]) {
                  fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
                }
                seen[fromDimIndex] = true;
              }
            }
          }

          getOutputShape(ctx, 0);

          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          for (int64_t i : perm) {
            appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    1,
    OpSchema()
        .SetDoc(kDoc_Transpose_ver13)
        .Attr(
            "perm",
            "A list of integers. By default, reverse the dimensions, "
            "otherwise permute the axes according to the values given.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Input(0, "data", "An input tensor.", "T")
        .Output(0, "transposed", "Transposed output.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          auto input_type = ctx.getInputType(0);
          const TensorShapeProto& shape = input_type->tensor_type().shape();
          std::vector<int64_t> perm;
          bool has_perm_attr = getRepeatedAttribute(ctx, "perm", perm);
          if (!has_perm_attr) {
            for (int i = shape.dim_size() - 1; i >= 0; --i)
              perm.push_back(i);
          } else if (!perm.empty()) {
            // check if every index is valid
            std::vector<bool> seen(shape.dim_size(), false);
            for (int64_t fromDimIndex : perm) {
              if (fromDimIndex < 0 || fromDimIndex >= shape.dim_size()) {
                std::ostringstream oss;
                oss << "Invalid attribute perm {" << perm[0];
                for (size_t i = 1; i != perm.size(); ++i) {
                  oss << ", " << perm[i];
                }
                oss << "}, input shape = {";
                if (shape.dim_size() > 0) {
                  oss << shape.dim(0).dim_value();
                  for (int i = 1; i != shape.dim_size(); ++i) {
                    oss << ", " << shape.dim(i).dim_value();
                  }
                  oss << "}";
                }
                fail_type_inference(oss.str());
              } else {
                // check if any perm is repeated
                if (seen[fromDimIndex]) {
                  fail_type_inference("Attribute perm for Transpose has repeated value: ", fromDimIndex);
                }
                seen[fromDimIndex] = true;
              }
            }
          }

          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          for (int64_t i : perm) {
            appendSingleDimCopiedFromInputTypeToOutputType(ctx, 0, 0, static_cast<size_t>(i));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    ScatterND,
    16,
    OpSchema()
        .SetDoc(kDoc_ScatterND_ver16)
        .Attr(
            "reduction",
            "Type of reduction to apply: none (default), add, mul. "
            "'none': no reduction applied. "
            "'add':  reduction using the addition operation. "
            "'mul': reduction using the multiplication operation.",
            AttributeProto::STRING,
            std::string("none"))
        .Input(0, "data", "Tensor of rank r >= 1.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "indices",
            "Tensor of rank q >= 1.",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "updates",
            "Tensor of rank q + r - indices_shape[-1] - 1.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(0, "output", "Tensor of rank r >= 1.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    ScatterND,
    13,
    OpSchema()
        .SetDoc(kDoc_ScatterND_ver13)
        .Input(0, "data", "Tensor of rank r >= 1.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "indices",
            "Tensor of rank q >= 1.",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "updates",
            "Tensor of rank q + r - indices_shape[-1] - 1.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(0, "output", "Tensor of rank r >= 1.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    ScatterND,
    11,
    OpSchema()
        .SetDoc(kDoc_ScatterND_ver13)
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(1, "indices", "Tensor of rank q >= 1.", "tensor(int64)")
        .Input(2, "updates", "Tensor of rank q + r - indices_shape[-1] - 1.", "T")
        .Output(0, "output", "Tensor of rank r >= 1.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    ScatterElements,
    16,
    OpSchema()
        .SetDoc(kDoc_ScatterElements_ver16)
        .Attr(
            "axis",
            "Which axis to scatter on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "reduction",
            "Type of reduction to apply: none (default), add, mul. "
            "'none': no reduction applied. "
            "'add':  reduction using the addition operation. "
            "'mul': reduction using the multiplication operation.",
            AttributeProto::STRING,
            std::string("none"))
        .Input(0, "data", "Tensor of rank r >= 1.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "indices",
            "Tensor of int32/int64 indices, of r >= 1 (same rank as input). All index values are expected to be "
            "within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds.",
            "Tind",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "updates",
            "Tensor of rank r >=1 (same rank and shape as indices)",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "Tensor of rank r >= 1 (same rank as input).",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Input and output types can be of any tensor type.")
        .TypeConstraint("Tind", {types::Int32, types::Int64}, "Constrain indices to integer types")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    ScatterElements,
    13,
    OpSchema()
        .SetDoc(kDoc_ScatterElements_ver13)
        .Attr(
            "axis",
            "Which axis to scatter on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "indices",
            "Tensor of int32/int64 indices, of r >= 1 (same rank as input). All index values are expected to be "
            "within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds.",
            "Tind",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "updates",
            "Tensor of rank r >=1 (same rank and shape as indices)",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "Tensor of rank r >= 1 (same rank as input).",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Input and output types can be of any tensor type.")
        .TypeConstraint("Tind", {types::Int32, types::Int64}, "Constrain indices to integer types")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    ScatterElements,
    11,
    OpSchema()
        .SetDoc(kDoc_ScatterElements_ver13)
        .Attr(
            "axis",
            "Which axis to scatter on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "indices",
            "Tensor of int32/int64 indices, of r >= 1 (same rank as input). All index values are expected to be "
            "within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds.",
            "Tind")
        .Input(2, "updates", "Tensor of rank r >=1 (same rank and shape as indices)", "T")
        .Output(0, "output", "Tensor of rank r >= 1 (same rank as input).", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Input and output types can be of any tensor type.")
        .TypeConstraint("Tind", {types::Int32, types::Int64}, "Constrain indices to integer types")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Gather,
    11,
    OpSchema()
        .SetDoc(kDoc_Gather_ver11)
        .Attr(
            "axis",
            "Which axis to gather on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "indices",
            "Tensor of int32/int64 indices, of any rank q. All index values are expected to be within bounds [-s, s-1] "
            "along axis of size s. It is an error if any of the index values are out of bounds.",
            "Tind")
        .Output(0, "output", "Tensor of rank q + (r - 1).", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to any tensor type.")
        .TypeConstraint("Tind", {types::Int32, types::Int64}, "Constrain indices to integer types")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 2)) {
            return;
          }
          const TensorShapeProto& data_shape = ctx.getInputType(0)->tensor_type().shape();
          const TensorShapeProto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
          int r = data_shape.dim_size();
          if (r < 1) {
            fail_shape_inference("data tensor must have rank >= 1");
          }
          int q = indices_shape.dim_size();
          int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
          if (axis < -r || axis >= r) {
            fail_shape_inference("axis must be in [-r, r-1]");
          }
          if (axis < 0) {
            axis += r;
          }
          int out_rank = q + r - 1;
          if (out_rank == 0) {
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          }
          for (int i = 0; i < out_rank; ++i) {
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = (i < axis) ? data_shape.dim(i)
                                                                                                  : // i < axis < r
                (i >= axis && i < axis + q) ? indices_shape.dim(i - axis)
                                            : // i - axis < q
                data_shape.dim(i - q + 1); // i < out_rank < q + r - 1
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) { GatherOp13DataPropagator(ctx); }));

ONNX_OPERATOR_SET_SCHEMA(
    GatherElements,
    11,
    OpSchema()
        .SetDoc(kDoc_GatherElements_ver11)
        .Attr(
            "axis",
            "Which axis to gather on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "indices",
            "Tensor of int32/int64 indices, with the same rank r as the input. All index values are expected to be "
            "within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds.",
            "Tind")
        .Output(0, "output", "Tensor of the same shape as indices.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to any tensor type.")
        .TypeConstraint("Tind", {types::Int32, types::Int64}, "Constrain indices to integer types")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          // propagate indices' shape to output if it exists
          if (hasInputShape(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 1, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    24,
    OpSchema()
        .SetDoc(kDoc_Squeeze_ver24)
        .Input(
            0,
            "data",
            "Tensors with at least max(dims) dimensions.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "axes",
            "1D tensor of integers indicating the dimensions to squeeze. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(data).",
            "tensor(int64)",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "squeezed",
            "Reshaped tensor with same data as input.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types_ir12(),
            "Constrain input and output types to all tensor types up to IRv12.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          std::vector<int64_t> axes;
          size_t num_inputs = ctx.getNumInputs();
          bool axes_not_specified = false;

          if ((num_inputs == 2) && ctx.getInputType(1)) { //'axes' is input
            auto axes_proto = ctx.getInputData(1);
            if (axes_proto == nullptr) {
              // skip if axes is not an initializer
              return;
            }
            axes = ParseData<int64_t>(axes_proto);
          } else {
            // axes not specified
            axes_not_specified = true;
          }

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_ndim = input_shape.dim_size();
          checkAxesRange(axes, input_ndim);
          adjustNegativeAxes(axes, input_ndim);

          for (int i = 0; i < input_ndim; ++i) {
            if (!input_shape.dim(i).has_dim_value() && axes_not_specified) {
              // if dim has a symbolic value and the axes spec want to act on all dims,
              // return early because we can't infer the shape
              return;
            }
          }

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          for (int i = 0; i < input_ndim; ++i) {
            if (axes_not_specified && input_shape.dim(i).dim_value() == 1) {
              // if axes not specified, do not keep shape if the dimension is equal to one
              continue;
            } else if (!axes_not_specified && std::find(axes.begin(), axes.end(), i) != axes.end()) {
              // if axes wants to explicitly act on this dim, fail explicitly only if the
              // dim is numerical and != 1. If the dim is 1 or symbolic, remove it. If
              // the dim is symbolic, runtime engines should check that the dimension is
              // actually 1 when the op is evaluated
              if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
                fail_shape_inference(
                    "Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
              }
            } else {
              *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
            }
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    23,
    OpSchema()
        .SetDoc(kDoc_Squeeze_ver24)
        .Input(
            0,
            "data",
            "Tensors with at least max(dims) dimensions.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "axes",
            "1D tensor of integers indicating the dimensions to squeeze. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(data).",
            "tensor(int64)",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "squeezed",
            "Reshaped tensor with same data as input.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types_ir11(),
            "Constrain input and output types to all tensor types up to IRv11.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          std::vector<int64_t> axes;
          size_t num_inputs = ctx.getNumInputs();
          bool axes_not_specified = false;

          if ((num_inputs == 2) && ctx.getInputType(1)) { //'axes' is input
            auto axes_proto = ctx.getInputData(1);
            if (axes_proto == nullptr) {
              // skip if axes is not an initializer
              return;
            }
            axes = ParseData<int64_t>(axes_proto);
          } else {
            // axes not specified
            axes_not_specified = true;
          }

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_ndim = input_shape.dim_size();
          checkAxesRange(axes, input_ndim);
          adjustNegativeAxes(axes, input_ndim);

          for (int i = 0; i < input_ndim; ++i) {
            if (!input_shape.dim(i).has_dim_value() && axes_not_specified) {
              // if dim has a symbolic value and the axes spec want to act on all dims,
              // return early because we can't infer the shape
              return;
            }
          }

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          for (int i = 0; i < input_ndim; ++i) {
            if (axes_not_specified && input_shape.dim(i).dim_value() == 1) {
              // if axes not specified, do not keep shape if the dimension is equal to one
              continue;
            } else if (!axes_not_specified && std::find(axes.begin(), axes.end(), i) != axes.end()) {
              // if axes wants to explicitly act on this dim, fail explicitly only if the
              // dim is numerical and != 1. If the dim is 1 or symbolic, remove it. If
              // the dim is symbolic, runtime engines should check that the dimension is
              // actually 1 when the op is evaluated
              if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
                fail_shape_inference(
                    "Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
              }
            } else {
              *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
            }
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    21,
    OpSchema()
        .SetDoc(kDoc_Squeeze_ver24)
        .Input(
            0,
            "data",
            "Tensors with at least max(dims) dimensions.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "axes",
            "List of integers indicating the dimensions to squeeze. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(data).",
            "tensor(int64)",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "squeezed",
            "Reshaped tensor with same data as input.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types_ir10(),
            "Constrain input and output types to all tensor types up to IRv10.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          std::vector<int64_t> axes;
          size_t num_inputs = ctx.getNumInputs();
          bool axes_not_specified = false;

          if ((num_inputs == 2) && ctx.getInputType(1)) { //'axes' is input
            auto axes_proto = ctx.getInputData(1);
            if (axes_proto == nullptr) {
              // skip if axes is not an initializer
              return;
            }
            axes = ParseData<int64_t>(axes_proto);
          } else {
            // axes not specified
            axes_not_specified = true;
          }

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_ndim = input_shape.dim_size();
          checkAxesRange(axes, input_ndim);
          adjustNegativeAxes(axes, input_ndim);

          for (int i = 0; i < input_ndim; ++i) {
            if (!input_shape.dim(i).has_dim_value() && axes_not_specified) {
              // if dim has a symbolic value and the axes spec want to act on all dims,
              // return early because we can't infer the shape
              return;
            }
          }

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          for (int i = 0; i < input_ndim; ++i) {
            if (axes_not_specified && input_shape.dim(i).dim_value() == 1) {
              // if axes not specified, do not keep shape if the dimension is equal to one
              continue;
            } else if (!axes_not_specified && std::find(axes.begin(), axes.end(), i) != axes.end()) {
              // if axes wants to explicitly act on this dim, fail explicitly only if the
              // dim is numerical and != 1. If the dim is 1 or symbolic, remove it. If
              // the dim is symbolic, runtime engines should check that the dimension is
              // actually 1 when the op is evaluated
              if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
                fail_shape_inference(
                    "Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
              }
            } else {
              *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
            }
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    13,
    OpSchema()
        .SetDoc(kDoc_Squeeze_ver24)
        .Input(
            0,
            "data",
            "Tensors with at least max(dims) dimensions.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "axes",
            "List of integers indicating the dimensions to squeeze. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(data).",
            "tensor(int64)",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "squeezed",
            "Reshaped tensor with same data as input.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          std::vector<int64_t> axes;
          size_t num_inputs = ctx.getNumInputs();
          bool axes_not_specified = false;

          if ((num_inputs == 2) && ctx.getInputType(1)) { //'axes' is input
            auto axes_proto = ctx.getInputData(1);
            if (axes_proto == nullptr) {
              // skip if axes is not an initializer
              return;
            }
            axes = ParseData<int64_t>(axes_proto);
          } else {
            // axes not specified
            axes_not_specified = true;
          }

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_ndim = input_shape.dim_size();
          checkAxesRange(axes, input_ndim);
          adjustNegativeAxes(axes, input_ndim);

          for (int i = 0; i < input_ndim; ++i) {
            if (!input_shape.dim(i).has_dim_value() && axes_not_specified) {
              // if dim has a symbolic value and the axes spec want to act on all dims,
              // return early because we can't infer the shape
              return;
            }
          }

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          for (int i = 0; i < input_ndim; ++i) {
            if (axes_not_specified && input_shape.dim(i).dim_value() == 1) {
              // if axes not specified, do not keep shape if the dimension is equal to one
              continue;
            } else if (!axes_not_specified && std::find(axes.begin(), axes.end(), i) != axes.end()) {
              // if axes wants to explicitly act on this dim, fail explicitly only if the
              // dim is numerical and != 1. If the dim is 1 or symbolic, remove it. If
              // the dim is symbolic, runtime engines should check that the dimension is
              // actually 1 when the op is evaluated
              if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
                fail_shape_inference(
                    "Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
              }
            } else {
              *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
            }
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    11,
    OpSchema()
        .Attr(
            "axes",
            "List of integers indicating the dimensions to squeeze. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .SetDoc(kDoc_Squeeze_ver11)
        .Input(0, "data", "Tensors with at least max(dims) dimensions.", "T")
        .Output(0, "squeezed", "Reshaped tensor with same data as input.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          if (!ctx.getInputType(0)->tensor_type().has_shape()) {
            return;
          }

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_ndim = input_shape.dim_size();
          std::vector<int64_t> axes;
          if (!getRepeatedAttribute(ctx, "axes", axes)) {
            for (int i = 0; i < input_ndim; ++i) {
              if (!input_shape.dim(i).has_dim_value()) {
                return;
              }
              if (input_shape.dim(i).dim_value() == 1) {
                axes.push_back(i);
              }
            }
          }

          std::transform(axes.begin(), axes.end(), axes.begin(), [&](int64_t axis) -> int64_t {
            return axis < 0 ? axis + input_ndim : axis;
          });

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          for (int i = 0; i < input_ndim; ++i) {
            if (std::find(axes.begin(), axes.end(), i) != axes.end()) {
              if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
                fail_shape_inference(
                    "Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
              }
            } else {
              *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    24,
    OpSchema()
        .SetDoc(kDoc_Unsqueeze_ver24)
        .Input(0, "data", "Original tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "axes",
            "1D tensor of integers indicating the dimensions to be inserted. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(expanded).",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "expanded",
            "Reshaped tensor with same data as input.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types_ir12(),
            "Constrain input and output types to all tensor types up to IRv12.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          std::vector<int64_t> axes;
          auto axes_proto = ctx.getInputData(1);
          if (axes_proto == nullptr) {
            // skip if axes is not an initializer
            return;
          }
          axes = ParseData<int64_t>(axes_proto);
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_ndim = input_shape.dim_size();
          const auto output_ndim = input_ndim + static_cast<int>(axes.size());
          checkAxesRange(axes, output_ndim);
          adjustNegativeAxes(axes, output_ndim);
          checkDuplicateAxes(axes, output_ndim);
          // sort after correcting negative axes values (if any)
          std::sort(axes.begin(), axes.end());

          int j = 0;
          for (int i = 0; i < input_ndim; ++i) {
            while (static_cast<size_t>(j) < axes.size() &&
                   axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
              ++j;
            }
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
                ctx.getInputType(0)->tensor_type().shape().dim(i);
          }
          while (static_cast<size_t>(j) < axes.size() &&
                 axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
            ++j;
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    23,
    OpSchema()
        .SetDoc(kDoc_Unsqueeze_ver24)
        .Input(0, "data", "Original tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "axes",
            "1D tensor of integers indicating the dimensions to be inserted. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(expanded).",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "expanded",
            "Reshaped tensor with same data as input.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types_ir11(),
            "Constrain input and output types to all tensor types up to IRv11.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          std::vector<int64_t> axes;
          auto axes_proto = ctx.getInputData(1);
          if (axes_proto == nullptr) {
            // skip if axes is not an initializer
            return;
          }
          axes = ParseData<int64_t>(axes_proto);
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_ndim = input_shape.dim_size();
          const auto output_ndim = input_ndim + static_cast<int>(axes.size());
          checkAxesRange(axes, output_ndim);
          adjustNegativeAxes(axes, output_ndim);
          checkDuplicateAxes(axes, output_ndim);
          // sort after correcting negative axes values (if any)
          std::sort(axes.begin(), axes.end());

          int j = 0;
          for (int i = 0; i < input_ndim; ++i) {
            while (static_cast<size_t>(j) < axes.size() &&
                   axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
              ++j;
            }
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
                ctx.getInputType(0)->tensor_type().shape().dim(i);
          }
          while (static_cast<size_t>(j) < axes.size() &&
                 axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
            ++j;
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    21,
    OpSchema()
        .SetDoc(kDoc_Unsqueeze_ver24)
        .Input(0, "data", "Original tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "axes",
            "List of integers indicating the dimensions to be inserted. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(expanded).",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "expanded",
            "Reshaped tensor with same data as input.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types_ir10(),
            "Constrain input and output types to all tensor types up to IRv10.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          std::vector<int64_t> axes;
          auto axes_proto = ctx.getInputData(1);
          if (axes_proto == nullptr) {
            // skip if axes is not an initializer
            return;
          }
          axes = ParseData<int64_t>(axes_proto);
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_ndim = input_shape.dim_size();
          const auto output_ndim = input_ndim + static_cast<int>(axes.size());
          checkAxesRange(axes, output_ndim);
          adjustNegativeAxes(axes, output_ndim);
          checkDuplicateAxes(axes, output_ndim);
          // sort after correcting negative axes values (if any)
          std::sort(axes.begin(), axes.end());

          int j = 0;
          for (int i = 0; i < input_ndim; ++i) {
            while (static_cast<size_t>(j) < axes.size() &&
                   axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
              ++j;
            }
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
                ctx.getInputType(0)->tensor_type().shape().dim(i);
          }
          while (static_cast<size_t>(j) < axes.size() &&
                 axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
            ++j;
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    13,
    OpSchema()
        .SetDoc(kDoc_Unsqueeze_ver24)
        .Input(0, "data", "Original tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "axes",
            "List of integers indicating the dimensions to be inserted. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(expanded).",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "expanded",
            "Reshaped tensor with same data as input.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          std::vector<int64_t> axes;
          auto axes_proto = ctx.getInputData(1);
          if (axes_proto == nullptr) {
            // skip if axes is not an initializer
            return;
          }
          axes = ParseData<int64_t>(axes_proto);
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_ndim = input_shape.dim_size();
          const auto output_ndim = input_ndim + static_cast<int>(axes.size());
          checkAxesRange(axes, output_ndim);
          adjustNegativeAxes(axes, output_ndim);
          checkDuplicateAxes(axes, output_ndim);
          // sort after correcting negative axes values (if any)
          std::sort(axes.begin(), axes.end());

          int j = 0;
          for (int i = 0; i < input_ndim; ++i) {
            while (static_cast<size_t>(j) < axes.size() &&
                   axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
              ++j;
            }
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
                ctx.getInputType(0)->tensor_type().shape().dim(i);
          }
          while (static_cast<size_t>(j) < axes.size() &&
                 axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
            ++j;
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          PropagateShapeDataFromInputToOutput(ctx, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    11,
    OpSchema()
        .Attr(
            "axes",
            "List of integers indicating the dimensions to be inserted. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(expanded).",
            AttributeProto::INTS)
        .SetDoc(kDoc_Unsqueeze_ver11)
        .Input(0, "data", "Original tensor", "T")
        .Output(0, "expanded", "Reshaped tensor with same data as input.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          std::vector<int64_t> axes;
          if (!getRepeatedAttribute(ctx, "axes", axes)) {
            return;
          }

          // validate 'axes' for duplicate entries
          std::unordered_set<int64_t> unique_values;
          for (const auto val : axes) {
            if (unique_values.find(val) != unique_values.end()) {
              fail_shape_inference("'axes' attribute must not contain any duplicates");
            }
            unique_values.insert(val);
          }

          if (!ctx.getInputType(0)->tensor_type().has_shape()) {
            return;
          }

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_ndim = input_shape.dim_size();
          const auto output_ndim = input_ndim + static_cast<int>(axes.size());
          for (auto& axe : axes) {
            if (axe < -output_ndim || axe >= output_ndim) {
              fail_shape_inference("values in 'axes' are beyond the bounds of the computed output shape");
            }
            if (axe < 0) {
              axe += output_ndim;
            }
          }

          // sort after correcting negative axes values (if any) in the previous
          // step
          std::sort(axes.begin(), axes.end());

          int j = 0;
          for (int i = 0; i < input_ndim; ++i) {
            while (static_cast<size_t>(j) < axes.size() &&
                   axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
              ++j;
            }
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
                ctx.getInputType(0)->tensor_type().shape().dim(i);
          }
          while (static_cast<size_t>(j) < axes.size() &&
                 axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
            ++j;
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    SpaceToDepth,
    1,
    OpSchema()
        .Attr("blocksize", "Blocks of [blocksize, blocksize] are moved.", AttributeProto::INT)
        .SetDoc(kDoc_SpaceToDepth_ver1)
        .Input(
            0,
            "input",
            "Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth"
            ", H is the height and W is the width.",
            "T")
        .Output(0, "output", "Output tensor of [N, C * blocksize * blocksize, H/blocksize, W/blocksize].", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto blocksize = getAttribute(ctx, "blocksize", 0);
          if (blocksize <= 0) {
            fail_shape_inference("Blocksize must be positive");
          }
          const auto block_area = checkedMultiply(blocksize, blocksize);
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = getInputShape(ctx, 0);
            if (input_shape.dim_size() == 4) {
              updateOutputShape(
                  ctx,
                  0,
                  {input_shape.dim(0),
                   input_shape.dim(1) * block_area,
                   input_shape.dim(2) / blocksize,
                   input_shape.dim(3) / blocksize});
            } else {
              fail_shape_inference("Input tensor must be 4-dimensional");
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    SpaceToDepth,
    13,
    OpSchema()
        .Attr("blocksize", "Blocks of [blocksize, blocksize] are moved.", AttributeProto::INT)
        .SetDoc(kDoc_SpaceToDepth_ver1)
        .Input(
            0,
            "input",
            "Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth"
            ", H is the height and W is the width.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "Output tensor of [N, C * blocksize * blocksize, H/blocksize, W/blocksize].",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto blocksize = getAttribute(ctx, "blocksize", 0);
          if (blocksize <= 0) {
            fail_shape_inference("Blocksize must be positive");
          }
          const auto block_area = checkedMultiply(blocksize, blocksize);
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = getInputShape(ctx, 0);
            if (input_shape.dim_size() == 4) {
              updateOutputShape(
                  ctx,
                  0,
                  {input_shape.dim(0),
                   input_shape.dim(1) * block_area,
                   input_shape.dim(2) / blocksize,
                   input_shape.dim(3) / blocksize});
            } else {
              fail_shape_inference("Input tensor must be 4-dimensional");
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    DepthToSpace,
    11,
    OpSchema()
        .Attr("blocksize", "Blocks of [blocksize, blocksize] are moved.", AttributeProto::INT)
        .Attr(
            "mode",
            "DCR (default) for depth-column-row order re-arrangement. Use CRD for column-row-depth order.",
            AttributeProto::STRING,
            std::string("DCR"))
        .SetDoc(kDoc_DepthToSpace_ver11)
        .Input(
            0,
            "input",
            "Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth"
            ", H is the height and W is the width.",
            "T")
        .Output(0, "output", "Output tensor of [N, C/(blocksize * blocksize), H * blocksize, W * blocksize].", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto blocksize = getAttribute(ctx, "blocksize", 0);
          if (blocksize <= 0) {
            fail_shape_inference("Blocksize must be positive");
          }
          const auto block_area = checkedMultiply(blocksize, blocksize);
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = getInputShape(ctx, 0);
            if (input_shape.dim_size() == 4) {
              updateOutputShape(
                  ctx,
                  0,
                  {input_shape.dim(0),
                   input_shape.dim(1) / block_area,
                   input_shape.dim(2) * blocksize,
                   input_shape.dim(3) * blocksize});
            } else {
              fail_shape_inference("Input tensor must be 4-dimensional");
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    DepthToSpace,
    13,
    OpSchema()
        .Attr("blocksize", "Blocks of [blocksize, blocksize] are moved.", AttributeProto::INT)
        .Attr(
            "mode",
            "DCR (default) for depth-column-row order re-arrangement. Use CRD for column-row-depth order.",
            AttributeProto::STRING,
            std::string("DCR"))
        .SetDoc(kDoc_DepthToSpace_ver13)
        .Input(
            0,
            "input",
            "Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth"
            ", H is the height and W is the width.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "Output tensor of [N, C/(blocksize * blocksize), H * blocksize, W * blocksize].",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto blocksize = getAttribute(ctx, "blocksize", 0);
          if (blocksize <= 0) {
            fail_shape_inference("Blocksize must be positive");
          }
          const auto block_area = checkedMultiply(blocksize, blocksize);
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = getInputShape(ctx, 0);
            if (input_shape.dim_size() == 4) {
              updateOutputShape(
                  ctx,
                  0,
                  {input_shape.dim(0),
                   input_shape.dim(1) / block_area,
                   input_shape.dim(2) * blocksize,
                   input_shape.dim(3) * blocksize});
            } else {
              fail_shape_inference("Input tensor must be 4-dimensional");
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Tile,
    6,
    OpSchema()
        .SetDoc(kDoc_Tile_ver6)
        .Input(0, "input", "Input tensor of any shape.", "T")
        .Input(
            1,
            "repeats",
            "1D int64 tensor of the same length as input's dimension number, "
            "includes numbers of repeated copies along input's dimensions.",
            "T1")
        .Output(
            0,
            "output",
            "Output tensor of the same dimensions and type as tensor input. "
            "output_dim[i] = input_dim[i] * repeats[i]",
            "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeConstraint("T1", {types::Int64}, "Constrain repeat's type to int64 tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          // Shape inference

          // Needs at least the first input to proceed
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_rank = input_shape.dim_size();

          const auto repeats_inputs = ctx.getInputData(1);

          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          if (nullptr != repeats_inputs && hasNInputShapes(ctx, 2)) {
            // shape inference is possible only when 'repeats' is an initializer
            const auto& repeats_shape = ctx.getInputType(1)->tensor_type().shape();
            if (repeats_shape.dim_size() != 1 || repeats_inputs->data_type() != TensorProto::INT64) {
              fail_shape_inference("'Repeats' input must be 1D tensor of type int64");
            }

            const auto repeats_data = ParseData<int64_t>(repeats_inputs);

            if (repeats_data.size() != static_cast<size_t>(input_rank)) {
              fail_shape_inference(
                  "'Repeats' input has incorrect number of values. "
                  "The number of values in 'repeats' must be equal "
                  "to the number of input dimensions.");
            }

            for (int i = 0; i < input_rank; ++i) {
              if (repeats_data[i] < 0) {
                fail_shape_inference("'Repeats' input must not contain negative values");
              }
              const auto& input_dim = input_shape.dim(i);
              auto output_dim = output_shape->add_dim();
              if (input_dim.has_dim_value()) {
                output_dim->set_dim_value(checkedMultiply(input_dim.dim_value(), repeats_data[i]));
              }
            }
          } else {
            // Infer output shape's rank in any case (if repeats data is not
            // available)
            auto output_shape_0 = getOutputShape(ctx, 0);
            for (int i = 0; i < input_rank; ++i) {
              output_shape_0->add_dim();
            }
          }
          return;
        }));

static constexpr const char* Resize_ver18_attr_coordinate_transformation_mode_doc = R"DOC(
This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor. <br/>

The coordinate of each dimension is transformed individually. Let's describe a case using axis x as an example.
Denote x_resized as the coordinate of axis x in the resized tensor, x_original as the coordinate of axis x in the original tensor, `length_original` as the length of the original tensor in axis x, length_resized as the length of the resized tensor in axis x, roi_x = (start_x, end_x) of the axis x in input "roi", `scale = length_resized / length_original`, <br/>

if coordinate_transformation_mode is `"half_pixel"`, <br/>
`x_original = (x_resized + 0.5) / scale - 0.5` <br/>

if coordinate_transformation_mode is `"pytorch_half_pixel"`, <br/>
`x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0` <br/>

if coordinate_transformation_mode is `"align_corners"`, <br/>
`x_original = x_resized * (length_original - 1) / (length_resized - 1)` <br/>

if coordinate_transformation_mode is `"asymmetric"`, <br/>
`x_original = x_resized / scale` <br/>

if coordinate_transformation_mode is `"tf_crop_and_resize"`, <br/>
`x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1)`
.)DOC";

static constexpr const char* Resize_ver18_attr_keep_aspect_ratio_policy_doc = R"DOC(
This attribute describes how to interpret the `sizes` input with regard to keeping the original aspect ratio of the input, and it is not applicable when
the `scales` input is used. <br/>

Given a set of `sizes`, associated with a subset of `axes` (explicitly provided or default), and assuming `d = axes[i]`, with `i` being the index of the provided `sizes`. <br/>

If `keep_aspect_ratio_policy` is `"stretch"`, the original aspect ratio is disregarded, and the input is resized to the specified size: <br/>
`out_size[d] = sizes[i]` <br/>

If `keep_aspect_ratio_policy` is `"not_larger"`, the sizes are adjusted so that no extent of the output is larger than the specified size, while keeping the original aspect ratio: <br/>
`scale = Min(sizes[i] / in_size[d])` <br/>
`out_size[d] = round_int(scale * in_size[d])` <br/>

If `keep_aspect_ratio_policy` is `"not_smaller"`, the sizes are adjusted so that no extent of the output is smaller than the specified size, while keeping the original aspect ratio: <br/>
`scale = Max(sizes[i] / in_size[d])` <br/>
`out_size[d] = round_int(scale * in_size[d])` <br/>

For non-resizable axes (those not specified in `axes`), the output size will be equal to the input size.

Note: `round_int` stands for computing the nearest integer value, rounding halfway cases up.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    18,
    OpSchema()
        .Attr(
            "mode",
            "Three interpolation modes: \"nearest\" (default), \"linear\" and \"cubic\". "
            "The \"linear\" mode includes linear interpolation for 1D tensor and N-linear interpolation for N-D tensor (for example, bilinear interpolation for 2D tensor). "
            "The \"cubic\" mode includes cubic interpolation for 1D tensor and N-cubic interpolation for N-D tensor (for example, bicubic interpolation for 2D tensor).",
            AttributeProto::STRING,
            std::string("nearest"))
        .Attr(
            "cubic_coeff_a",
            "The coefficient 'a' used in cubic interpolation. Two common choice are -0.5 (in some cases of TensorFlow) and -0.75"
            " (in PyTorch). Check out Equation (4) in https://ieeexplore.ieee.org/document/1163711 for the details. "
            "This attribute is valid only if mode is \"cubic\".",
            AttributeProto::FLOAT,
            static_cast<float>(-0.75))
        .Attr(
            "exclude_outside",
            "If set to 1, the weight of sampling locations outside the tensor will be set to 0"
            " and the weight will be renormalized so that their sum is 1.0. The default value is 0.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "coordinate_transformation_mode",
            Resize_ver18_attr_coordinate_transformation_mode_doc,
            AttributeProto::STRING,
            std::string("half_pixel"))
        .Attr(
            "nearest_mode",
            "Four modes: \"round_prefer_floor\" (default, as known as round half down), \"round_prefer_ceil\" (as known as round half up), \"floor\", \"ceil\". Only used by nearest interpolation. It indicates how to get \"nearest\" pixel in input tensor from x_original, so this attribute is valid only if \"mode\" is \"nearest\".",
            AttributeProto::STRING,
            std::string("round_prefer_floor"))
        .Attr(
            "extrapolation_value",
            "When coordinate_transformation_mode is \"tf_crop_and_resize\" and x_original is outside the range [0, length_original - 1], this value is used as the corresponding output value. Default is 0.0f.",
            AttributeProto::FLOAT,
            static_cast<float>(0))
        .Attr(
            "antialias",
            "If set to 1, \"linear\" and \"cubic\" interpolation modes will use an antialiasing filter when downscaling. "
            "Antialiasing is achieved by stretching the resampling filter by a factor max(1, 1 / scale) for each dimension, "
            "where the scale for each dimension may be provided explicitly through the \"scales\" input "
            "or implicitly derived from the ratio of output size to input size when the \"sizes\" input is used. "
            "This means that when downsampling, more input pixels contribute to an output pixel.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "axes",
            "If provided, it specifies a subset of axes that 'roi', 'scales' and 'sizes' refer to. "
            "If not provided, all axes are assumed [0, 1, ..., r-1], where r = rank(data). "
            "Non-specified dimensions are interpreted as non-resizable. "
            "Negative value means counting dimensions from the back. Accepted range is [-r, r-1], where r = rank(data). "
            "Behavior is undefined if an axis is repeated.",
            AttributeProto::INTS,
            false)
        .Attr(
            "keep_aspect_ratio_policy",
            Resize_ver18_attr_keep_aspect_ratio_policy_doc,
            AttributeProto::STRING,
            std::string("stretch"))
        .Input(0, "X", "N-D tensor", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "roi",
            "1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the rank of X or the length of axes, if provided. "
            "The RoIs' coordinates are normalized in the coordinate system of the input image. It only takes effect when coordinate_transformation_mode is \"tf_crop_and_resize\"",
            "T2",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "scales",
            "The scale array along each dimension. It takes value greater than 0. If it's less than 1,"
            " it's sampling down, otherwise, it's upsampling. The number of elements of 'scales' should"
            " be the same as the rank of input 'X' or the length of 'axes', if provided. "
            "One of 'scales' and 'sizes' MUST be specified and it is an error if both are specified. If 'sizes' is needed, the user can use an empty string as the name of 'scales' in this operator's input list.",
            "tensor(float)",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            3,
            "sizes",
            "Target size of the output tensor. Its interpretation depends on the 'keep_aspect_ratio_policy' value."
            "The number of elements of 'sizes' should be the same as the"
            " rank of input 'X', or the length of 'axes', if provided. Only one of 'scales' and 'sizes' can be specified. ",
            "tensor(int64)",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "Y", "N-D tensor after resizing", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            OpSchema::all_tensor_types_ir4(),
            "Constrain input 'X' and output 'Y' to all tensor types.")
        .TypeConstraint("T2", {types::Float16, types::Float, types::Double}, "Constrain roi type to float or double.")
        .SetDoc(kDoc_Resize_ver18)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { resizeShapeInference_opset13_to_18(ctx); }));

static constexpr const char* Resize_ver13_attr_coordinate_transformation_mode_doc = R"DOC(
This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor. <br/>

The coordinate of each dimension is transformed individually. Let's describe a case using axis x as an example.
Denote x_resized as the coordinate of axis x in the resized tensor, x_original as the coordinate of axis x in the original tensor, length_original as the length of the original tensor in axis x, length_resized as the length of the resized tensor in axis x, roi_x = (start_x, end_x) of the axis x in input "roi", scale = length_resized / length_original, <br/>

if coordinate_transformation_mode is "half_pixel", <br/>
x_original = (x_resized + 0.5) / scale - 0.5, <br/>

if coordinate_transformation_mode is "pytorch_half_pixel", <br/>
x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0, <br/>

if coordinate_transformation_mode is "align_corners", <br/>
x_original = x_resized * (length_original - 1) / (length_resized - 1), <br/>

if coordinate_transformation_mode is "asymmetric", <br/>
x_original = x_resized / scale, <br/>

if coordinate_transformation_mode is "tf_crop_and_resize", <br/>
x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1).)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    13,
    OpSchema()
        .Attr(
            "mode",
            "Three interpolation modes: nearest (default), linear and cubic. "
            "The \"linear\" mode includes linear interpolation for 1D tensor and N-linear interpolation for N-D tensor (for example, bilinear interpolation for 2D tensor). "
            "The \"cubic\" mode includes cubic interpolation for 1D tensor and N-cubic interpolation for N-D tensor (for example, bicubic interpolation for 2D tensor).",
            AttributeProto::STRING,
            std::string("nearest"))
        .Attr(
            "cubic_coeff_a",
            "The coefficient 'a' used in cubic interpolation. Two common choice are -0.5 (in some cases of TensorFlow) and -0.75"
            " (in PyTorch). Check out Equation (4) in https://ieeexplore.ieee.org/document/1163711 for the details. "
            "This attribute is valid only if \"mode\" is \"cubic\".",
            AttributeProto::FLOAT,
            static_cast<float>(-0.75))
        .Attr(
            "exclude_outside",
            "If set to 1, the weight of sampling locations outside the tensor will be set to 0"
            " and the weight will be renormalized so that their sum is 1.0. The default value is 0.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "coordinate_transformation_mode",
            Resize_ver13_attr_coordinate_transformation_mode_doc,
            AttributeProto::STRING,
            std::string("half_pixel"))
        .Attr(
            "nearest_mode",
            "Four modes: round_prefer_floor (default, as known as round half down), round_prefer_ceil (as known as round half up), floor, ceil. Only used by nearest interpolation. It indicates how to get \"nearest\" pixel in input tensor from x_original, so this attribute is valid only if \"mode\" is \"nearest\".",
            AttributeProto::STRING,
            std::string("round_prefer_floor"))
        .Attr(
            "extrapolation_value",
            "When coordinate_transformation_mode is \"tf_crop_and_resize\" and x_original is outside the range [0, length_original - 1], this value is used as the corresponding output value. Default is 0.0f.",
            AttributeProto::FLOAT,
            static_cast<float>(0))
        .Input(0, "X", "N-D tensor", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "roi",
            "1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the rank of X. The RoIs' coordinates are normalized in the coordinate system of the input image. It only takes effect when coordinate_transformation_mode is \"tf_crop_and_resize\"",
            "T2",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "scales",
            "The scale array along each dimension. It takes value greater than 0. If it's less than 1,"
            " it's sampling down, otherwise, it's upsampling. The number of elements of 'scales' should"
            " be the same as the rank of input 'X'. One of 'scales' and 'sizes' MUST be specified and it is an error if both are specified. If 'sizes' is needed, the user can use an empty string as the name of 'scales' in this operator's input list.",
            "tensor(float)",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            3,
            "sizes",
            "The size of the output tensor. The number of elements of 'sizes' should be the same as the"
            " rank of input 'X'. Only one of 'scales' and 'sizes' can be specified.",
            "tensor(int64)",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "Y", "N-D tensor after resizing", "T1", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "T1",
            OpSchema::all_tensor_types_ir4(),
            "Constrain input 'X' and output 'Y' to all tensor types.")
        .TypeConstraint("T2", {types::Float16, types::Float, types::Double}, "Constrain roi type to float or double.")
        .SetDoc(kDoc_Resize_ver13)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { resizeShapeInference_opset13_to_18(ctx); }));

static constexpr const char* Resize_attr_coordinate_transformation_mode_ver11_doc = R"DOC(
This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor. <br/>

The coordinate of each dimension is transformed individually. Let's describe a case using axis x as an example.
Denote x_resized as the coordinate of axis x in the resized tensor, x_original as the coordinate of axis x in the original tensor, length_original as the length of the original tensor in axis x, length_resized as the length of the resized tensor in axis x, roi_x = (start_x, end_x) of the axis x in input "roi", scale = length_resized / length_original, <br/>

if coordinate_transformation_mode is "half_pixel", <br/>
x_original = (x_resized + 0.5) / scale - 0.5, <br/>

if coordinate_transformation_mode is "pytorch_half_pixel", <br/>
x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0, <br/>

if coordinate_transformation_mode is "align_corners", <br/>
x_original = x_resized * (length_original - 1) / (length_resized - 1), <br/>

if coordinate_transformation_mode is "asymmetric", <br/>
x_original = x_resized / scale, <br/>

if coordinate_transformation_mode is "tf_half_pixel_for_nn", <br/>
x_original = (x_resized + 0.5) / scale, <br/>

if coordinate_transformation_mode is "tf_crop_and_resize", <br/>
x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1).)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    11,
    OpSchema()
        .Attr(
            "mode",
            "Three interpolation modes: nearest (default), linear and cubic. "
            "The \"linear\" mode includes linear interpolation for 1D tensor and N-linear interpolation for N-D tensor (for example, bilinear interpolation for 2D tensor). "
            "The \"cubic\" mode includes cubic interpolation for 1D tensor and N-cubic interpolation for N-D tensor (for example, bicubic interpolation for 2D tensor).",
            AttributeProto::STRING,
            std::string("nearest"))
        .Attr(
            "cubic_coeff_a",
            "The coefficient 'a' used in cubic interpolation. Two common choice are -0.5 (in some cases of TensorFlow) and -0.75"
            " (in PyTorch). Check out Equation (4) in https://ieeexplore.ieee.org/document/1163711 for the details. "
            "This attribute is valid only if \"mode\" is \"cubic\".",
            AttributeProto::FLOAT,
            static_cast<float>(-0.75))
        .Attr(
            "exclude_outside",
            "If set to 1, the weight of sampling locations outside the tensor will be set to 0"
            " and the weight will be renormalized so that their sum is 1.0. The default value is 0.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "coordinate_transformation_mode",
            Resize_attr_coordinate_transformation_mode_ver11_doc,
            AttributeProto::STRING,
            std::string("half_pixel"))
        .Attr(
            "nearest_mode",
            "Four modes: round_prefer_floor (default, as known as round half down), round_prefer_ceil (as known as round half up), floor, ceil. Only used by nearest interpolation. It indicates how to get \"nearest\" pixel in input tensor from x_original, so this attribute is valid only if \"mode\" is \"nearest\".",
            AttributeProto::STRING,
            std::string("round_prefer_floor"))
        .Attr(
            "extrapolation_value",
            "When coordinate_transformation_mode is \"tf_crop_and_resize\" and x_original is outside the range [0, length_original - 1], this value is used as the corresponding output value. Default is 0.0f.",
            AttributeProto::FLOAT,
            static_cast<float>(0))
        .Input(0, "X", "N-D tensor", "T1")
        .Input(
            1,
            "roi",
            "1-D tensor given as [start1, ..., startN, end1, ..., endN], where N is the rank of X. The RoIs' coordinates are normalized in the coordinate system of the input image. It only takes effect when coordinate_transformation_mode is \"tf_crop_and_resize\"",
            "T2")
        .Input(
            2,
            "scales",
            "The scale array along each dimension. It takes value greater than 0. If it's less than 1,"
            " it's sampling down, otherwise, it's upsampling. The number of elements of 'scales' should"
            " be the same as the rank of input 'X'. If 'size' is needed, the user must set 'scales' to an empty tensor.",
            "tensor(float)")
        .Input(
            3,
            "sizes",
            "The size of the output tensor. The number of elements of 'sizes' should be the same as the"
            " rank of input 'X'. May only be set if 'scales' is set to an empty tensor.",
            "tensor(int64)",
            OpSchema::Optional)
        .Output(0, "Y", "N-D tensor after resizing", "T1")
        .TypeConstraint("T1", OpSchema::all_tensor_types(), "Constrain input 'X' and output 'Y' to all tensor types.")
        .TypeConstraint("T2", {types::Float16, types::Float, types::Double}, "Constrain roi type to float or double.")
        .SetDoc(kDoc_Resize_ver13)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { resizeShapeInference_opset11_to_12(ctx); }));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    24,
    OpSchema()
        .SetDoc(kDoc_Identity_ver25)
        .Input(0, "input", "Input tensor", "V", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "output", "Tensor to copy input into.", "V", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "V",
            []() {
              auto t = OpSchema::all_tensor_types_ir12();
              auto s = OpSchema::all_tensor_sequence_types();
              auto o = OpSchema::all_optional_types();
              t.insert(t.end(), s.begin(), s.end());
              t.insert(t.end(), o.begin(), o.end());
              return t;
            }(),
            "Constrain input and output types to all tensor, sequence, and optional types.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    23,
    OpSchema()
        .SetDoc(kDoc_Identity_ver25)
        .Input(0, "input", "Input tensor", "V", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "output", "Tensor to copy input into.", "V", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "V",
            []() {
              auto t = OpSchema::all_tensor_types_ir11();
              auto s = OpSchema::all_tensor_sequence_types();
              auto o = OpSchema::all_optional_types();
              t.insert(t.end(), s.begin(), s.end());
              t.insert(t.end(), o.begin(), o.end());
              return t;
            }(),
            "Constrain input and output types to all tensor, sequence, and optional types.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    21,
    OpSchema()
        .SetDoc(kDoc_Identity_ver25)
        .Input(0, "input", "Input tensor", "V", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "output", "Tensor to copy input into.", "V", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "V",
            []() {
              auto t = OpSchema::all_tensor_types_ir10();
              auto s = OpSchema::all_tensor_sequence_types();
              auto o = OpSchema::all_optional_types();
              t.insert(t.end(), s.begin(), s.end());
              t.insert(t.end(), o.begin(), o.end());
              return t;
            }(),
            "Constrain input and output types to all tensor, sequence, and optional types.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    19,
    OpSchema()
        .SetDoc(kDoc_Identity_ver25)
        .Input(0, "input", "Input tensor", "V", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "output", "Tensor to copy input into.", "V", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "V",
            []() {
              auto t = OpSchema::all_tensor_types_ir9();
              auto s = OpSchema::all_tensor_sequence_types();
              auto o = OpSchema::all_optional_types();
              t.insert(t.end(), s.begin(), s.end());
              t.insert(t.end(), o.begin(), o.end());
              return t;
            }(),
            "Constrain input and output types to all tensor, sequence, and optional types.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    13,
    OpSchema()
        .SetDoc(kDoc_Identity_ver25)
        .Input(0, "input", "Input tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "output", "Tensor to copy input into.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    1,
    OpSchema()
        .SetDoc(kDoc_Identity_ver25)
        .Input(0, "input", "Input tensor", "T")
        .Output(0, "output", "Tensor to copy input into.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    IsNaN,
    9,
    OpSchema()
        .SetDoc(kDoc_IsNaN_ver20)
        .Input(0, "X", "input", "T1")
        .Output(0, "Y", "output", "T2")
        .TypeConstraint("T1", {types::Float16, types::Float, types::Double}, "Constrain input types to float tensors.")
        .TypeConstraint("T2", {types::Bool}, "Constrain output types to boolean tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          updateOutputElemType(ctx, 0, TensorProto::BOOL);
          if (hasInputShape(ctx, 0)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    IsNaN,
    13,
    OpSchema()
        .SetDoc(kDoc_IsNaN_ver20)
        .Input(0, "X", "input", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "Y", "output", "T2", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T1",
            {types::Float16, types::Float, types::Double, types::BFloat16},
            "Constrain input types to float tensors.")
        .TypeConstraint("T2", {types::Bool}, "Constrain output types to boolean tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          updateOutputElemType(ctx, 0, TensorProto::BOOL);
          if (hasInputShape(ctx, 0)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    IsInf,
    10,
    OpSchema()
        .SetDoc(kDoc_IsInf_ver20)
        .Input(0, "X", "input", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "Y", "output", "T2", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Attr(
            "detect_positive",
            "(Optional) Whether map positive infinity to true. Default to 1 "
            "so that positive infinity induces true. Set this attribute to 0 "
            "if positive infinity should be mapped to false.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "detect_negative",
            "(Optional) Whether map negative infinity to true. Default to 1 "
            "so that negative infinity induces true. Set this attribute to 0 "
            "if negative infinity should be mapped to false.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .TypeConstraint("T1", {types::Float, types::Double}, "Constrain input types to float tensors.")
        .TypeConstraint("T2", {types::Bool}, "Constrain output types to boolean tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          updateOutputElemType(ctx, 0, TensorProto::BOOL);
          if (hasInputShape(ctx, 0)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    NonZero,
    9,
    OpSchema()
        .SetDoc(kDoc_NonZero_ver9)
        .Input(0, "X", "input", "T")
        .Output(0, "Y", "output", "tensor(int64)")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          updateOutputElemType(ctx, 0, TensorProto::INT64);
          TensorShapeProto output_shape;
          auto dim = output_shape.add_dim();
          if (hasInputShape(ctx, 0)) {
            const TensorShapeProto& input_shape = getInputShape(ctx, 0);
            dim->set_dim_value(input_shape.dim_size());
          }
          output_shape.add_dim();
          updateOutputShape(ctx, 0, output_shape);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    GatherND,
    12,
    OpSchema()
        .SetDoc(kDoc_GatherND_ver12)
        .Attr(
            "batch_dims",
            "The number of batch dimensions. The gather of indexing starts from dimension of data[batch_dims:]",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "indices",
            "Tensor of rank q >= 1. All index values are expected to be within bounds [-s, s-1] "
            "along axis of size s. It is an error if any of the index values are out of bounds.",
            "tensor(int64)")
        .Output(0, "output", "Tensor of rank q + r - indices_shape[-1] - 1.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          if (!hasNInputShapes(ctx, 2)) {
            // cannot proceed with shape or rank inference
            return;
          }

          const auto& data_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto data_rank = data_shape.dim_size();

          const auto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
          const auto indices_rank = indices_shape.dim_size();

          int64_t batch_dims_data = getAttribute(ctx, "batch_dims", 0);
          if (data_rank < 1 || indices_rank < 1) {
            fail_shape_inference(
                "Both `data` and `indices` input tensors in GatherND op "
                "need to have rank larger than 0.");
          }

          // cannot ascertain if the input shapes are valid if shape of
          // `indices` is missing last dimension value so return at this point
          if (!indices_shape.dim(indices_rank - 1).has_dim_value()) {
            return;
          }

          const auto last_index_dimension =
              checkedAdd(indices_shape.dim(indices_rank - 1).dim_value(), batch_dims_data);

          if (last_index_dimension > data_rank) {
            fail_shape_inference(
                "Last dimension of `indices` input tensor in GatherND op "
                "must not be larger than the rank of `data` tensor");
          }

          for (int i = 0; i < indices_rank - 1; ++i) {
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = indices_shape.dim(i);
          }

          for (int i = static_cast<int>(last_index_dimension); i < data_rank; ++i) {
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = data_shape.dim(i);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    24,
    OpSchema().FillUsing(PadDocGenerator(
        kDoc_Pad_ver24,
        "Supported modes: `constant`(default), `reflect`, `edge`, `wrap`",
        OpSchema::all_tensor_types_ir12(),
        "Constrain input and output types to all tensor types up to IRv12.")));

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    23,
    OpSchema().FillUsing(PadDocGenerator(
        kDoc_Pad_ver24,
        "Supported modes: `constant`(default), `reflect`, `edge`, `wrap`",
        OpSchema::all_tensor_types_ir11(),
        "Constrain input and output types to all tensor types up to IRv11.")));

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    21,
    OpSchema().FillUsing(PadDocGenerator(
        kDoc_Pad_ver24,
        "Supported modes: `constant`(default), `reflect`, `edge`, `wrap`",
        OpSchema::all_tensor_types_ir10(),
        "Constrain input and output types to all tensor types up to IRv10.")));

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    19,
    OpSchema().FillUsing(
        PadDocGenerator(kDoc_Pad_ver24, "Supported modes: `constant`(default), `reflect`, `edge`, `wrap`")));

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    11,
    OpSchema()
        .Attr(
            "mode",
            "Supported modes: `constant`(default), `reflect`, `edge`",
            AttributeProto::STRING,
            std::string("constant"))
        .SetDoc(kDoc_Pad_ver11)
        .Input(0, "data", "Input tensor.", "T")
        .Input(
            1,
            "pads",
            "Tensor of integers indicating the number of padding elements to add or remove (if negative) "
            "at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. "
            "`pads` should be a 1D tensor of shape [2 * input_rank]. "
            "`pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...], "
            "where xi_begin is the number of pad values added at the beginning of axis `i` and "
            "xi_end, the number of pad values added at the end of axis `i`.",
            "tensor(int64)")
        .Input(
            2,
            "constant_value",
            "(Optional) A scalar value to be used if the mode chosen is `constant` (by default it is 0).",
            "T",
            OpSchema::Optional)
        .Output(0, "output", "Tensor after padding.", "T")
        .TypeConstraint("T", OpSchema::all_numeric_types(), "Constrain input and output to only numeric types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          // Shape inference needs the input data shape
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_rank = input_shape.dim_size();

          // Infer output shape if 'pads' tensor is available
          const auto pads_initializer = ctx.getInputData(1);
          if (nullptr != pads_initializer) {
            if (pads_initializer->dims_size() != 1 || pads_initializer->data_type() != TensorProto::INT64) {
              fail_shape_inference("'pads' input must be a 1D (shape: [2 * input_rank]) tensor of type int64");
            }

            const auto pads_data = ParseData<int64_t>(pads_initializer);
            if (pads_data.size() != static_cast<size_t>(2 * input_rank)) {
              fail_shape_inference("Pads has incorrect number of values");
            }

            auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
            for (int i = 0; i < input_rank; ++i) {
              const auto& input_dim = input_shape.dim(i);
              auto output_dim = output_shape->add_dim();
              const auto total_pad = checkedAdd(pads_data[i], pads_data[i + input_rank]);
              if (input_dim.has_dim_value()) {
                output_dim->set_dim_value(checkedAdd(input_dim.dim_value(), total_pad));
              } else if (total_pad == 0) {
                *output_dim = input_dim;
              }
            }
          } else {
            // Infer output shapes' rank in any case
            auto output_shape_0 = getOutputShape(ctx, 0);
            for (int i = 0; i < input_rank; ++i) {
              output_shape_0->add_dim();
            }
          }
          return;
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    1,
    OpSchema()
        .SetDoc(kDoc_Cast_ver1)
        .Attr(
            "to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::STRING)
        .Input(0, "input", "Input tensor to be cast.", "T1")
        .Output(
            0,
            "output",
            "Output tensor with the same shape as input with type "
            "specified by the 'to' argument",
            "T2")
        .TypeConstraint(
            "T1",
            {types::Float16,
             types::Float,
             types::Double,
             types::Int8,
             types::Int16,
             types::Int32,
             types::Int64,
             types::UInt8,
             types::UInt16,
             types::UInt32,
             types::UInt64,
             types::Bool},
            "Constrain input types. Casting from strings and complex are not supported.")
        .TypeConstraint(
            "T2",
            {types::Float16,
             types::Float,
             types::Double,
             types::Int8,
             types::Int16,
             types::Int32,
             types::Int64,
             types::UInt8,
             types::UInt16,
             types::UInt32,
             types::UInt64,
             types::Bool},
            "Constrain output types. Casting to strings and complex are not supported."));

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    6,
    OpSchema()
        .SetDoc(kDoc_Cast_ver1)
        .Attr(
            "to",
            "The data type to which the elements of the input tensor are cast. "
            "Strictly must be one of the types from DataType enum in TensorProto",
            AttributeProto::INT)
        .Input(0, "input", "Input tensor to be cast.", "T1")
        .Output(
            0,
            "output",
            "Output tensor with the same shape as input with type "
            "specified by the 'to' argument",
            "T2")
        .TypeConstraint(
            "T1",
            {types::Float16,
             types::Float,
             types::Double,
             types::Int8,
             types::Int16,
             types::Int32,
             types::Int64,
             types::UInt8,
             types::UInt16,
             types::UInt32,
             types::UInt64,
             types::Bool},
            "Constrain input types. Casting from strings and complex are not supported.")
        .TypeConstraint(
            "T2",
            {types::Float16,
             types::Float,
             types::Double,
             types::Int8,
             types::Int16,
             types::Int32,
             types::Int64,
             types::UInt8,
             types::UInt16,
             types::UInt32,
             types::UInt64,
             types::Bool},
            "Constrain output types. Casting to strings and complex are not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromAttributeToOutput(ctx, "to", 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Concat,
    1,
    OpSchema()
        .Attr("axis", "Which axis to concat on.  Default value is 1.", AttributeProto::INT, OPTIONAL_VALUE)
        .SetDoc(kDoc_Concat_ver1)
        .Input(0, "inputs", "List of tensors for concatenation", "T", OpSchema::Variadic)
        .Output(0, "concat_result", "Concatenated tensor", "T")
        .TypeConstraint(
            "T",
            {types::Float16, types::Float, types::Double},
            "Constrain output types to float tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    Concat,
    4,
    OpSchema()
        .Attr("axis", "Which axis to concat on", AttributeProto::INT)
        .SetDoc(kDoc_Concat_ver4)
        .Input(0, "inputs", "List of tensors for concatenation", "T", OpSchema::Variadic)
        .Output(0, "concat_result", "Concatenated tensor", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain output types to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto numInputs = ctx.getNumInputs();
          if (numInputs < 1 || !hasNInputShapes(ctx, numInputs)) {
            return;
          }

          auto rank = ctx.getInputType(0)->tensor_type().shape().dim_size();

          auto axisAttr = ctx.getAttribute("axis");
          if (!axisAttr) {
            fail_shape_inference("Required attribute axis is missing");
          }
          int axis = static_cast<int>(axisAttr->i());
          if (rank <= axis) {
            fail_shape_inference("rank must be greater than axis");
          }
          if (axis < 0) {
            return;
          }

          bool all_lengths_known = true;
          int64_t total_length = 0;

          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          for (int64_t i = 0; i < rank; ++i) {
            output_shape->add_dim();
          }

          for (size_t i = 0; i < numInputs; i++) {
            const auto& shape = ctx.getInputType(i)->tensor_type().shape();
            if (shape.dim_size() != rank) {
              fail_shape_inference("All inputs to Concat must have same rank");
            }
            for (int j = 0; j < rank; j++) {
              if (j == axis) {
                if (shape.dim(j).has_dim_value()) {
                  const int64_t dim_val = shape.dim(j).dim_value();
                  if (dim_val < 0) {
                    fail_shape_inference("Negative dimension value on Concat axis");
                  }
                  if (checked_add_overflow(total_length, dim_val, &total_length)) {
                    fail_shape_inference("Integer overflow computing Concat output length");
                  }
                } else {
                  all_lengths_known = false;
                }
              } else {
                auto& output_dim = *output_shape->mutable_dim(j);
                const auto& input_dim = shape.dim(j);
                mergeInDimensionInfo(input_dim, output_dim, j);
              }
            }
          }

          if (all_lengths_known) {
            output_shape->mutable_dim(axis)->set_dim_value(total_length);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    1,
    OpSchema()
        .Input(0, "input", "The tensor to split", "T")
        .Input(1, "split", "Optional list of output lengths (see also arg 'split')", "T", OpSchema::Optional)
        .Output(0, "outputs...", "One or more outputs forming list of tensors after splitting", "T", OpSchema::Variadic)
        .TypeConstraint("T", {types::Float16, types::Float, types::Double}, "Constrain input types to float tensors.")
        .Attr("axis", "Which axis to split on", AttributeProto::INT, OPTIONAL_VALUE)
        .Attr("split", "length of each output", AttributeProto::INTS, OPTIONAL_VALUE)
        .SetDoc(kDoc_Split_ver1));

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    1,
    OpSchema()
        .Attr(
            "paddings",
            "List of integers indicate the padding element count at the "
            "beginning and end of each axis, for 2D it is the number of pixel. "
            "`paddings` rank should be double of the input's rank. `paddings` format should be as follow "
            "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
            "added at the beginning of axis `i` and xi_end, the number of pixels added at "
            "the end of axis `i`.",
            AttributeProto::INTS)
        .Attr("mode", "Three modes: constant(default), reflect, edge", AttributeProto::STRING, std::string("constant"))
        .Attr("value", "One float, indicates the value to be filled, default is 0", AttributeProto::FLOAT, 0.0f)
        .SetDoc(kDoc_Pad_ver1)
        .Input(0, "data", "Input tensor.", "T")
        .Output(0, "output", "Tensor after padding.", "T")
        .TypeConstraint(
            "T",
            {types::Float16, types::Float, types::Double},
            "Constrain input and output types to float tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    1,
    OpSchema()
        .SetDoc(kDoc_Reshape_ver1)
        .Attr("shape", "New shape", AttributeProto::INTS, OPTIONAL_VALUE)
        // This attribute was added via AllowConsumed API in OpSchema.
        // After removing the API, we're now using the Attr API to simulate the
        // old definition.
        .Attr("consumed_inputs", "legacy optimization attribute.", AttributeProto::INTS, OPTIONAL_VALUE)
        .Input(0, "data", "An input tensor.", "T")
        .Output(0, "reshaped", "Reshaped data.", "T")
        .TypeConstraint(
            "T",
            {types::Float16, types::Float, types::Double},
            "Constrain input and output types to float tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    Tile,
    1,
    OpSchema()
        .SetDoc(kDoc_Tile_ver1)
        .Input(0, "input", "Input tensor of any shape.", "T")
        .Input(1, "tiles", "Number of repeated copies to make of the input tensor.", "T")
        .Input(2, "axis", "Axis along which to repeat.", "T")
        .Output(0, "output", "Output tensor of same shape and type as input.", "T")
        .TypeConstraint("T", {types::Float16, types::Float, types::Double}, "Constrain input types to float tensors.")
        .TypeConstraint("T1", {types::Int64}, "Constrain tiles and axis's type to int64 tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          // Only rank of output can be inferred. We can do better if second
          // input is a constant, but this requires extending InferenceContext
          // interface to get values of constant inputs.
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Upsample,
    1,
    OpSchema()
        .SetSupportLevel(OpSchema::SupportType::EXPERIMENTAL)
        .Attr(
            "width_scale",
            "The scale along width dimension. It takes value greater than or equal to 1.",
            AttributeProto::FLOAT)
        .Attr(
            "height_scale",
            "The scale along height dimension. It takes value greater than or equal to 1.",
            AttributeProto::FLOAT)
        .Attr(
            "mode",
            "Two interpolation modes: nearest(default), bilinear",
            AttributeProto::STRING,
            std::string("nearest"))
        .Input(0, "X", "4-D tensor, [N,C,H,W]", "T")
        .Output(0, "Y", "4-D tensor after resizing, [N,C,H,W]", "T")
        .TypeConstraint(
            "T",
            {types::Bool, types::Int32, types::Int64, types::Float16, types::Float, types::Double},
            "Constrain output types to bool, int32, int64, float16, float, double tensors.")
        .SetDoc(kDoc_Upsample_ver1));

ONNX_OPERATOR_SET_SCHEMA(
    Upsample,
    7,
    OpSchema()
        .Attr(
            "scales",
            "The scale array along each dimension. It takes value greater than or equal to 1."
            " The number of elements of 'scales' should be the same as the rank of input 'X'.",
            AttributeProto::FLOATS)
        .Attr(
            "mode",
            "Two interpolation modes: nearest (default), and linear (including bilinear, trilinear, etc)",
            AttributeProto::STRING,
            std::string("nearest"))
        .Input(0, "X", "N-D tensor", "T")
        .Output(0, "Y", "N-D tensor after resizing", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .SetDoc(kDoc_Upsample_ver7)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          const auto& input_shape = getInputShape(ctx, 0);
          auto output_shape = getOutputShape(ctx, 0);
          const auto scales = ctx.getAttribute("scales");

          if (output_shape->dim_size() > 0) {
            if (output_shape->dim_size() != input_shape.dim_size()) {
              fail_shape_inference(
                  "Ranks inferred (",
                  input_shape.dim_size(),
                  ") is not equal to the existing rank value (",
                  output_shape->dim_size(),
                  ").");
            }
          } else { // Infer the rank of output anyway
            for (int i = 0; i < input_shape.dim_size(); ++i) {
              output_shape->add_dim();
            }
          }

          if (nullptr != scales) {
            // Infer output shape's dimension value if 'scales' is known.
            if (scales->type() == AttributeProto_AttributeType_FLOATS) {
              const std::vector<float> scales_data(scales->floats().begin(), scales->floats().end());
              if (scales_data.size() != static_cast<size_t>(input_shape.dim_size())) {
                fail_shape_inference("Number of elements of attribute 'scales' must be same as rank of input 'X'");
              }
              resizeShapeInferenceHelper_opset7_to_10(input_shape, scales_data, output_shape);
            } else {
              fail_shape_inference("Attribute 'scales' must have floats type.");
            } // scales->type() == float
          } else {
            fail_shape_inference("Attribute 'scales' is required.");
          } // nullptr != scales
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Upsample,
    9,
    OpSchema()
        .Attr(
            "mode",
            "Two interpolation modes: nearest (default), and linear (including bilinear, trilinear, etc)",
            AttributeProto::STRING,
            std::string("nearest"))
        .Input(0, "X", "N-D tensor", "T")
        .Input(
            1,
            "scales",
            "The scale array along each dimension. It takes value greater than or equal to 1."
            " The number of elements of 'scales' should be the same as the rank of input 'X'.",
            "tensor(float)")
        .Output(0, "Y", "N-D tensor after resizing", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input 'X' and output 'Y' to all tensor types.")
        .SetDoc(kDoc_Upsample_ver7)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { resizeShapeInference_opset7_to_10(ctx); }));

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    10,
    OpSchema()
        .Attr(
            "mode",
            "Two interpolation modes: nearest (default), and linear (including bilinear, trilinear, etc)",
            AttributeProto::STRING,
            std::string("nearest"))
        .Input(0, "X", "N-D tensor", "T")
        .Input(
            1,
            "scales",
            "The scale array along each dimension. It takes value greater than 0. If it's less than 1,"
            " it's sampling down, otherwise, it's upsampling. The number of elements of 'scales' should"
            " be the same as the rank of input 'X'.",
            "tensor(float)")
        .Output(0, "Y", "N-D tensor after resizing", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input 'X' and output 'Y' to all tensor types.")
        .SetDoc(kDoc_Resize_ver10)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { resizeShapeInference_opset7_to_10(ctx); }));

ONNX_OPERATOR_SET_SCHEMA(
    Slice,
    1,
    OpSchema()
        .SetDoc(kDoc_Slice_ver1)
        .Input(0, "data", "Tensor of data to extract slices from.", "T")
        .Attr(
            "axes",
            "Axes that `starts` and `ends` apply to. "
            "It's optional. If not present, will be treated as "
            "[0, 1, ..., len(`starts`) - 1].",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Attr("starts", "Starting indices of corresponding axis in `axes`", AttributeProto::INTS)
        .Attr("ends", "Ending indices (exclusive) of corresponding axis in axes`", AttributeProto::INTS)
        .Output(0, "output", "Sliced data tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          std::vector<int64_t> starts;
          std::vector<int64_t> ends;
          if (!getRepeatedAttribute(ctx, "starts", starts) || !getRepeatedAttribute(ctx, "ends", ends) ||
              starts.size() != ends.size()) {
            fail_shape_inference("Incorrect or missing attribute value for starts and ends");
          }

          std::vector<int64_t> axes;
          if (!getRepeatedAttribute(ctx, "axes", axes)) {
            for (size_t i = 0; i < starts.size(); ++i) {
              axes.push_back(i);
            }
          } else if (axes.size() != starts.size()) {
            fail_shape_inference("Attribute axes has incorrect length");
          } else if (!std::is_sorted(axes.begin(), axes.end())) {
            // TODO(ONNX) support shape inference for unsorted axes
            return;
          }

          auto is_negative = [](int64_t index) { return index < 0; };
          if (std::any_of(axes.begin(), axes.end(), is_negative)) {
            // Negative axes were not explicitly discussed in the spec before opset-10.
            // Hence, they are officially not part of the spec, but some models/runtimes may use them.
            // So we perform simple rank inference in this case.
            for (size_t i = 0; static_cast<int64_t>(i) < ctx.getInputType(0)->tensor_type().shape().dim_size(); ++i) {
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
            }
            return;
          }

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          for (size_t i = 0, j = 0; static_cast<int64_t>(i) < ctx.getInputType(0)->tensor_type().shape().dim_size();
               ++i) {
            auto newdim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
            if (j < axes.size() && static_cast<size_t>(axes[j]) == i) {
              // There's a lot of potential behaviors. For now just
              // handle some simple cases.
              const auto& dim = ctx.getInputType(0)->tensor_type().shape().dim(i);
              if (dim.has_dim_value()) {
                auto dim_value = dim.dim_value();
                if (starts[j] < 0) {
                  starts[j] += dim_value;
                }
                if (ends[j] < 0) {
                  ends[j] += dim_value;
                }
                if (starts[j] >= 0 && ends[j] >= 0) {
                  auto newval = std::min(dim_value, ends[j]) - starts[j];
                  if (newval >= 0) {
                    newdim->set_dim_value(newval);
                  }
                }
              }
              ++j;
            } else {
              *newdim = ctx.getInputType(0)->tensor_type().shape().dim(i);
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Slice,
    10,
    OpSchema()
        .SetDoc(kDoc_Slice_ver10)
        .Input(0, "data", "Tensor of data to extract slices from.", "T")
        .Input(1, "starts", "1-D tensor of starting indices of corresponding axis in `axes`", "Tind")
        .Input(2, "ends", "1-D tensor of ending indices (exclusive) of corresponding axis in `axes`", "Tind")
        .Input(3, "axes", "1-D tensor of axes that `starts` and `ends` apply to.", "Tind", OpSchema::Optional)
        .Input(
            4,
            "steps",
            "1-D tensor of slice step of corresponding axis in `axes`. Default to 1. ",
            "Tind",
            OpSchema::Optional)
        .Output(0, "output", "Sliced data tensor.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeConstraint("Tind", {types::Int32, types::Int64}, "Constrain indices to integer types")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          size_t num_inputs = ctx.getNumInputs();
          if (num_inputs != 3 && num_inputs != 4 && num_inputs != 5) {
            fail_type_inference("Slice op must have either three, four or five inputs.");
          }
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          // Shape Inference if
          //     1. 2nd and 3rd input data (starts, ends) are available.
          // and 2. 4th and 5th optional input (axes, steps) are either not set,
          // or set and is initializer.
          const TensorProto* startsInitializer = ctx.getInputData(1);
          const TensorProto* endsInitializer = ctx.getInputData(2);
          const TensorProto* axesInitializer = hasInputShape(ctx, 3) ? ctx.getInputData(3) : nullptr;
          const TensorProto* stepsInitializer = hasInputShape(ctx, 4) ? ctx.getInputData(4) : nullptr;

          if (!startsInitializer || !endsInitializer || (hasInputShape(ctx, 3) && !ctx.getInputData(3)) ||
              (hasInputShape(ctx, 4) && !ctx.getInputData(4))) {
            const auto input_rank = ctx.getInputType(0)->tensor_type().shape().dim_size();
            // we can infer the output rank - it never changes
            for (int i = 0; i < input_rank; ++i) {
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
            }
            return;
          }

          // don't know data_type- can't proceed
          if (!startsInitializer->has_data_type())
            return;

          auto get_initializer_data = [](const TensorProto* initializer) -> std::vector<int64_t> {
            std::vector<int64_t> vec;
            if (initializer->data_type() == TensorProto::INT64) {
              const auto data = ParseData<int64_t>(initializer);
              vec.insert(vec.end(), data.begin(), data.end());
            } else if (initializer->data_type() == TensorProto::INT32) {
              const auto data = ParseData<int32_t>(initializer);
              vec.insert(vec.end(), data.begin(), data.end());
            } else {
              // unaccepted data type
              fail_shape_inference("Only supports `int32_t` or `int64_t` inputs for starts/ends/axes/steps");
            }
            return vec;
          };

          std::vector<int64_t> starts = get_initializer_data(startsInitializer);
          std::vector<int64_t> ends = get_initializer_data(endsInitializer);

          if (starts.size() != ends.size()) {
            fail_shape_inference("Incorrect or missing input value for starts and ends");
          }

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_rank = input_shape.dim_size();
          std::vector<int64_t> axes(starts.size());
          if (!axesInitializer) {
            std::iota(axes.begin(), axes.end(), 0);
          } else {
            axes = get_initializer_data(axesInitializer);
            if (axes.size() != starts.size()) {
              fail_shape_inference("Input axes has incorrect length");
            }
          }

          std::vector<int64_t> steps;
          if (!stepsInitializer) {
            steps = std::vector<int64_t>(starts.size(), 1);
          } else {
            steps = get_initializer_data(stepsInitializer);
            if (steps.size() != axes.size()) {
              fail_shape_inference("Input steps has incorrect length");
            }
          }

          for (int i = 0; i < input_rank; ++i) {
            // first update rank of output dim
            auto output_dim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
            const auto& input_dim = input_shape.dim(i);
            if (input_dim.has_dim_value()) {
              output_dim->set_dim_value(input_dim.dim_value());
            } else if (input_dim.has_dim_param()) {
              output_dim->set_dim_param(input_dim.dim_param());
            }
          }

          std::unordered_set<int64_t> unique_axes;
          size_t axes_size = axes.size();
          for (size_t axis_index = 0; axis_index < axes_size; ++axis_index) {
            auto axis = axes[axis_index] < 0 ? axes[axis_index] + static_cast<int64_t>(input_rank) : axes[axis_index];

            if (axis >= static_cast<int64_t>(input_rank) || axis < 0) {
              fail_shape_inference("Input axes has invalid data");
            }

            if (unique_axes.find(axis) != unique_axes.end()) {
              fail_shape_inference("'axes' has duplicates");
            }

            unique_axes.insert(axis);
            auto input_dim = ctx.getInputType(0)->tensor_type().shape().dim(axis);

            // input dim value is missing - cannot perform shape inference for
            // this axis
            if (!input_dim.has_dim_value())
              continue;

            const auto input_dim_value = input_dim.dim_value();

            // Empty dimension: clamp bounds are invalid when dimension size is 0,
            // so short-circuit to produce a zero-length output.
            if (input_dim_value == 0) {
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(0);
              continue;
            }

            // process step
            auto step = steps[axis_index];
            if (step == 0) {
              fail_shape_inference("'step' cannot be 0");
            }

            // process start
            auto start = starts[axis_index];
            if (start < 0)
              start += input_dim_value;
            if (step < 0)
              start = std::clamp(start, static_cast<int64_t>(0), input_dim_value - 1);
            else
              start = std::clamp(start, static_cast<int64_t>(0), input_dim_value);

            // process end
            auto end = ends[axis_index];
            if (end < static_cast<int64_t>(0))
              end += input_dim_value;
            if (step < static_cast<int64_t>(0))
              end = std::clamp(end, static_cast<int64_t>(-1), input_dim_value);
            else
              end = std::clamp(end, static_cast<int64_t>(0), input_dim_value);

            // find output dim value for this axis
            auto temp = static_cast<int64_t>(ceil(1.0 * (end - start) / step));
            if (temp < 0)
              temp = 0;

            // assign output value
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(temp);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Scatter,
    9,
    OpSchema()
        .SetDoc(kDoc_Scatter_ver9)
        .Attr(
            "axis",
            "Which axis to scatter on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1]",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(1, "indices", "Tensor of int32/int64 indices, of r >= 1 (same rank as input).", "Tind")
        .Input(2, "updates", "Tensor of rank r >=1 (same rank and shape as indices)", "T")
        .Output(0, "output", "Tensor of rank r >= 1 (same rank as input).", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Input and output types can be of any tensor type.")
        .TypeConstraint("Tind", {types::Int32, types::Int64}, "Constrain indices to integer types")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    DepthToSpace,
    1,
    OpSchema()
        .Attr("blocksize", "Blocks of [blocksize, blocksize] are moved.", AttributeProto::INT)
        .SetDoc(kDoc_DepthToSpace_ver1)
        .Input(
            0,
            "input",
            "Input tensor of [N,C,H,W], where N is the batch axis, C is the channel or depth"
            ", H is the height and W is the width.",
            "T")
        .Output(0, "output", "Output tensor of [N, C/(blocksize * blocksize), H * blocksize, W * blocksize].", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto blocksize = getAttribute(ctx, "blocksize", 0);
          if (blocksize <= 0) {
            fail_shape_inference("Blocksize must be positive");
          }
          const auto block_area = checkedMultiply(blocksize, blocksize);
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = getInputShape(ctx, 0);
            if (input_shape.dim_size() == 4) {
              updateOutputShape(
                  ctx,
                  0,
                  {input_shape.dim(0),
                   input_shape.dim(1) / block_area,
                   input_shape.dim(2) * blocksize,
                   input_shape.dim(3) * blocksize});
            } else {
              fail_shape_inference("Input tensor must be 4-dimensional");
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Gather,
    1,
    OpSchema()
        .SetDoc(kDoc_Gather_ver1)
        .Attr(
            "axis",
            "Which axis to gather on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1]",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "indices",
            "Tensor of int32/int64 indices, of any rank q. All index values are expected to be within bounds. "
            "It is an error if any of the index values are out of bounds.",
            "Tind")
        .Output(0, "output", "Tensor of rank q + (r - 1).", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to any tensor type.")
        .TypeConstraint("Tind", {types::Int32, types::Int64}, "Constrain indices to integer types")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 2)) {
            return;
          }
          const TensorShapeProto& data_shape = ctx.getInputType(0)->tensor_type().shape();
          const TensorShapeProto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
          int r = data_shape.dim_size();
          if (r < 1) {
            fail_shape_inference("data tensor must have rank >= 1");
          }
          int q = indices_shape.dim_size();
          int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
          if (axis < -r || axis >= r) {
            fail_shape_inference("axis must be in [-r, r-1]");
          }
          if (axis < 0) {
            axis += r;
          }
          int out_rank = q + r - 1;

          if (out_rank == 0) {
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          }
          for (int i = 0; i < out_rank; ++i) {
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = (i < axis) ? data_shape.dim(i)
                                                                                                  : // i < axis < r
                (i >= axis && i < axis + q) ? indices_shape.dim(i - axis)
                                            : // i - axis < q
                data_shape.dim(i - q + 1); // i < out_rank < q + r - 1
          }
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) { GatherOp13DataPropagator(ctx); }));

ONNX_OPERATOR_SET_SCHEMA(
    Squeeze,
    1,
    OpSchema()
        .Attr(
            "axes",
            "List of non-negative integers, indicate the dimensions to squeeze.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .SetDoc(kDoc_Squeeze_ver11)
        .Input(0, "data", "Tensors with at least max(dims) dimensions.", "T")
        .Output(0, "squeezed", "Reshaped tensor with same data as input.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          if (!ctx.getInputType(0)->tensor_type().has_shape()) {
            return;
          }

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_ndim = input_shape.dim_size();
          std::vector<int64_t> axes;
          if (!getRepeatedAttribute(ctx, "axes", axes)) {
            for (int i = 0; i < input_ndim; ++i) {
              if (!input_shape.dim(i).has_dim_value()) {
                return;
              }
              if (input_shape.dim(i).dim_value() == 1) {
                axes.push_back(i);
              }
            }
          }

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          for (int i = 0, j = 0; i < input_shape.dim_size(); ++i) {
            if (static_cast<size_t>(j) < axes.size() && axes[j] == i) {
              if (input_shape.dim(i).has_dim_value() && input_shape.dim(i).dim_value() != 1) {
                fail_shape_inference(
                    "Dimension of input ", i, " must be 1 instead of ", input_shape.dim(i).dim_value());
              }
              ++j;
            } else {
              *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = input_shape.dim(i);
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Unsqueeze,
    1,
    OpSchema()
        .Attr("axes", "List of non-negative integers, indicate the dimensions to be inserted", AttributeProto::INTS)
        .SetDoc(kDoc_Unsqueeze_ver1)
        .Input(0, "data", "Original tensor", "T")
        .Output(0, "expanded", "Reshaped tensor with same data as input.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          std::vector<int64_t> axes;
          if (!getRepeatedAttribute(ctx, "axes", axes)) {
            return;
          }
          std::sort(axes.begin(), axes.end());

          if (!ctx.getInputType(0)->tensor_type().has_shape()) {
            return;
          }

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

          int j = 0;
          for (int i = 0; i < ctx.getInputType(0)->tensor_type().shape().dim_size(); ++i) {
            while (static_cast<size_t>(j) < axes.size() &&
                   axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
              ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
              ++j;
            }
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() =
                ctx.getInputType(0)->tensor_type().shape().dim(i);
          }
          while (static_cast<size_t>(j) < axes.size() &&
                 axes[j] == ctx.getOutputType(0)->tensor_type().shape().dim_size()) {
            ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
            ++j;
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    OneHot,
    9,
    OpSchema()
        .SetDoc(kDoc_OneHot_ver9)
        .Attr(
            "axis",
            "(Optional) Axis along which one-hot representation in added. Default: axis=-1. "
            "axis=-1 means that the additional dimension will be inserted as the "
            "innermost/last dimension in the output tensor.",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Input(
            0,
            "indices",
            "Input tensor containing indices. The values must be non-negative integers. "
            "Any entries in the 'indices' input tensor with values outside the range [0, depth) "
            "will result in one-hot representation with all 'off_value' values in the output tensor."
            "In case 'indices' is of non-integer type, the values will be casted to int64 before use.",
            "T1")
        .Input(
            1,
            "depth",
            "Scalar or rank 1 tensor containing exactly one element, specifying the number of classes "
            "in one-hot tensor. This is also the size of the one-hot dimension (specified by 'axis' attribute) "
            "added on in the output tensor. The values in the 'indices' input tensor are expected to be "
            "in the range [0, depth). "
            "In case 'depth' is of non-integer type, it will be casted to int64 before use.",
            "T2")
        .Input(
            2,
            "values",
            "Rank 1 tensor containing exactly two elements, in the format [off_value, on_value], "
            "where 'on_value' is the value used for filling locations specified in 'indices' input "
            "tensor, and 'off_value' is the value used for filling locations other than those specified "
            "in 'indices' input tensor. ",
            "T3")
        .Output(
            0,
            "output",
            "Tensor of rank one greater than input tensor 'indices', i.e. rank(output) = rank(indices) + 1. "
            "The data type for the elements of the output tensor is the same as the type of input 'values' "
            "is used.",
            "T3")
        .TypeConstraint("T1", OpSchema::all_numeric_types(), "Constrain input to only numeric types.")
        .TypeConstraint("T2", OpSchema::all_numeric_types(), "Constrain input to only numeric types.")
        .TypeConstraint("T3", OpSchema::all_tensor_types(), "Constrain to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { oneHotShapeInference(ctx, 9); }));

ONNX_OPERATOR_SET_SCHEMA(
    Compress,
    9,
    OpSchema()
        .SetDoc(kDoc_Compress_ver9)
        .Attr(
            "axis",
            "(Optional) Axis along which to take slices. If not specified, "
            "input is flattened before elements being selected.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Input(0, "input", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "condition",
            "Rank 1 tensor of booleans to indicate which slices or data elements to be selected. "
            "Its length can be less than the input length alone the axis "
            "or the flattened input size if axis is not specified. "
            "In such cases data slices or elements exceeding the condition length are discarded.",
            "T1")
        .Output(0, "output", "Tensor of rank r if axis is specified. Otherwise output is a Tensor of rank 1.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeConstraint("T1", {types::Bool}, "Constrain to boolean tensors."));

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    2,
    OpSchema()
        .Input(0, "input", "The tensor to split", "T")
        .Output(0, "outputs", "One or more outputs forming list of tensors after splitting", "T", OpSchema::Variadic)
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .Attr("axis", "Which axis to split on. ", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("split", "length of each output", AttributeProto::INTS, OPTIONAL_VALUE)
        .SetDoc(kDoc_Split_ver11)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getNumOutputs() == 0) {
            fail_shape_inference("Split must have at least one output");
          }
          for (int i = 0; i < static_cast<int>(ctx.getNumOutputs()); ++i) {
            propagateElemTypeFromInputToOutput(ctx, 0, i);
          }
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          const auto& shape = ctx.getInputType(0)->tensor_type().shape();
          int rank = shape.dim_size();
          int axis = static_cast<int>(getAttribute(ctx, "axis", 0));
          if (axis < -rank || axis >= rank) {
            fail_type_inference("Invalid value of attribute 'axis'. Rank=", rank, " Value=", axis);
          }
          // Previously Split-2 does not mention how to deal with negative axis
          // However, there is an existing test onnx/backend/test/data/pytorch-converted/test_GLU
          // using Split-2 with negative axis and it is hard to be regenerated.
          // To compromise, handle negative axis for Split-2 here.
          if (axis < 0) {
            axis += rank;
          }
          const auto& split_dim = shape.dim(axis);
          if (!split_dim.has_dim_value()) {
            for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
              *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
              ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->Clear();
            }
            return;
          }
          int split_dim_value = static_cast<int>(split_dim.dim_value());

          std::vector<int64_t> split;
          if (getRepeatedAttribute(ctx, "split", split)) {
            if (split.size() != ctx.getNumOutputs()) {
              fail_shape_inference(
                  "Mismatch between number of splits (", split.size(), ") and outputs (", ctx.getNumOutputs(), ")");
            }
            int64_t total_dim = 0;
            for (int64_t d : split) {
              total_dim += d;
            }
            if (total_dim != split_dim_value) {
              fail_shape_inference(
                  "Mismatch between the sum of 'split' (",
                  total_dim,
                  ") and the split dimension of the input (",
                  split_dim_value,
                  ")");
            }
          } else {
            int num_outputs = static_cast<int>(ctx.getNumOutputs());
            if (split_dim_value % num_outputs != 0) {
              fail_shape_inference("The input is not evenly splittable");
            }
            int chunk_size = split_dim_value / num_outputs;
            for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
              split.push_back(chunk_size);
            }
          }
          for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
            *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
            ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(split[i]);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    2,
    OpSchema()
        .Attr(
            "pads",
            "List of integers indicating the number of padding elements to add or remove (if negative) "
            "at the beginning and end of each axis. For 2D it is the number of pixels. "
            "`pads` rank should be double of the input's rank. `pads` format should be as follow "
            "[x1_begin, x2_begin...x1_end, x2_end,...], where xi_begin the number of pixels "
            "added at the beginning of axis `i` and xi_end, the number of pixels added at "
            "the end of axis `i`.",
            AttributeProto::INTS)
        .Attr("mode", "Three modes: constant(default), reflect, edge", AttributeProto::STRING, std::string("constant"))
        .Attr("value", "One float, indicates the value to be filled.", AttributeProto::FLOAT, 0.0f)
        .SetDoc(kDoc_Pad_ver2)
        .Input(0, "data", "Input tensor.", "T")
        .Output(0, "output", "Tensor after padding.", "T")
        .TypeConstraint(
            "T",
            {types::Float16, types::Float, types::Double},
            "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          auto& input_shape = ctx.getInputType(0)->tensor_type().shape();

          std::vector<int64_t> pads;
          if (!getRepeatedAttribute(ctx, "pads", pads)) {
            fail_shape_inference("Attribute value for pads is required");
          }
          if (pads.size() != static_cast<size_t>(input_shape.dim_size() * 2)) {
            fail_shape_inference("Attribute pads has incorrect length");
            ;
          }

          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          for (int i = 0; i < input_shape.dim_size(); ++i) {
            auto newdim = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim();
            const auto total_pad = checkedAdd(pads[i], pads[input_shape.dim_size() + i]);
            if (ctx.getInputType(0)->tensor_type().shape().dim(i).has_dim_value()) {
              newdim->set_dim_value(
                  checkedAdd(ctx.getInputType(0)->tensor_type().shape().dim(i).dim_value(), total_pad));
            } else if (total_pad == 0) {
              *newdim = input_shape.dim(i);
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    GatherND,
    11,
    OpSchema()
        .SetDoc(kDoc_GatherND_ver11)
        .Input(0, "data", "Tensor of rank r >= 1.", "T")
        .Input(
            1,
            "indices",
            "Tensor of rank q >= 1. All index values are expected to be within bounds [-s, s-1] "
            "along axis of size s. It is an error if any of the index values are out of bounds.",
            "tensor(int64)")
        .Output(0, "output", "Tensor of rank q + r - indices_shape[-1] - 1.", "T")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Shape inference
          if (!hasNInputShapes(ctx, 2)) {
            // cannot proceed with shape or rank inference
            return;
          }

          const auto& data_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto data_rank = data_shape.dim_size();

          const auto& indices_shape = ctx.getInputType(1)->tensor_type().shape();
          const auto indices_rank = indices_shape.dim_size();

          if (data_rank < 1 || indices_rank < 1) {
            fail_shape_inference(
                "Both `data` and `indices` input tensors in GatherND op "
                "need to have rank larger than 0.");
          }

          // cannot ascertain if the input shapes are valid if shape of
          // `indices` is missing last dimension value so return at this point
          if (!indices_shape.dim(indices_rank - 1).has_dim_value()) {
            return;
          }

          const auto last_index_dimension = indices_shape.dim(indices_rank - 1).dim_value();

          if (last_index_dimension > data_rank) {
            fail_shape_inference(
                "Last dimension of `indices` input tensor in GatherND op "
                "must not be larger than the rank of `data` tensor");
          }

          for (int i = 0; i < indices_rank - 1; ++i) {
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = indices_shape.dim(i);
          }

          for (int i = static_cast<int>(last_index_dimension); i < data_rank; ++i) {
            *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape()->add_dim() = data_shape.dim(i);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    14,
    OpSchema()
        .SetDoc(kDoc_Identity_ver25)
        .Input(0, "input", "Input tensor", "V", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "output", "Tensor to copy input into.", "V", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "V",
            []() {
              auto t = OpSchema::all_tensor_types_ir4();
              auto s = OpSchema::all_tensor_sequence_types();
              t.insert(t.end(), s.begin(), s.end());
              return t;
            }(),
            "Constrain input and output types to all tensor and sequence types.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Where,
    9,
    OpSchema()
        .SetDoc(kDoc_Where_ver9 + GenerateBroadcastingDocMul())
        .Input(
            0,
            "condition",
            "When True (nonzero), yield X, otherwise yield Y",
            "B",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            1,
            "X",
            "values selected at indices where condition is True",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            2,
            "Y",
            "values selected at indices where condition is False",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Output(
            0,
            "output",
            "Tensor of shape equal to the broadcasted shape of condition, X, and Y.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("B", {types::Bool}, "Constrain to boolean tensors.")
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 1, 0);
          if (hasNInputShapes(ctx, 3)) {
            std::vector<const TensorShapeProto*> shapes;
            shapes.push_back(&ctx.getInputType(0)->tensor_type().shape());
            shapes.push_back(&ctx.getInputType(1)->tensor_type().shape());
            shapes.push_back(&ctx.getInputType(2)->tensor_type().shape());
            multidirectionalBroadcastShapeInference(
                shapes, *ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape());
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    13,
    OpSchema()
        .Attr(
            "mode",
            "Supported modes: `constant`(default), `reflect`, `edge`",
            AttributeProto::STRING,
            std::string("constant"))
        .SetDoc(kDoc_Pad_ver13)
        .Input(0, "data", "Input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "pads",
            "Tensor of integers indicating the number of padding elements to add or remove (if negative) "
            "at the beginning and end of each axis. For 2D input tensor, it is the number of pixels. "
            "`pads` should be a 1D tensor of shape [2 * input_rank]. "
            "`pads` format should be: [x1_begin, x2_begin,...,x1_end, x2_end,...], "
            "where xi_begin is the number of pad values added at the beginning of axis `i` and "
            "xi_end, the number of pad values added at the end of axis `i`.",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "constant_value",
            "(Optional) A scalar value to be used if the mode chosen is `constant` (by default it is 0, "
            "empty string or False).",
            "T",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "output", "Tensor after padding.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          // Shape inference needs the input data shape
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const auto input_rank = input_shape.dim_size();

          // Infer output shape if 'pads' tensor is available
          const auto pads_initializer = ctx.getInputData(1);
          if (nullptr != pads_initializer) {
            if (pads_initializer->dims_size() != 1 || pads_initializer->data_type() != TensorProto::INT64) {
              fail_shape_inference("'pads' input must be a 1D (shape: [2 * input_rank]) tensor of type int64");
            }

            const auto pads_data = ParseData<int64_t>(pads_initializer);
            if (pads_data.size() != static_cast<size_t>(2 * input_rank)) {
              fail_shape_inference("Pads has incorrect number of values");
            }

            auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
            for (int i = 0; i < input_rank; ++i) {
              const auto& input_dim = input_shape.dim(i);
              auto output_dim = output_shape->add_dim();
              const auto total_pad = checkedAdd(pads_data[i], pads_data[i + input_rank]);
              if (input_dim.has_dim_value()) {
                output_dim->set_dim_value(checkedAdd(input_dim.dim_value(), total_pad));
              } else if (total_pad == 0) {
                *output_dim = input_dim;
              }
            }
          } else {
            // Infer output shapes' rank in any case
            auto output_shape_0 = getOutputShape(ctx, 0);
            for (int i = 0; i < input_rank; ++i) {
              output_shape_0->add_dim();
            }
          }
          return;
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Pad,
    18,
    OpSchema().FillUsing(PadDocGenerator(kDoc_Pad_ver18, "Supported modes: `constant`(default), `reflect`, `edge`")));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    16,
    OpSchema()
        .SetDoc(kDoc_Identity_ver25)
        .Input(0, "input", "Input tensor", "V", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "output", "Tensor to copy input into.", "V", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "V",
            []() {
              auto t = OpSchema::all_tensor_types_ir4();
              auto s = OpSchema::all_tensor_sequence_types();
              auto o = OpSchema::all_optional_types();
              t.insert(t.end(), s.begin(), s.end());
              t.insert(t.end(), o.begin(), o.end());
              return t;
            }(),
            "Constrain input and output types to all tensor, sequence, and optional types.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    14,
    OpSchema()
        .SetDoc(kDoc_Reshape_ver24)
        .Attr(
            "allowzero",
            "(Optional) By default, when any value in the 'shape' input is equal to zero "
            "the corresponding dimension value is copied from the input tensor dynamically. "
            "allowzero=1 indicates that if any value in the 'shape' input is set to zero, "
            "the zero value is honored, similar to NumPy.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "shape",
            "Specified shape for output.",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "reshaped", "Reshaped data.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          bool found;
          TensorShapeProto targetShapeProto = getShapeInput(ctx, 1, found);
          if (!found) {
            return;
          }

          int allowzero = static_cast<int>(getAttribute(ctx, "allowzero", 0));

          // Iterate through targetShape, adding dimensions in the outputShape
          // TensorProto. If the targetShape dimension is -1, we do not set the
          // dimension value in this iteration, but we record the Dimension. If
          // targetShape dimension is 0, we attempt to propagate the dimension
          // value/param. If the value cannot be inferred, we set the flag in
          // the unresolveZeros vector. If targetShape dimension is positive, we
          // set the dimension value in the outputShape. We track the product of
          // the dimensions we are setting outputShape in the outputProduct
          // variable. The outputProduct will potentially be used for inferring
          // a dimension marked -1.
          auto outputShape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          TensorShapeProto::Dimension* negativeOneDim = nullptr;
          const auto& dataInputTensorType = ctx.getInputType(0)->tensor_type();
          std::vector<bool> unresolvedZeros(targetShapeProto.dim_size(), false);
          int64_t outputProduct = 1;
          bool outputProductValid = true;
          for (int i = 0; i < static_cast<int>(targetShapeProto.dim_size()); ++i) {
            // Add a new dimension to outputShape
            auto new_dim = outputShape->add_dim();
            if (targetShapeProto.dim(i).has_dim_param()) {
              // There is a tricky edge case here. It is possible that the value of
              // symbolic dim can be -1 or 0 at runtime. In that case simply propagating this
              // symbol can be erroneous. This should be a very rare scenario and in such a
              // case an option is to turn off data propagation during shape inference.
              new_dim->set_dim_param(targetShapeProto.dim(i).dim_param());
              outputProductValid = false;
            } else {
              if (!targetShapeProto.dim(i).has_dim_value()) {
                outputProductValid = false;
                // treat this dim as unknown dim
                continue;
              }

              const auto dim_value = targetShapeProto.dim(i).dim_value();

              if (dim_value == -1) {
                // Check if multiple -1's. If not, set negativeOneDim, marking
                // this dimension to potentially be filled in later.
                if (negativeOneDim) {
                  fail_shape_inference("Target shape may not have multiple -1 dimensions.");
                }
                negativeOneDim = new_dim;
              } else if (dim_value == 0) {
                // Check if data input has a shape and if the index i is within
                // its bounds. If these conditions are satisfied, any dimension
                // value/param should be propagated. If dimension value cannot be
                // inferred, set the corresponding  unresolvedZeros flag to true.
                // If allowzero is set however, do not propagate values, since output
                // dimension is explicitly zero.
                if (allowzero == 0) {
                  unresolvedZeros[i] = true;
                  if (dataInputTensorType.has_shape()) {
                    if (i >= dataInputTensorType.shape().dim_size()) {
                      fail_shape_inference("Invalid position of 0.");
                    }
                    if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                      const auto& input_dim_value = dataInputTensorType.shape().dim(i).dim_value();
                      new_dim->set_dim_value(input_dim_value);
                      checked_mul_into(outputProduct, input_dim_value);
                      unresolvedZeros[i] = false;
                    } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
                      new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
                    }
                  }
                } else {
                  new_dim->set_dim_value(dim_value);
                  checked_mul_into(outputProduct, dim_value);
                }
              } else if (dim_value > 0) {
                // Set the dimension value to dim_value
                new_dim->set_dim_value(dim_value);
                checked_mul_into(outputProduct, dim_value);
              } else {
                // Check if value is less than -1; fail if so
                fail_shape_inference("Invalid dimension value: ", dim_value);
              }
            }
          }
          // If negativeOneDim has been set, we attempt to infer its value. This
          // can be done if all dimension values for the data input tensor shape
          // are known other than the ones corresponding to unresolvedZeros
          // flags.
          if (negativeOneDim && outputProductValid) {
            // First, attempt to compute product of data input shape dimensions
            // that are not marked by unresolvedZeros. If not possible, set the
            // inputProductValid flag to false.
            if (!outputProduct) {
              fail_shape_inference("Invalid Target shape product of 0. Product cannot be 0 in combination with -1");
            }
            int64_t inputProduct = 1;
            bool inputProductValid = true;
            if (!dataInputTensorType.has_shape()) {
              inputProductValid = false;
            } else {
              for (int i = 0; i < dataInputTensorType.shape().dim_size(); ++i) {
                if (dataInputTensorType.shape().dim(i).has_dim_value()) {
                  checked_mul_into(inputProduct, dataInputTensorType.shape().dim(i).dim_value());
                } else if (i >= static_cast<int>(unresolvedZeros.size()) || !unresolvedZeros[i]) {
                  inputProductValid = false;
                  break;
                }
              }
            }
            if (inputProductValid) {
              if (inputProduct % outputProduct != 0) {
                fail_shape_inference("Dimension could not be inferred: incompatible shapes");
              }
              negativeOneDim->set_dim_value(inputProduct / outputProduct);
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    21,
    OpSchema()
        .SetDoc(kDoc_Shape_ver24)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "shape", "Shape of the input tensor", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Attr(
            "start",
            "(Optional) Starting axis for slicing the shape. Default value is 0."
            "Negative value means counting dimensions from the back.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "end",
            "(Optional) Ending axis for slicing the shape. "
            "Negative value means counting dimensions from the back. "
            "If omitted, sizes of all axes upto (including) the last one will be included.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir10(), "Input tensor can be of arbitrary type.")
        .TypeConstraint("T1", {types::Int64}, "Constrain output to int64 tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          auto output_length = output_shape->add_dim();

          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
          int64_t start = getAttribute(ctx, "start", 0);
          if (start < 0)
            start += rank;
          start = (start < 0) ? 0 : (start > rank) ? rank : start;
          int64_t end = getAttribute(ctx, "end", rank);
          if (end < 0)
            end += rank;
          end = (end < 0) ? 0 : (end > rank) ? rank : end;
          output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            int64_t rank = static_cast<int64_t>(input_shape.dim_size());
            int64_t start = getAttribute(ctx, "start", 0);
            if (start < 0)
              start += rank;
            start = (start < 0) ? 0 : (start > rank) ? rank : start;
            int64_t end = getAttribute(ctx, "end", rank);
            if (end < 0)
              end += rank;
            end = (end < 0) ? 0 : (end > rank) ? rank : end;
            TensorShapeProto output_shape;
            for (int64_t d = start; d < end; ++d) {
              *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
            }
            ctx.addOutputData(0, std::move(output_shape));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    19,
    OpSchema()
        .SetDoc(kDoc_Shape_ver24)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "shape", "Shape of the input tensor", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Attr(
            "start",
            "(Optional) Starting axis for slicing the shape. Default value is 0."
            "Negative value means counting dimensions from the back.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "end",
            "(Optional) Ending axis for slicing the shape. "
            "Negative value means counting dimensions from the back. "
            "If omitted, sizes of all axes upto (including) the last one will be included.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir9(), "Input tensor can be of arbitrary type.")
        .TypeConstraint("T1", {types::Int64}, "Constrain output to int64 tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          auto output_length = output_shape->add_dim();

          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
          int64_t start = getAttribute(ctx, "start", 0);
          if (start < 0)
            start += rank;
          start = (start < 0) ? 0 : (start > rank) ? rank : start;
          int64_t end = getAttribute(ctx, "end", rank);
          if (end < 0)
            end += rank;
          end = (end < 0) ? 0 : (end > rank) ? rank : end;
          output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            int64_t rank = static_cast<int64_t>(input_shape.dim_size());
            int64_t start = getAttribute(ctx, "start", 0);
            if (start < 0)
              start += rank;
            start = (start < 0) ? 0 : (start > rank) ? rank : start;
            int64_t end = getAttribute(ctx, "end", rank);
            if (end < 0)
              end += rank;
            end = (end < 0) ? 0 : (end > rank) ? rank : end;
            TensorShapeProto output_shape;
            for (int64_t d = start; d < end; ++d) {
              *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
            }
            ctx.addOutputData(0, std::move(output_shape));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Shape,
    15,
    OpSchema()
        .SetDoc(kDoc_Shape_ver24)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "shape", "Shape of the input tensor", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Attr(
            "start",
            "(Optional) Starting axis for slicing the shape. Default value is 0."
            "Negative value means counting dimensions from the back.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "end",
            "(Optional) Ending axis for slicing the shape. "
            "Negative value means counting dimensions from the back. "
            "If omitted, sizes of all axes upto (including) the last one will be included.",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Input tensor can be of arbitrary type.")
        .TypeConstraint("T1", {types::Int64}, "Constrain output to int64 tensor.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          auto output_length = output_shape->add_dim();

          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          int64_t rank = static_cast<int64_t>(ctx.getInputType(0)->tensor_type().shape().dim_size());
          int64_t start = getAttribute(ctx, "start", 0);
          if (start < 0)
            start += rank;
          start = (start < 0) ? 0 : (start > rank) ? rank : start;
          int64_t end = getAttribute(ctx, "end", rank);
          if (end < 0)
            end += rank;
          end = (end < 0) ? 0 : (end > rank) ? rank : end;
          output_length->set_dim_value((end - start) < 0 ? 0 : (end - start));
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          if (hasInputShape(ctx, 0)) {
            auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            int64_t rank = static_cast<int64_t>(input_shape.dim_size());
            int64_t start = getAttribute(ctx, "start", 0);
            if (start < 0)
              start += rank;
            start = (start < 0) ? 0 : (start > rank) ? rank : start;
            int64_t end = getAttribute(ctx, "end", rank);
            if (end < 0)
              end += rank;
            end = (end < 0) ? 0 : (end > rank) ? rank : end;
            TensorShapeProto output_shape;
            for (int64_t d = start; d < end; ++d) {
              *output_shape.add_dim() = input_shape.dim(static_cast<int>(d));
            }
            ctx.addOutputData(0, std::move(output_shape));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    21,
    OpSchema()
        .SetDoc(kDoc_Size_ver24)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(
            0,
            "size",
            "Total number of elements of the input tensor",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir10(), "Input tensor can be of arbitrary type.")
        .TypeConstraint("T1", {types::Int64}, "Constrain output to int64 tensor, which should be a scalar though.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          const auto input_data = ctx.getInputData(0);
          if (input_data != nullptr) {
            TensorShapeProto tsp;
            tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
            ctx.addOutputData(0, std::move(tsp));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    19,
    OpSchema()
        .SetDoc(kDoc_Size_ver24)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(
            0,
            "size",
            "Total number of elements of the input tensor",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir9(), "Input tensor can be of arbitrary type.")
        .TypeConstraint("T1", {types::Int64}, "Constrain output to int64 tensor, which should be a scalar though.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          const auto input_data = ctx.getInputData(0);
          if (input_data != nullptr) {
            TensorShapeProto tsp;
            tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
            ctx.addOutputData(0, std::move(tsp));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Size,
    13,
    OpSchema()
        .SetDoc(kDoc_Size_ver24)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(
            0,
            "size",
            "Total number of elements of the input tensor",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Input tensor can be of arbitrary type.")
        .TypeConstraint("T1", {types::Int64}, "Constrain output to int64 tensor, which should be a scalar though.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          ctx.getOutputType(0)->mutable_tensor_type()->set_elem_type(TensorProto::INT64);
          ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          const auto input_data = ctx.getInputData(0);
          if (input_data != nullptr) {
            TensorShapeProto tsp;
            tsp.mutable_dim()->Add()->set_dim_value(input_data->dim_size());
            ctx.addOutputData(0, std::move(tsp));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Compress,
    11,
    OpSchema()
        .SetDoc(kDoc_Compress_ver9)
        .Attr(
            "axis",
            "(Optional) Axis along which to take slices. If not specified, "
            "input is flattened before elements being selected. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Input(0, "input", "Tensor of rank r >= 1.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "condition",
            "Rank 1 tensor of booleans to indicate which slices or data elements to be selected. "
            "Its length can be less than the input length along the axis "
            "or the flattened input size if axis is not specified. "
            "In such cases data slices or elements exceeding the condition length are discarded.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Tensor of rank r if axis is specified. Otherwise output is a Tensor of rank 1.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input and output types to all tensor types.")
        .TypeConstraint("T1", {types::Bool}, "Constrain to boolean tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          auto axisAttr = ctx.getAttribute("axis");
          if (hasInputShape(ctx, 0)) {
            const TensorShapeProto& indices_shape = ctx.getInputType(0)->tensor_type().shape();
            int r = indices_shape.dim_size();
            if (r < 1) {
              fail_shape_inference("Indices tensor must have rank >= 1");
            }
            if (axisAttr) {
              int axis = static_cast<int>(axisAttr->i());
              if (axis < -r || axis >= r) {
                fail_shape_inference("'axis' must be in [-rank(indices), rank(indices)-1]");
              }
              if (axis < 0) {
                axis += r;
              }
              TensorShapeProto* shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
              for (int i = 0; i < indices_shape.dim_size(); i++) {
                auto dim = shape->add_dim();
                if (i != axis) {
                  *dim = indices_shape.dim(i);
                }
              }
            }
          }
          if (!axisAttr) {
            updateOutputShape(ctx, 0, {Dim()});
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    OneHot,
    11,
    OpSchema()
        .SetDoc(kDoc_OneHot_ver11)
        .Attr(
            "axis",
            "(Optional) Axis along which one-hot representation in added. Default: axis=-1. "
            "axis=-1 means that the additional dimension will be inserted as the "
            "innermost/last dimension in the output tensor. Negative value means counting dimensions "
            "from the back. Accepted range is [-r-1, r] where r = rank(indices).",
            AttributeProto::INT,
            static_cast<int64_t>(-1))
        .Input(
            0,
            "indices",
            "Input tensor containing indices. Any entries in the 'indices' input tensor with "
            "values outside the range [-depth, depth-1] will result in one-hot representation with all "
            "'off_value' values in the output tensor."
            "In case 'indices' is of non-integer type, the values will be casted to int64 before use.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            1,
            "depth",
            "Scalar or Rank 1 tensor containing exactly one element, specifying the number of classes "
            "in one-hot tensor. This is also the size of the one-hot dimension (specified by 'axis' attribute) "
            "added on in the output tensor. The values in the 'indices' input tensor are expected to be "
            "in the range [-depth, depth-1]. "
            "In case 'depth' is of non-integer type, it will be casted to int64 before use.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "values",
            "Rank 1 tensor containing exactly two elements, in the format [off_value, on_value], "
            "where 'on_value' is the value used for filling locations specified in 'indices' input "
            "tensor, and 'off_value' is the value used for filling locations other than those specified "
            "in 'indices' input tensor. ",
            "T3",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Tensor of rank one greater than input tensor 'indices', i.e. rank(output) = rank(indices) + 1. "
            "The data type for the elements of the output tensor is the same as the type of input 'values' "
            "is used.",
            "T3",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint("T1", OpSchema::all_numeric_types(), "Constrain input to only numeric types.")
        .TypeConstraint("T2", OpSchema::all_numeric_types(), "Constrain input to only numeric types.")
        .TypeConstraint("T3", OpSchema::all_tensor_types(), "Constrain to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { oneHotShapeInference(ctx, 11); }));

ONNX_OPERATOR_SET_SCHEMA(
    ReverseSequence,
    10,
    OpSchema()
        .SetDoc(kDoc_ReverseSequence_ver10)
        .Attr(
            "time_axis",
            "(Optional) Specify which axis is time axis. Must be one of 0 (default), or 1.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "batch_axis",
            "(Optional) Specify which axis is batch axis. Must be one of 1 (default), or 0.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Input(0, "input", "Tensor of rank r >= 2.", "T", OpSchema::Single)
        .Input(
            1,
            "sequence_lens",
            "Tensor specifying lengths of the sequences in a batch. It has shape `[batch_size]`.",
            "tensor(int64)",
            OpSchema::Single)
        .Output(0, "Y", "Tensor with same shape of input.", "T", OpSchema::Single)
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Input and output types can be of any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 2)) {
            return;
          }

          auto& first_input_shape = getInputShape(ctx, 0);
          if (first_input_shape.dim_size() < 2) {
            fail_shape_inference("'input' must have rank >= 2");
          }
          auto& seq_len_input_shape = getInputShape(ctx, 1);
          if (seq_len_input_shape.dim_size() != 1) {
            fail_shape_inference("'sequence_lens' must have rank of 1");
          }

          propagateShapeFromInputToOutput(ctx, 0, 0);
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Unique,
    11,
    OpSchema()
        .SetDoc(kDoc_Unique_ver11)
        .Attr(
            "sorted",
            "(Optional) Whether to sort the unique elements in ascending order before returning as output. "
            "Must be one of 0, or 1 (default).",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Attr(
            "axis",
            "(Optional) The dimension to apply unique. If not specified, the unique elements of the "
            "flattened input are returned. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(input).",
            AttributeProto::INT,
            OPTIONAL_VALUE)
        .Input(
            0,
            "X",
            "A N-D input tensor that is to be processed.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "Y",
            "A tensor of the same type as 'X' "
            "containing all the unique values or subtensors sliced along a provided 'axis' in 'X', either sorted "
            "or maintained in the same order they occur in input 'X'",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            1,
            "indices",
            "A 1-D INT64 tensor "
            "containing indices of 'Y' elements' first occurrence in 'X'. "
            "When 'axis' is provided, it contains indices to subtensors in input 'X' on the 'axis'. "
            "When 'axis' is not provided, it contains indices to values in the flattened input tensor. ",
            "tensor(int64)",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            2,
            "inverse_indices",
            "A 1-D INT64 tensor "
            "containing, for elements of 'X', its corresponding indices in 'Y'. "
            "When 'axis' is provided, it contains indices to subtensors in output 'Y' on the 'axis'. "
            "When 'axis' is not provided, it contains indices to values in output 'Y'. ",
            "tensor(int64)",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            3,
            "counts",
            "A 1-D INT64 tensor containing "
            "the count of each element "
            "of 'Y' in input 'X'",
            "tensor(int64)",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Input can be of any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          const TypeProto* xTensorProto = ctx.getInputType(0);
          TypeProto* yTensorProto = ctx.getOutputType(0);
          TypeProto* indicesTensorProto = nullptr;
          TypeProto* inverseIndicesTensorProto = nullptr;
          TypeProto* countsTensorProto = nullptr;

          auto num_outputs = ctx.getNumOutputs();
          if (num_outputs >= 2) {
            indicesTensorProto = ctx.getOutputType(1);
            updateOutputElemType(ctx, 1, TensorProto::INT64);
            indicesTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
          }

          if (num_outputs >= 3) {
            inverseIndicesTensorProto = ctx.getOutputType(2);
            updateOutputElemType(ctx, 2, TensorProto::INT64);
            inverseIndicesTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
          }

          if (num_outputs >= 4) {
            countsTensorProto = ctx.getOutputType(3);
            updateOutputElemType(ctx, 3, TensorProto::INT64);
            countsTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
          }

          auto axisAttr = ctx.getAttribute("axis");
          if (!axisAttr) {
            yTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
          } else {
            int axis = static_cast<int>(axisAttr->i());
            if (!xTensorProto->tensor_type().has_shape()) {
              return;
            }
            const TensorShapeProto& input_shape = xTensorProto->tensor_type().shape();
            int rank = input_shape.dim_size();
            if (axis < 0)
              axis += rank;
            if (axis < 0 || axis >= rank) {
              fail_shape_inference("Invalid value for attribute axis");
            }
            for (int i = 0; i < input_shape.dim_size(); i++) {
              auto dim = yTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
              if (i != axis) {
                *dim = input_shape.dim(i);
              }
            }
          }
        }));

} // namespace ONNX_NAMESPACE
