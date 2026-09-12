// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "onnx/common/safe_math.h"
#include "onnx/defs/data_propagators.h"
#include "onnx/defs/doc_strings.h"
#include "onnx/defs/function.h"
#include "onnx/defs/tensor/utils.h"
#include "onnx/defs/tensor_proto_util.h"
#include "onnx/defs/type_builders.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    Cast,
    28,
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
            OpSchema::all_non_complex_tensor_types_ir14(),
            "Constrain input types. Casting from complex is not supported.")
        .TypeConstraint(
            "T2",
            OpSchema::all_non_complex_tensor_types_ir14(),
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
    25,
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
            OpSchema::all_non_complex_tensor_types_ir13(),
            "Constrain input types. Casting from complex is not supported.")
        .TypeConstraint(
            "T2",
            OpSchema::all_non_complex_tensor_types_ir13(),
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
    BitCast,
    26,
    OpSchema()
        .SetDoc(kDoc_BitCast_ver26)
        .Attr(
            "to",
            "The data type to which the input tensor is bitwise reinterpreted. "
            "Must be one of the non-string types from DataType enum in TensorProto. "
            "The target type must have the same bit-width as the input type.",
            AttributeProto::INT)
        .Input(0, "input", "Input tensor to be bitcast.", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Output tensor with the same shape as the input.",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .TypeConstraint(
            "T1",
            OpSchema::all_non_string_tensor_types_ir13(),
            "Constrain input types. Bitcasting from string is not supported.")
        .TypeConstraint(
            "T2",
            OpSchema::all_non_string_tensor_types_ir13(),
            "Constrain output types. Bitcasting to string is not supported.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          auto get_bit_size = [](int element_type) -> int {
            switch (element_type) {
              case TensorProto::FLOAT:
              case TensorProto::INT32:
              case TensorProto::UINT32:
                return 32;
              case TensorProto::DOUBLE:
              case TensorProto::INT64:
              case TensorProto::UINT64:
              case TensorProto::COMPLEX64:
                return 64;
              case TensorProto::COMPLEX128:
                return 128;
              case TensorProto::FLOAT16:
              case TensorProto::BFLOAT16:
              case TensorProto::INT16:
              case TensorProto::UINT16:
                return 16;
              case TensorProto::INT8:
              case TensorProto::UINT8:
              case TensorProto::BOOL:
              case TensorProto::FLOAT8E4M3FN:
              case TensorProto::FLOAT8E4M3FNUZ:
              case TensorProto::FLOAT8E5M2:
              case TensorProto::FLOAT8E5M2FNUZ:
              case TensorProto::FLOAT8E8M0:
                return 8;
              case TensorProto::INT4:
              case TensorProto::UINT4:
              case TensorProto::FLOAT4E2M1:
                return 4;
              case TensorProto::INT2:
              case TensorProto::UINT2:
                return 2;
              default:
                return 0;
            }
          };

          // Validate that input and output types have the same bit-width
          auto input_type = ctx.getInputType(0);
          if (input_type && input_type->has_tensor_type()) {
            auto input_element_type = input_type->tensor_type().elem_type();
            auto* to_attr = ctx.getAttribute("to");
            if (to_attr) {
              auto output_element_type = static_cast<int32_t>(to_attr->i());
              int input_element_bit_size = get_bit_size(input_element_type);
              int output_element_bit_size = get_bit_size(output_element_type);
              if (input_element_bit_size != 0 && output_element_bit_size != 0 &&
                  input_element_bit_size != output_element_bit_size) {
                fail_shape_inference(
                    "BitCast requires input and output types to have the same bit-width, but input type has ",
                    input_element_bit_size,
                    " bits and output type has ",
                    output_element_bit_size,
                    " bits.");
              }
            }
          }

          propagateElemTypeFromAttributeToOutput(ctx, "to", 0);

          // Same bit-width: output shape equals input shape
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        })
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic));

ONNX_OPERATOR_SET_SCHEMA(
    Reshape,
    25,
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
        .TypeConstraint("T", OpSchema::all_tensor_types_ir13(), "Constrain input and output types to all tensor types.")
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
                      if (checked_mul_overflow(outputProduct, input_dim_value, &outputProduct)) {
                        fail_shape_inference("Dimension product overflow in Reshape");
                      }
                      unresolvedZeros[i] = false;
                    } else if (dataInputTensorType.shape().dim(i).has_dim_param()) {
                      new_dim->set_dim_param(dataInputTensorType.shape().dim(i).dim_param());
                    }
                  }
                } else {
                  new_dim->set_dim_value(dim_value);
                  if (checked_mul_overflow(outputProduct, dim_value, &outputProduct)) {
                    fail_shape_inference("Dimension product overflow in Reshape");
                  }
                }
              } else if (dim_value > 0) {
                // Set the dimension value to dim_value
                new_dim->set_dim_value(dim_value);
                if (checked_mul_overflow(outputProduct, dim_value, &outputProduct)) {
                  fail_shape_inference("Dimension product overflow in Reshape");
                }
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
                  if (checked_mul_overflow(
                          inputProduct, dataInputTensorType.shape().dim(i).dim_value(), &inputProduct)) {
                    fail_shape_inference("Dimension product overflow in Reshape");
                  }
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
    25,
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
        .TypeConstraint("T", OpSchema::all_tensor_types_ir13(), "Input tensor can be of arbitrary type.")
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
    25,
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
        .TypeConstraint("T", OpSchema::all_tensor_types_ir13(), "Input tensor can be of arbitrary type.")
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
    Concat,
    13,
    OpSchema()
        .Attr(
            "axis",
            "Which axis to concat on. A negative value means counting dimensions from the back. "
            "Accepted range is [-r, r-1] where r = rank(inputs)..",
            AttributeProto::INT)
        .SetDoc(kDoc_Concat_ver13)
        .Input(
            0,
            "inputs",
            "List of tensors for concatenation",
            "T",
            OpSchema::Variadic,
            true,
            1,
            OpSchema::Differentiable)
        .Output(0, "concat_result", "Concatenated tensor", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain output types to any tensor type.")
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
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          if (!axisIsZero(ctx)) {
            return;
          }
          TensorShapeProto tsp;
          for (size_t i = 0; i < ctx.getNumInputs(); ++i) {
            const auto input_data = ctx.getInputData(i);
            if (input_data == nullptr) {
              return;
            }
            for (int j = 0; j < input_data->dim_size(); ++j) {
              *tsp.add_dim() = input_data->dim(j);
            }
          }
          if (tsp.dim_size() > 0) {
            ctx.addOutputData(0, std::move(tsp));
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Split,
    18,
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
        .Attr(
            "num_outputs",
            "Number of outputs to split parts of the tensor into. "
            "If the tensor is not evenly splittable the last chunk will be smaller.",
            AttributeProto::INT,
            false)
        .SetDoc(kDoc_Split_ver18)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          for (size_t i = 0; i < ctx.getNumOutputs(); ++i) {
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
          const auto num_outputs_attr = ctx.getAttribute("num_outputs");
          if (ctx.hasInput(1) && num_outputs_attr) {
            fail_shape_inference("Both 'split' input and 'num_outputs' attribute were given");
          }
          if (ctx.hasInput(1)) { //'split' is input
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
            if (num_outputs_attr) {
              const auto num_outputs = num_outputs_attr->i();
              if (num_outputs < 1) {
                fail_shape_inference("Attribute `num_outputs` value cannot be lower than 1");
              }
              if (split_dim_value % num_outputs == 0) { // tensor is evenly splittable
                int chunk_size = split_dim_value / num_outputs;
                split.resize(num_outputs, chunk_size);
              } else { // tensor needs to be split unevenly
                int chunk_size = (split_dim_value / num_outputs) + 1;
                int last_chunk_size = split_dim_value - (chunk_size * (num_outputs - 1));
                split.resize(num_outputs - 1, chunk_size);
                split.push_back(last_chunk_size);
              }
            } else {
              fail_shape_inference("Neither 'split' input nor 'num_outputs' attribute has been given");
            }
          }
          for (size_t i = 0; i < ctx.getNumOutputs(); i++) {
            *ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape() = shape;
            ctx.getOutputType(i)->mutable_tensor_type()->mutable_shape()->mutable_dim(axis)->set_dim_value(split[i]);
          }
        }));

static void processSliceInputs(const int64_t input_dim_size_or_value, int64_t& start, int64_t& end, int64_t step) {
  // process step
  if (step == 0) {
    fail_shape_inference("'step' cannot be 0 for Slice");
  }
  // Empty dimension: clamp bounds are invalid when dimension size is 0,
  // so short-circuit to produce a zero-length output.
  if (input_dim_size_or_value == 0) {
    start = 0;
    end = 0;
    return;
  }
  // process start
  if (start < 0)
    start += input_dim_size_or_value;
  if (step < 0)
    start = std::clamp(start, static_cast<int64_t>(0), input_dim_size_or_value - 1);
  else
    start = std::clamp(start, static_cast<int64_t>(0), input_dim_size_or_value);
  // process end
  if (end < 0)
    end += input_dim_size_or_value;
  if (step < 0)
    end = std::clamp(end, static_cast<int64_t>(-1), input_dim_size_or_value - 1);
  else
    end = std::clamp(end, static_cast<int64_t>(0), input_dim_size_or_value);
}

ONNX_OPERATOR_SET_SCHEMA(
    Slice,
    13,
    OpSchema()
        .SetDoc(kDoc_Slice_ver13)
        .Input(
            0,
            "data",
            "Tensor of data to extract slices from.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "starts",
            "1-D tensor of starting indices of corresponding axis in `axes`",
            "Tind",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            2,
            "ends",
            "1-D tensor of ending indices (exclusive) of corresponding axis in `axes`",
            "Tind",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            3,
            "axes",
            "1-D tensor of axes that `starts` and `ends` apply to. Negative value means counting dimensions "
            "from the back. Accepted range is [-r, r-1] where r = rank(data). Behavior is undefined if an "
            "axis is repeated.",
            "Tind",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            4,
            "steps",
            "1-D tensor of slice step of corresponding axis in `axes`. "
            "Negative value means slicing backward. 'steps' cannot be 0. "
            "Defaults to 1s.",
            "Tind",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "output", "Sliced data tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
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
          checkAxesRange(axes, input_rank);
          adjustNegativeAxes(axes, input_rank);
          checkDuplicateAxes(axes, input_rank);
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

          size_t axes_size = axes.size();
          for (size_t axis_index = 0; axis_index < axes_size; ++axis_index) {
            auto axis = axes[axis_index] < 0 ? axes[axis_index] + static_cast<int64_t>(input_rank) : axes[axis_index];

            auto input_dim = ctx.getInputType(0)->tensor_type().shape().dim(static_cast<int>(axis));

            // input dim value is missing - cannot perform shape inference for
            // this axis
            if (!input_dim.has_dim_value()) {
              // Clear any previously propagated dim_param and leave this dimension "empty",
              // before moving on to the next dimension
              ctx.getOutputType(0)
                  ->mutable_tensor_type()
                  ->mutable_shape()
                  ->mutable_dim(static_cast<int>(axis))
                  ->clear_dim_param();
              continue;
            }
            auto start = starts[axis_index];
            auto end = ends[axis_index];
            auto step = steps[axis_index];
            processSliceInputs(input_dim.dim_value(), start, end, step);

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
        })
        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {
          const auto input_data = ctx.getInputData(0);
          const auto starts = ctx.getInputData(1);
          const auto ends = ctx.getInputData(2);
          bool axes_specified = ctx.getNumInputs() >= 4;
          bool steps_specified = ctx.getNumInputs() >= 5;

          const TensorShapeProto* axes = nullptr;
          const TensorShapeProto* steps = nullptr;
          if (axes_specified) {
            axes = ctx.getInputData(3);
            if (axes == nullptr) {
              return;
            }
          }
          if (steps_specified) {
            steps = ctx.getInputData(4);
            if (steps == nullptr) {
              return;
            }
          }

          if (input_data == nullptr || starts == nullptr || ends == nullptr) {
            return;
          }
          if (starts->dim_size() != ends->dim_size()) {
            fail_shape_inference(
                "Input rank for starts and ends should be the same: (",
                starts->dim_size(),
                ") vs (",
                ends->dim_size(),
                ").");
          }
          // Only supports axis = 0 since the data comes from Shape
          if ((!axes_specified || (axes->dim_size() == 1 && axes->dim(0).dim_value() == 0)) &&
              starts->dim_size() == 1 && ends->dim_size() == 1) {
            auto start = starts->dim(0).dim_value();
            auto end = ends->dim(0).dim_value();
            int64_t step = 1; // Default step is 1
            if (steps_specified) {
              if (steps->dim_size() != 1) {
                return;
              }
              if (!steps->dim(0).has_dim_value()) {
                return;
              }
              step = steps->dim(0).dim_value();
            }
            processSliceInputs(input_data->dim_size(), start, end, step);

            TensorShapeProto tsp;
            if (step > 0) {
              for (int i = start; i < end; i += step) {
                *tsp.add_dim() = input_data->dim(i);
              }
            } else {
              for (int i = start; i > end; i += step) {
                *tsp.add_dim() = input_data->dim(i);
              }
            }
            if (tsp.dim_size() > 0) {
              ctx.addOutputData(0, std::move(tsp));
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Transpose,
    25,
    OpSchema()
        .SetDoc(kDoc_Transpose_ver13)
        .Attr(
            "perm",
            "A list of integers. By default, reverse the dimensions; "
            "otherwise permute the axes according to the values given. "
            "Its length must be equal to the rank of the input, and each "
            "value must be in the range `[0, rank-1]`.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "transposed", "Transposed output.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir13(), "Constrain input and output types to all tensor types.")
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
    Scatter,
    11,
    OpSchema()
        .Deprecate()
        .SetDoc(kDoc_Scatter_ver11)
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
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Input and output types can be of any tensor type.")
        .TypeConstraint("Tind", {types::Int32, types::Int64}, "Constrain indices to integer types")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    ScatterND,
    18,
    OpSchema()
        .SetDoc(kDoc_ScatterND_ver18)
        .Attr(
            "reduction",
            "Type of reduction to apply: none (default), add, mul, max, min. "
            "'none': no reduction applied. "
            "'add':  reduction using the addition operation. "
            "'mul':  reduction using the addition operation. "
            "'max': reduction using the maximum operation."
            "'min': reduction using the minimum operation.",
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
    ScatterElements,
    18,
    OpSchema()
        .SetDoc(kDoc_ScatterElements_ver18)
        .Attr(
            "axis",
            "Which axis to scatter on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Attr(
            "reduction",
            "Type of reduction to apply: none (default), add, mul, max, min. "
            "'none': no reduction applied. "
            "'add':  reduction using the addition operation. "
            "'mul': reduction using the multiplication operation."
            "'max': reduction using the maximum operation."
            "'min': reduction using the minimum operation.",
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
    Gather,
    13,
    OpSchema()
        .SetDoc(kDoc_Gather_ver13)
        .Attr(
            "axis",
            "Which axis to gather on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "indices",
            "Tensor of int32/int64 indices, of any rank q. All index values are expected to be within bounds [-s, s-1] "
            "along axis of size s. It is an error if any of the index values are out of bounds.",
            "Tind",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "output", "Tensor of rank q + (r - 1).", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to any tensor type.")
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
    13,
    OpSchema()
        .SetDoc(kDoc_GatherElements_ver13)
        .Attr(
            "axis",
            "Which axis to gather on. Negative value means "
            "counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(data).",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "indices",
            "Tensor of int32/int64 indices, with the same rank r as the input. All index values are expected to be "
            "within bounds [-s, s-1] along axis of size s. It is an error if any of the index values are out of bounds.",
            "Tind",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Tensor of the same shape as indices.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to any tensor type.")
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
    25,
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
            OpSchema::all_tensor_types_ir13(),
            "Constrain input and output types to all tensor types up to IRv13.")
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
    Unsqueeze,
    25,
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
            OpSchema::all_tensor_types_ir13(),
            "Constrain input and output types to all tensor types up to IRv13.")
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

static bool BuildContextDependentFunctionBodySpaceToDepth(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  const auto* blocksize_attr = ctx.getAttribute("blocksize");
  if (blocksize_attr == nullptr || !blocksize_attr->has_i() || blocksize_attr->i() <= 0) {
    return false;
  }
  const auto blocksize = blocksize_attr->i();
  const auto block_area = checkedMultiply(blocksize, blocksize);
  const auto* mode_attr = ctx.getAttribute("mode");
  const std::string mode = (mode_attr != nullptr && mode_attr->has_s()) ? mode_attr->s() : "DCR";
  if (mode != "DCR" && mode != "CRD") {
    return false;
  }

  FunctionBuilder builder(functionProto);
  builder.Const1D("blocksize", blocksize)
      .Const1D("block_area", block_area)
      .Const1D("zero", int64_t{0})
      .Const1D("one", int64_t{1})
      .Const1D("two", int64_t{2})
      .Const1D("three", int64_t{3})
      .Const1D("four", int64_t{4})
      .Add("input_shape = Shape (input)")
      .Add("N = Slice (input_shape, zero, one)")
      .Add("C = Slice (input_shape, one, two)")
      .Add("H = Slice (input_shape, two, three)")
      .Add("W = Slice (input_shape, three, four)")
      .Add("H_block = Div (H, blocksize)")
      .Add("W_block = Div (W, blocksize)")
      .Add("C_block = Mul (C, block_area)")
      .Add("tmp_shape = Concat <axis = 0> (N, C, H_block, blocksize, W_block, blocksize)")
      .Add("tmp = Reshape <allowzero = 1> (input, tmp_shape)");
  if (mode == "DCR") {
    builder.Add("tmp_transposed = Transpose <perm = [0, 3, 5, 1, 2, 4]> (tmp)");
  } else {
    builder.Add("tmp_transposed = Transpose <perm = [0, 1, 3, 5, 2, 4]> (tmp)");
  }
  builder.Add("output_shape = Concat <axis = 0> (N, C_block, H_block, W_block)")
      .Add("output = Reshape <allowzero = 1> (tmp_transposed, output_shape)");
  schema.BuildFunction(functionProto);
  return true;
}

static bool BuildContextDependentFunctionBodyDepthToSpace(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  const auto* blocksize_attr = ctx.getAttribute("blocksize");
  if (blocksize_attr == nullptr || !blocksize_attr->has_i() || blocksize_attr->i() <= 0) {
    return false;
  }
  const auto blocksize = blocksize_attr->i();
  const auto block_area = checkedMultiply(blocksize, blocksize);
  const auto* mode_attr = ctx.getAttribute("mode");
  const std::string mode = (mode_attr != nullptr && mode_attr->has_s()) ? mode_attr->s() : "DCR";
  if (mode != "DCR" && mode != "CRD") {
    return false;
  }

  FunctionBuilder builder(functionProto);
  builder.Const1D("blocksize", blocksize)
      .Const1D("block_area", block_area)
      .Const1D("zero", int64_t{0})
      .Const1D("one", int64_t{1})
      .Const1D("two", int64_t{2})
      .Const1D("three", int64_t{3})
      .Const1D("four", int64_t{4})
      .Add("input_shape = Shape (input)")
      .Add("N = Slice (input_shape, zero, one)")
      .Add("C = Slice (input_shape, one, two)")
      .Add("H = Slice (input_shape, two, three)")
      .Add("W = Slice (input_shape, three, four)")
      .Add("H_block = Mul (H, blocksize)")
      .Add("W_block = Mul (W, blocksize)")
      .Add("C_block = Div (C, block_area)");
  if (mode == "DCR") {
    builder.Add("tmp_shape = Concat <axis = 0> (N, blocksize, blocksize, C_block, H, W)")
        .Add("tmp = Reshape <allowzero = 1> (input, tmp_shape)")
        .Add("tmp_transposed = Transpose <perm = [0, 3, 4, 1, 5, 2]> (tmp)");
  } else {
    builder.Add("tmp_shape = Concat <axis = 0> (N, C_block, blocksize, blocksize, H, W)")
        .Add("tmp = Reshape <allowzero = 1> (input, tmp_shape)")
        .Add("tmp_transposed = Transpose <perm = [0, 1, 4, 2, 5, 3]> (tmp)");
  }
  builder.Add("output_shape = Concat <axis = 0> (N, C_block, H_block, W_block)")
      .Add("output = Reshape <allowzero = 1> (tmp_transposed, output_shape)");
  schema.BuildFunction(functionProto);
  return true;
}

ONNX_OPERATOR_SET_SCHEMA(
    SpaceToDepth,
    28,
    OpSchema()
        .Attr("blocksize", "Blocks of [blocksize, blocksize] are moved.", AttributeProto::INT)
        .Attr(
            "mode",
            "DCR (default) for depth-column-row order re-arrangement. Use CRD for column-row-depth order.",
            AttributeProto::STRING,
            std::string("DCR"))
        .SetDoc(kDoc_SpaceToDepth_ver28)
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
        })
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodySpaceToDepth));

ONNX_OPERATOR_SET_SCHEMA(
    DepthToSpace,
    28,
    OpSchema()
        .Attr("blocksize", "Blocks of [blocksize, blocksize] are moved.", AttributeProto::INT)
        .Attr(
            "mode",
            "DCR (default) for depth-column-row order re-arrangement. Use CRD for column-row-depth order.",
            AttributeProto::STRING,
            std::string("DCR"))
        .SetDoc(kDoc_DepthToSpace_ver28)
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
        })
        .SetContextDependentFunctionBodyBuilder(BuildContextDependentFunctionBodyDepthToSpace));

ONNX_OPERATOR_SET_SCHEMA(
    Tile,
    13,
    OpSchema()
        .SetDoc(kDoc_Tile_ver6)
        .Input(0, "input", "Input tensor of any shape.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "repeats",
            "1D int64 tensor of the same length as input's dimension number, "
            "includes numbers of repeated copies along input's dimensions.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Output tensor of the same dimensions and type as tensor input. "
            "output_dim[i] = input_dim[i] * repeats[i]",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
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

ONNX_OPERATOR_SET_SCHEMA(
    Upsample,
    10,
    OpSchema()
        .Deprecate()
        .Attr(
            "mode",
            "Two interpolation modes: nearest (default), and linear (including bilinear, trilinear, etc)",
            AttributeProto::STRING,
            std::string("nearest"))
        .Input(0, "X", "N-D tensor", "T", OpSchema::Single)
        .Input(
            1,
            "scales",
            "The scale array along each dimension. It takes value greater than or equal to 1."
            " The number of elements of 'scales' should be the same as the rank of input 'X'.",
            "tensor(float)",
            OpSchema::Single)
        .Output(0, "Y", "N-D tensor after resizing", "T", OpSchema::Single)
        .TypeConstraint("T", OpSchema::all_tensor_types(), "Constrain input 'X' and output 'Y' to all tensor types.")
        .SetDoc(kDoc_Upsample_ver7)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { resizeShapeInference_opset7_to_10(ctx); }));

static constexpr const char* Resize_ver19_attr_coordinate_transformation_mode_doc = R"DOC(
This attribute describes how to transform the coordinate in the resized tensor to the coordinate in the original tensor.

The coordinate of each dimension is transformed individually. Let's describe a case using axis x as an example.
Denote `x_resized` as the coordinate of axis x in the resized tensor,
 `x_original` as the coordinate of axis x in the original tensor,
 `length_original` as the length of the original tensor in axis x,
 `length_resized` as the length of the resized tensor in axis x,
 `scale = length_resized / length_original`,
 `output_width` the target length on the axis x which can be a fractional number when it is calculated out of a scale factor,
 and `output_width_int` the effective output width as an integer.

if coordinate_transformation_mode is `"half_pixel"`,
```
x_original = (x_resized + 0.5) / scale - 0.5
```

if coordinate_transformation_mode is `"half_pixel_symmetric"`,
```
adjustment = output_width_int / output_width
center = input_width / 2
offset = center * (1 - adjustment)
x_ori = offset + (x + 0.5) / scale - 0.5
```

if coordinate_transformation_mode is `"pytorch_half_pixel"`,
```
x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0
```

if coordinate_transformation_mode is `"align_corners"`,
```
x_original = x_resized * (length_original - 1) / (length_resized - 1)
```

if coordinate_transformation_mode is `"asymmetric"`,
```
x_original = x_resized / scale
```

if coordinate_transformation_mode is `"tf_crop_and_resize"`,
```
x_original = length_resized > 1 ? start_x * (length_original - 1) + x_resized * (end_x - start_x) * (length_original - 1) / (length_resized - 1) : 0.5 * (start_x + end_x) * (length_original - 1)
```
.)DOC";

static constexpr const char* Resize_ver19_attr_keep_aspect_ratio_policy_doc = R"DOC(
This attribute describes how to interpret the `sizes` input with regard to keeping the original aspect ratio of the input, and it is not applicable when
the `scales` input is used.

Given a set of `sizes`, associated with a subset of `axes` (explicitly provided or default), and assuming `d = axes[i]`, with `i` being the index of the provided `sizes`.

If `keep_aspect_ratio_policy` is `"stretch"`, the original aspect ratio is disregarded, and the input is resized to the specified size:
`out_size[d] = sizes[i]`

If `keep_aspect_ratio_policy` is `"not_larger"`, the sizes are adjusted so that no extent of the output is larger than the specified size, while keeping the original aspect ratio:
```
scale = Min(sizes[i] / in_size[d])
out_size[d] = round_int(scale * in_size[d])
```

If `keep_aspect_ratio_policy` is `"not_smaller"`, the sizes are adjusted so that no extent of the output is smaller than the specified size, while keeping the original aspect ratio:
```
scale = Max(sizes[i] / in_size[d])
out_size[d] = round_int(scale * in_size[d])
```

For non-resizable axes (those not specified in `axes`), the output size will be equal to the input size.

Note: `round_int` stands for computing the nearest integer value, rounding halfway cases up.)DOC";

ONNX_OPERATOR_SET_SCHEMA(
    Resize,
    19,
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
            Resize_ver19_attr_coordinate_transformation_mode_doc,
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
            Resize_ver19_attr_keep_aspect_ratio_policy_doc,
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
        .SetDoc(kDoc_Resize_ver19)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { resizeShapeInference_opset18_to_19(ctx); }));

ONNX_OPERATOR_SET_SCHEMA(
    GridSample,
    22,
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
            OpSchema::all_tensor_types_ir4(),
            "Constrain input `X` and output `Y` types to all tensor types.")
        .TypeConstraint("T2", OpSchema::all_float_types_ir4(), "Constrain grid types to float tensors.")
        .SetDoc(kDoc_GridSample_ver20)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { gridSampleShapeInference(ctx); }));

ONNX_OPERATOR_SET_SCHEMA(
    AffineGrid,
    20,
    OpSchema()
        .Attr(
            "align_corners",
            "if align_corners=1, consider -1 and 1 to refer to the centers of the corner pixels. "
            "if align_corners=0, consider -1 and 1 to refer to the outer edge the corner pixels.",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(
            0,
            "theta",
            "input batch of affine matrices with shape (N, 2, 3) for 2D or (N, 3, 4) for 3D",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Input(
            1,
            "size",
            "the target output image size (N, C, H, W) for 2D or (N, C, D, H, W) for 3D",
            "T2",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "grid",
            "output tensor of shape (N, H, W, 2) of 2D sample coordinates or (N, D, H, W, 3) of 3D sample coordinates.",
            "T1",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T1", OpSchema::all_float_types_ir4(), "Constrain grid types to float tensors.")
        .TypeConstraint("T2", {types::Int64}, "Constrain size's type to int64 tensors.")
        .SetDoc(kDoc_AffineGrid_ver20)
        .FunctionBody(R"ONNX(
        {
          # naming one: 1, one_f: 1.0, one_1d: [1], one_f_1d: [1.0]
          one = Constant <value_int: int=1> ()
          two = Constant <value_int: int=2> ()
          zero = Constant <value_int: int=0> ()
          four = Constant <value_int: int=4> ()
          one_1d = Constant <value_ints: ints = [1]> ()
          zero_1d = Constant <value_ints: ints = [0]> ()

          minus_one = Constant <value_int: int=-1> ()
          minus_one_f = CastLike (minus_one, theta)
          zero_f = CastLike (zero, theta)
          one_f = CastLike (one, theta)
          two_f = CastLike (two, theta)

          constant_align_corners = Constant <value_int: int=@align_corners> ()
          constant_align_corners_equal_zero = Equal (constant_align_corners, zero)

          size_ndim = Size (size)
          condition_is_2d = Equal (size_ndim, four)

          N, C, D, H, W = If (condition_is_2d) <
              then_branch = g1 () => (N_then, C_then, D_then, H_then, W_then) {
                  N_then, C_then, H_then, W_then = Split <num_outputs: int=4> (size)
                  D_then = Identity (one_1d)
              },
              else_branch = g2 () => (N_else, C_else, D_else, H_else, W_else) {
                  N_else, C_else, D_else, H_else, W_else = Split <num_outputs: int=5> (size)
              }
          >
          size_NCDHW = Concat <axis=0> (N, C, D, H, W)

          theta_3d = If (condition_is_2d) <
              then_branch = g3 () => (theta_then) { # theta: N by 2 by 3 => N by 3 by 4
                  # use of thetaN23 is a way to make shape inference happy when theta is N by 3 by 4.
                  gather_idx_6 = Constant <value_ints: ints = [0, 1, 2, 0, 1, 2]> ()
                  shape_23 = Constant <value_ints: ints = [2, 3]> ()
                  gather_idx_23 = Reshape (gather_idx_6, shape_23)
                  shape_N23 = Concat <axis=0>(N, shape_23)
                  gather_idx_N23 = Expand (gather_idx_23, shape_N23)
                  thetaN23 = GatherElements <axis=2> (theta, gather_idx_N23) # N by 2 by 3 => N by 3 by 2

                  r1, r2 = Split <axis: int=1, num_outputs: int=2> (thetaN23) # N by 1 by 3
                  r1_ = Squeeze (r1) # N by 3
                  r2_ = Squeeze (r2)
                  r11, r12, t1 = Split <axis: int=1, num_outputs: int=3> (r1_) # N by 1
                  r21, r22, t2 = Split <axis: int=1, num_outputs: int=3> (r2_)

                  r11_shape = Shape (r21)
                  float_zero_1d_ = ConstantOfShape (r11_shape) # N by 1
                  float_zero_1d = CastLike (float_zero_1d_, theta)
                  float_one_1d = Add (float_zero_1d, one_f) # N by 1

                  R1 = Concat <axis=1>(r11, r12, float_zero_1d, t1) # N by 4
                  R2 = Concat <axis=1>(r21, r22, float_zero_1d, t2)
                  R3 = Concat <axis=1>(float_zero_1d, float_zero_1d, float_one_1d, float_zero_1d)

                  R1_ = Unsqueeze (R1, one_1d) # N by 1 by 4
                  R2_ = Unsqueeze (R2, one_1d)
                  R3_ = Unsqueeze (R3, one_1d)
                  theta_then = Concat <axis=1> (R1_, R2_, R3_) # N by 3 by 4
                  # theta_then = Identity (theta)
              },
              else_branch = g4 () => (theta_else) {
                  theta_else = Identity (theta)
              }
          >

          two_1d = Constant <value_ints=[2]> ()
          three_1d = Constant <value_ints=[3]> ()
          five_1d = Constant <value_ints=[5]> ()
          constant_D_H_W_shape = Slice (size_NCDHW, two_1d, five_1d) # [N, C, D, H, W] => [D, H, W]
          zeros_D_H_W_ = ConstantOfShape (constant_D_H_W_shape)
          zeros_D_H_W = CastLike (zeros_D_H_W_, theta)
          ones_D_H_W = Add (zeros_D_H_W, one_f)

          D_float = CastLike (D, zero_f)
          H_float = CastLike (H, zero_f)
          W_float = CastLike (W, zero_f)
          start_d, step_d, start_h, step_h, start_w, step_w = If (constant_align_corners_equal_zero) <
              then_branch = h1 () => (start_d_then, step_d_then, start_h_then, step_h_then, start_w_then, step_w_then) { # => (float, float, float, float, float, float)
                  step_d_then = Div (two_f, D_float)
                  step_h_then = Div (two_f, H_float)
                  step_w_then = Div (two_f, W_float)

                  step_d_half = Div (step_d_then, two_f)
                  start_d_then = Add (minus_one_f, step_d_half)

                  step_h_half = Div (step_h_then, two_f)
                  start_h_then = Add (minus_one_f, step_h_half)

                  step_w_half = Div (step_w_then, two_f)
                  start_w_then = Add (minus_one_f, step_w_half)
              },
              else_branch = h2 () => (start_d_else, step_d_else, start_h_else, step_h_else, start_w_else, step_w_else) { # => (float, float, float, float, float, float)
                  D_float_nimus_one = Sub (D_float, one_f)
                  H_float_nimus_one = Sub (H_float, one_f)
                  W_float_nimus_one = Sub (W_float, one_f)
                  # avoid divide by 0
                  D_equals_one = Equal (D, one)
                  step_d_else = If (D_equals_one) <
                      then_branch = g5 () => (step_d_else_then) {
                          step_d_else_then = Identity (zero_f)
                      },
                      else_branch = g6 () => (step_d_else_else) {
                          step_d_else_else = Div (two_f, D_float_nimus_one)
                      }
                  >
                  step_h_else = Div (two_f, H_float_nimus_one)
                  step_w_else = Div (two_f, W_float_nimus_one)
                  start_d_else = Identity (minus_one_f)
                  start_h_else = Identity (minus_one_f)
                  start_w_else = Identity (minus_one_f)
              }
          >
          grid_w_steps_int = Range (zero, W, one)
          grid_w_steps_float = CastLike (grid_w_steps_int, step_w)
          grid_w_steps = Mul (grid_w_steps_float, step_w)
          grid_w_0 = Add (start_w, grid_w_steps)

          grid_h_steps_int = Range (zero, H, one)
          grid_h_steps_float = CastLike (grid_h_steps_int, step_h)
          grid_h_steps = Mul (grid_h_steps_float, step_h)
          grid_h_0 = Add (start_h, grid_h_steps)

          grid_d_steps_int = Range (zero, D, one)
          grid_d_steps_float = CastLike (grid_d_steps_int, step_d)
          grid_d_steps = Mul (grid_d_steps_float, step_d)
          grid_d_0 = Add (start_d, grid_d_steps)

          zeros_H_W_D = Transpose <perm = [1, 2, 0]> (zeros_D_H_W)
          grid_d_1 = Add (zeros_H_W_D, grid_d_0)
          grid_d = Transpose <perm = [2, 0, 1]> (grid_d_1)

          zeros_D_W_H = Transpose <perm = [0, 2, 1]> (zeros_D_H_W)
          grid_h_1 = Add (zeros_D_W_H, grid_h_0)
          grid_h = Transpose <perm = [0, 2, 1]> (grid_h_1)

          grid_w = Add (grid_w_0, zeros_D_H_W)

          grid_w_usqzed = Unsqueeze (grid_w, minus_one)
          grid_h_usqzed = Unsqueeze (grid_h, minus_one)
          grid_d_usqzed = Unsqueeze (grid_d, minus_one)
          ones_D_H_W_usqzed = Unsqueeze (ones_D_H_W, minus_one)
          original_grid = Concat <axis=-1> (grid_w_usqzed, grid_h_usqzed, grid_d_usqzed, ones_D_H_W_usqzed)

          constant_shape_DHW_4 = Constant <value_ints: ints = [-1, 4]> ()
          original_grid_DHW_4 = Reshape (original_grid, constant_shape_DHW_4)
          original_grid_4_DHW_ = Transpose (original_grid_DHW_4)

          original_grid_4_DHW = CastLike (original_grid_4_DHW_, theta_3d)
          grid_N_3_DHW = MatMul (theta_3d, original_grid_4_DHW)
          grid_N_DHW_3 = Transpose <perm = [0, 2, 1]> (grid_N_3_DHW)
          N_D_H_W_3 = Concat <axis=-1> (N, D, H, W, three_1d)
          grid_3d_else_ = Reshape (grid_N_DHW_3, N_D_H_W_3)
          grid_3d = CastLike (grid_3d_else_, theta_3d)

          # grid = Identity (grid_3d)
          grid = If (condition_is_2d) <
              then_branch = g1 () => (grid_then) { # [N, D=1, H, W, 3] => [N, H, W, 2]
                  grid_squeezed = Squeeze (grid_3d, one_1d)  # [N, H, W, 3]
                  grid_then = Slice (grid_squeezed, zero_1d, two_1d, three_1d) # [N, H, W, 2]
              },
              else_branch = g2 () => (grid_else) {
                  grid_else = Identity (grid_3d)
              }
          >
        }
        )ONNX")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }

          checkInputRank(ctx, 1, 1);

          bool found;
          TensorShapeProto size_proto = getShapeInput(ctx, 1, found);
          if (!found) {
            return;
          }

          const auto size_length = size_proto.dim_size();
          if (size_length != 4 && size_length != 5) {
            fail_shape_inference("Length of input 'size' is ", size_length, ". It must be 4 for 2D or 5 for 5D.");
          }

          auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
          const auto& N = size_proto.dim(0);
          *output_shape->add_dim() = N;
          // const auto& C = size_proto.dim(1); // C is not used
          if (size_length == 4) {
            // 2D case: size shape (N, C, H, W), output shape (N, C, H, W, 2)
            const auto& H = size_proto.dim(2);
            const auto& W = size_proto.dim(3);
            *output_shape->add_dim() = H;
            *output_shape->add_dim() = W;
            output_shape->add_dim()->set_dim_value(2);
          } else if (size_length == 5) {
            // 3D case: size shape (N, C, D, H, W), output shape (N, C, D, H, W, 3)
            const auto& D = size_proto.dim(2);
            const auto& H = size_proto.dim(3);
            const auto& W = size_proto.dim(4);
            *output_shape->add_dim() = D;
            *output_shape->add_dim() = H;
            *output_shape->add_dim() = W;
            output_shape->add_dim()->set_dim_value(3);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Identity,
    25,
    OpSchema()
        .SetDoc(kDoc_Identity_ver25)
        .Input(0, "input", "Input tensor", "V", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Output(0, "output", "Tensor to copy input into.", "V", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .TypeConstraint(
            "V",
            []() {
              auto t = OpSchema::all_tensor_types_ir13();
              auto s = OpSchema::all_tensor_sequence_types();
              auto o = OpSchema::all_optional_types();
              t.insert(t.end(), s.begin(), s.end());
              t.insert(t.end(), o.begin(), o.end());
              return t;
            }(),
            "Constrain input and output types to all tensor, sequence, and optional types.")
        .TypeAndShapeInferenceFunction(propagateShapeAndTypeFromFirstInput));

ONNX_OPERATOR_SET_SCHEMA(
    Compress,
    28,
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
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
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
    28,
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
        .TypeConstraint("T3", OpSchema::all_tensor_types_ir4(), "Constrain to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) { oneHotShapeInference(ctx, 11); }));

ONNX_OPERATOR_SET_SCHEMA(
    IsNaN,
    20,
    OpSchema()
        .SetDoc(kDoc_IsNaN_ver20)
        .Input(0, "X", "input", "T1", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "Y", "output", "T2", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint("T1", OpSchema::all_float_types_ir9(), "Constrain input types to float tensors.")
        .TypeConstraint("T2", {types::Bool}, "Constrain output types to boolean tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          updateOutputElemType(ctx, 0, TensorProto::BOOL);
          if (hasInputShape(ctx, 0)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    IsInf,
    20,
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
        .TypeConstraint("T1", OpSchema::all_float_types_ir9(), "Constrain input types to float tensors.")
        .TypeConstraint("T2", {types::Bool}, "Constrain output types to boolean tensors.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          updateOutputElemType(ctx, 0, TensorProto::BOOL);
          if (hasInputShape(ctx, 0)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    Where,
    16,
    OpSchema()
        .SetDoc(GET_OP_DOC_STR(std::string(kDoc_Where_ver9) + GenerateBroadcastingDocMul()))
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
        .TypeConstraint(
            "T",
            OpSchema::all_tensor_types_ir4(),
            "Constrain input and output types to all tensor types (including bfloat).")
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
    NonZero,
    13,
    OpSchema()
        .SetDoc(kDoc_NonZero_ver9)
        .Input(0, "X", "input", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .Output(0, "Y", "output", "tensor(int64)", OpSchema::Single, true, 1, OpSchema::NonDifferentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain to all tensor types.")
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
    ReverseSequence,
    28,
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
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Input and output types can be of any tensor type.")
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
    28,
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
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Input can be of any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          const TypeProto* xTensorProto = ctx.getInputType(0);
          TypeProto* yTensorProto = ctx.getOutputType(0);
          TypeProto* indicesTensorProto = nullptr;
          TypeProto* inverseIndicesTensorProto = nullptr;
          TypeProto* countsTensorProto = nullptr;

          // 'indices', 'inverse_indices', and 'counts' are 1-D tensors of
          // unknown dimension.
          // Shape inference will happen even in case of empty optional outputs,
          // graph-level shape inference should not propagate the shape downstream for empty optional outputs.
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
            // 'axis' is not provided. Input 'X' is flattened.
            // 'Y' is a 1-D tensor of unknown dimension.
            yTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
          } else {
            // 'axis' is provided.
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
            // 'Y' has the same shape as 'X' except in the 'axis' dimension
            // which is unknown.
            for (int i = 0; i < input_shape.dim_size(); i++) {
              auto dim = yTensorProto->mutable_tensor_type()->mutable_shape()->add_dim();
              if (i != axis) {
                *dim = input_shape.dim(i);
              }
            }
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    GatherND,
    13,
    OpSchema()
        .SetDoc(kDoc_GatherND_ver13)
        .Attr(
            "batch_dims",
            "The number of batch dimensions. The gather of indexing starts from dimension of data[batch_dims:]",
            AttributeProto::INT,
            static_cast<int64_t>(0))
        .Input(0, "data", "Tensor of rank r >= 1.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Input(
            1,
            "indices",
            "Tensor of rank q >= 1. All index values are expected to be within bounds [-s, s-1] "
            "along axis of size s. It is an error if any of the index values are out of bounds.",
            "tensor(int64)",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Tensor of rank q + r - indices_shape[-1] - 1.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to any tensor type.")
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
    25,
    OpSchema().FillUsing(PadDocGenerator(
        kDoc_Pad_ver24,
        "Supported modes: `constant`(default), `reflect`, `edge`, `wrap`",
        OpSchema::all_tensor_types_ir13(),
        "Constrain input and output types to all tensor types up to IRv13.")));

ONNX_OPERATOR_SET_SCHEMA(
    Trilu,
    14,
    OpSchema()
        .SetDoc(kDoc_Trilu_ver14)
        .Attr(
            "upper",
            "Boolean. Indicates whether upper or lower part of matrix is retained. Default is true.",
            AttributeProto::INT,
            static_cast<int64_t>(1))
        .Input(
            0,
            "input",
            "Input tensor of rank 2 or higher.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "k",
            "A 0-D tensor containing a single value corresponding to the number diagonals above or below the main diagonal to exclude or include. "
            "Default value is 0 if it's not specified.",
            "tensor(int64)",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "output",
            "Output tensor of the same type and shape as the input tensor.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          // Type inference
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          // Shape inference needs the input data shape
          if (hasInputShape(ctx, 0)) {
            const TensorShapeProto& input_shape = ctx.getInputType(0)->tensor_type().shape();
            const int rank = static_cast<int>(input_shape.dim_size());
            if (rank < 2) {
              fail_shape_inference("Input rank must be >= 2.")
            }
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

ONNX_OPERATOR_SET_SCHEMA(
    CenterCropPad,
    18,
    OpSchema()
        .SetDoc(kDoc_CenterCropPad_ver18)
        .Input(
            0,
            "input_data",
            "Input to extract the centered crop from.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "shape",
            "1-D tensor representing the cropping window dimensions.",
            "Tind",
            OpSchema::Single,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(0, "output_data", "Output data.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable)
        .Attr(
            "axes",
            "If provided, it specifies a subset of axes that 'shape' refer to. "
            "If not provided, all axes are assumed [0, 1, ..., r-1], where r = rank(data). "
            "Negative value means counting dimensions from the back. Accepted range is [-r, r-1], where r = rank(data). "
            "Behavior is undefined if an axis is repeated.",
            AttributeProto::INTS,
            OPTIONAL_VALUE)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir4(), "Constrain input and output types to all tensor types.")
        .TypeConstraint("Tind", {types::Int32, types::Int64}, "Constrain indices to integer types")
        .SetNodeDeterminism(OpSchema::NodeDeterminism::Deterministic)
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          if (ctx.getNumInputs() != 2) {
            fail_type_inference("CenterCropPad op must have 2 inputs.");
          }
          propagateElemTypeFromInputToOutput(ctx, 0, 0);
          if (!hasNInputShapes(ctx, 1)) {
            return;
          }
          // Shape Inference if shape is initializer
          const TensorProto* cropShapeInitializer = ctx.getInputData(1);
          if (!cropShapeInitializer) {
            return;
          }

          // don't know data_type - can't proceed
          if (!cropShapeInitializer->has_data_type())
            return;

          const auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
          const int64_t input_rank = input_shape.dim_size();

          std::vector<int64_t> shape;
          if (cropShapeInitializer->data_type() == TensorProto::INT64) {
            const auto data = ParseData<int64_t>(cropShapeInitializer);
            shape.insert(shape.end(), data.begin(), data.end());
          } else if (cropShapeInitializer->data_type() == TensorProto::INT32) {
            const auto data = ParseData<int32_t>(cropShapeInitializer);
            shape.insert(shape.end(), data.begin(), data.end());
          } else {
            // unaccepted data type
            fail_shape_inference("`shape` only supports `int32_t` or `int64_t` inputs");
          }

          auto axes_attr = ctx.getAttribute("axes");
          std::vector<int64_t> axes;
          if (axes_attr) {
            axes = RetrieveValues<int64_t>(*axes_attr);
            checkAxesRange(axes, input_rank);
            adjustNegativeAxes(axes, input_rank);
            checkDuplicateAxes(axes, input_rank);
          } else {
            axes.resize(input_rank);
            std::iota(axes.begin(), axes.end(), 0);
          }

          if (shape.size() != axes.size()) {
            fail_shape_inference(
                "Number of elements of input 'shape' (",
                shape.size(),
                ") does not match the number of axes (",
                axes.size(),
                ").");
          }

          // Populating default dims
          std::vector<TensorShapeProto_Dimension*> out_dims(input_rank);
          auto output_shape = getOutputShape(ctx, 0);
          for (int i = 0; i < input_rank; ++i) {
            out_dims[i] = output_shape->add_dim();
            const auto& input_dim = input_shape.dim(i);
            if (input_dim.has_dim_value()) {
              out_dims[i]->set_dim_value(input_dim.dim_value());
            } else if (input_dim.has_dim_param()) {
              out_dims[i]->set_dim_param(input_dim.dim_param());
            }
          }
          int j = 0;
          for (int axis : axes) {
            out_dims[axis]->set_dim_value(shape[j++]);
          }
        })
        .SetContextDependentFunctionBodyBuilder([](const FunctionBodyBuildContext& ctx,
                                                   const OpSchema& schema,
                                                   FunctionProto& functionProto) {
          FunctionBuilder builder(functionProto);
          builder.Const("k2", std::vector<int64_t>{2});

          auto axes_attr = ctx.getAttribute("axes");
          if (axes_attr) { // axes provided, need to work on a subset of dimensions
            builder.Add("axes_input = Constant <value_ints : ints = @axes>()");
            builder.Add("x_shape_alldims = Shape (input_data)").Add("x_shape = Gather (x_shape_alldims, axes_input)");
          } else { // axes not provided, assuming all dims
            builder.Add("x_shape = Shape (input_data)");
          }

          // First: Pad step
          builder.Add("padded_sh = Max(x_shape, shape)")
              .Add("pad_amount = Sub(padded_sh, x_shape)")
              .Add("pad_amount_left = Div(pad_amount, k2)")
              .Add("pad_amount_right = Sub(pad_amount, pad_amount_left)")
              .Add("pads = Concat <axis = 0> (pad_amount_left, pad_amount_right)");
          if (axes_attr)
            builder.Add("padded_input = Pad (input_data, pads, , axes_input)");
          else
            builder.Add("padded_input = Pad (input_data, pads)");

          // Second: Slice step
          if (axes_attr) {
            builder.Add("x_shape_alldims2 = Shape (padded_input)")
                .Add("x_shape2 = Gather (x_shape_alldims2, axes_input)");
          } else {
            builder.Add("x_shape2 = Shape (padded_input)");
          }

          builder.Add("sh_diff = Sub (x_shape2, shape)")
              .Add("start_dims = Div (sh_diff, k2)")
              .Add("end_dims = Add (start_dims, shape)");
          if (axes_attr)
            builder.Add("output_data = Slice (padded_input, start_dims, end_dims, axes_input)");
          else
            builder.Add("output_data = Slice (padded_input, start_dims, end_dims)");

          schema.BuildFunction(functionProto);
          return true;
        }));

ONNX_OPERATOR_SET_SCHEMA(
    TensorScatter,
    24,
    OpSchema()
        .SetDoc(kDoc_TensorScatter_ver24)
        .Attr(
            "axis",
            "Sequence dimension of the `past_cache` and `update` tensors. It cannot be 0 (the batch dimension). Default is -2.",
            AttributeProto::INT,
            static_cast<int64_t>(-2))
        .Attr(
            "mode",
            "Write mode of cache update. Supported modes include `linear` and `circular`. `linear` mode requires "
            "write_indices+sequence_length<=max_sequence_length. For `circular` mode, the updates happen in "
            "wrap-around fashion, ie, the update index is modulo `max_sequence_length`",
            AttributeProto::STRING,
            std::string("linear"))
        .Input(
            0,
            "past_cache",
            "Past state cache for key or value with shape `(batch_size, D1, D2, ..., max_sequence_length, ..., Dn)`.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            1,
            "update",
            "New update tensor with shape `(batch_size, D1, D2, ..., sequence_length, ..., Dn)`.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .Input(
            2,
            "write_indices",
            "Write indices for the incoming update tensor in the cache. Shape is `(batch_size,)`. Assumed to be all zeros if not provided.",
            "tensor(int64)",
            OpSchema::Optional,
            true,
            1,
            OpSchema::NonDifferentiable)
        .Output(
            0,
            "present_cache",
            "Updated cache. Same shape as `past_cache`.",
            "T",
            OpSchema::Single,
            true,
            1,
            OpSchema::Differentiable)
        .TypeConstraint("T", OpSchema::all_tensor_types_ir12(), "Constrain input and output types to any tensor type.")
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
          propagateElemTypeFromInputToOutput(ctx, 0, 0);

          // Check that updates and past_cache have the same rank
          if (hasInputShape(ctx, 0) && hasInputShape(ctx, 1)) {
            const auto& past_cache_shape = getInputShape(ctx, 0);
            const auto& update_shape = getInputShape(ctx, 1);
            const int past_cache_rank = static_cast<int>(past_cache_shape.dim_size());
            const int update_rank = static_cast<int>(update_shape.dim_size());

            if (past_cache_rank != update_rank) {
              fail_shape_inference(
                  "past_cache and update must have the same rank. past_cache_rank=",
                  past_cache_rank,
                  ", update_rank=",
                  update_rank);
            }

            // Check that all dimensions match except the axis dimension
            // Read axis from attribute, default is -2
            auto axis_attr = ctx.getAttribute("axis");
            int64_t axis = axis_attr ? axis_attr->i() : -2;

            // Handle negative axis
            if (axis < 0) {
              axis += past_cache_rank;
            }

            // Validate axis is within bounds
            if (axis <= 0 || axis >= past_cache_rank) {
              fail_shape_inference("axis ", axis, " is out of range for rank ", past_cache_rank);
            }
            for (int i = 0; i < past_cache_rank; ++i) {
              if (i != axis) {
                // All dimensions except axis should match
                Dim dim;
                unifyInputDim(ctx, 0, i, dim);
                unifyInputDim(ctx, 1, i, dim);
              }
            }
          }

          // Check that write_indices is of shape (batch_size,)
          if (hasInputShape(ctx, 2)) {
            checkInputRank(ctx, 2, 1);
            Dim batch_size;
            unifyInputDim(ctx, 0, 0, batch_size);
            unifyInputDim(ctx, 2, 0, batch_size);
          }

          if (hasNInputShapes(ctx, 1)) {
            propagateShapeFromInputToOutput(ctx, 0, 0);
          }
        }));

} // namespace ONNX_NAMESPACE
