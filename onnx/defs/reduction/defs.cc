/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <functional>
#include "onnx/defs/function.h"
#include "onnx/defs/reduction/utils.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

std::function<void(OpSchema&)> ReduceDocGeneratorWithFunctionBody(const char* name, const char* func_body) {
  return ReduceDocGenerator_opset13_18(name, false, false, func_body);
}

std::function<void(OpSchema&)> ReduceDocGeneratorWithFunctionBuilder(
    const char* name,
    ContextDependentFunctionBodyBuilder functionBuilder) {
  return ReduceDocGenerator_opset13_18(name, false, false, nullptr, functionBuilder);
}

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 13, OpSchema().FillUsing(ReduceDocGenerator_opset13_18("max", true)));

ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 13, OpSchema().FillUsing(ReduceDocGenerator_opset13_18("min", true)));

ONNX_OPERATOR_SET_SCHEMA(ReduceSum, 13, OpSchema().FillUsing(ReduceDocGenerator_opset13_18("sum", false, true)));

const char* reduce_sum_square_func_body = R"ONNX(
  {
    has_axes = AttributeHasValue <value_ints: ints = @axes>()
    reduced = If (has_axes) <
      then_branch = g1 () => (reduced_then)
      {
        data_square = Mul(data, data)
        axes = Constant <value_ints: ints = @axes>()
        reduced_then = ReduceSum<keepdims: int = @keepdims>(data_square, axes)
      },
      else_branch = g2 () => (reduced_else)
      {
        data_square = Mul(data, data)
        reduced_else = ReduceSum<keepdims: int = @keepdims>(data_square)
      }
    >
  }
  )ONNX";

ONNX_OPERATOR_SET_SCHEMA(
    ReduceSumSquare,
    18,
    OpSchema().FillUsing(ReduceDocGeneratorWithFunctionBody("sum square", reduce_sum_square_func_body)));

ONNX_OPERATOR_SET_SCHEMA(ReduceMean, 13, OpSchema().FillUsing(ReduceDocGenerator_opset13_18("mean")));

ONNX_OPERATOR_SET_SCHEMA(ReduceProd, 13, OpSchema().FillUsing(ReduceDocGenerator_opset13_18("product")));

const char* reduce_log_sum_func_body = R"ONNX(
  {
    has_axes = AttributeHasValue <value_ints: ints = @axes>()
    reduced = If (has_axes) <
      then_branch = g1 () => (reduced_then)
      {
        axes = Constant <value_ints: ints = @axes>()
        reduced_sum = ReduceSum<keepdims: int = @keepdims>(data, axes)
        reduced_then = Log (reduced_sum)
      },
      else_branch = g2 () => (reduced_else)
      {
        reduced_sum = ReduceSum<keepdims: int = @keepdims>(data)
        reduced_else = Log (reduced_sum)
      }
    >
  }
  )ONNX";
ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSum,
    18,
    OpSchema().FillUsing(ReduceDocGeneratorWithFunctionBody("log sum", reduce_log_sum_func_body)));

const char* reduce_log_sum_exp_func_body = R"ONNX(
  {
    has_axes = AttributeHasValue <value_ints: ints = @axes>()
    reduced = If (has_axes) <
      then_branch = g1 () => (reduced_then)
      {
        data_exp = Exp (data)
        axes = Constant <value_ints: ints = @axes>()
        reduced_sum = ReduceSum<keepdims: int = @keepdims>(data_exp, axes)
        reduced_then = Log (reduced_sum)
      },
      else_branch = g2 () => (reduced_else)
      {
        data_exp = Exp (data)
        reduced_sum = ReduceSum<keepdims: int = @keepdims>(data_exp)
        reduced_else = Log (reduced_sum)
      }
    >
  }
  )ONNX";
ONNX_OPERATOR_SET_SCHEMA(
    ReduceLogSumExp,
    18,
    OpSchema().FillUsing(ReduceDocGeneratorWithFunctionBody("log sum exponent", reduce_log_sum_exp_func_body)));

const char* reduce_l1_func_body = R"ONNX(
  {
    has_axes = AttributeHasValue <value_ints: ints = @axes>()
    reduced = If (has_axes) <
      then_branch = g1 () => (reduced_then)
      {
        data_abs = Abs(data)
        axes = Constant <value_ints: ints = @axes>()
        reduced_then = ReduceSum<keepdims: int = @keepdims>(data_abs, axes)
      },
      else_branch = g2 () => (reduced_else)
      {
        data_abs = Abs(data)
        reduced_else = ReduceSum<keepdims: int = @keepdims>(data_abs)
      }
    >
  }
  )ONNX";
ONNX_OPERATOR_SET_SCHEMA(
    ReduceL1,
    18,
    OpSchema().FillUsing(ReduceDocGeneratorWithFunctionBody("L1 norm", reduce_l1_func_body)));

const char* reduce_l2_func_body = R"ONNX(
  {
    sum_square = ReduceSumSquare<axes: ints = @axes, keepdims: int = @keepdims>(data)
    sum_square_dbl = Cast <to = 1>(sum_square)
    sqrt = Sqrt(sum_square_dbl)
    reduced = CastLike(sqrt, data)
  }
  )ONNX";
ONNX_OPERATOR_SET_SCHEMA(
    ReduceL2,
    18,
    OpSchema().FillUsing(ReduceDocGeneratorWithFunctionBody("L2 norm", reduce_l2_func_body)));

std::function<void(OpSchema&)> ArgReduceDocGenerator(const char* name) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Computes the indices of the {name} elements of the input tensor's element along the
provided axis. The resulting tensor has the same rank as the input if keepdims equals 1.
If keepdims equals 0, then the resulting tensor has the reduced dimension pruned.
If select_last_index is True (default False), the index of the last occurrence of the {name}
is selected if the {name} appears more than once in the input. Otherwise the index of the
first occurrence is selected.
The type of the output tensor is integer.)DOC";
                        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc.c_str());
    schema.Attr(
        "axis",
        "The axis in which to compute the arg indices. Accepted range is [-r, r-1] where r = rank(data).",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Attr(
        "keepdims",
        "Keep the reduced dimension or not, default 1 means keep reduced dimension.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.Attr(
        "select_last_index",
        "Whether to select the last index or the first index if the {name} appears in multiple indices, default is False (first index).",
        AttributeProto::INT,
        static_cast<int64_t>(0));
    schema.Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::NonDifferentiable);
    schema.Output(
        0,
        "reduced",
        "Reduced output tensor with integer data type.",
        "tensor(int64)",
        OpSchema::Single,
        true,
        1,
        OpSchema::NonDifferentiable);
    schema.TypeConstraint(
        "T", OpSchema::all_numeric_types_with_bfloat(), "Constrain input and output types to all numeric tensors.");
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      // set output element type to int64
      updateOutputElemType(ctx, 0, TensorProto_DataType_INT64);

      if (!hasNInputShapes(ctx, 1)) {
        return;
      }

      auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
      int64_t input_ndim = input_shape.dim_size();
      int64_t axis = 0; // default to 0
      auto axis_proto = ctx.getAttribute("axis");
      if (axis_proto) {
        axis = axis_proto->i();
        if (axis < -input_ndim || axis >= input_ndim) {
          fail_shape_inference("'axis' must be in [-rank(indices), rank(indices)-1]");
        }
        if (axis < 0)
          axis += input_ndim;
      }

      int64_t keep_dims = 1;
      auto attr_proto = ctx.getAttribute("keepdims");
      if (attr_proto) {
        keep_dims = attr_proto->i();
      }
      // do we need handle negative axis?
      for (int i = 0; i < input_ndim; ++i) {
        if (i != axis) {
          auto dim = output_shape->add_dim();
          dim->CopyFrom(input_shape.dim(i));
        } else {
          if (keep_dims == 1) {
            auto dim = output_shape->add_dim();
            dim->set_dim_value(1);
          }
        }
      }
    });
  };
} // namespace ONNX_NAMESPACE

ONNX_OPERATOR_SET_SCHEMA(ArgMax, 13, OpSchema().FillUsing(ArgReduceDocGenerator("max")));

ONNX_OPERATOR_SET_SCHEMA(ArgMin, 13, OpSchema().FillUsing(ArgReduceDocGenerator("min")));

} // namespace ONNX_NAMESPACE
