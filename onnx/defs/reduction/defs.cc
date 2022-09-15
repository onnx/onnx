/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <functional>
#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/tensor_proto_util.h"

namespace ONNX_NAMESPACE {

std::vector<std::string> GetSupportedDataTypesForReductionOps(bool supports8bit) {
  if (supports8bit) {
    auto data_types = OpSchema::numeric_types_for_math_reduction_with_bfloat();
    data_types.push_back("tensor(uint8)");
    data_types.push_back("tensor(int8)");

    return data_types;
  }

  return OpSchema::numeric_types_for_math_reduction_with_bfloat();
}

std::function<void(OpSchema&)>
ReduceDocGenerator(const char* name, bool supports_8bit_datatypes = false, bool axes_input = false,
const char* func_body = nullptr, ContextDependentFunctionBodyBuilder function_builder = nullptr) {
  return [=](OpSchema& schema) {
    std::string doc;
    POPULATE_OP_DOC_STR(doc = R"DOC(
Computes the {name} of the input tensor's element along the provided axes. The resulting
tensor has the same rank as the input if keepdims equals 1. If keepdims equals 0, then
the resulting tensor has the reduced dimension pruned.

The above behavior is similar to numpy, with the exception that numpy defaults keepdims to
False instead of True.)DOC";
                        ReplaceAll(doc, "{name}", name););
    schema.SetDoc(doc.c_str());
    schema.Attr(
        "keepdims",
        "Keep the reduced dimension or not, default 1 means keep reduced dimension.",
        AttributeProto::INT,
        static_cast<int64_t>(1));
    schema.Input(0, "data", "An input tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable);
    if (axes_input) {
      schema.Attr(
          "noop_with_empty_axes",
          "Defines behaviour if 'axes' is empty. Default behaviour with 'false' is to reduce all axes. "
          "When axes is empty and this attribute is set to true, input tensor will not be reduced,"
          "and the output tensor would be equivalent to input tensor.",
          AttributeProto::INT,
          static_cast<int64_t>(0));
      schema.Input(
          1,
          "axes",
          "Optional input list of integers, along which to reduce. "
          "The default is to reduce over all the dimensions of the input tensor if 'noop_with_empty_axes' is false, "
          "else act as an Identity op when 'noop_with_empty_axes' is true. "
          "Accepted range is [-r, r-1] where r = rank(data).",
          "tensor(int64)",
          OpSchema::Optional,
          true,
          1,
          OpSchema::NonDifferentiable);
    } else {
      schema.Attr(
          "axes",
          "A list of integers, along which to reduce. The default is to reduce over "
          "all the dimensions of the input tensor. Accepted range is [-r, r-1] where r = rank(data).",
          AttributeProto::INTS,
          OPTIONAL_VALUE);
    }
    schema.Output(0, "reduced", "Reduced output tensor.", "T", OpSchema::Single, true, 1, OpSchema::Differentiable);
    schema.TypeConstraint(
        "T",
        GetSupportedDataTypesForReductionOps(supports_8bit_datatypes),
        supports_8bit_datatypes ? "Constrain input and output types to high-precision and 8 bit numeric tensors."
                                : "Constrain input and output types to high-precision numeric tensors.");
    if (func_body) {
      schema.FunctionBody(func_body);
    }
    else if(function_builder) {
      std::cout << "schema.SetContextDependentFunctionBodyBuilder(function_builder);" << std::endl;
      schema.SetContextDependentFunctionBodyBuilder(function_builder);
    }
    schema.TypeAndShapeInferenceFunction([](InferenceContext& ctx) {
      propagateElemTypeFromInputToOutput(ctx, 0, 0);
      if (!hasNInputShapes(ctx, 1)) {
        return;
      }

      int64_t keep_dims = 1, noop_with_empty_axes = 0;
      auto attr_proto = ctx.getAttribute("keepdims");
      if (attr_proto) {
        keep_dims = attr_proto->i();
      }
      auto noop_attr_proto = ctx.getAttribute("noop_with_empty_axes");
      if (noop_attr_proto) {
        noop_with_empty_axes = noop_attr_proto->i();
      }
      std::vector<int64_t> axes;
      size_t num_inputs = ctx.getNumInputs();
      if ((num_inputs == 2) && ctx.getInputType(1)) { // axes is input
        if (ctx.getAttribute("axes")) {
          fail_shape_inference("axes as an input and attribute cannot be specified at the same time.");
        }

        const TensorProto* axesInitializer = ctx.getInputData(1);
        if (axesInitializer == nullptr) {
          // skip if axes is not an initializer
          return;
        }
        std::vector<int64_t> axes_values = ParseData<int64_t>(axesInitializer);
        axes.assign(axes_values.begin(), axes_values.end());
      } else { // axes is attribute
        auto axes_proto = ctx.getAttribute("axes");
        if (axes_proto)
          axes.assign(axes_proto->ints().begin(), axes_proto->ints().end());
      }
      auto& input_shape = ctx.getInputType(0)->tensor_type().shape();
      if (noop_with_empty_axes && axes.empty()) {
        propagateShapeFromInputToOutput(ctx, 0, 0);
        return;
      }
      int64_t input_ndim = input_shape.dim_size();
      auto output_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();

      for (size_t i = 0; i < axes.size(); ++i) {
        if (axes[i] < -input_ndim || axes[i] >= input_ndim) {
          fail_shape_inference("axis must be in [-rank, rank-1]. input rank was ", input_ndim);
        }
        if (axes[i] < 0)
          axes[i] += input_ndim;
      }
      for (int i = 0; i < input_ndim; ++i) {
        // axes empty means reduce all dim
        if (!axes.empty() && std::find(axes.begin(), axes.end(), i) == axes.end()) {
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
}

std::function<void(OpSchema&)>
ReduceDocGenerator(const char* name, const char* func_body) {
  return ReduceDocGenerator(name, false, false, func_body);
}

std::function<void(OpSchema&)>
ReduceDocGeneratorWithFunctionBuilder(const char* name, ContextDependentFunctionBodyBuilder functionBuilder) {
  std::cout << "ReduceDocGenerator(name, false, false, nullptr, functionBuilder);" << std::endl;
  return ReduceDocGenerator(name, false, false, nullptr, functionBuilder);
}

ONNX_OPERATOR_SET_SCHEMA(ReduceMax, 13, OpSchema().FillUsing(ReduceDocGenerator("max", true)));

ONNX_OPERATOR_SET_SCHEMA(ReduceMin, 13, OpSchema().FillUsing(ReduceDocGenerator("min", true)));

ONNX_OPERATOR_SET_SCHEMA(ReduceSum, 13, OpSchema().FillUsing(ReduceDocGenerator("sum", false, true)));

const char* reduce_sum_square_func_body = R"ONNX(
  {
    data_square = Mul(data, data)
    axes = Constant <value_ints = @axes>()
    reduced = ReduceSum<keepdims = @keepdims>(data_square, axes)
  }
  )ONNX";

bool BuildContextDependentFunctionBodyReduceSumSquare(
    const FunctionBodyBuildContext& ctx,
    const OpSchema& schema,
    FunctionProto& functionProto) {
  FunctionBuilder builder(functionProto);
  auto* axes_attr = ctx.getAttribute("axes");
  if (axes_attr) {
    builder.Add("data_square = Mul(data, data)");
    builder.Add("axes = Constant <value_ints = @axes>()");
    builder.Add("reduced = ReduceSum<keepdims = @keepdims>(data_square, axes)");
  } else {
    builder.Add("data_square = Mul(data, data)");
    builder.Add("reduced = ReduceSum<keepdims = @keepdims>(data_square)");
  }
  schema.BuildFunction(functionProto);
  return true;
}

// ONNX_OPERATOR_SET_SCHEMA(ReduceSumSquare, 13, OpSchema().FillUsing(ReduceDocGenerator("sum square", reduce_sum_square_func_body)));
ONNX_OPERATOR_SET_SCHEMA(ReduceSumSquare, 13, OpSchema().FillUsing(
  ReduceDocGeneratorWithFunctionBuilder("sum square", BuildContextDependentFunctionBodyReduceSumSquare)));

ONNX_OPERATOR_SET_SCHEMA(ReduceMean, 13, OpSchema().FillUsing(ReduceDocGenerator("mean")));

ONNX_OPERATOR_SET_SCHEMA(ReduceProd, 13, OpSchema().FillUsing(ReduceDocGenerator("product")));

ONNX_OPERATOR_SET_SCHEMA(ReduceLogSum, 13, OpSchema().FillUsing(ReduceDocGenerator("log sum")));

ONNX_OPERATOR_SET_SCHEMA(ReduceLogSumExp, 13, OpSchema().FillUsing(ReduceDocGenerator("log sum exponent")));

// const char* reduce_l1_func_body = R"ONNX(
//   {
//     data_abs = Abs(data)
//     reduced = ReduceSum(data_square, axes, keepdims=keepdims)
//   }
//   )ONNX";
ONNX_OPERATOR_SET_SCHEMA(ReduceL1, 13, OpSchema().FillUsing(ReduceDocGenerator("L1 norm")));

// const char* reduce_l2_func_body = R"ONNX(
//   {
//     sum_square = ReduceSumSquare(data, axes, keepdims=keepdims)
//     sum_square_dbl = Cast (sum_square, to=1)
//     sqrt = Sqrt(sum_square_dbl)
//     reduced = CastLike(sqrt, data)
//   }
//   )ONNX";
ONNX_OPERATOR_SET_SCHEMA(ReduceL2, 13, OpSchema().FillUsing(ReduceDocGenerator("L2 norm")));

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
