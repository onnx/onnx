/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <cmath>
#include <numeric>

#include "onnx/defs/schema.h"

namespace ONNX_NAMESPACE {

ONNX_PNP_OPERATOR_SET_SCHEMA(LinalgSVD, 1,
                            OpSchema()
                                .SetDoc(R"DOC(For internal use.)DOC")
                                .Attr(
                                    "full_matrices",
                                    "",
                                    AttributeProto::INT,
                                    static_cast<int64_t>(1))
                                .Input(
                                    0,
                                    "A",
                                    "",
                                    "T")
                                .Output(
                                    0,
                                    "U",
                                    "",
                                    "T")
                                .Output(
                                    1,
                                    "S",
                                    "",
                                    "T")
                                .Output(
                                    2,
                                    "Vh",
                                    "",
                                    "T")
                                .TypeConstraint(
                                    "T",
                                    {"tensor(float)", "tensor(double)"},
                                    "")
                                .TypeAndShapeInferenceFunction([](ONNX_NAMESPACE::InferenceContext& ctx) {
                                  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 0);
                                  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 1);
                                  ONNX_NAMESPACE::propagateElemTypeFromInputToOutput(ctx, 0, 2);
                                  int64_t full_matrices = ctx.getAttribute("full_matrices")->i();

                                  const TensorShapeProto& A_shape = ctx.getInputType(0)->tensor_type().shape();
                                  const auto& M = A_shape.dim(A_shape.dim_size() - 2);
                                  const auto& N = A_shape.dim(A_shape.dim_size() - 1);
                                  if (!M.has_dim_value() || !N.has_dim_value()) {
                                    // cannot do shape inference without knowing dimension values
                                    return;
                                  }
                                  const auto& K = M.dim_value() < N.dim_value() ? M : N;
                                  auto* u_shape = ctx.getOutputType(0)->mutable_tensor_type()->mutable_shape();
                                  auto* s_shape = ctx.getOutputType(1)->mutable_tensor_type()->mutable_shape();
                                  auto* v_shape = ctx.getOutputType(2)->mutable_tensor_type()->mutable_shape();
                                  if (A_shape.dim_size() == 3) {
                                    const auto& batch_dim = A_shape.dim(0);
                                    *u_shape->add_dim() = batch_dim;
                                    *s_shape->add_dim() = batch_dim;
                                    *v_shape->add_dim() = batch_dim;
                                  }
                                  *u_shape->add_dim() = M;
                                  *u_shape->add_dim() = full_matrices ? M : K;
                                  *s_shape->add_dim() = K;
                                  *v_shape->add_dim() = full_matrices ? N : K;
                                  *v_shape->add_dim() = N;
                                }));

} // namespace ONNX_NAMESPACE