/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

namespace ONNX_NAMESPACE {

template <typename T>
void LoadProtoFromPath(const std::string proto_path, T& proto);

TensorProto* LoadExternalTensor(const TensorProto& external_tensor,
  const std::string model_dir);

} // namespace ONNX_NAMESPACE