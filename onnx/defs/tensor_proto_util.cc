// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "tensor_proto_util.h"

namespace ONNX_NAMESPACE {

#define DEFINE_TO_TENSOR_ONE(type, enumType, field)      \
  template <>                                            \
  static TensorProto ToTensor<type>(const type& value) { \
    TensorProto t;                                       \
    t.set_data_type(##enumType);                         \
    t.add_##field##_data(value);                         \
    return t;                                            \
  }

#define DEFINE_TO_TENSOR_LIST(type, enumType, field)                   \
  template <>                                                          \
  static TensorProto ToTensor<type>(const std::vector<type>& values) { \
    TensorProto t;                                                     \
    t.clear_##field##_data();                                          \
    t.set_data_type(##enumType);                                       \
    for (const type& val : values) {                                   \
      t.add_##field##_data(val);                                       \
    }                                                                  \
    return t;                                                          \
  }

DEFINE_TO_TENSOR_ONE(float, TensorProto_DataType_FLOAT, float);
DEFINE_TO_TENSOR_ONE(bool, TensorProto_DataType_BOOL, int32);
DEFINE_TO_TENSOR_ONE(int32_t, TensorProto_DataType_INT32, int32);
DEFINE_TO_TENSOR_ONE(int64_t, TensorProto_DataType_INT64, int64);
DEFINE_TO_TENSOR_ONE(uint64_t, TensorProto_DataType_UINT64, uint64);
DEFINE_TO_TENSOR_ONE(double, TensorProto_DataType_DOUBLE, double);
DEFINE_TO_TENSOR_ONE(std::string, TensorProto_DataType_STRING, string);

DEFINE_TO_TENSOR_LIST(float, TensorProto_DataType_FLOAT, float);
DEFINE_TO_TENSOR_LIST(bool, TensorProto_DataType_BOOL, int32);
DEFINE_TO_TENSOR_LIST(int32_t, TensorProto_DataType_INT32, int32);
DEFINE_TO_TENSOR_LIST(int64_t, TensorProto_DataType_INT64, int64);
DEFINE_TO_TENSOR_LIST(uint64_t, TensorProto_DataType_UINT64, uint64);
DEFINE_TO_TENSOR_LIST(double, TensorProto_DataType_DOUBLE, double);
DEFINE_TO_TENSOR_LIST(std::string, TensorProto_DataType_STRING, string);
  
} // namespace ONNX_NAMESPACE
