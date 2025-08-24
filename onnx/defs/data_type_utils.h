/*
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ONNX_DATA_TYPE_UTILS_H
#define ONNX_DATA_TYPE_UTILS_H

#include <mutex>
#include <string>
#include <unordered_map>

#include "onnx/common/common.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE {
// String pointer as unique TypeProto identifier.
using DataType = const std::string*;

namespace Utils {

// Data type utility, which maintains a global type string to TypeProto map.
// DataType (string pointer) is used as unique data type identifier for
// efficiency.
//
// Grammar for data type string:
// <type> ::= <data_type> |
//            tensor(<data_type>) |
//            seq(<type>) |
//            map(<data_type>, <type>)
// <data_type> :: = float | int32 | string | bool | uint8
//                | int8 | uint16 | int16 | int64 | float16 | double
//
// NOTE: <type> ::= <data_type> means the data is scalar (zero dimension).
//
// Example: float, tensor(float), etc.
//
class DataTypeUtils final {
 public:
  // If the DataType input is invalid, this function will throw std::invalid_argument exception.
  // If ONNX_NO_EXCEPTIONS is set it will abort.
  static DataType ToType(const std::string& type_str);

  // If the DataType input is invalid, this function will throw std::invalid_argument exception.
  // If ONNX_NO_EXCEPTIONS is set it will abort.
  static DataType ToType(const TypeProto& type_proto);

  // If the DataType input is invalid, this function will throw std::invalid_argument exception.
  // If ONNX_NO_EXCEPTIONS is set it will abort.
  static const TypeProto& ToTypeProto(const DataType& data_type);
  static std::string ToDataTypeString(int32_t tensor_data_type);

 private:
  static void FromString(const std::string& type_str, TypeProto& type_proto);

  static int32_t FromDataTypeString(const std::string& type_str);

  static std::string ToString(const TypeProto& type_proto, const std::string& left = "", const std::string& right = "");

  // If int32_t input is invalid, this function will throw an exception.
  // If ONNX_NO_EXCEPTIONS is set it will abort.

  static bool IsValidDataType(int32_t tensor_data_type) {
    switch (tensor_data_type) {
      case TensorProto_DataType_FLOAT:
      case TensorProto_DataType_UINT8:
      case TensorProto_DataType_INT8:
      case TensorProto_DataType_UINT16:
      case TensorProto_DataType_INT16:
      case TensorProto_DataType_INT32:
      case TensorProto_DataType_INT64:
      case TensorProto_DataType_STRING:
      case TensorProto_DataType_BOOL:
      case TensorProto_DataType_FLOAT16:
      case TensorProto_DataType_DOUBLE:
      case TensorProto_DataType_UINT32:
      case TensorProto_DataType_UINT64:
      case TensorProto_DataType_COMPLEX64:
      case TensorProto_DataType_COMPLEX128:
      case TensorProto_DataType_BFLOAT16:
      case TensorProto_DataType_FLOAT8E4M3FN:
      case TensorProto_DataType_FLOAT8E4M3FNUZ:
      case TensorProto_DataType_FLOAT8E5M2:
      case TensorProto_DataType_FLOAT8E5M2FNUZ:
      case TensorProto_DataType_FLOAT8E8M0:
      case TensorProto_DataType_UINT4:
      case TensorProto_DataType_INT4:
      case TensorProto_DataType_FLOAT4E2M1:
      case TensorProto_DataType_FLOAT6E2M3:
      case TensorProto_DataType_FLOAT6E3M2:
        return true;
      default:
        return false;
    }
  }

  static std::unordered_map<std::string, TypeProto>& GetTypeStrToProtoMap();

  // Returns lock used for concurrent updates to TypeStrToProtoMap.
  static std::mutex& GetTypeStrLock();
};
} // namespace Utils
} // namespace ONNX_NAMESPACE

#endif // ! ONNX_DATA_TYPE_UTILS_H
