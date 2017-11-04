// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ONNX_DATA_TYPE_UTILS_H
#define ONNX_DATA_TYPE_UTILS_H

#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "onnx/onnx.pb.h"

namespace onnx {
// String pointer as unique TypeProto identifier.
typedef const std::string* DataType;

namespace Utils {
// Singleton wrapper around allowed data types.
// This implements construct on first use which is needed to ensure
// static objects are initialized before use. Ops registration does not work
// properly without this.
class TypesWrapper {
 public:
  static TypesWrapper& GetTypesWrapper();

  // DataType strings. These should match the DataTypes defined in onnx.proto
  const std::string kFloat16 = "float16";
  const std::string kFloat = "float";
  const std::string kDouble = "double";
  const std::string kInt8 = "int8";
  const std::string kInt16 = "int16";
  const std::string kInt32 = "int32";
  const std::string kInt64 = "int64";
  const std::string kUint8 = "uint8";
  const std::string kUint16 = "uint16";
  const std::string kUint32 = "uint32";
  const std::string kUint64 = "uint64";
  const std::string kComplex64 = "complex64";
  const std::string kComplex128 = "complex128";
  const std::string kString = "string";
  const std::string kBool = "bool";

  std::unordered_set<std::string>& GetAllowedDataTypes();

  ~TypesWrapper() = default;
  TypesWrapper(const TypesWrapper&) = delete;
  void operator=(const TypesWrapper&) = delete;

 private:
  TypesWrapper() = default;
};

// Data type utility, which maintains a global type string to TypeProto map.
// DataType (string pointer) is used as unique data type identifier for
// efficiency.
//
// Grammar for data type string:
// <type> ::= <data_type> | tensor(<data_type>) | sparse(<data_type>)
// <data_type> :: = float | int32 | string | bool | uint8
//                | int8 | uint16 | int16 | int64 | float16 | double
//
// NOTE: <type> ::= <data_type> means the data is scalar (zero dimension).
//
// Example: float, tensor(float), sparse(double), etc.
//
class DataTypeUtils {
 public:
  static DataType ToType(const std::string& type_str);

  static DataType ToType(const TypeProto& type_proto);

  static const TypeProto& ToTypeProto(const DataType& data_type);

 private:
  static void FromString(const std::string& type_str, TypeProto& type_proto);

  static void FromDataTypeString(
      const std::string& type_str,
      TensorProto::DataType& tensor_data_type);

  static std::string ToString(
      const TypeProto& type_proto,
      const std::string& left = "",
      const std::string& right = "");

  static std::string ToDataTypeString(const TensorProto::DataType& tensor_data_type);

  static bool IsValidDataTypeString(const std::string& type_str);

  static std::unordered_map<std::string, TypeProto>& GetTypeStrToProtoMap();

  // Returns lock used for concurrent updates to TypeStrToProtoMap.
  static std::mutex& GetTypeStrLock();
};
} // namespace Utils
} // namespace onnx

#endif // ! ONNX_DATA_TYPE_UTILS_H
