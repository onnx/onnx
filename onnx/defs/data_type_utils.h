// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ONNX_DATA_TYPE_UTILS_H
#define ONNX_DATA_TYPE_UTILS_H

#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "onnx/onnx_pb.h"

namespace onnx {

// ONNX domain.
const char* const ONNX_DOMAIN = "";
struct DataType;

// DataType pointer as unique TypeProto identifier.
typedef const DataType* PDataType;

namespace Utils {
// Data type utility, which maintains a global type string to TypeProto map.
// DataType (string pointer) is used as unique data type identifier for
// efficiency.
class DataTypeUtils {
 public:
  static void Register(PDataType p_data_type);

  static PDataType ToType(const std::string& type_id);

  static PDataType ToType(const TypeProto& type_proto);

  static std::string GetElementTypeStr(TensorProto_DataType elem_type);

 private:
  static std::string GetId(const TypeProto& type_proto);

  static std::unordered_map<std::string, PDataType>& GetTypeIdToDataMap();
};
} // namespace Utils

struct DataType {
  virtual ~DataType() {}
  const std::string& Domain() const;
  const std::string& Id() const;
  const std::string& Description() const;
  const TypeProto& ToProto() const;

  static PDataType Tensor_FLOAT;
  static PDataType Tensor_UINT8;
  static PDataType Tensor_INT8;
  static PDataType Tensor_UINT16;
  static PDataType Tensor_INT16;
  static PDataType Tensor_INT32;
  static PDataType Tensor_INT64;
  static PDataType Tensor_STRING;
  static PDataType Tensor_BOOL;
  static PDataType Tensor_FLOAT16;
  static PDataType Tensor_DOUBLE;
  static PDataType Tensor_UINT32;
  static PDataType Tensor_UINT64;
  static PDataType Tensor_COMPLEX64;
  static PDataType Tensor_COMPLEX128;
  static PDataType Map_Int64_String;
  static PDataType Map_Int64_Float;
  static PDataType Map_String_Int64;
  static PDataType Map_Int64_Double;
  static PDataType Map_String_Float;
  static PDataType Map_String_Double;

 protected:
  DataType() {}
  std::string description;
  std::string domain;
  std::string id;
  TypeProto type_proto;
};

template <int elemT>
struct TensorType : public DataType {
  static PDataType Type();

 private:
  TensorType() {
    domain = ONNX_DOMAIN;
    description = "tensor(" +
        Utils::DataTypeUtils::GetElementTypeStr(
                      static_cast<TensorProto_DataType>(elemT)) +
        ")";
    id = description;
    type_proto.mutable_tensor_type()->set_elem_type(
        (TensorProto_DataType)elemT);
    Utils::DataTypeUtils::Register(this);
  }
};

template <typename T>
struct Abstract : public DataType {
  static PDataType Type(
      const std::string& description,
      const std::string& domain = ONNX_DOMAIN);

 private:
  Abstract(
      const std::string& description_,
      const std::string& domain_ = ONNX_DOMAIN) {
    domain = domain_;
    description = description_;
    id = std::string(typeid(T).name());
    type_proto.mutable_abs_type()->set_domain(domain);
    type_proto.mutable_abs_type()->set_identifier(id);
    Utils::DataTypeUtils::Register(this);
  }
};

} // namespace onnx

#endif // ! ONNX_DATA_TYPE_UTILS_H