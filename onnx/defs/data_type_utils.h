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

// String pointer as unique TypeProto identifier.
typedef const std::string* DataType;

template<int elemT>
struct TensorType {
    static DataType Type() {
        static TensorType tensor_type;
        return tensor_type.TypeInternal();
    }

private:

    TensorType() {
        tensor_type_key = "tensor(" + Utils::DataTypeUtils::GetElementTypeStr(static_cast<TensorProto_DataType>(elemT)) + ")";
        TypeProto tensor_type;
        tensor_type.mutable_tensor_type()->set_elem_type((TensorProto_DataType)elemT);
        Utils::DataTypeUtils::Register(TypeInternal(), tensor_type);
    }

    DataType TypeInternal() const {
        return &tensor_type_key;
    }

    std::string tensor_type_key;
};

template<typename T>
struct Abstract {
    static DataType Type(const std::string& domain = ONNX_DOMAIN) {
        static Abstract abs_type(domain);
        return abs_type.TypeInternal();
    }
private:

    Abstract(const std::string& domain = ONNX_DOMAIN) {
        abs_type_key = std::string(typeid(T).name());
        TypeProto abs_type;
        abs_type.mutable_abs_type()->set_domain(domain);
        abs_type.mutable_abs_type()->set_identifier(abs_type_key);
        Utils::DataTypeUtils::Register(TypeInternal(), abs_type);
    }

    DataType TypeInternal() const {
        return &abs_type_key;
    }

    std::string abs_type_key;
};

namespace Utils {
// Data type utility, which maintains a global type string to TypeProto map.
// DataType (string pointer) is used as unique data type identifier for
// efficiency.
class DataTypeUtils {
 public:
  
  static void Register(DataType type_key, const TypeProto& type_proto) {
      static std::mutex mutex;
      std::lock_guard<std::mutex> lock(mutex);
      if (GetTypeStrToProtoMap().find(*type_key) == GetTypeStrToProtoMap().end()) {
          GetTypeStrToProtoMap()[*type_key] = type_proto;
      }
      else {
          // One type is prevented from being registered multiple times
          // from different domain intentionally.
          assert(false);
      }
  }

  static DataType ToType(const std::string& type_str);

  static DataType ToType(const TypeProto& type_proto);

  static const TypeProto& ToTypeProto(const DataType& data_type);

  static std::string GetElementTypeStr(TensorProto_DataType elem_type);

 private:

  static std::string ToString(const TypeProto& type_proto);

  static std::unordered_map<std::string, TypeProto>& GetTypeStrToProtoMap();

};
} // namespace Utils
} // namespace onnx

#endif // ! ONNX_DATA_TYPE_UTILS_H