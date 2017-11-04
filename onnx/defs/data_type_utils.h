// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#ifndef ONNX_DATA_TYPE_UTILS_H
#define ONNX_DATA_TYPE_UTILS_H

#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "onnx/onnx.pb.h"

namespace onnx
{
    // String pointer as unique TypeProto identifier.
    typedef const std::string* DataType;

    namespace Utils
    {
        // Singleton wrapper around allowed data types.
        // This implements construct on first use which is needed to ensure
        // static objects are initialized before use. Ops registration does not work
        // properly without this.
        class TypesWrapper
        {
        public:

            static TypesWrapper& GetTypesWrapper();

            // DataType strings. These should match the DataTypes defined in onnx.proto
            const std::string c_float16 = "float16";
            const std::string c_float = "float";
            const std::string c_double = "double";
            const std::string c_int8 = "int8";
            const std::string c_int16 = "int16";
            const std::string c_int32 = "int32";
            const std::string c_int64 = "int64";
            const std::string c_uint8 = "uint8";
            const std::string c_uint16 = "uint16";
            const std::string c_uint32 = "uint32";
            const std::string c_uint64 = "uint64";
            const std::string c_complex64 = "complex64";
            const std::string c_complex128 = "complex128";
            const std::string c_string = "string";
            const std::string c_bool = "bool";

            std::unordered_set<std::string>& GetAllowedDataTypes();

            ~TypesWrapper() = default;
            TypesWrapper(const TypesWrapper&) = delete;
            void operator=(const TypesWrapper&) = delete;
        private:
            TypesWrapper() = default;
        };

        // Data type utility, which maintains a global type string to TypeProto map.
        // DataType (string pointer) is used as unique data type identifier for efficiency.
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
        class DataTypeUtils
        {
        public:
            static DataType ToType(const std::string& p_type);

            static DataType ToType(const TypeProto& p_type);

            static const TypeProto& ToTypeProto(const DataType& p_type);

        private:
            static void FromString(const std::string& p_src, TypeProto& p_type);

            static void FromDataTypeString(const std::string& p_src, TensorProto::DataType& p_type);

            static std::string ToString(const TypeProto& p_type, const std::string& left = "", const std::string& right = "");

            static std::string ToDataTypeString(const TensorProto::DataType& p_type);

            static bool IsValidDataTypeString(const std::string& p_dataType);

            static std::unordered_map<std::string, TypeProto>& GetTypeStrToProtoMap();

            // Returns lock used for concurrent updates to TypeStrToProtoMap.
            static std::mutex& GetTypeStrLock();
        };
    }
}

#endif // ! ONNX_DATA_TYPE_UTILS_H
