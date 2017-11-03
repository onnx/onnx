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
    typedef const std::string* DTYPE;

    namespace Utils
    {
        // Simple class which contains pointers to external string buffer and a size.
        // This can be used to track a "valid" range/slice of the string.
        // Caller should ensure StringRange is not used after external storage has
        // been freed.
        class StringRange
        {
        public:
            StringRange();
            StringRange(const char* p_data, size_t p_size);
            StringRange(const std::string& p_str);
            StringRange(const char* p_data);
            const char* Data() const;
            size_t Size() const;
            bool Empty() const;
            char operator[](size_t p_idx) const;
            void Reset();
            void Reset(const char* p_data, size_t p_size);
            void Reset(const std::string& p_str);
            bool StartsWith(const StringRange& p_str) const;
            bool EndsWith(const StringRange& p_str) const;
            bool LStrip();
            bool LStrip(size_t p_size);
            bool LStrip(StringRange p_str);
            bool RStrip();
            bool RStrip(size_t p_size);
            bool RStrip(StringRange p_str);
            bool LAndRStrip();
            void ParensWhitespaceStrip();
            size_t Find(const char p_ch) const;

            // These methods provide a way to return the range of the string
            // which was discarded by LStrip(). i.e. We capture the string
            // range which was discarded.
            StringRange GetCaptured();
            void RestartCapture();

        private:
            // m_data + size tracks the "valid" range of the external string buffer.
            const char* m_data;
            size_t m_size;

            // m_start and m_end track the captured range.
            // m_end advances when LStrip() is called.
            const char* m_start;
            const char* m_end;
        };

        // Singleton wrapper around allowed data types.
        // This implements construct on first use which is needed to ensure
        // static objects are initialized before use. Ops registration does not work
        // properly without this.
        class TypesWrapper
        {
        public:

            static TypesWrapper& GetTypesWrapper();

            // DataType strings. These should match the DataTypes defined in Data.proto
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
        // DTYPE (string pointer) is used as unique data type identifier for efficiency.
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
            static DTYPE ToType(const std::string& p_type);

            static DTYPE ToType(const TypeProto& p_type);

            static const TypeProto& ToTypeProto(const DTYPE& p_type);

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
