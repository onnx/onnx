// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Experimental language syntax and parser for ONNX. Please note that the syntax as formalized
// by this parser is preliminary and may change.

#pragma once

#include <algorithm>
#include <string>
#include <string_view>
#include <unordered_map>

#include "onnx/common/common.h"
#include "onnx/common/status.h"
#include "onnx/onnx_pb.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {

// Locale-independent string-to-float/double conversion (defined in parser.cc).
ONNX_API float LocaleIndependentStof(const std::string& s);
ONNX_API double LocaleIndependentStod(const std::string& s);

using IdList = google::protobuf::RepeatedPtrField<std::string>;

using NodeList = google::protobuf::RepeatedPtrField<NodeProto>;

using AttrList = google::protobuf::RepeatedPtrField<AttributeProto>;

using ValueInfoList = google::protobuf::RepeatedPtrField<ValueInfoProto>;

using TensorList = google::protobuf::RepeatedPtrField<TensorProto>;

using OpsetIdList = google::protobuf::RepeatedPtrField<OperatorSetIdProto>;

using StringStringList = google::protobuf::RepeatedPtrField<StringStringEntryProto>;

#define CHECK_PARSER_STATUS(status) \
  {                                 \
    auto local_status_ = status;    \
    if (!local_status_.IsOK())      \
      return local_status_;         \
  }

template <typename Map>
// NOLINTNEXTLINE(bugprone-crtp-constructor-accessibility)
class StringIntMap {
 public:
  static const std::unordered_map<std::string, int32_t>& Instance() {
    static Map instance;
    return instance.map_;
  }

  static int32_t Lookup(const std::string& dtype) {
    auto it = Instance().find(dtype);
    if (it != Instance().end())
      return it->second;
    return 0;
  }

  static const std::string& ToString(int32_t dtype) {
    static std::string undefined("undefined");
    for (const auto& [name, value] : Instance()) {
      if (value == dtype)
        return name;
    }
    return undefined;
  }

 protected:
  std::unordered_map<std::string, int32_t> map_;
};

class PrimitiveTypeNameMap : public StringIntMap<PrimitiveTypeNameMap> {
 public:
  PrimitiveTypeNameMap() : StringIntMap() {
    map_["float"] = TensorProto_DataType_FLOAT;
    map_["uint8"] = TensorProto_DataType_UINT8;
    map_["int8"] = TensorProto_DataType_INT8;
    map_["uint16"] = TensorProto_DataType_UINT16;
    map_["int16"] = TensorProto_DataType_INT16;
    map_["int32"] = TensorProto_DataType_INT32;
    map_["int64"] = TensorProto_DataType_INT64;
    map_["string"] = TensorProto_DataType_STRING;
    map_["bool"] = TensorProto_DataType_BOOL;
    map_["float16"] = TensorProto_DataType_FLOAT16;
    map_["double"] = TensorProto_DataType_DOUBLE;
    map_["uint32"] = TensorProto_DataType_UINT32;
    map_["uint64"] = TensorProto_DataType_UINT64;
    map_["complex64"] = TensorProto_DataType_COMPLEX64;
    map_["complex128"] = TensorProto_DataType_COMPLEX128;
    map_["bfloat16"] = TensorProto_DataType_BFLOAT16;
    map_["float8e4m3fn"] = TensorProto_DataType_FLOAT8E4M3FN;
    map_["float8e4m3fnuz"] = TensorProto_DataType_FLOAT8E4M3FNUZ;
    map_["float8e5m2"] = TensorProto_DataType_FLOAT8E5M2;
    map_["float8e5m2fnuz"] = TensorProto_DataType_FLOAT8E5M2FNUZ;
    map_["float8e8m0"] = TensorProto_DataType_FLOAT8E8M0;
    map_["uint4"] = TensorProto_DataType_UINT4;
    map_["int4"] = TensorProto_DataType_INT4;
    map_["float4e2m1"] = TensorProto_DataType_FLOAT4E2M1;
    map_["uint2"] = TensorProto_DataType_UINT2;
    map_["int2"] = TensorProto_DataType_INT2;
  }

  static bool IsTypeName(const std::string& dtype) {
    return Lookup(dtype) != 0;
  }
};

class AttributeTypeNameMap : public StringIntMap<AttributeTypeNameMap> {
 public:
  AttributeTypeNameMap() : StringIntMap() {
    map_["float"] = AttributeProto_AttributeType_FLOAT;
    map_["int"] = AttributeProto_AttributeType_INT;
    map_["string"] = AttributeProto_AttributeType_STRING;
    map_["tensor"] = AttributeProto_AttributeType_TENSOR;
    map_["graph"] = AttributeProto_AttributeType_GRAPH;
    map_["sparse_tensor"] = AttributeProto_AttributeType_SPARSE_TENSOR;
    map_["type_proto"] = AttributeProto_AttributeType_TYPE_PROTO;
    map_["floats"] = AttributeProto_AttributeType_FLOATS;
    map_["ints"] = AttributeProto_AttributeType_INTS;
    map_["strings"] = AttributeProto_AttributeType_STRINGS;
    map_["tensors"] = AttributeProto_AttributeType_TENSORS;
    map_["graphs"] = AttributeProto_AttributeType_GRAPHS;
    map_["sparse_tensors"] = AttributeProto_AttributeType_SPARSE_TENSORS;
    map_["type_protos"] = AttributeProto_AttributeType_TYPE_PROTOS;
  }
};

class KeyWordMap {
 public:
  enum class KeyWord : std::uint8_t {
    NONE,
    IR_VERSION,
    OPSET_IMPORT,
    PRODUCER_NAME,
    PRODUCER_VERSION,
    DOMAIN_KW,
    MODEL_VERSION,
    DOC_STRING,
    METADATA_PROPS,
    SEQ_TYPE,
    MAP_TYPE,
    OPTIONAL_TYPE,
    SPARSE_TENSOR_TYPE,
    OVERLOAD_KW
  };

  KeyWordMap() {
    map_["ir_version"] = KeyWord::IR_VERSION;
    map_["opset_import"] = KeyWord::OPSET_IMPORT;
    map_["producer_name"] = KeyWord::PRODUCER_NAME;
    map_["producer_version"] = KeyWord::PRODUCER_VERSION;
    map_["domain"] = KeyWord::DOMAIN_KW;
    map_["model_version"] = KeyWord::MODEL_VERSION;
    map_["doc_string"] = KeyWord::DOC_STRING;
    map_["metadata_props"] = KeyWord::METADATA_PROPS;
    map_["seq"] = KeyWord::SEQ_TYPE;
    map_["map"] = KeyWord::MAP_TYPE;
    map_["optional"] = KeyWord::OPTIONAL_TYPE;
    map_["sparse_tensor"] = KeyWord::SPARSE_TENSOR_TYPE;
    map_["overload"] = KeyWord::OVERLOAD_KW;
  }

  static const std::unordered_map<std::string, KeyWord>& Instance();

  static KeyWord Lookup(const std::string& id) {
    auto it = Instance().find(id);
    if (it != Instance().end())
      return it->second;
    return KeyWord::NONE;
  }

  static const std::string& ToString(KeyWord kw);

 private:
  std::unordered_map<std::string, KeyWord> map_;
};

// Locale-independent ASCII classification, so that the text format does not
// vary with the process locale. Accepts char or the int returned by
// ParserBase::NextChar; non-ASCII bytes classify as false either way.
constexpr bool IsSpace(int c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\v' || c == '\f' || c == '\r';
}

constexpr bool IsAlpha(int c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

constexpr bool IsDigit(int c) {
  return c >= '0' && c <= '9';
}

constexpr bool IsAlnum(int c) {
  return IsAlpha(c) || IsDigit(c);
}

class ParserBase {
 public:
  explicit ParserBase(std::string_view input) : input_(input) {}

  void SavePos() {
    saved_pos_ = pos_;
  }

  void RestorePos() {
    pos_ = saved_pos_;
  }

  std::string GetCurrentPos() {
    uint32_t line = 1, col = 1;
    for (size_t i = 0; i < pos_; ++i) {
      if (input_[i] == '\n') {
        ++line;
        col = 1;
      } else {
        ++col;
      }
    }
    return ONNX_NAMESPACE::MakeString("(line: ", line, " column: ", col, ")");
  }

  // Return a suitable suffix of what has been parsed to provide error message context:
  // return the line containing the last non-space character preceding the error (if it exists).
  std::string GetErrorContext() {
    if (input_.empty())
      return std::string();
    // Special case: a parse-error at end of input starts from the last character.
    size_t p = (pos_ < input_.size()) ? pos_ : input_.size() - 1;
    while ((p > 0) && IsSpace(input_[p]))
      --p;
    while ((p > 0) && (input_[p] != '\n'))
      --p;
    // Start at character after '\n' unless we are at start of input
    size_t context_start = (p > 0) ? (p + 1) : 0;
    size_t context_end = context_start;
    while ((context_end < input_.size()) && (input_[context_end] != '\n'))
      ++context_end;
    return std::string(input_.substr(context_start, context_end - context_start));
  }

  template <typename... Args>
  Common::Status ParseError(const Args&... args) {
    return Common::Status(
        Common::StatusCategory::NONE,
        Common::StatusCode::FAIL,
        ONNX_NAMESPACE::MakeString(
            "[ParseError at position ", GetCurrentPos(), "]\n", "Error context: ", GetErrorContext(), "\n", args...));
  }

  void SkipWhiteSpace() {
    do {
      while (!AtEnd() && IsSpace(Cur()))
        ++pos_;
      if (AtEnd() || (Cur() != '#'))
        return;
      // Skip rest of the line; the loop then consumes the newline as whitespace.
      pos_ = std::min(input_.find('\n', pos_), input_.size());
    } while (true);
  }

  int NextChar(bool skipspace = true) {
    if (skipspace)
      SkipWhiteSpace();
    // Return as unsigned char so byte values are non-negative.
    return AtEnd() ? 0 : static_cast<unsigned char>(Cur());
  }

  bool Matches(char ch, bool skipspace = true) {
    if (skipspace)
      SkipWhiteSpace();
    if (!AtEnd() && (Cur() == ch)) {
      ++pos_;
      return true;
    }
    return false;
  }

  Common::Status Match(char ch, bool skipspace = true) {
    if (!Matches(ch, skipspace))
      return ParseError("Expected character ", ch, " not found.");
    return Common::Status::OK();
  }

  bool EndOfInput() {
    SkipWhiteSpace();
    return AtEnd();
  }

  enum class LiteralType : std::uint8_t { UNDEFINED, INT_LITERAL, FLOAT_LITERAL, STRING_LITERAL };

  struct Literal {
    LiteralType type{LiteralType::UNDEFINED};
    std::string value;
  };

  Common::Status Parse(Literal& result);

  Common::Status Parse(int64_t& val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal))
    if (literal.type != LiteralType::INT_LITERAL)
      return ParseError("Integer value expected, but not found.");
    val = std::stoll(literal.value);
    return Common::Status::OK();
  }

  Common::Status Parse(uint64_t& val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal))
    if (literal.type != LiteralType::INT_LITERAL)
      return ParseError("Integer value expected, but not found.");
    val = std::stoull(literal.value);
    return Common::Status::OK();
  }

  Common::Status Parse(float& val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal))
    switch (literal.type) {
      case LiteralType::INT_LITERAL:
      case LiteralType::FLOAT_LITERAL:
        val = LocaleIndependentStof(literal.value);
        break;
      default:
        return ParseError("Unexpected literal type.");
    }
    return Common::Status::OK();
  }

  Common::Status Parse(double& val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal))
    switch (literal.type) {
      case LiteralType::INT_LITERAL:
      case LiteralType::FLOAT_LITERAL:
        val = LocaleIndependentStod(literal.value);
        break;
      default:
        return ParseError("Unexpected literal type.");
    }
    return Common::Status::OK();
  }

  // Parse a string-literal enclosed within double-quotes.
  Common::Status Parse(std::string& val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal))
    if (literal.type != LiteralType::STRING_LITERAL)
      return ParseError("String value expected, but not found.");
    val = literal.value;
    return Common::Status::OK();
  }

  // Parse an identifier, including keywords. If none found, this will
  // return an empty-string identifier.
  std::string ParseOptionalIdentifier() {
    SkipWhiteSpace();
    size_t from = pos_;
    if (!AtEnd() && (IsAlpha(Cur()) || (Cur() == '_'))) {
      ++pos_;
      while (!AtEnd() && (IsAlnum(Cur()) || (Cur() == '_')))
        ++pos_;
    }
    return std::string(input_.substr(from, pos_ - from));
  }

  Common::Status ParseIdentifier(std::string& id) {
    id = ParseOptionalIdentifier();
    if (id.empty())
      return ParseError("Identifier expected but not found.");
    return Common::Status::OK();
  }

  Common::Status ParseQuotableIdentifier(std::string& id) {
    if (NextChar() == '"') {
      return Parse(id);
    }
    return ParseIdentifier(id);
  }

  Common::Status ParseOptionalQuotableIdentifier(std::string& id) {
    if (NextChar() == '"') {
      return Parse(id);
    }
    id = ParseOptionalIdentifier();
    return Common::Status::OK();
  }

  // Parse an optional quotable identifier, and return whether an identifier was found
  // in the output parameter 'id_found'.
  // A empty string followed by a comma is considered to be a valid, but empty, identifier.
  // This helps handle the following different cases:
  // "Op()" has no operands
  // "Op(,x)" has two operands, the first being empty.
  // 'Op("")' has one operand, which is an empty string.
  // 'Op(,)' has one operand, which is an empty string.
  // Thus, this will also allow a trailing comma after a non-empty identifier with no effect.
  // 'Op(x,)' has one operand, which is 'x'.
  //
  // This is mostly for some backward compatibility. "" is a simpler way to represent an
  // empty identifier that is less confusing and is recommended.
  Common::Status ParseOptionalQuotableIdentifier(std::string& id, bool& id_found) {
    if (NextChar() == '"') {
      id_found = true;
      return Parse(id);
    }
    id = ParseOptionalIdentifier();
    id_found = !id.empty() || NextChar() == ',';
    return Common::Status::OK();
  }

  std::string PeekIdentifier() {
    SavePos();
    auto id = ParseOptionalIdentifier();
    RestorePos();
    return id;
  }

  Common::Status Parse(KeyWordMap::KeyWord& keyword) {
    std::string id;
    CHECK_PARSER_STATUS(ParseIdentifier(id))
    keyword = KeyWordMap::Lookup(id);
    return Common::Status::OK();
  }

 protected:
  // True when the cursor has consumed all input.
  bool AtEnd() const {
    return pos_ >= input_.size();
  }
  // Character at the cursor; only valid when !AtEnd().
  char Cur() const {
    return input_[pos_];
  }
  std::string_view input_;
  size_t pos_ = 0;
  size_t saved_pos_ = 0;

  bool NextIsValidFloatString();
};

class OnnxParser : public ParserBase {
 public:
  using ParserBase::ParserBase;

  ONNX_API Common::Status Parse(TensorShapeProto& shape);

  ONNX_API Common::Status Parse(TypeProto& typeProto);

  ONNX_API Common::Status Parse(StringStringList& stringStringList);

  ONNX_API Common::Status Parse(TensorProto& tensorProto);

  ONNX_API Common::Status Parse(AttributeProto& attr);

  ONNX_API Common::Status Parse(AttributeProto& attr, std::string& name);

  ONNX_API Common::Status Parse(AttrList& attrlist);

  ONNX_API Common::Status Parse(NodeProto& node);

  ONNX_API Common::Status Parse(NodeList& nodelist);

  ONNX_API Common::Status Parse(GraphProto& graph);

  ONNX_API Common::Status Parse(FunctionProto& fn);

  ONNX_API Common::Status Parse(ModelProto& model);

  template <typename T>
  static Common::Status Parse(T& parsedData, std::string_view input) {
    OnnxParser parser(input);
    return parser.Parse(parsedData);
  }

 private:
  Common::Status Parse(std::string name, GraphProto& graph);

  Common::Status Parse(IdList& idlist);

  Common::Status Parse(char open, IdList& idlist, char close);

  Common::Status Parse(IdList& idlist, AttrList& attrlist);

  Common::Status Parse(char open, IdList& idlist, AttrList& attrlist, char close);

  Common::Status ParseSingleAttributeValue(AttributeProto& attr, AttributeProto_AttributeType expected);

  Common::Status Parse(ValueInfoProto& valueinfo);

  Common::Status ParseGraphInputOutput(ValueInfoList& vilist);

  Common::Status ParseFunctionInputOutput(IdList& idlist, ValueInfoList& vilist);

  Common::Status Parse(char open, ValueInfoList& vilist, char close);

  Common::Status ParseInput(ValueInfoList& inputs, TensorList& initializers);

  Common::Status ParseValueInfo(ValueInfoList& value_infos, TensorList& initializers);

  Common::Status Parse(TensorProto& tensorProto, const TypeProto& tensorTypeProto);

  Common::Status Parse(OpsetIdList& opsets);

  bool NextIsType();

  bool NextIsIdentifier();
};

} // namespace ONNX_NAMESPACE
