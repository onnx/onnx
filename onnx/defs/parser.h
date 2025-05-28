/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Experimental language syntax and parser for ONNX. Please note that the syntax as formalized
// by this parser is preliminary and may change.

#pragma once

#include <cctype>
#include <string>
#include <unordered_map>

#include "onnx/common/status.h"
#include "onnx/onnx_pb.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {

using namespace ONNX_NAMESPACE::Common;

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
    for (const auto& pair : Instance()) {
      if (pair.second == dtype)
        return pair.first;
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
    map_["uint4"] = TensorProto_DataType_UINT4;
    map_["int4"] = TensorProto_DataType_INT4;
    map_["float4e2m1"] = TensorProto_DataType_FLOAT4E2M1;
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

class ParserBase {
 public:
  explicit ParserBase(const std::string& str)
      : start_(str.data()), next_(str.data()), end_(str.data() + str.length()), saved_pos_(next_) {}

  explicit ParserBase(const char* cstr) : start_(cstr), next_(cstr), end_(cstr + strlen(cstr)), saved_pos_(next_) {}

  void SavePos() {
    saved_pos_ = next_;
  }

  void RestorePos() {
    next_ = saved_pos_;
  }

  std::string GetCurrentPos() {
    uint32_t line = 1, col = 1;
    for (const char* p = start_; p < next_; ++p) {
      if (*p == '\n') {
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
    // Special cases: empty input string, and parse-error at first character.
    const char* p = next_ < end_ ? next_ : next_ - 1;
    while ((p > start_) && isspace(*p))
      --p;
    while ((p > start_) && (*p != '\n'))
      --p;
    // Start at character after '\n' unless we are at start of input
    const char* context_start = (p > start_) ? (p + 1) : start_;
    for (p = context_start; (p < end_) && (*p != '\n'); ++p)
      ;
    return std::string(context_start, p - context_start);
  }

  template <typename... Args>
  Status ParseError(const Args&... args) {
    return Status(
        StatusCategory::NONE,
        StatusCode::FAIL,
        ONNX_NAMESPACE::MakeString(
            "[ParseError at position ", GetCurrentPos(), "]\n", "Error context: ", GetErrorContext(), "\n", args...));
  }

  void SkipWhiteSpace() {
    do {
      while ((next_ < end_) && (isspace(*next_)))
        ++next_;
      if ((next_ >= end_) || ((*next_) != '#'))
        return;
      // Skip rest of the line:
      while ((next_ < end_) && ((*next_) != '\n'))
        ++next_;
    } while (true);
  }

  int NextChar(bool skipspace = true) {
    if (skipspace)
      SkipWhiteSpace();
    return (next_ < end_) ? *next_ : 0;
  }

  bool Matches(char ch, bool skipspace = true) {
    if (skipspace)
      SkipWhiteSpace();
    if ((next_ < end_) && (*next_ == ch)) {
      ++next_;
      return true;
    }
    return false;
  }

  Status Match(char ch, bool skipspace = true) {
    if (!Matches(ch, skipspace))
      return ParseError("Expected character ", ch, " not found.");
    return Status::OK();
  }

  bool EndOfInput() {
    SkipWhiteSpace();
    return (next_ >= end_);
  }

  enum class LiteralType : std::uint8_t { UNDEFINED, INT_LITERAL, FLOAT_LITERAL, STRING_LITERAL };

  struct Literal {
    LiteralType type{LiteralType::UNDEFINED};
    std::string value;
  };

  Status Parse(Literal& result);

  Status Parse(int64_t& val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal));
    if (literal.type != LiteralType::INT_LITERAL)
      return ParseError("Integer value expected, but not found.");
    std::string s = literal.value;
    val = std::stoll(s);
    return Status::OK();
  }

  Status Parse(uint64_t& val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal));
    if (literal.type != LiteralType::INT_LITERAL)
      return ParseError("Integer value expected, but not found.");
    std::string s = literal.value;
    val = std::stoull(s);
    return Status::OK();
  }

  Status Parse(float& val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal));
    switch (literal.type) {
      case LiteralType::INT_LITERAL:
      case LiteralType::FLOAT_LITERAL:
        val = std::stof(literal.value);
        break;
      default:
        return ParseError("Unexpected literal type.");
    }
    return Status::OK();
  }

  Status Parse(double& val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal));
    switch (literal.type) {
      case LiteralType::INT_LITERAL:
      case LiteralType::FLOAT_LITERAL:
        val = std::stod(literal.value);
        break;
      default:
        return ParseError("Unexpected literal type.");
    }
    return Status::OK();
  }

  // Parse a string-literal enclosed within double-quotes.
  Status Parse(std::string& val) {
    Literal literal;
    CHECK_PARSER_STATUS(Parse(literal));
    if (literal.type != LiteralType::STRING_LITERAL)
      return ParseError("String value expected, but not found.");
    val = literal.value;
    return Status::OK();
  }

  // Parse an identifier, including keywords. If none found, this will
  // return an empty-string identifier.
  std::string ParseOptionalIdentifier() {
    SkipWhiteSpace();
    auto from = next_;
    if ((next_ < end_) && (isalpha(*next_) || (*next_ == '_'))) {
      ++next_;
      while ((next_ < end_) && (isalnum(*next_) || (*next_ == '_')))
        ++next_;
    }
    return std::string(from, next_ - from);
  }

  Status ParseIdentifier(std::string& id) {
    id = ParseOptionalIdentifier();
    if (id.empty())
      return ParseError("Identifier expected but not found.");
    return Status::OK();
  }

  Status ParseQuotableIdentifier(std::string& id) {
    if (NextChar() == '"') {
      return Parse(id);
    }
    return ParseIdentifier(id);
  }

  Status ParseOptionalQuotableIdentifier(std::string& id) {
    if (NextChar() == '"') {
      return Parse(id);
    }
    id = ParseOptionalIdentifier();
    return Status::OK();
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
  Status ParseOptionalQuotableIdentifier(std::string& id, bool& id_found) {
    if (NextChar() == '"') {
      id_found = true;
      return Parse(id);
    }
    id = ParseOptionalIdentifier();
    id_found = !id.empty() || NextChar() == ',';
    return Status::OK();
  }

  std::string PeekIdentifier() {
    SavePos();
    auto id = ParseOptionalIdentifier();
    RestorePos();
    return id;
  }

  Status Parse(KeyWordMap::KeyWord& keyword) {
    std::string id;
    CHECK_PARSER_STATUS(ParseIdentifier(id));
    keyword = KeyWordMap::Lookup(id);
    return Status::OK();
  }

 protected:
  const char* start_;
  const char* next_;
  const char* end_;
  const char* saved_pos_;

  bool NextIsValidFloatString();
};

class OnnxParser : public ParserBase {
 public:
  explicit OnnxParser(const char* cstr) : ParserBase(cstr) {}

  Status Parse(TensorShapeProto& shape);

  Status Parse(TypeProto& typeProto);

  Status Parse(StringStringList& stringStringList);

  Status Parse(TensorProto& tensorProto);

  Status Parse(AttributeProto& attr);

  Status Parse(AttributeProto& attr, std::string& name);

  Status Parse(AttrList& attrlist);

  Status Parse(NodeProto& node);

  Status Parse(NodeList& nodelist);

  Status Parse(GraphProto& graph);

  Status Parse(FunctionProto& fn);

  Status Parse(ModelProto& model);

  template <typename T>
  static Status Parse(T& parsedData, const char* input) {
    OnnxParser parser(input);
    return parser.Parse(parsedData);
  }

 private:
  Status Parse(std::string name, GraphProto& graph);

  Status Parse(IdList& idlist);

  Status Parse(char open, IdList& idlist, char close);

  Status Parse(IdList& idlist, AttrList& attrlist);

  Status Parse(char open, IdList& idlist, AttrList& attrlist, char close);

  Status ParseSingleAttributeValue(AttributeProto& attr, AttributeProto_AttributeType expected);

  Status Parse(ValueInfoProto& valueinfo);

  Status ParseGraphInputOutput(ValueInfoList& vilist);

  Status ParseFunctionInputOutput(IdList& idlist, ValueInfoList& vilist);

  Status Parse(char open, ValueInfoList& vilist, char close);

  Status ParseInput(ValueInfoList& vilist, TensorList& initializers);

  Status ParseValueInfo(ValueInfoList& vilist, TensorList& initializers);

  Status Parse(TensorProto& tensorProto, const TypeProto& tensorTypeProto);

  Status Parse(OpsetIdList& opsets);

  bool NextIsType();

  bool NextIsIdentifier();
};

} // namespace ONNX_NAMESPACE
