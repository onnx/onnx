/*
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <ctype.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "onnx/onnx_pb.h"
#include "onnx/string_utils.h"

namespace ONNX_NAMESPACE {

using IdList = google::protobuf::RepeatedPtrField<std::string>;

using NodeList = google::protobuf::RepeatedPtrField<NodeProto>;

using AttrList = google::protobuf::RepeatedPtrField<AttributeProto>;

using ValueInfoList = google::protobuf::RepeatedPtrField<ValueInfoProto>;

// Exception class used for handling parse errors

class ParseError final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;

  ParseError(const std::string& message) : std::runtime_error(message) {}

  const char* what() const noexcept override {
    if (!expanded_message_.empty()) {
      return expanded_message_.c_str();
    }
    return std::runtime_error::what();
  }

  void AppendContext(const std::string& context) {
    expanded_message_ = ONNX_NAMESPACE::MakeString(std::runtime_error::what(), "\n\n==> Context: ", context);
  }

 private:
  std::string expanded_message_;
};

#define parse_error(...)                                                                                          \
  do {                                                                                                            \
    throw ParseError(ONNX_NAMESPACE::MakeString("[ParseError at position ", (next_ - start_), "]", __VA_ARGS__)); \
  } while (0)

#define parser_check(cond, msg) \
  if (!(cond))                  \
  parse_error(msg)

class PrimitiveTypeNameMap {
 public:
  PrimitiveTypeNameMap() {
    map_["float"] = 1;
    map_["uint8"] = 2;
    map_["int8"] = 3;
    map_["uint16"] = 4;
    map_["int16"] = 5;
    map_["int32"] = 6;
    map_["int64"] = 7;
    map_["string"] = 8;
    map_["bool"] = 9;
    map_["float16"] = 10;
    map_["double"] = 11;
    map_["uint32"] = 12;
    map_["uint64"] = 13;
    map_["complex64"] = 14;
    map_["complex128"] = 15;
    map_["bfloat16"] = 16;
  }

  static const std::unordered_map<std::string, int32_t>& Instance() {
    static PrimitiveTypeNameMap instance;
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

 private:
  std::unordered_map<std::string, int32_t> map_;
};

class KeyWordMap {
 public:
  enum class KeyWord {
    NONE,
    IR_VERSION,
    OPSET_IMPORT,
    PRODUCER_NAME,
    PRODUCER_VERSION,
    DOMAIN_KW,
    MODEL_VERSION,
    DOC_STRING,
    METADATA_PROPS
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
  }

  static const std::unordered_map<std::string, KeyWord>& Instance() {
    static KeyWordMap instance;
    return instance.map_;
  }

  static KeyWord Lookup(const std::string& id) {
    auto it = Instance().find(id);
    if (it != Instance().end())
      return it->second;
    return KeyWord::NONE;
  }

 private:
  std::unordered_map<std::string, KeyWord> map_;
};

class ParserBase {
 public:
  ParserBase(const std::string& str) : start_(str.data()), next_(str.data()), end_(str.data() + str.length()) {}

  ParserBase(const char* cstr) : start_(cstr), next_(cstr), end_(cstr + strlen(cstr)) {}

  void SkipWhiteSpace() {
    while ((next_ < end_) && (isspace(*next_)))
      next_++;
  }

  int NextChar(bool skipspace = true) {
    if (skipspace)
      SkipWhiteSpace();
    return (next_ < end_) ? *next_ : 0;
  }

  std::string ParseIdentifier() {
    SkipWhiteSpace();
    auto from = next_;
    if ((next_ < end_) && (isalpha(*next_) || (*next_ == '_'))) {
      next_++;
      while ((next_ < end_) && (isalnum(*next_) || (*next_ == '_')))
        next_++;
    }
    if (next_ == from)
      parse_error("Identifier expected but not found.");
    return std::string(from, next_ - from);
  }

  KeyWordMap::KeyWord ParseKeyWord() {
    return KeyWordMap::Lookup(ParseIdentifier());
  }

  enum class TokenType { INT_LITERAL, FLOAT_LITERAL, STRING_LITERAL };

  struct Token {
    TokenType type;
    std::string value;
  };

  bool ParseIntValue(uint64_t& val) {
    SkipWhiteSpace();
    auto from = next_;
    while ((next_ < end_) && (isdigit(*next_)))
      next_++;
    if (next_ == from)
      return false;
    val = std::stol(std::string(from, next_ - from));
    return true;
  }

  uint64_t ParseIntValue() {
    uint64_t result;
    if (!ParseIntValue(result))
      parse_error("Integer value expected but not found.");
    return result;
  }

  Token ParseValue() {
    Token result;
    bool decimal_point = false;
    auto nextch = NextChar();
    auto from = next_;
    if (nextch == '"') {
      next_++;
      // TODO: Handle escape characters
      while ((next_ < end_) && (*next_ != '"')) {
        next_++;
      }
      next_++;
      result.type = TokenType::STRING_LITERAL;
      result.value = std::string(from + 1, next_ - from - 2); // skip enclosing quotes

    } else if ((isdigit(nextch) || (nextch == '-'))) {
      next_++;

      while ((next_ < end_) && (isdigit(*next_) || (*next_ == '.'))) {
        if (*next_ == '.') {
          if (decimal_point)
            break; // Only one decimal point allowed in numeric literal
          decimal_point = true;
        }
        next_++;
      }

      if (next_ == from)
        parse_error("Value expected but not found.");

      result.value = std::string(from, next_ - from);
      result.type = decimal_point ? TokenType::FLOAT_LITERAL : TokenType::INT_LITERAL;
    }
    return result;
  }

  float ParseFloatValue() {
    auto token = ParseValue();
    switch (token.type) {
      case TokenType::INT_LITERAL:
      case TokenType::FLOAT_LITERAL:
        return std::stof(token.value);
        break;
      default:
        parse_error("Unexpected literal type.");
    }
  }

  std::string ParseString() {
    auto token = ParseValue();
    if (token.type != TokenType::STRING_LITERAL)
      parse_error("String value expected, but not found.");
    return token.value;
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

  void Match(char ch, bool skipspace = true) {
    if (!Matches(ch, skipspace))
      parse_error("Expected character %c but not found", ch);
  }

  bool EndOfInput() {
    SkipWhiteSpace();
    return (next_ >= end_);
  }

 protected:
  const char* start_;
  const char* next_;
  const char* end_;
};

class OnnxParser : public ParserBase {
 public:
  OnnxParser(const char* cstr) : ParserBase(cstr) {}

  void ParseIdList(IdList& idlist);

  void Parse(TensorShapeProto& shape);

  void Parse(TypeProto& typeProto);

  void Parse(TensorProto& tensorProto);

  void Parse(ValueInfoProto& valueinfo);

  void Parse(ValueInfoList& vilist);

  void ParseSingleAttributeValue(AttributeProto& attr);

  void Parse(AttributeProto& attr);

  void Parse(AttrList& attrlist);

  void Parse(NodeProto& node);

  void Parse(NodeList& nodelist);

  void Parse(GraphProto& graph);

  void Parse(ModelProto& model);

  template <typename T>
  static void Parse(T& parsedData, const char* input) {
    OnnxParser parser(input);
    parser.Parse(parsedData);
  }
};

} // namespace ONNX_NAMESPACE