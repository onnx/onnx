// Copyright (c) ONNX Project Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Experimental language syntax and parser for ONNX. Please note that the syntax as formalized
// by this parser is preliminary and may change.

#include "onnx/defs/parser.h"

// POSIX locale extensions (newlocale, freelocale, strtof_l, strtod_l) are used
// to parse floating-point literals independently of the global locale.
#include <locale.h> // NOLINT(modernize-deprecated-headers)

#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_map>

#include "onnx/common/common.h"

// Locale-independent float/double parsing implementation.
// Uses std::from_chars when the stdlib supports it (__cpp_lib_to_chars >= 201611L),
// otherwise strtof_l/strtod_l with an explicit "C" locale. The strto*_l helpers
// also resolve from_chars range errors: underflow rounds to zero or a subnormal
// and is accepted; only overflow is an error.

namespace {

#ifdef _WIN32
struct CLocale {
  _locale_t loc;
  CLocale() : loc(_create_locale(LC_ALL, "C")) {}
  ~CLocale() {
    if (loc)
      _free_locale(loc);
  }
  CLocale(const CLocale&) = delete;
  CLocale& operator=(const CLocale&) = delete;
};
#else // POSIX (Linux, macOS)
struct CLocale {
  locale_t loc;
  CLocale() : loc(newlocale(LC_ALL_MASK, "C", nullptr)) {}
  ~CLocale() {
    if (loc)
      freelocale(loc);
  }
  CLocale(const CLocale&) = delete;
  CLocale& operator=(const CLocale&) = delete;
};
#endif

const CLocale& GetCLocale() {
  static const CLocale instance;
  return instance;
}

float StrtofC(const std::string& s) {
  const auto& cloc = GetCLocale();
  if (!cloc.loc) {
    ONNX_THROW("Failed to create C locale for float parsing");
  }
  char* end = nullptr;
  errno = 0;
#ifdef _WIN32
  const float val = _strtof_l(s.c_str(), &end, cloc.loc);
#else
  const float val = strtof_l(s.c_str(), &end, cloc.loc);
#endif
  // ERANGE with a finite result is underflow: the value is still the correctly
  // rounded zero or subnormal. Only overflow (±inf) is an error.
  if (end == s.c_str() || end != s.c_str() + s.size() || (errno == ERANGE && std::isinf(val))) {
    ONNX_THROW("Failed to parse float from string: " + s);
  }
  return val;
}

double StrtodC(const std::string& s) {
  const auto& cloc = GetCLocale();
  if (!cloc.loc) {
    ONNX_THROW("Failed to create C locale for double parsing");
  }
  char* end = nullptr;
  errno = 0;
#ifdef _WIN32
  const double val = _strtod_l(s.c_str(), &end, cloc.loc);
#else
  const double val = strtod_l(s.c_str(), &end, cloc.loc);
#endif
  if (end == s.c_str() || end != s.c_str() + s.size() || (errno == ERANGE && std::isinf(val))) {
    ONNX_THROW("Failed to parse double from string: " + s);
  }
  return val;
}

} // anonymous namespace

namespace ONNX_NAMESPACE {

#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L

float LocaleIndependentStof(const std::string& s) {
  float val = 0.0f;
  const char* const begin = s.data();
  const char* const end = begin + s.size();
  const auto result = std::from_chars(begin, end, val);
  if (result.ec == std::errc::result_out_of_range && result.ptr == end) {
    // from_chars leaves val unspecified on range errors; re-parse to accept
    // underflow (rounds to zero/subnormal) and reject only overflow.
    return StrtofC(s);
  }
  if (result.ec != std::errc{} || result.ptr != end) {
    ONNX_THROW("Failed to parse float from string: " + s);
  }
  return val;
}

double LocaleIndependentStod(const std::string& s) {
  double val = 0.0;
  const char* const begin = s.data();
  const char* const end = begin + s.size();
  const auto result = std::from_chars(begin, end, val);
  if (result.ec == std::errc::result_out_of_range && result.ptr == end) {
    return StrtodC(s);
  }
  if (result.ec != std::errc{} || result.ptr != end) {
    ONNX_THROW("Failed to parse double from string: " + s);
  }
  return val;
}

#else // No floating-point from_chars (e.g. Apple Clang)

float LocaleIndependentStof(const std::string& s) {
  return StrtofC(s);
}

double LocaleIndependentStod(const std::string& s) {
  return StrtodC(s);
}

#endif // __cpp_lib_to_chars

} // namespace ONNX_NAMESPACE

#define PARSE_TOKEN(x) CHECK_PARSER_STATUS(ParserBase::Parse(x))
#define PARSE(...) CHECK_PARSER_STATUS(Parse(__VA_ARGS__))
#define MATCH(...) CHECK_PARSER_STATUS(Match(__VA_ARGS__))

namespace ONNX_NAMESPACE {

Common::Status ParserBase::Parse(Literal& result) {
  bool decimal_point = false;
  auto nextch = NextChar();
  size_t from = pos_;
  if (nextch == '"') {
    ++pos_;
    bool has_escape = false;
    while (!AtEnd() && (Cur() != '"')) {
      if (Cur() == '\\') {
        has_escape = true;
        ++pos_;
        if (AtEnd())
          return ParseError("Incomplete string literal.");
      }
      ++pos_;
    }
    if (AtEnd())
      return ParseError("Incomplete string literal.");
    ++pos_;
    result.type = LiteralType::STRING_LITERAL;
    if (has_escape) {
      std::string& target = result.value;
      target.clear();
      target.reserve(pos_ - from - 2); // upper bound
      // input_[from] is the starting quote. input_[pos_-1] is the ending quote.
      // Copy what is in-between, except for the escape character
      while (++from < pos_ - 1) {
        // Copy current char, if not escape, or next char otherwise.
        target.push_back(input_[from] != '\\' ? input_[from] : input_[++from]);
      }
    } else {
      result.value = std::string(input_.substr(from + 1, pos_ - from - 2)); // skip enclosing quotes
    }
    return Common::Status::OK();
  }

  // Simplify the next ifs by consuming a possible negative sign.
  if (nextch == '-') {
    ++pos_;
    nextch = NextChar();
  }

  // Check for float literals that start with alphabet characters.
  if (IsAlpha(nextch)) {
    // Has to be a special float literal now: (-)*(nan|inf|infinity).
    if (NextIsValidFloatString()) {
      while (!AtEnd() && IsAlpha(Cur())) {
        ++pos_;
      }
      ONNX_TRY {
        static_cast<void>(LocaleIndependentStof(std::string(input_.substr(from, pos_ - from))));
        result.type = LiteralType::FLOAT_LITERAL;
        result.value = std::string(input_.substr(from, pos_ - from));
      }
      ONNX_CATCH(...) {
        ONNX_HANDLE_EXCEPTION([&]() { return ParseError("Encountered invalid float literal!"); });
      }
    } else {
      return ParseError("Encountered invalid float literal!");
    }
    return Common::Status::OK();
  }

  // Checking for numeric ints or float literal.
  if (IsDigit(nextch)) {
    ++pos_;

    while (!AtEnd() && (IsDigit(Cur()) || (Cur() == '.'))) {
      if (Cur() == '.') {
        if (decimal_point)
          break; // Only one decimal point allowed in numeric literal
        decimal_point = true;
      }
      ++pos_;
    }

    if (pos_ == from)
      return ParseError("Value expected but not found.");

    // Optional exponent syntax: (e|E)(+|-)?[0-9]+
    if (!AtEnd() && ((Cur() == 'e') || (Cur() == 'E'))) {
      decimal_point = true; // treat as float-literal
      ++pos_;
      if (!AtEnd() && ((Cur() == '+') || (Cur() == '-')))
        ++pos_;
      while (!AtEnd() && IsDigit(Cur()))
        ++pos_;
    }

    result.value = std::string(input_.substr(from, pos_ - from));
    result.type = decimal_point ? LiteralType::FLOAT_LITERAL : LiteralType::INT_LITERAL;
  }
  return Common::Status::OK();
}

bool ParserBase::NextIsValidFloatString() {
  auto nextch = NextChar();
  const size_t from = pos_;
  constexpr int INFINITY_LENGTH = 8;

  if (IsAlpha(nextch)) {
    while (!AtEnd() && IsAlpha(Cur()) && (pos_ - from) <= INFINITY_LENGTH) {
      ++pos_;
    }

    if (!AtEnd() && IsDigit(Cur())) { // No trailing digits
      pos_ = from;
      return false;
    }

    std::string candidate = std::string(input_.substr(from, pos_ - from));

    // Reset parser location before continuing.
    pos_ = from;

    std::transform(candidate.begin(), candidate.end(), candidate.begin(), [](char c) {
      // ASCII-only lowercasing; std::tolower is locale-dependent.
      return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    });
    if (candidate == std::string_view("inf") || candidate == std::string_view("infinity") ||
        candidate == std::string_view("nan")) {
      return true;
    }
  }
  return false;
}

// Parsing an IdList (list of identifiers separated by commas, where identifiers are allowed to be empty).
// Used to represent the list of inputs or outputs of a node.
// An empty identifier may be represented by an empty string "" or by nothing followed by a single comma.
// "Op()" has no operands
// "Op(,x)" has two operands, the first being empty.
// 'Op("")' has one operand, which is an empty string.
// 'Op(,)' has one operand, which is an empty string.
// Thus, this will also allow a trailing comma after a non-empty identifier with no effect.
// 'Op(x,)' has one operand, which is 'x'.
//
// This is mostly for some backward compatibility. "" is a simpler way to represent an
// empty identifier that is less confusing and is recommended.

Common::Status OnnxParser::Parse(IdList& idlist) {
  idlist.Clear();
  std::string id;
  bool found = false;
  CHECK_PARSER_STATUS(ParseOptionalQuotableIdentifier(id, found));
  if (!found)
    return Common::Status::OK();
  *idlist.Add() = id;
  while (Matches(',')) {
    CHECK_PARSER_STATUS(ParseOptionalQuotableIdentifier(id, found));
    if (!found)
      break;
    *idlist.Add() = id;
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(char open, IdList& idlist, char close) {
  idlist.Clear();
  if (Matches(open)) {
    PARSE(idlist);
    MATCH(close);
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(IdList& idlist, AttrList& attrlist) {
  idlist.Clear();
  attrlist.Clear();
  do {
    std::string id;
    CHECK_PARSER_STATUS(ParseQuotableIdentifier(id));
    auto next = NextChar();
    if (next == ':' || next == '=')
      Parse(*attrlist.Add(), id);
    else
      *idlist.Add() = id;
  } while (Matches(','));
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(char open, IdList& idlist, AttrList& attrlist, char close) {
  if (Matches(open)) {
    PARSE(idlist, attrlist);
    MATCH(close);
  } else {
    idlist.Clear();
    attrlist.Clear();
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(TensorShapeProto& shape) {
  shape.clear_dim();
  do {
    if (Matches('?')) {
      shape.add_dim();
    } else if (NextChar() == '"') {
      // Check for a quoted string as symbolic dim ...
      std::string id;
      CHECK_PARSER_STATUS(ParserBase::Parse(id));
      shape.add_dim()->set_dim_param(id);
    } else {
      // Check for a symbolic identifier ...
      auto id = ParseOptionalIdentifier();
      if (!id.empty()) {
        shape.add_dim()->set_dim_param(id);
      } else {
        // ...or a integer value
        int64_t dimval = 0;
        PARSE_TOKEN(dimval);
        shape.add_dim()->set_dim_value(dimval);
      }
    }
  } while (Matches(','));
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(TypeProto& typeProto) {
  std::string id;
  CHECK_PARSER_STATUS(ParseIdentifier(id));
  int dtype = PrimitiveTypeNameMap::Lookup(id);
  if (dtype != 0) {
    auto* tensortype = typeProto.mutable_tensor_type();
    tensortype->set_elem_type(dtype);
    tensortype->clear_shape();
    // Grammar:
    // float indicates scalar (rank 0)
    // float [] indicates unknown rank tensor (not a zero rank tensor)
    // float [one-or-more-dimensions] indicates tensor of known rank > 0.
    if (Matches('[')) {
      if (!Matches(']')) {
        PARSE(*tensortype->mutable_shape());
        MATCH(']');
      }
    } else {
      // Create shape with zero dimensions for scalar
      (void)(tensortype->mutable_shape());
    }
  } else {
    switch (KeyWordMap::Lookup(id)) {
      case KeyWordMap::KeyWord::SEQ_TYPE: {
        // Grammar: seq ( type )
        MATCH('(');
        auto* seqtype = typeProto.mutable_sequence_type();
        PARSE(*seqtype->mutable_elem_type());
        MATCH(')');
        break;
      }
      case KeyWordMap::KeyWord::MAP_TYPE: {
        // Grammar: map ( prim-type , type )
        MATCH('(');
        auto* maptype = typeProto.mutable_map_type();
        CHECK_PARSER_STATUS(ParseIdentifier(id));
        dtype = PrimitiveTypeNameMap::Lookup(id);
        if (dtype == 0) {
          return ParseError("Expecting primitive type as map key type.");
        }
        maptype->set_key_type(dtype);
        MATCH(',');
        PARSE(*maptype->mutable_value_type());
        MATCH(')');
        break;
      }
      case KeyWordMap::KeyWord::OPTIONAL_TYPE: {
        // Grammar: optional ( type )
        MATCH('(');
        auto* opttype = typeProto.mutable_optional_type();
        PARSE(*opttype->mutable_elem_type());
        MATCH(')');
        break;
      }
      case KeyWordMap::KeyWord::SPARSE_TENSOR_TYPE: {
        // Grammar: sparse_tensor ( tensor-type )
        MATCH('(');
        CHECK_PARSER_STATUS(ParseIdentifier(id));
        dtype = PrimitiveTypeNameMap::Lookup(id);
        if (dtype != 0) {
          auto* sparsetype = typeProto.mutable_sparse_tensor_type();
          sparsetype->set_elem_type(dtype);
          sparsetype->clear_shape();
          // Grammar:
          // float indicates scalar (rank 0)
          // float [] indicates unknown rank tensor (not a zero rank tensor)
          // float [one-or-more-dimensions] indicates tensor of known rank > 0.
          if (Matches('[')) {
            if (!Matches(']')) {
              PARSE(*sparsetype->mutable_shape());
              MATCH(']');
            }
          } else {
            // Create shape with zero dimensions for scalar
            (void)(sparsetype->mutable_shape());
          }
        } else {
          return ParseError("Unexpected type in sparse-tensor element type.");
        }
        MATCH(')');
        break;
      }
      default:
        return ParseError("Unexpected type.");
    }
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(ValueInfoProto& valueinfo) {
  if (NextIsType())
    PARSE(*valueinfo.mutable_type());
  std::string name;
  CHECK_PARSER_STATUS(ParseQuotableIdentifier(name));
  valueinfo.set_name(name);
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(char open, ValueInfoList& vilist, char close) {
  MATCH(open);
  if (!Matches(close)) {
    do {
      PARSE(*vilist.Add());
    } while (Matches(','));
    MATCH(close);
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::ParseGraphInputOutput(ValueInfoList& vilist) {
  vilist.Clear();
  PARSE('(', vilist, ')');
  return Common::Status::OK();
}

Common::Status OnnxParser::ParseFunctionInputOutput(IdList& idlist, ValueInfoList& vilist) {
  // Do not clear vilist, as it accumulates values over inputs and outputs.
  idlist.Clear();
  MATCH('(');
  if (!Matches(')')) {
    do {
      // Function inputs/outputs can be optionally typed.
      // Syntax: Name | Type Name
      // The name is added to idlist. If the optional type is present, an entry is
      // added to vilist.

      std::string* name = idlist.Add();
      ValueInfoProto* vi = nullptr;

      if (NextIsType()) {
        vi = vilist.Add();
        PARSE(*(vi->mutable_type()));
      }
      CHECK_PARSER_STATUS(ParseQuotableIdentifier(*name));
      if (vi != nullptr)
        vi->set_name(*name);
    } while (Matches(','));
    MATCH(')');
  }
  return Common::Status::OK();
}

// Each input element is a value-info with an optional initializer of the form "= initial-value".
// The value-info is added to the "inputs", while the initializer is added to initializers.
Common::Status OnnxParser::ParseInput(ValueInfoList& inputs, TensorList& initializers) {
  inputs.Clear();
  if (Matches('(')) {
    if (!Matches(')')) {
      do {
        ValueInfoProto vi;
        PARSE(vi);
        *inputs.Add() = vi;
        if (Matches('=')) {
          // default value for input
          TensorProto& tp = *initializers.Add();
          tp.set_name(vi.name());
          CHECK_PARSER_STATUS(Parse(tp, vi.type()));
        }
      } while (Matches(','));
      MATCH(')');
    }
  }
  return Common::Status::OK();
}

// This is handled slightly different from the inputs.
// Each element is either a value-info or an initializer.
// A value-info is added to the "value_infos", while an initializer is added to initializers.
Common::Status OnnxParser::ParseValueInfo(ValueInfoList& value_infos, TensorList& initializers) {
  value_infos.Clear();
  if (Matches('<')) {
    if (!Matches('>')) {
      do {
        ValueInfoProto vi;
        PARSE(vi);
        if (Matches('=')) {
          // initializer
          TensorProto& tp = *initializers.Add();
          tp.set_name(vi.name());
          CHECK_PARSER_STATUS(Parse(tp, vi.type()));
        } else {
          // valueinfo
          *value_infos.Add() = vi;
        }
      } while (Matches(','));
      MATCH('>');
    }
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(StringStringList& stringStringList) {
  std::string strval;
  do {
    auto* metadata = stringStringList.Add();
    PARSE_TOKEN(strval);
    metadata->set_key(strval);
    MATCH(':');
    PARSE_TOKEN(strval);
    metadata->set_value(strval);
  } while (Matches(','));
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(TensorProto& tensorProto) {
  tensorProto = TensorProto();
  // Parse the concrete tensor-type with numeric dimensions:
  TypeProto typeProto;
  PARSE(typeProto);
  *tensorProto.mutable_name() = ParseOptionalIdentifier();
  (void)Matches('='); // Optional, to unify handling of initializers as well as tensor-protos in other contexts
  return Parse(tensorProto, typeProto);
}

// Parse TensorProto data given its type:
Common::Status OnnxParser::Parse(TensorProto& tensorProto, const TypeProto& tensorTypeProto) {
  if (!tensorTypeProto.has_tensor_type())
    return ParseError("Error parsing TensorProto (expected a tensor type).");
  auto elem_type = tensorTypeProto.tensor_type().elem_type();
  tensorProto.set_data_type(elem_type);
  if (!tensorTypeProto.tensor_type().has_shape())
    return ParseError("Error parsing TensorProto (expected a tensor shape).");
  for (const auto& dim : tensorTypeProto.tensor_type().shape().dim()) {
    if (!dim.has_dim_value())
      return ParseError("Error parsing TensorProto shape (expected numeric dimension).");
    auto dimval = dim.dim_value();
    tensorProto.add_dims(dimval);
  }

  // tensorProto.mutable_int64_data()->Reserve(n);
  // Parse the actual values:

  int64_t intval = 0;
  uint64_t uintval = 0;
  float floatval = 0.0;
  double dblval = 0.0;
  std::string strval;
  if (Matches('{')) {
    if (!Matches('}')) {
      do {
        switch (static_cast<TensorProto::DataType>(elem_type)) {
          case TensorProto::DataType::TensorProto_DataType_INT2:
          case TensorProto::DataType::TensorProto_DataType_INT4:
          case TensorProto::DataType::TensorProto_DataType_INT8:
          case TensorProto::DataType::TensorProto_DataType_INT16:
          case TensorProto::DataType::TensorProto_DataType_INT32:
          case TensorProto::DataType::TensorProto_DataType_UINT2:
          case TensorProto::DataType::TensorProto_DataType_UINT4:
          case TensorProto::DataType::TensorProto_DataType_UINT8:
          case TensorProto::DataType::TensorProto_DataType_UINT16:
          case TensorProto::DataType::TensorProto_DataType_FLOAT16:
          case TensorProto::DataType::TensorProto_DataType_BFLOAT16:
          case TensorProto::DataType::TensorProto_DataType_FLOAT8E4M3FN:
          case TensorProto::DataType::TensorProto_DataType_FLOAT8E4M3FNUZ:
          case TensorProto::DataType::TensorProto_DataType_FLOAT8E5M2:
          case TensorProto::DataType::TensorProto_DataType_FLOAT8E5M2FNUZ:
          case TensorProto::DataType::TensorProto_DataType_FLOAT6E2M3:
          case TensorProto::DataType::TensorProto_DataType_FLOAT6E3M2:
          case TensorProto::DataType::TensorProto_DataType_FLOAT8E8M0:
          case TensorProto::DataType::TensorProto_DataType_BOOL:
          case TensorProto::DataType::TensorProto_DataType_FLOAT4E2M1:
            PARSE_TOKEN(intval);
            if (intval > std::numeric_limits<int32_t>::max() || intval < std::numeric_limits<int32_t>::min()) {
              return ParseError("Mismatch between data type and value: %d, %d", elem_type, intval);
            }
            // NOLINTNEXTLINE(bugprone-narrowing-conversions)
            tensorProto.add_int32_data(intval);
            break;
          case TensorProto::DataType::TensorProto_DataType_INT64:
            PARSE_TOKEN(intval);
            tensorProto.add_int64_data(intval);
            break;
          case TensorProto::DataType::TensorProto_DataType_UINT32:
          case TensorProto::DataType::TensorProto_DataType_UINT64:
            PARSE_TOKEN(uintval);
            tensorProto.add_uint64_data(uintval);
            break;
          case TensorProto::DataType::TensorProto_DataType_COMPLEX64:
          case TensorProto::DataType::TensorProto_DataType_FLOAT:
            PARSE_TOKEN(floatval);
            tensorProto.add_float_data(floatval);
            break;
          case TensorProto::DataType::TensorProto_DataType_COMPLEX128:
          case TensorProto::DataType::TensorProto_DataType_DOUBLE:
            PARSE_TOKEN(dblval);
            tensorProto.add_double_data(dblval);
            break;
          case TensorProto::DataType::TensorProto_DataType_STRING:
            PARSE_TOKEN(strval);
            tensorProto.add_string_data(strval);
            break;
          default:
            return ParseError("Unhandled type: %d", elem_type);
        }
      } while (Matches(','));
      MATCH('}');
    }
  } else if (Matches('[')) {
    tensorProto.set_data_location(TensorProto::DataLocation::TensorProto_DataLocation_EXTERNAL);
    auto& externalData = *tensorProto.mutable_external_data();
    PARSE(externalData);
    MATCH(']');
  }
  return Common::Status::OK();
}

bool OnnxParser::NextIsIdentifier() {
  auto id = PeekIdentifier();
  return !(id.empty());
}

bool OnnxParser::NextIsType() {
  auto id = PeekIdentifier();
  if (PrimitiveTypeNameMap::IsTypeName(id))
    return true;
  switch (KeyWordMap::Lookup(id)) {
    case KeyWordMap::KeyWord::SEQ_TYPE:
    case KeyWordMap::KeyWord::MAP_TYPE:
    case KeyWordMap::KeyWord::OPTIONAL_TYPE:
    case KeyWordMap::KeyWord::SPARSE_TENSOR_TYPE:
      return true;
    default:
      return false;
  }
}

Common::Status OnnxParser::ParseSingleAttributeValue(AttributeProto& attr, AttributeProto_AttributeType expected) {
  // Parse a single-value
  auto next = NextChar();
  if (IsAlpha(next) || next == '_') {
    if (NextIsType()) {
      TypeProto typeProto;
      CHECK_PARSER_STATUS(Parse(typeProto));
      next = NextChar();
      if ((next == '{') || (next == '=') || (NextIsIdentifier())) {
        attr.set_type(AttributeProto_AttributeType_TENSOR);
        auto& tensorProto = *attr.mutable_t();
        CHECK_PARSER_STATUS(ParseOptionalQuotableIdentifier(*tensorProto.mutable_name()));
        (void)Matches('='); // Optional, to unify handling of initializers
        CHECK_PARSER_STATUS(Parse(tensorProto, typeProto));
      } else {
        attr.set_type(AttributeProto_AttributeType_TYPE_PROTO);
        attr.mutable_tp()->CopyFrom(typeProto);
      }
    } else {
      if (NextIsValidFloatString()) {
        Literal literal;
        PARSE_TOKEN(literal);
        attr.set_type(AttributeProto_AttributeType_FLOAT);
        attr.set_f(LocaleIndependentStof(literal.value));
      } else {
        attr.set_type(AttributeProto_AttributeType_GRAPH);
        PARSE(*attr.mutable_g());
      }
    }
  } else if (Matches('@')) {
    std::string name;
    CHECK_PARSER_STATUS(ParseQuotableIdentifier(name));
    attr.set_ref_attr_name(name);
  } else {
    Literal literal;
    PARSE_TOKEN(literal);
    switch (literal.type) {
      case LiteralType::UNDEFINED:
        return ParseError("Internal error");
      case LiteralType::INT_LITERAL:
        if (expected == AttributeProto_AttributeType_FLOAT) {
          // Implicit INT->FLOAT cast; parse the text as float to preserve "-0".
          attr.set_type(AttributeProto_AttributeType_FLOAT);
          attr.set_f(LocaleIndependentStof(literal.value));
        } else {
          attr.set_type(AttributeProto_AttributeType_INT);
          attr.set_i(std::stoll(literal.value));
        }
        break;
      case LiteralType::FLOAT_LITERAL:
        attr.set_type(AttributeProto_AttributeType_FLOAT);
        attr.set_f(LocaleIndependentStof(literal.value));
        break;
      case LiteralType::STRING_LITERAL:
        attr.set_type(AttributeProto_AttributeType_STRING);
        attr.set_s(literal.value);
        break;
    }
  }
  if ((expected != AttributeProto_AttributeType_UNDEFINED) && (expected != attr.type())) {
    return ParseError(
        "Mismatch between expected type ",
        AttributeProto_AttributeType_Name(expected),
        " and specified value's type",
        AttributeProto_AttributeType_Name(attr.type()));
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(AttributeProto& attr) {
  attr.Clear();
  std::string name;
  CHECK_PARSER_STATUS(ParseIdentifier(name));
  return Parse(attr, name);
}

static bool IsSingletonAttribute(AttributeProto_AttributeType type) {
  switch (type) {
    case AttributeProto_AttributeType_FLOAT:
    case AttributeProto_AttributeType_INT:
    case AttributeProto_AttributeType_STRING:
    case AttributeProto_AttributeType_TENSOR:
    case AttributeProto_AttributeType_GRAPH:
    case AttributeProto_AttributeType_SPARSE_TENSOR:
    case AttributeProto_AttributeType_TYPE_PROTO:
      return true;
    default:
      return false;
  }
}

static AttributeProto_AttributeType ToSingletonType(AttributeProto_AttributeType type) {
  switch (type) {
    case AttributeProto_AttributeType_FLOATS:
      return AttributeProto_AttributeType_FLOAT;
    case AttributeProto_AttributeType_INTS:
      return AttributeProto_AttributeType_INT;
    case AttributeProto_AttributeType_STRINGS:
      return AttributeProto_AttributeType_STRING;
    case AttributeProto_AttributeType_TENSORS:
      return AttributeProto_AttributeType_TENSOR;
    case AttributeProto_AttributeType_GRAPHS:
      return AttributeProto_AttributeType_GRAPH;
    case AttributeProto_AttributeType_SPARSE_TENSORS:
      return AttributeProto_AttributeType_SPARSE_TENSOR;
    case AttributeProto_AttributeType_TYPE_PROTOS:
      return AttributeProto_AttributeType_TYPE_PROTO;
    default:
      return type;
  }
}

Common::Status OnnxParser::Parse(AttributeProto& attr, std::string& name) {
  attr.set_name(name);
  if (Matches(':')) {
    CHECK_PARSER_STATUS(ParseIdentifier(name));
    int attrtype = AttributeTypeNameMap::Lookup(name);
    if (attrtype != 0) {
      attr.set_type(static_cast<AttributeProto_AttributeType>(attrtype));
    } else {
      return ParseError("Unexpected attribute type.");
    }
  }
  MATCH('=');
  if (NextChar() == '[') {
    // Parse a list of values. For an empty list, the type MUST be specified
    // using the type-annotation syntax of ": type".
    MATCH('[');
    if (NextChar() != ']') {
      do {
        AttributeProto nextval;
        auto expected_type = ToSingletonType(attr.type());
        CHECK_PARSER_STATUS(ParseSingleAttributeValue(nextval, expected_type));
        switch (nextval.type()) {
          case AttributeProto_AttributeType_INT:
            attr.set_type(AttributeProto_AttributeType_INTS);
            attr.add_ints(nextval.i());
            break;
          case AttributeProto_AttributeType_FLOAT:
            attr.set_type(AttributeProto_AttributeType_FLOATS);
            attr.add_floats(nextval.f());
            break;
          case AttributeProto_AttributeType_STRING:
            attr.add_strings(nextval.s());
            attr.set_type(AttributeProto_AttributeType_STRINGS);
            break;
          default:
            break;
        }
      } while (Matches(','));
    } else {
      if (attr.type() == AttributeProto_AttributeType_UNDEFINED)
        return ParseError("Empty list attribute value requires type annotation.");
      if (IsSingletonAttribute(attr.type()))
        return ParseError("Singleton attribute value cannot be specified as a list.");
    }
    MATCH(']');
  } else {
    CHECK_PARSER_STATUS(ParseSingleAttributeValue(attr, attr.type()));
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(AttrList& attrlist) {
  attrlist.Clear();
  if (Matches('<')) {
    do {
      PARSE(*attrlist.Add());
    } while (Matches(','));
    MATCH('>');
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(NodeProto& node) {
  if (Matches('[')) {
    CHECK_PARSER_STATUS(ParseOptionalQuotableIdentifier(*node.mutable_name()));
    MATCH(']');
  }
  PARSE(*node.mutable_output());
  MATCH('=');
  std::string domain;
  std::string id = ParseOptionalIdentifier();
  while (Matches('.')) {
    if (!domain.empty())
      domain += ".";
    domain += id;
    CHECK_PARSER_STATUS(ParseIdentifier(id));
  }
  node.set_domain(domain);
  node.set_op_type(id);

  if (Matches(':')) {
    std::string overload;
    CHECK_PARSER_STATUS(ParseIdentifier(overload));
    node.set_overload(overload);
  }
  PARSE(*node.mutable_attribute());
  MATCH('(');
  PARSE(*node.mutable_input());
  MATCH(')');
  if (node.attribute_size() == 0) {
    // Permit attributes to be specified before or after parameters.
    PARSE(*node.mutable_attribute());
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(NodeList& nodelist) {
  nodelist.Clear();
  MATCH('{');
  while (!Matches('}')) {
    PARSE(*nodelist.Add());
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(GraphProto& graph) {
  std::string id;
  CHECK_PARSER_STATUS(ParseQuotableIdentifier(id));
  return Parse(id, graph);
}

Common::Status OnnxParser::Parse(std::string name, GraphProto& graph) {
  graph.set_name(name);
  graph.mutable_initializer()->Clear();
  CHECK_PARSER_STATUS(ParseInput(*graph.mutable_input(), *graph.mutable_initializer()));
  MATCH('=');
  MATCH('>', false);
  CHECK_PARSER_STATUS(ParseGraphInputOutput(*graph.mutable_output()));
  CHECK_PARSER_STATUS(ParseValueInfo(*graph.mutable_value_info(), *graph.mutable_initializer()));
  return Parse(*graph.mutable_node());
}

Common::Status OnnxParser::Parse(FunctionProto& fn) {
  fn.Clear();
  std::string strval;
  if (Matches('<')) {
    do {
      KeyWordMap::KeyWord keyword = KeyWordMap::KeyWord::NONE;
      PARSE_TOKEN(keyword);
      MATCH(':');
      switch (keyword) {
        case KeyWordMap::KeyWord::OPSET_IMPORT:
          PARSE(*fn.mutable_opset_import());
          break;
        case KeyWordMap::KeyWord::DOC_STRING:
          PARSE_TOKEN(strval);
          fn.set_doc_string(strval);
          break;
        case KeyWordMap::KeyWord::DOMAIN_KW:
          PARSE_TOKEN(strval);
          fn.set_domain(strval);
          break;
        case KeyWordMap::KeyWord::OVERLOAD_KW:
          PARSE_TOKEN(strval);
          fn.set_overload(strval);
          break;
        default:
          return ParseError("Unhandled keyword.");
      }
    } while (Matches(','));
    MATCH('>');
  }
  std::string id;
  CHECK_PARSER_STATUS(ParseQuotableIdentifier(id));
  fn.set_name(id);

  PARSE('<', *fn.mutable_attribute(), *fn.mutable_attribute_proto(), '>');
  fn.mutable_value_info()->Clear();
  CHECK_PARSER_STATUS(ParseFunctionInputOutput(*fn.mutable_input(), *fn.mutable_value_info()));
  MATCH('=');
  MATCH('>', false);
  CHECK_PARSER_STATUS(ParseFunctionInputOutput(*fn.mutable_output(), *fn.mutable_value_info()));
  if (NextChar() == '<') {
    PARSE('<', *fn.mutable_value_info(), '>');
  }
  return Parse(*fn.mutable_node());
}

Common::Status OnnxParser::Parse(OpsetIdList& opsets) {
  std::string strval;
  int64_t intval = 0;
  MATCH('[');
  if (!Matches(']')) {
    do {
      auto* import = opsets.Add();
      PARSE_TOKEN(strval);
      import->set_domain(strval);
      MATCH(':');
      PARSE_TOKEN(intval);
      import->set_version(intval);
    } while (Matches(','));
    MATCH(']');
  }
  return Common::Status::OK();
}

Common::Status OnnxParser::Parse(ModelProto& model) {
  model.Clear();
  std::string strval;
  int64_t intval = 0;
  if (Matches('<')) {
    do {
      KeyWordMap::KeyWord keyword = KeyWordMap::KeyWord::NONE;
      PARSE_TOKEN(keyword);
      MATCH(':');
      switch (keyword) {
        case KeyWordMap::KeyWord::IR_VERSION:
          PARSE_TOKEN(intval);
          model.set_ir_version(intval);
          break;
        case KeyWordMap::KeyWord::OPSET_IMPORT:
          PARSE(*model.mutable_opset_import());
          break;
        case KeyWordMap::KeyWord::PRODUCER_NAME:
          PARSE_TOKEN(strval);
          model.set_producer_name(strval);
          break;
        case KeyWordMap::KeyWord::PRODUCER_VERSION:
          PARSE_TOKEN(strval);
          model.set_producer_version(strval);
          break;
        case KeyWordMap::KeyWord::DOMAIN_KW:
          PARSE_TOKEN(strval);
          model.set_domain(strval);
          break;
        case KeyWordMap::KeyWord::MODEL_VERSION:
          PARSE_TOKEN(intval);
          model.set_model_version(intval);
          break;
        case KeyWordMap::KeyWord::DOC_STRING:
          PARSE_TOKEN(strval);
          model.set_doc_string(strval);
          break;
        case KeyWordMap::KeyWord::METADATA_PROPS: {
          auto& metadata_props = *model.mutable_metadata_props();
          MATCH('[');
          if (!Matches(']')) {
            PARSE(metadata_props);
            MATCH(']');
          }
          break;
        }
        default:
          return ParseError("Unhandled keyword.");
      }
    } while (Matches(','));
    MATCH('>');
  }
  PARSE(*model.mutable_graph());

  auto* functions = model.mutable_functions();
  while (!EndOfInput()) {
    PARSE(*functions->Add());
  }
  return Common::Status::OK();
}
const std::unordered_map<std::string, KeyWordMap::KeyWord>& KeyWordMap::Instance() {
  static KeyWordMap instance;
  return instance.map_;
}

const std::string& KeyWordMap::ToString(KeyWord kw) {
  static std::string undefined("undefined");
  for (const auto& pair : Instance()) {
    if (pair.second == kw)
      return pair.first;
  }
  return undefined;
}
} // namespace ONNX_NAMESPACE
