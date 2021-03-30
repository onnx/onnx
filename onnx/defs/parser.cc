/*
 * SPDX-License-Identifier: Apache-2.0
 */

// Experimental language syntax and parser for ONNX. Please note that the syntax as formalized
// by this parser is preliminary and may change.

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "onnx/onnx_pb.h"
#include "onnx/string_utils.h"

#include "onnx/defs/parser.h"

#define PARSE_TOKEN(x) CHECK_PARSER_STATUS(ParserBase::Parse(x))
#define PARSE(x) CHECK_PARSER_STATUS(Parse(x))
#define MATCH(...) CHECK_PARSER_STATUS(Match(__VA_ARGS__))

namespace ONNX_NAMESPACE {

Status OnnxParser::Parse(IdList& idlist) {
  idlist.Clear();
  std::string id;
  ParseOptionalIdentifier(id);
  if (id.empty())
    return Status::OK(); // Treat as empty list of identifiers
  *idlist.Add() = id;
  while (Matches(',')) {
    ParseOptionalIdentifier(id);
    *idlist.Add() = id;
  }
  return Status::OK();
}

Status OnnxParser::Parse(TensorShapeProto& shape) {
  shape.clear_dim();
  do {
    if (Matches('?')) {
      shape.add_dim();
    } else {
      // Check for a symbolic identifier ...
      std::string id;
      CHECK_PARSER_STATUS(ParseOptionalIdentifier(id));
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
  return Status::OK();
}

Status OnnxParser::Parse(TypeProto& typeProto) {
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
  } else
    return ParseError("Unexpected type.");
  return Status::OK();
}

Status OnnxParser::Parse(ValueInfoProto& valueinfo) {
  PARSE(*valueinfo.mutable_type());
  std::string name;
  CHECK_PARSER_STATUS(ParseIdentifier(name));
  valueinfo.set_name(name);
  return Status::OK();
}

Status OnnxParser::Parse(ValueInfoList& vilist) {
  vilist.Clear();
  MATCH('(');
  if (!Matches(')')) {
    do {
      PARSE(*vilist.Add());
    } while (Matches(','));
    MATCH(')');
  }
  return Status::OK();
}

Status OnnxParser::Parse(TensorProto& tensorProto) {
  tensorProto = TensorProto();
  // Parse the concrete tensor-type with numeric dimensions:
  TypeProto typeProto;
  PARSE(typeProto);
  if (!typeProto.has_tensor_type())
    return ParseError("Error parsing TensorProto (expected a tensor type).");
  auto elem_type = typeProto.tensor_type().elem_type();
  tensorProto.set_data_type(elem_type);
  if (!typeProto.tensor_type().has_shape())
    return ParseError("Error parsing TensorProto (expected a tensor shape).");
  uint64_t n = 1;
  for (auto& dim : typeProto.tensor_type().shape().dim()) {
    if (!dim.has_dim_value())
      return ParseError("Error parsing TensorProto shape (expected numeric dimension).");
    auto dimval = dim.dim_value();
    tensorProto.add_dims(dimval);
    n *= dimval;
  }

  // tensorProto.mutable_int64_data()->Reserve(n);
  // Parse the actual values:

  int64_t intval;
  uint64_t uintval;
  float floatval;
  double dblval;
  std::string strval;
  MATCH('{');
  if (!Matches('}')) {
    do {
      switch (static_cast<TensorProto::DataType>(elem_type)) {
        case TensorProto::DataType::TensorProto_DataType_INT8:
        case TensorProto::DataType::TensorProto_DataType_INT16:
        case TensorProto::DataType::TensorProto_DataType_INT32:
        case TensorProto::DataType::TensorProto_DataType_UINT8:
        case TensorProto::DataType::TensorProto_DataType_UINT16:
        case TensorProto::DataType::TensorProto_DataType_BOOL:
          PARSE_TOKEN(intval);
          // TODO: check values are in the correct range.
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
        case TensorProto::DataType::TensorProto_DataType_FLOAT:
          PARSE_TOKEN(floatval);
          tensorProto.add_float_data(floatval);
          break;
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
  return Status::OK();
}

Status OnnxParser::ParseSingleAttributeValue(AttributeProto& attr) {
  // Parse a single-value
  auto next = NextChar();
  if (isalpha(next) || next == '_') {
    std::string id("");
    (void)PeekIdentifier(id);
    if (PrimitiveTypeNameMap::IsTypeName(id)) {
      attr.set_type(AttributeProto_AttributeType_TENSOR);
      Parse(*attr.mutable_t());
    } else {
      attr.set_type(AttributeProto_AttributeType_GRAPH);
      Parse(*attr.mutable_g());
    }
  } else {
    Literal literal;
    PARSE_TOKEN(literal);
    switch (literal.type) {
      case LiteralType::INT_LITERAL:
        attr.set_type(AttributeProto_AttributeType_INT);
        attr.set_i(std::stol(literal.value));
        break;
      case LiteralType::FLOAT_LITERAL:
        attr.set_type(AttributeProto_AttributeType_FLOAT);
        attr.set_f(static_cast<float>(std::stof(literal.value)));
        break;
      case LiteralType::STRING_LITERAL:
        attr.set_type(AttributeProto_AttributeType_STRING);
        attr.set_s(literal.value);
        break;
      default:
        return ParseError("Unexpected literal type.");
    }
  }
  return Status::OK();
}

Status OnnxParser::Parse(AttributeProto& attr) {
  std::string name;
  CHECK_PARSER_STATUS(ParseIdentifier(name));
  attr.set_name(name);
  MATCH('=');
  if (NextChar() == '[') {
    // Parse a list of values. For now, empty list is not allowed, as we need to
    // figure out a type for the attribute.
    std::vector<Literal> vals;
    MATCH('[');
    do {
      AttributeProto nextval;
      CHECK_PARSER_STATUS(ParseSingleAttributeValue(nextval));
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
    MATCH(']');
  } else {
    CHECK_PARSER_STATUS(ParseSingleAttributeValue(attr));
  }
  return Status::OK();
}

Status OnnxParser::Parse(AttrList& attrlist) {
  attrlist.Clear();
  if (Matches('<')) {
    do {
      PARSE(*attrlist.Add());
    } while (Matches(','));
    MATCH('>');
  }
  return Status::OK();
}

Status OnnxParser::Parse(NodeProto& node) {
  PARSE(*node.mutable_output());
  MATCH('=');
  std::string domain("");
  std::string id;
  ParseIdentifier(id);
  while (Matches('.')) {
    if (!domain.empty())
      domain += ".";
    domain += id;
    ParseIdentifier(id);
  }
  node.set_domain(domain); // TODO
  node.set_op_type(id);
  PARSE(*node.mutable_attribute());
  MATCH('(');
  PARSE(*node.mutable_input());
  MATCH(')');
  if (node.attribute_size() == 0) {
    // Permit attributes to be specified before or after parameters.
    PARSE(*node.mutable_attribute());
  }
  return Status::OK();
}

Status OnnxParser::Parse(NodeList& nodelist) {
  nodelist.Clear();
  MATCH('{');
  while (!Matches('}')) {
    PARSE(*nodelist.Add());
  }
  return Status::OK();
}

Status OnnxParser::Parse(GraphProto& graph) {
  std::string id;
  ParseIdentifier(id);
  return Parse(id, graph);
}

Status OnnxParser::Parse(std::string name, GraphProto& graph) {
  graph.set_name(name);
  PARSE(*graph.mutable_input());
  MATCH('=');
  MATCH('>', false);
  PARSE(*graph.mutable_output());
  return Parse(*graph.mutable_node());
}

Status OnnxParser::Parse(ModelProto& model) {
  std::string strval;
  int64_t intval;
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
        case KeyWordMap::KeyWord::OPSET_IMPORT: {
          auto& imports = *model.mutable_opset_import();
          MATCH('[');
          while (!Matches(']')) {
            auto* import = imports.Add();
            PARSE_TOKEN(strval);
            import->set_domain(strval);
            MATCH(':');
            PARSE_TOKEN(intval);
            import->set_version(intval);
          }
          break;
        }
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
            do {
              auto* metadata = metadata_props.Add();
              PARSE_TOKEN(strval);
              metadata->set_key(strval);
              MATCH(':');
              PARSE_TOKEN(strval);
              metadata->set_value(strval);
            } while (Matches(','));
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
  return Parse(*model.mutable_graph());
}

} // namespace ONNX_NAMESPACE
