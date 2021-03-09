/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdexcept>
#include <string>
#include <unordered_map>

#include "onnx/onnx_pb.h"
#include "onnx/string_utils.h"

#include "onnx/defs/parser.h"

namespace ONNX_NAMESPACE {

void OnnxParser::ParseIdList(IdList& idlist) {
  idlist.Clear();
  // Allow empty identifier (used for missing optional parameters) except
  // in special case. TODO: fix this.
  std::string id = ParseIdentifier();
  if (id.empty())
    return;
  *idlist.Add() = id;
  while (Matches(',')) {
    *idlist.Add() = ParseIdentifier();
  }
}

void OnnxParser::Parse(TensorShapeProto& shape) {
  shape.clear_dim();
  do {
    uint64_t dimval;
    if (Matches('?')) {
      shape.add_dim();
    } else if (ParseIntValue(dimval)) {
      shape.add_dim()->set_dim_value(dimval);
    } else {
      shape.add_dim()->set_dim_param(ParseIdentifier());
    }
  } while (Matches(','));
}

void OnnxParser::Parse(TypeProto& typeProto) {
  std::string id = ParseIdentifier();
  int dtype = PrimitiveTypeNameMap::Lookup(id);
  if (dtype != 0) {
    auto* tensortype = typeProto.mutable_tensor_type();
    tensortype->set_elem_type(dtype);
    tensortype->clear_shape();
    // Grammar:
    // FLOAT indicates scalar (rank 0)
    // FLOAT [] indicates unknown rank tensor
    // FLOAT [one-or-more-dimensions] indicates tensor of known rank > 0.
    if (Matches('[')) {
      if (!Matches(']')) {
        Parse(*tensortype->mutable_shape());
        Match(']');
      }
    } else {
      // Create shape with zero dimensions for scalar
      (void)(tensortype->mutable_shape());
    }
  } else
    parse_error("Unexpected type.");
}

void OnnxParser::Parse(ValueInfoProto& valueinfo) {
  Parse(*valueinfo.mutable_type());
  valueinfo.set_name(ParseIdentifier());
}

void OnnxParser::Parse(ValueInfoList& vilist) {
  vilist.Clear();
  Match('(');
  while (!Matches(')')) {
    Parse(*vilist.Add());
    (void)Matches(','); // skip optional comma if present
  }
}

void OnnxParser::Parse(TensorProto& tensorProto) {
  // Parse the concrete tensor-type with numeric dimensions:
  TypeProto typeProto;
  Parse(typeProto);
  parser_check(typeProto.has_tensor_type(), "Error parsing TensorProto (expected a tensor type).");
  auto elem_type = typeProto.tensor_type().elem_type();
  tensorProto.set_data_type(elem_type);
  parser_check(typeProto.tensor_type().has_shape(), "Error parsing TensorProto (expected a tensor shape).");
  uint64_t n = 1;
  for (auto& dim : typeProto.tensor_type().shape().dim()) {
    parser_check(dim.has_dim_value(), "Error parsing TensorProto shape (expected numeric dimension).");
    auto dimval = dim.dim_value();
    tensorProto.add_dims(dimval);
    n *= dimval;
  }

  // tensorProto.mutable_int64_data()->Reserve(n);
  // Parse the actual values:
  Match('{');
  while (!Matches('}')) {
    switch (static_cast<TensorProto::DataType>(elem_type)) {
      case TensorProto::DataType::TensorProto_DataType_INT32:
        tensorProto.add_int32_data(ParseIntValue());
        break;
      case TensorProto::DataType::TensorProto_DataType_INT64:
        tensorProto.add_int64_data(ParseIntValue());
        break;
      case TensorProto::DataType::TensorProto_DataType_FLOAT:
        tensorProto.add_float_data(ParseFloatValue());
        break;
      default:
        parse_error("Unhandled type: %d", elem_type);
    }

    (void)Matches(',');
  }
}

void OnnxParser::ParseSingleAttributeValue(AttributeProto& attr) {
  // Parse a single-value
  auto token = ParseValue();
  switch (token.type) {
    case TokenType::INT_LITERAL:
      attr.set_type(AttributeProto_AttributeType_INT);
      attr.set_i(std::stol(token.value));
      break;
    case TokenType::FLOAT_LITERAL:
      attr.set_type(AttributeProto_AttributeType_FLOAT);
      attr.set_f(static_cast<float>(std::stof(token.value)));
      break;
    case TokenType::STRING_LITERAL:
      attr.set_type(AttributeProto_AttributeType_STRING);
      attr.set_s(token.value);
      break;
    default:
      parse_error("Unexpected literal type.");
  }
}

void OnnxParser::Parse(AttributeProto& attr) {
  attr.set_name(ParseIdentifier());
  Match('=');
  if (NextChar() == '[') {
    // Parse a list of values
    std::vector<Token> vals;
    Match('[');
    while (!Matches(']')) {
      AttributeProto nextval;
      ParseSingleAttributeValue(nextval);
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
      (void)Matches(',');
    }
  } else {
    ParseSingleAttributeValue(attr);
  }
}

void OnnxParser::Parse(AttrList& attrlist) {
  attrlist.Clear();
  if (Matches('<')) {
    while (!Matches('>')) {
      Parse(*attrlist.Add());
      (void)Matches(','); // skip optional comma if present
    }
  }
}

void OnnxParser::Parse(NodeProto& node) {
  ParseIdList(*node.mutable_output());
  Match('=');
  std::string domain("");
  auto id = ParseIdentifier();
  while (Matches('.')) {
    if (!domain.empty())
      domain += ".";
    domain += id;
    id = ParseIdentifier();
  }
  node.set_domain(domain); // TODO
  node.set_op_type(id);
  Parse(*node.mutable_attribute());
  Match('(');
  ParseIdList(*node.mutable_input());
  Match(')');
}

void OnnxParser::Parse(NodeList& nodelist) {
  nodelist.Clear();
  Match('{');
  while (!Matches('}')) {
    Parse(*nodelist.Add());
    (void)Matches(';'); // skip optional semicolon if present
  }
}

void OnnxParser::Parse(GraphProto& graph) {
  graph.set_name(ParseIdentifier());
  Parse(*graph.mutable_input());
  Match('=');
  Match('>', false);
  Parse(*graph.mutable_output());
  Parse(*graph.mutable_node());
}

void OnnxParser::Parse(ModelProto& model) {
  if (Matches('<')) {
    while (!Matches('>')) {
      auto keyword = ParseKeyWord();
      Match(':');
      switch (keyword) {
        case KeyWordMap::KeyWord::IR_VERSION:
          model.set_ir_version(ParseIntValue());
          break;
        case KeyWordMap::KeyWord::OPSET_IMPORT: {
          auto& imports = *model.mutable_opset_import();
          Match('[');
          while (!Matches(']')) {
            auto* import = imports.Add();
            import->set_domain(ParseString());
            Match(':');
            import->set_version(ParseIntValue());
          }
          break;
        }
        case KeyWordMap::KeyWord::PRODUCER_NAME:
          model.set_producer_name(ParseString());
          break;
        case KeyWordMap::KeyWord::PRODUCER_VERSION:
          model.set_producer_version(ParseString());
          break;
        case KeyWordMap::KeyWord::DOMAIN_KW:
          model.set_domain(ParseString());
          break;
        case KeyWordMap::KeyWord::MODEL_VERSION:
          model.set_model_version(ParseIntValue());
          break;
        case KeyWordMap::KeyWord::DOC_STRING:
          model.set_doc_string(ParseString());
          break;
        case KeyWordMap::KeyWord::METADATA_PROPS: {
          auto& metadata_props = *model.mutable_metadata_props();
          Match('[');
          while (!Matches(']')) {
            auto* metadata = metadata_props.Add();
            metadata->set_key(ParseString());
            Match(':');
            metadata->set_value(ParseString());
          }
          break;
        }
        default:
          parse_error("Unhandled keyword.");
      }
    }
  }
  Parse(*model.mutable_graph());
}

} // namespace ONNX_NAMESPACE
