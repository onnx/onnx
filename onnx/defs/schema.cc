// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
#include <stdexcept>
#include <unordered_set>
#include "onnx/checker.h"

namespace ONNX_NAMESPACE {

OpSchema::FormalParameter::FormalParameter(
    const std::string& name,
    const DataTypeSet& allowed_type_set,
    const std::string& type_str,
    const std::string& description,
    FormalParameterOption param_option)
    : name_(name),
      type_set_(allowed_type_set),
      type_str_(type_str),
      description_(description),
      param_option_(param_option) {}

OpSchema::FormalParameter::FormalParameter(
    const std::string& name,
    const std::string& description,
    const std::string& type_str,
    FormalParameterOption param_option)
    : name_(name),
      type_str_(type_str),
      description_(description),
      param_option_(param_option) {}

const std::string& OpSchema::FormalParameter::GetName() const {
  return name_;
}

const DataTypeSet& OpSchema::FormalParameter::GetTypes() const {
  return type_set_;
}

DataTypeSet& OpSchema::FormalParameter::MutableTypes() {
  return type_set_;
}

const std::string& OpSchema::FormalParameter::GetTypeStr() const {
  return type_str_;
}

const std::string& OpSchema::FormalParameter::GetDescription() const {
  return description_;
}

OpSchema::FormalParameterOption OpSchema::FormalParameter::GetOption() const {
  return param_option_;
}

void OpSchema::Verify(const NodeProto& node) const {
  // Check the number of inputs.
  if (node.input_size() < min_input_ || node.input_size() > max_input_) {
    fail_check(
        "Input size ",
        node.input_size(),
        " not in range [min=",
        min_input_,
        ", max=",
        max_input_,
        "].");
  }

  if (!num_inputs_allowed_(node.input_size())) {
    fail_check(
        "Input size ", node.input_size(), " not in allowed input sizes.");
  }

  // Check the number of outputs.
  if (node.output_size() < min_output_ || node.output_size() > max_output_) {
    fail_check(
        "Output size ",
        node.output_size(),
        " not in range [min=",
        min_output_,
        ", max=",
        max_output_,
        "].");
  }

  if (!num_outputs_allowed_(node.output_size())) {
    fail_check(
        "Output size ", node.output_size(), " not in allowed output sizes.");
  }

  // Check the values of inputs / outputs
  for (int in_idx = 0; in_idx < node.input_size(); ++in_idx) {
    if (in_idx >= static_cast<int>(inputs_.size())) {
      if (inputs_.size() > 0 && Variadic == inputs_.back().GetOption()) {
        // The last input formal parameter should be variadic.
        break;
      } else {
        fail_check(
            "Node (",
            node.name(),
            ") has more inputs (",
            node.input_size(),
            ") than declared (",
            inputs_.size(),
            ") in op definition.");
      }
    }
    if (node.input(in_idx).empty() && (Single == inputs_[in_idx].GetOption())) {
      fail_check(
          "Input ",
          in_idx,
          " is marked single but has an empty string in the graph");
    }
  }

  for (int out_idx = 0; out_idx < node.output_size(); ++out_idx) {
    if (out_idx >= static_cast<int>(outputs_.size())) {
        if (outputs_.size() > 0 && Variadic == outputs_.back().GetOption()) {
            // The last output formal parameter should be variadic.
            break;
        }
        else {
            fail_check(
                "Node (",
                node.name(),
                ") has more outputs (",
                node.output_size(),
                ") than declared (",
                outputs_.size(),
                ") in op definition.");
        }
    }

    if (node.output(out_idx).empty() &&
        (Single == outputs_[out_idx].GetOption())) {
      fail_check(
          "Output ",
          out_idx,
          " is marked single but has an empty string in the graph");
    }
  }

  // Check attributes
  std::unordered_set<std::string> seen_attr_names{};
  const AttributeProto* consume_attr = nullptr;

  for (const auto& attr_proto : node.attribute()) {
    const auto& name = attr_proto.name();

    if (!seen_attr_names.insert(name).second) {
      fail_check("Attribute '", name, "' appeared multiple times.");
    };

    const auto& search = attributes_.find(name);
    AttributeProto::AttributeType expected_type;
    if (search != attributes_.end()) {
      expected_type = search->second.type;
    } else if (allows_unchecked_attributes_) {
      continue;
    } else if (name == "consumed_inputs") {
      expected_type = AttributeProto::INTS;
      consume_attr = &attr_proto;
      if (attr_proto.ints().size() != node.input_size()) {
        fail_check(
            "Attribute consumed_inputs (length ",
            attr_proto.ints().size(),
            ") is not the same length as inputs (length ",
            node.input_size(),
            ")");
      }
    } else {
      fail_check("Unrecognized attribute: ", name);
    }

    switch (expected_type) {
      case AttributeProto::FLOAT:
        if (!attr_proto.has_f()) {
          fail_check("Attribute '", name, "' is expected to have field 'f'");
        }
        break;
      case AttributeProto::INT:
        if (!attr_proto.has_i()) {
          fail_check("Attribute '", name, "' is expected to have field 'i'");
        }
        break;
      case AttributeProto::STRING:
        if (!attr_proto.has_s()) {
          fail_check("Attribute '", name, "' is expected to have field 's'");
        }
        break;
      case AttributeProto::TENSOR:
        if (!attr_proto.has_t()) {
          fail_check("Attribute '", name, "' is expected to have field 't'");
        }
        break;
      case AttributeProto::GRAPH:
        if (!attr_proto.has_g()) {
          fail_check("Attribute '", name, "' is expected to have field 'g'");
        }
        break;
      case AttributeProto::FLOATS:
        if (!attr_proto.floats_size()) {
          fail_check(
              "Attribute '", name, "' is expected to have field 'floats'");
        }
        break;
      case AttributeProto::INTS:
        if (!attr_proto.ints_size()) {
          fail_check("Attribute '", name, "' is expected to have field 'ints'");
        }
        break;
      case AttributeProto::STRINGS:
        if (!attr_proto.strings_size()) {
          fail_check(
              "Attribute '", name, "' is expected to have field 'strings'");
        }
        break;
      case AttributeProto::TENSORS:
        if (!attr_proto.tensors_size()) {
          fail_check(
              "Attribute '", name, "' is expected to have field 'tensors'");
        }
        break;
      case AttributeProto::GRAPHS:
        if (!attr_proto.graphs_size()) {
          fail_check(
              "Attribute '", name, "' is expected to have field 'graphs'");
        }
        break;
      default:
        fail_check("Attribute '", name, " has unknown expected type");
    }
  }
  for (const auto& pair : attributes_) {
    const auto& attr = pair.second;
    if (!attr.required) {
      continue;
    }
    if (!seen_attr_names.count(attr.name)) {
      fail_check("Required attribute '", attr.name, "' is missing.");
    }
  }

  // Check in-place settings.
  for (int in_idx = 0; in_idx < node.input_size(); ++in_idx) {
    bool consumed = consume_attr ? consume_attr->ints(in_idx) : false;
    auto use_type = consumed_(in_idx);
    switch (use_type.first) {
      case UseType::DEFAULT:
        if (consumed) {
          fail_check(
              "Input index ",
              in_idx,
              " is set to consumed but ",
              "is not supported by op ",
              node.op_type());
        }
        break;
      case UseType::CONSUME_ENFORCED:
        if (!consumed) {
          fail_check(
              "Input index ",
              in_idx,
              " must be set to consumed ",
              "for operator ",
              node.op_type());
        }
        break;
      case UseType::CONSUME_ALLOWED:
        // pass, either is allowed
        break;
    }
  }
  // Phew. All verifications passed.
}

OpSchema& OpSchema::SinceVersion(OperatorSetVersion v) {
  since_version_ = v;
  return *this;
}

OpSchema& OpSchema::NumInputs(std::set<int> allowed_input_nums) {
  num_inputs_allowed_ = [allowed_input_nums](int n) -> bool {
    return allowed_input_nums.count(n);
  };
  return *this;
}

OpSchema& OpSchema::NumOutputs(std::set<int> allowed_output_nums) {
  num_outputs_allowed_ = [allowed_output_nums](int n) -> bool {
    return allowed_output_nums.count(n);
  };
  return *this;
}

OpSchema& OpSchema::AllowConsumed(
    std::function<std::pair<bool, int>(int)> inplace) {
  consumed_ = [inplace](int idx) {
    auto r = inplace(idx);
    return std::make_pair(
        r.first ? UseType::CONSUME_ALLOWED : UseType::DEFAULT, r.second);
  };
  return *this;
}

OpSchema& OpSchema::AllowConsumed(std::unordered_map<int, int> inplace) {
  return AllowConsumed([inplace](int idx) {
    auto it = inplace.find(idx);
    if (it != inplace.end()) {
      return std::make_pair(true, it->second);
    }
    return std::make_pair(false, 0);
  });
}

OpSchema& OpSchema::AllowOneToOneConsumed() {
  return AllowConsumed([](int i) { return std::make_pair(true, i); });
}

OpSchema& OpSchema::EnforceConsumed(
    std::function<std::pair<bool, int>(int)> inplace) {
  consumed_ = [inplace](int idx) {
    auto r = inplace(idx);
    return std::make_pair(
        r.first ? UseType::CONSUME_ENFORCED : UseType::DEFAULT, r.second);
  };
  return *this;
}

OpSchema& OpSchema::EnforceConsumed(std::unordered_map<int, int> inplace) {
  return EnforceConsumed([inplace](int idx) {
    auto it = inplace.find(idx);
    if (it != inplace.end()) {
      return std::make_pair(true, it->second);
    }
    return std::make_pair(false, 0);
  });
}

OpSchema& OpSchema::EnforceOneToOneConsumed() {
  return EnforceConsumed([](int i) { return std::make_pair(true, i); });
}

OpSchema& OpSchema::SetSupportLevel(SupportType support) {
  support_ = support;
  return *this;
}

OpSchema& OpSchema::SetDoc(const std::string& doc) {
  doc_ = doc;
  return *this;
}

OpSchema& OpSchema::SetDomain(const std::string& domain) {
  domain_ = domain;
  return *this;
}

OpSchema& OpSchema::Attr(const Attribute& attr) {
  attributes_.insert(std::make_pair(attr.name, attr));
  return *this;
}

OpSchema& OpSchema::Attr(
    const std::string& name,
    const std::string& description,
    AttributeProto::AttributeType type,
    bool required) {
  Attr(Attribute{name, description, type, required});
  return *this;
}

#define ATTR_SETTER_WITH_SINGLE_VALUE(type, field, attrtype) \
  OpSchema& OpSchema::Attr(                                  \
      const std::string& name,                               \
      const std::string& description,                        \
      AttributeProto::AttributeType attr_type,               \
      const type& default_value) {                           \
    if (attrtype != attr_type) {                             \
      std::cerr << "Attribute specification type mismatch."; \
      abort();                                               \
    }                                                        \
    AttributeProto a;                                        \
    a.set_name(name);                                        \
    a.set_##field(default_value);                            \
    a.set_type(attr_type);                                   \
    Attr(Attribute(name, description, a));                   \
    return *this;                                            \
  }

#define ATTR_SETTER_WITH_LIST_VALUE(type, field, attrtype)   \
  OpSchema& OpSchema::Attr(                                  \
      const std::string& name,                               \
      const std::string& description,                        \
      AttributeProto::AttributeType attr_type,               \
      const std::vector<type>& default_value) {              \
    if (attrtype != attr_type) {                             \
      std::cerr << "Attribute specification type mismatch."; \
      abort();                                               \
    }                                                        \
    AttributeProto a;                                        \
    a.set_name(name);                                        \
    a.set_type(attr_type);                                   \
    for (const auto& v : default_value) {                    \
      a.add_##field(v);                                      \
    }                                                        \
    Attr(Attribute(name, description, a));                   \
    return *this;                                            \
  }

#define ATTR_SETTER_WITH_SINGLE_COMPLEXVALUE(type, field, attrtype) \
  OpSchema& OpSchema::Attr(                                         \
      const std::string& name,                                      \
      const std::string& description,                               \
      AttributeProto::AttributeType attr_type,                      \
      const type& default_value) {                                  \
    if (attrtype != attr_type) {                                    \
      std::cerr << "Attribute specification type mismatch.";        \
      abort();                                                      \
    }                                                               \
    AttributeProto a;                                               \
    a.set_name(name);                                               \
    *(a.mutable_##field()) = default_value;                         \
    a.set_type(attr_type);                                          \
    Attr(Attribute(name, description, a));                          \
    return *this;                                                   \
  }

#define ATTR_SETTER_WITH_LIST_COMPLEXVALUE(type, field, attrtype) \
  OpSchema& OpSchema::Attr(                                       \
      const std::string& name,                                    \
      const std::string& description,                             \
      AttributeProto::AttributeType attr_type,                    \
      const std::vector<type>& default_value) {                   \
    if (attrtype != attr_type) {                                  \
      std::cerr << "Attribute specification type mismatch.";      \
      abort();                                                    \
    }                                                             \
    AttributeProto a;                                             \
    a.set_name(name);                                             \
    a.set_type(attr_type);                                        \
    for (const auto& v : default_value) {                         \
      *(a.add_##field()) = v;                                     \
    }                                                             \
    Attr(Attribute(name, description, a));                        \
    return *this;                                                 \
  }

ATTR_SETTER_WITH_SINGLE_VALUE(int64_t, i, AttributeProto::INT)
ATTR_SETTER_WITH_SINGLE_VALUE(float, f, AttributeProto::FLOAT)
ATTR_SETTER_WITH_SINGLE_VALUE(std::string, s, AttributeProto::STRING)
ATTR_SETTER_WITH_SINGLE_COMPLEXVALUE(TensorProto, t, AttributeProto::TENSOR)
ATTR_SETTER_WITH_SINGLE_COMPLEXVALUE(GraphProto, g, AttributeProto::GRAPH)
ATTR_SETTER_WITH_LIST_VALUE(int64_t, ints, AttributeProto::INTS)
ATTR_SETTER_WITH_LIST_VALUE(float, floats, AttributeProto::FLOATS)
ATTR_SETTER_WITH_LIST_COMPLEXVALUE(
    std::string,
    strings,
    AttributeProto::STRINGS)
ATTR_SETTER_WITH_LIST_COMPLEXVALUE(
    TensorProto,
    tensors,
    AttributeProto::TENSORS)
ATTR_SETTER_WITH_LIST_COMPLEXVALUE(GraphProto, graphs, AttributeProto::GRAPHS)

OpSchema& OpSchema::AllowUncheckedAttributes() {
  allows_unchecked_attributes_ = true;
  return *this;
}

OpSchema& OpSchema::Input(
    const int n,
    const std::string& name,
    const std::string& description,
    const std::string& type_str,
    OpSchema::FormalParameterOption param_option) {
  if (int(inputs_.size()) <= n) {
    inputs_.resize(n + 1);
  }
  inputs_[n] = FormalParameter(name, description, type_str, param_option);
  return *this;
}

OpSchema& OpSchema::Output(
    const int n,
    const std::string& name,
    const std::string& description,
    const std::string& type_str,
    OpSchema::FormalParameterOption param_option) {
  if (int(outputs_.size()) <= n) {
    outputs_.resize(n + 1);
  }
  outputs_[n] = FormalParameter(name, description, type_str, param_option);
  return *this;
}

OpSchema& OpSchema::TypeConstraint(
    const std::string& type_str,
    const std::vector<std::string>& constraints,
    const std::string& description) {
  assert(type_constraints_.end() == type_constraints_.find(type_str));
  DataTypeSet d;
  for (const auto& t : constraints) {
    d.insert(Utils::DataTypeUtils::ToType(t));
  }
  type_constraints_.insert(
      std::make_pair(type_str, std::make_pair(d, description)));
  type_constraint_params_.push_back(
      TypeConstraintParam(type_str, constraints, description));
  return *this;
}

void OpSchema::ParseAndSetTypes(
    /*out*/ std::vector<OpSchema::FormalParameter>* formal_parameters) {
  for (auto& formal_parameter : *formal_parameters) {
    auto& type = formal_parameter.GetTypeStr();
    DataTypeSet allowed_types;
    auto it = type_constraints_.find(type);
    if (it != type_constraints_.end()) {
      allowed_types = it->second.first;
    } else {
      allowed_types.emplace(Utils::DataTypeUtils::ToType(type));
    }

    formal_parameter.MutableTypes() = allowed_types;
  }
}

OpSchema& OpSchema::FillUsing(std::function<void(OpSchema&)> populator) {
  if (populator) {
    populator(*this);
  }
  return *this;
}

void OpSchema::Finalize() {
#define ENFORCE(x)                                                          \
  do {                                                                      \
    if (!(x))                                                               \
      throw std::logic_error(                                               \
          "ONNX Schema " + name_ + ": failed validating the check: " + #x); \
  } while (0)

  // Calculate min/max number of inputs.
  // <Min number of inputs> = <number of "single" inputs> + <number of
  // "optional" but not trailing inputs>. <Max number of inputs> = <number of
  // all inputs or std::numeric_limits<int>::max() (if the last input is
  // variadic).

  // Flag indicates whether an optional input is trailing one (there's no single
  // or variadic input behind).
  for (size_t i = 0; i < inputs_.size(); ++i) {
    switch (inputs_[i].GetOption()) {
      case OpSchema::Single:
        ++max_input_;
        min_input_ = max_input_;
        break;
      case OpSchema::Optional:
        ++max_input_;
        break;
      case OpSchema::Variadic:
        // Only last input formal parameter could be variadic.
        ENFORCE((inputs_.size() - 1) == i);
        min_input_ = max_input_ + 1;
        max_input_ = std::numeric_limits<int>::max();
        break;
    }
  }

  // Calculate min/max number of outputs.
  for (size_t i = 0; i < outputs_.size(); ++i) {
    switch (outputs_[i].GetOption()) {
      case OpSchema::Single:
        ++max_output_;
        min_output_ = max_output_;
        break;
      case OpSchema::Optional:
        ++max_output_;
        break;
      case OpSchema::Variadic:
        // Only last output formal parameter could be variadic.
        ENFORCE((outputs_.size() - 1) == i);
        min_output_ = max_output_ + 1;
        max_output_ = std::numeric_limits<int>::max();
        break;
    }
  }

  // all inputs and outputs have names
  for (const auto& it : inputs_) {
    ENFORCE(!(it.GetName().empty()));
  }
  for (const auto& it : outputs_) {
    ENFORCE(!(it.GetName().empty()));
  }

  ParseAndSetTypes(&inputs_);
  ParseAndSetTypes(&outputs_);
}

std::ostream& operator<<(std::ostream& out, const OpSchema& schema) {
  if (!schema.attributes_.empty()) {
    out << "Attributes:" << std::endl;
    for (const auto& pair : schema.attributes_) {
      out << "  " << pair.second.name << " : " << pair.second.description
          << std::endl;
    }
  }
  if (schema.max_input_ > 0) {
    out << "Inputs:" << std::endl;
    if (!schema.inputs_.empty()) {
      for (size_t i = 0; i < schema.inputs_.size(); ++i) {
        const auto& p = schema.inputs_[i];
        auto& name = p.GetName();
        auto& description = p.GetDescription();
        auto& type_str = p.GetTypeStr();
        out << "  " << i << ", " << ("" != name ? name : "(unnamed)") << " : "
            << ("" != description ? description : "(no doc)") << " : "
            << ("" != type_str ? type_str : "(no type)") << std::endl;
      }
    } else {
      out << "  (no explicit description available)" << std::endl;
    }
  }
  if (schema.max_output_ > 0) {
    out << "Outputs:" << std::endl;
    if (!schema.outputs_.empty()) {
      for (size_t i = 0; i < schema.outputs_.size(); ++i) {
        const auto& p = schema.outputs_[i];
        auto& name = p.GetName();
        auto& description = p.GetDescription();
        auto& type_str = p.GetTypeStr();
        out << "  " << i << ", " << ("" != name ? name : "(unnamed)") << " : "
            << ("" != description ? description : "(no doc)") << " : "
            << ("" != type_str ? type_str : "(no type)") << std::endl;
      }
    } else {
      out << "  (no explicit description available)" << std::endl;
    }
  }
  out << std::endl;
  if (schema.doc()) {
    out << schema.doc();
  } else {
    out << "(no documentation yet)" << std::endl;
  }
  out << std::endl;
  if (schema.line_) {
    out << "Defined at " << schema.file_ << ":" << schema.line_ << std::endl;
  }
  return out;
}

OpName_Domain_Version_Schema_Map& OpSchemaRegistry::map() {
  static OpName_Domain_Version_Schema_Map map;
  return map;
}

size_t ReplaceAll(std::string& s, const char* from, const char* to) {
  size_t numReplaced = 0;
  std::string::size_type lenFrom = std::strlen(from);
  std::string::size_type lenTo = std::strlen(to);
  for (std::string::size_type pos = s.find(from); pos != std::string::npos;
       pos = s.find(from, pos + lenTo)) {
    s.replace(pos, lenFrom, to);
    numReplaced++;
  }
  return numReplaced;
}
} // namespace ONNX_NAMESPACE
