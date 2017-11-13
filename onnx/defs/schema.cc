// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "onnx/defs/schema.h"
#include <stdexcept>
#include <unordered_set>
#include "onnx/checker.h"

namespace onnx {
OpSchema::FormalParameter::FormalParameter(
    const std::string& name,
    const DataTypeSet& allowed_type_set,
    const std::string& type_str,
    const std::string& description,
    bool optional)
    : name_(name),
      type_set_(allowed_type_set),
      type_str_(type_str),
      description_(description),
      is_optional_(optional) {}

OpSchema::FormalParameter::FormalParameter(
    const std::string& name,
    const std::string& description,
    const std::string& type_str,
    bool optional)
    : name_(name),
      type_str_(type_str),
      description_(description),
      is_optional_(optional) {}

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

bool OpSchema::FormalParameter::IsOptional() const {
  return is_optional_;
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
  if (!num_inputs_outputs_allowed_(node.input_size(), node.output_size())) {
    fail_check(
        "Combination of input size ",
        node.input_size(),
        "and output size ",
        node.output_size(),
        " not in allowed.");
  }
  // If the number of outputs can be calculated, check if the number matches.
  if (calculate_output_) {
    int expected_nout = calculate_output_(node.input_size());
    if (expected_nout != kCannotComputeNumOutputs &&
        node.output_size() != expected_nout) {
      fail_check(
          "Output size ",
          node.output_size(),
          " not matching expected output size, which is ",
          expected_nout);
    }
  }

  // Check the values of inputs / outputs
  for (int in_idx = 0; in_idx < node.input_size(); ++in_idx) {
    if (node.input(in_idx).empty() && !(inputs_[in_idx].IsOptional())) {
      fail_check(
          "Input ",
          in_idx,
          " is not marked optional but has an empty string in the graph");
    }
  }
  for (int out_idx = 0; out_idx < node.output_size(); ++out_idx) {
    if (node.output(out_idx).empty()) {
      fail_check("Output ", out_idx, " has an empty string in the graph");
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
    AttrType expected_type;
    if (search != attributes_.end()) {
      expected_type = search->second.type;
    } else if (allows_unchecked_attributes_) {
      continue;
    } else if (name == "consumed_inputs") {
      expected_type = AttrType::INTS;
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
      case AttrType::FLOAT:
        if (!attr_proto.has_f()) {
          fail_check("Attribute '", name, "' is expected to have field 'f'");
        }
        break;
      case AttrType::INT:
        if (!attr_proto.has_i()) {
          fail_check("Attribute '", name, "' is expected to have field 'i'");
        }
        break;
      case AttrType::STRING:
        if (!attr_proto.has_s()) {
          fail_check("Attribute '", name, "' is expected to have field 's'");
        }
        break;
      case AttrType::TENSOR:
        if (!attr_proto.has_t()) {
          fail_check("Attribute '", name, "' is expected to have field 't'");
        }
        break;
      case AttrType::GRAPH:
        if (!attr_proto.has_g()) {
          fail_check("Attribute '", name, "' is expected to have field 'g'");
        }
        break;
      case AttrType::FLOATS:
        if (!attr_proto.floats_size()) {
          fail_check(
              "Attribute '", name, "' is expected to have field 'floats'");
        }
        break;
      case AttrType::INTS:
        if (!attr_proto.ints_size()) {
          fail_check("Attribute '", name, "' is expected to have field 'ints'");
        }
        break;
      case AttrType::STRINGS:
        if (!attr_proto.strings_size()) {
          fail_check(
              "Attribute '", name, "' is expected to have field 'strings'");
        }
        break;
      case AttrType::TENSORS:
        if (!attr_proto.tensors_size()) {
          fail_check(
              "Attribute '", name, "' is expected to have field 'tensors'");
        }
        break;
      case AttrType::GRAPHS:
        if (!attr_proto.graphs_size()) {
          fail_check(
              "Attribute '", name, "' is expected to have field 'graphs'");
        }
        break;
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

OpSchema& OpSchema::NumInputs(int min, int max) {
  min_input_ = min;
  max_input_ = max;
  return *this;
}

OpSchema& OpSchema::NumInputs(int n) {
  return NumInputs(n, n);
}

OpSchema& OpSchema::NumInputs(std::function<bool(int)> func) {
  num_inputs_allowed_ = func;
  return *this;
}

OpSchema& OpSchema::NumInputs(std::set<int> allowed_input_nums) {
  return NumInputs([allowed_input_nums](int n) -> bool {
    return allowed_input_nums.count(n);
  });
}

OpSchema& OpSchema::NumOutputs(int min, int max) {
  min_output_ = min;
  max_output_ = max;
  return *this;
}

OpSchema& OpSchema::NumOutputs(int n) {
  return NumOutputs(n, n);
}

OpSchema& OpSchema::NumOutputs(std::function<bool(int)> func) {
  num_outputs_allowed_ = func;
  return *this;
}

OpSchema& OpSchema::NumOutputs(std::set<int> allowed_output_nums) {
  return NumOutputs([allowed_output_nums](int n) -> bool {
    return allowed_output_nums.count(n);
  });
}

OpSchema& OpSchema::NumInputsOutputs(std::function<bool(int, int)> func) {
  num_inputs_outputs_allowed_ = func;
  return *this;
}

OpSchema& OpSchema::OutputCalculator(std::function<int(int)> calc) {
  calculate_output_ = calc;
  return *this;
}

OpSchema& OpSchema::SameNumberOfOutput() {
  return OutputCalculator([](int n) -> int { return n; });
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

OpSchema& OpSchema::Attr(const Attribute& attr) {
  attributes_.insert(std::make_pair(attr.name, attr));
  return *this;
}

OpSchema& OpSchema::Attr(
    const char* name,
    const char* description,
    AttrType type,
    bool required) {
  Attr(Attribute{name, description, type, required});
  return *this;
}

OpSchema& OpSchema::AllowUncheckedAttributes() {
  allows_unchecked_attributes_ = true;
  return *this;
}

OpSchema& OpSchema::Input(
    const int n,
    const std::string& name,
    const std::string& description,
    const std::string& type_str,
    bool optional) {
  if (int(inputs_.size()) <= n) {
    inputs_.resize(n + 1);
  }
  inputs_[n] = FormalParameter(name, description, type_str, optional);
  return *this;
}

OpSchema& OpSchema::Output(
    const int n,
    const std::string& name,
    const std::string& description,
    const std::string& type_str) {
  if (int(outputs_.size()) <= n) {
    outputs_.resize(n + 1);
  }
  outputs_[n] = FormalParameter(name, description, type_str, false);
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

int OpSchema::CalculateOutput(int num_input) const {
  if (min_output_ == max_output_) {
    return min_output_;
  } else if (calculate_output_) {
    return calculate_output_(num_input);
  } else {
    return kCannotComputeNumOutputs;
  }
}

void OpSchema::Finalize() {
#define ENFORCE(x)                                                          \
  do {                                                                      \
    if (!(x))                                                               \
      throw std::logic_error(                                               \
          "ONNX Schema " + name_ + ": failed validating the check: " + #x); \
  } while (0)
  ENFORCE(min_input_ <= max_input_);
  ENFORCE(min_output_ <= max_output_);
  ENFORCE(inputs_.size() >= min_input_);
  ENFORCE(outputs_.size() >= min_output_);
  ENFORCE(inputs_.size() <= max_input_);
  ENFORCE(outputs_.size() <= max_output_);
  // if max limit is finite - all names should be specified
  if (max_input_ < std::numeric_limits<int>::max()) {
    ENFORCE(inputs_.size() == max_input_);
  }
  if (max_output_ < std::numeric_limits<int>::max()) {
    ENFORCE(outputs_.size() == max_output_);
  }
  // all inputs and outputs have names
  for (const auto& it : inputs_) {
    ENFORCE(!(it.GetName().empty()));
  }
  for (const auto& it : outputs_) {
    ENFORCE(!(it.GetName().empty()));
  }
  // TODO: also cover checks for arbitrary number of inputs
  // allow extra tailing inputs not be present if all inputs at the end are
  // marked as optional
  if (max_input_ < std::numeric_limits<int>::max()) {
    int ind = max_input_;
    for (auto& input : inputs_) {
      if (input.IsOptional() && ind > 0) {
        --ind;
      }
    }
    min_input_ = std::min(min_input_, ind);
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

std::unordered_map<std::string, OpSchema>& OpSchemaRegistry::map() {
  static std::unordered_map<std::string, OpSchema> map;
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
} // namespace onnx
