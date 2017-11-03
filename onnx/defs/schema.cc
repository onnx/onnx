// Copyright (c) Facebook Inc. and Microsoft Corporation.
// Licensed under the MIT license.

#include "schema.h"
#include <unordered_set>
#include <stdexcept>

namespace onnx {

bool OpSchema::Verify(const NodeProto& node) const {
  // Check the number of inputs.
  if (node.input_size() < min_input_ || node.input_size() > max_input_) {
    std::cerr << "Input size " << node.input_size()
                    << " not in range [min=" << min_input_ << ", max="
                    << max_input_ << "].";
    return false;
  }
  if (!num_inputs_allowed_(node.input_size())) {
    std::cerr << "Input size " << node.input_size()
                    << " not in allowed input sizes.";
    return false;
  }
  // Check the number of outputs.
  if (node.output_size() < min_output_ || node.output_size() > max_output_) {
    std::cerr << "Output size " << node.output_size()
                    << " not in range [min=" << min_output_ << ", max="
                    << max_output_ << "].";
    return false;
  }
  if (!num_outputs_allowed_(node.output_size())) {
    std::cerr << "Output size " << node.output_size()
                    << " not in allowed output sizes.";
    return false;
  }
  if (!num_inputs_outputs_allowed_(node.input_size(), node.output_size())) {
    std::cerr << "Combination of input size " << node.input_size()
               << "and output size " << node.output_size() << " not in allowed.";
    return false;
  }
  // If the number of outputs can be calculated, check if the number matches.
  if (calculate_output_) {
    int expected_nout = calculate_output_(node.input_size());
    if (expected_nout != kCannotComputeNumOutputs &&
        node.output_size() != expected_nout) {
      std::cerr << "Output size " << node.output_size()
                      << " not matching expected output size, which is "
                      << expected_nout;
      return false;
    }
  }

  // Check the values of inputs / outputs
  for (int in_idx = 0; in_idx < node.input_size(); ++in_idx) {
    if (node.input(in_idx).empty() && !optional_inputs_.count(in_idx)) {
      std::cerr
          << "Input " << in_idx
          << " is not marked optional but has an empty string in the graph";
      return false;
    }
  }
  for (int out_idx = 0; out_idx < node.output_size(); ++out_idx) {
    if (node.output(out_idx).empty()) {
        std::cerr << "Output " << out_idx
                  << " has an empty string in the graph";
        return false;
      }
  }

  // Check attributes
  std::unordered_set<std::string> seen_attr_names{};
  const AttributeProto * consume_attr = nullptr;

  for (const auto& attr_proto : node.attribute()) {
      const auto& name = attr_proto.name();

      if (!seen_attr_names.insert(name).second) {
          std::cerr << "Attribute '"
                    << name
                    << "' appeared multiple times.";
          return false;
      };

      if (!IsAttributeLegal(attr_proto)) {
          std::cerr << "Attribute '"
                    << name
                    << "' is not legal"
                    << std::endl;
          return false;
      }

      const auto& search = attributes_.find(name);
      AttrType expected_type;
      if (search != attributes_.end()) {
          expected_type = search->second.type;
      } else if (allows_unchecked_attributes_){
        continue;
      } else if (name == "consumed_inputs") {
          expected_type = AttrType::INTS;
          consume_attr = &attr_proto;
          if(attr_proto.ints().size() != node.input_size()) {
            std::cerr << "Attribute consumed_inputs (length "
                      << attr_proto.ints().size()
                      << ") is not the same length as inputs (length "
                      << node.input_size() << ")" << std::endl;
            return false;
          }
      } else {
          std::cerr << "Unrecognized attribute: " << name << std::endl;
          return false;
      }

      switch (expected_type) {
      case AttrType::FLOAT:
          if (!attr_proto.has_f()) {
              std::cerr << "Attribute '"
                        << name
                        << "' is expected to have field 'f'"
                        << std::endl;
              return false;
          }
          break;
      case AttrType::INT:
          if (!attr_proto.has_i()) {
              std::cerr << "Attribute '"
                        << name
                        << "' is expected to have field 'i'"
                        << std::endl;
              return false;
          }
          break;
      case AttrType::STRING:
          if (!attr_proto.has_s()) {
              std::cerr << "Attribute '"
                        << name
                        << "' is expected to have field 's'"
                        << std::endl;
              return false;
          }
          break;
      case AttrType::TENSOR:
          if (!attr_proto.has_t()) {
              std::cerr << "Attribute '"
                        << name
                        << "' is expected to have field 't'"
                        << std::endl;
              return false;
          }
          break;
      case AttrType::GRAPH:
          if (!attr_proto.has_g()) {
              std::cerr << "Attribute '"
                        << name
                        << "' is expected to have field 'g'"
                        << std::endl;
              return false;
          }
          break;
      case AttrType::FLOATS:
          if (!attr_proto.floats_size()) {
              std::cerr << "Attribute '"
                        << name
                        << "' is expected to have field 'floats'"
                        << std::endl;
              return false;
          }
          break;
      case AttrType::INTS:
          if (!attr_proto.ints_size()) {
              std::cerr << "Attribute '"
                        << name
                        << "' is expected to have field 'ints'"
                        << std::endl;
              return false;
          }
          break;
      case AttrType::STRINGS:
          if (!attr_proto.strings_size()) {
              std::cerr << "Attribute '"
                        << name
                        << "' is expected to have field 'strings'"
                        << std::endl;
              return false;
          }
          break;
      case AttrType::TENSORS:
          if (!attr_proto.tensors_size()) {
              std::cerr << "Attribute '"
                        << name
                        << "' is expected to have field 'tensors'"
                        << std::endl;
              return false;
          }
          break;
      case AttrType::GRAPHS:
          if (!attr_proto.graphs_size()) {
              std::cerr << "Attribute '"
                        << name
                        << "' is expected to have field 'graphs'"
                        << std::endl;
              return false;
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
          std::cerr << "Required attribute '"
                    << attr.name
                    << "' is missing." << std::endl;
          return false;
      }
  }


  // Check in-place settings.
  for (int in_idx = 0; in_idx < node.input_size(); ++in_idx) {
    bool consumed = consume_attr ? consume_attr->ints(in_idx) : false;
    auto use_type = consumed_(in_idx);
    switch(use_type.first) {
      case UseType::DEFAULT:
        if(consumed) {
          std::cerr << "Input index " << in_idx << " is set to consumed but "
                    << "is not supported by op " << node.op_type();
          return false;
        } break;
      case UseType::CONSUME_ENFORCED:
        if(!consumed) {
          std::cerr << "Input index " << in_idx << " must be set to consumed "
                    << "for operator " << node.op_type();
          return false;
        } break;
      case UseType::CONSUME_ALLOWED:
        // pass, either is allowed
        break;
    }
  }

  // Phew. All verifications passed.
  return true;
}

bool OpSchema::IsAttributeLegal(const AttributeProto& proto) {
    if (proto.name().empty()) {
        std::cerr << "Attribute should set name field." << std::endl;
        return false;
    }

    int used_fields =
        proto.has_f() +
        proto.has_i() +
        proto.has_s() +
        proto.has_t() +
        proto.has_g() +
        (proto.floats_size() > 0) +
        (proto.ints_size() > 0) +
        (proto.strings_size() > 0) +
        (proto.tensors_size() > 0) +
        (proto.graphs_size() > 0);
    if (used_fields != 1) {
        std::cerr << "Attribute should contain one and only one value field.\n"
                  << "AttributeProto:\n"
                  << proto.SerializeAsString()
                  << "\n"
                  << std::endl;
        return false;
    }
    return true;
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
  return NumInputs(
      [allowed_input_nums](int n)->bool {
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
  return NumOutputs(
      [allowed_output_nums](int n)->bool {
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
  return OutputCalculator([](int n)->int { return n; } );
}

OpSchema& OpSchema::AllowConsumed(std::function<std::pair<bool,int>(int)> inplace) {
  consumed_ = [inplace](int idx) {
    auto r = inplace(idx);
    return std::make_pair(
      r.first ? UseType::CONSUME_ALLOWED : UseType::DEFAULT,
      r.second);
  };
  return *this;
}

OpSchema& OpSchema::AllowConsumed(std::unordered_map<int, int> inplace) {
  return AllowConsumed(
      [inplace](int idx) {
        auto it = inplace.find(idx);
        if(it != inplace.end()) {
          return std::make_pair(true, it->second);
        }
        return std::make_pair(false,0);
      });
}

OpSchema& OpSchema::AllowOneToOneConsumed() {
  return AllowConsumed([](int i){
    return std::make_pair(true,i);
  });
}

OpSchema& OpSchema::EnforceConsumed(std::function<std::pair<bool,int>(int)> inplace) {
  consumed_ = [inplace](int idx) {
    auto r = inplace(idx);
    return std::make_pair(
      r.first ? UseType::CONSUME_ENFORCED : UseType::DEFAULT,
      r.second);
  };
  return *this;
}

OpSchema& OpSchema::EnforceConsumed(std::unordered_map<int, int> inplace) {
  return EnforceConsumed(
      [inplace](int idx) {
        auto it = inplace.find(idx);
        if(it != inplace.end()) {
          return std::make_pair(true, it->second);
        }
        return std::make_pair(false,0);
      });
}

OpSchema& OpSchema::EnforceOneToOneConsumed() {
  return EnforceConsumed([](int i){
    return std::make_pair(true,i);
  });
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

OpSchema& OpSchema::Attr(const char* name,
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

OpSchema& OpSchema::Input(const int n, const char* name, const char* description, bool optional) {
  if (int(input_desc_.size()) <= n) {
    input_desc_.resize(n + 1);
  }
  input_desc_[n] = std::make_pair(name, description);
  if (optional) {
    optional_inputs_.insert(n);
  }
  return *this;
}

OpSchema& OpSchema::Output(const int n, const char* name, const char* description) {
  if (int(output_desc_.size()) <= n) {
    output_desc_.resize(n + 1);
  }
  output_desc_[n] = std::make_pair(name, description);
  return *this;
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
#define ENFORCE(x) do { if (!(x)) throw std::logic_error("ONNX Schema " + name_ + ": failed validating the check: " + #x); } while (0)
  ENFORCE(min_input_ <= max_input_);
  ENFORCE(min_output_ <= max_output_);
  ENFORCE(input_desc_.size() >= min_input_);
  ENFORCE(output_desc_.size() >= min_output_);
  ENFORCE(input_desc_.size() <= max_input_);
  ENFORCE(output_desc_.size() <= max_output_);
  // if max limit is finite - all names should be specified
  if (max_input_ < std::numeric_limits<int>::max()) {
    ENFORCE(input_desc_.size() == max_input_);
  }
  if (max_output_ < std::numeric_limits<int>::max()) {
    ENFORCE(output_desc_.size() == max_output_);
  }
  // all inputs and outputs have names
  for (const auto& it : input_desc_) {
    ENFORCE(it.first);
  }
  for (const auto& it : output_desc_) {
    ENFORCE(it.first);
  }
  // TODO: also cover checks for arbitrary number of inputs
  // allow extra tailing inputs not be present if all inputs at the end are
  // marked as optional
  if (max_input_ < std::numeric_limits<int>::max()) {
    int ind = max_input_;
    while (ind > 0 && optional_inputs_.count(ind-1)) {
      --ind;
    }
    min_input_ = std::min(min_input_, ind);
  }
}


std::ostream& operator<<(std::ostream& out, const OpSchema& schema) {
  if (!schema.attributes_.empty()) {
    out << "Attributes:" << std::endl;
    for (const auto& pair : schema.attributes_) {
      out << "  " << pair.second.name << " : " << pair.second.description << std::endl;
    }
  }
  if (schema.max_input_ > 0) {
    out << "Inputs:" << std::endl;
    if (!schema.input_desc_.empty()) {
      for (size_t i = 0; i < schema.input_desc_.size(); ++i) {
        const auto& p = schema.input_desc_[i];
        out << "  " << i << ", " << (p.first ? p.first : "(unnamed)") << " : "
            << (p.second ? p.second : "(no doc)") << std::endl;
      }
    } else {
      out << "  (no explicit description available)" << std::endl;
    }
  }
  if (schema.max_output_ > 0) {
    out << "Outputs:" << std::endl;
    if (!schema.output_desc_.empty()) {
      for (size_t i = 0; i < schema.output_desc_.size(); ++i) {
        const auto& p = schema.output_desc_[i];
        out << "  " << i << ", " << (p.first ? p.first : "(unnamed)") << " : "
            << (p.second ? p.second : "(no doc)") << std::endl;
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

}  // namespace onnx
