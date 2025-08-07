#include "onnx2_helper.h"

#include <filesystem>

namespace onnx2 {
bool IteratorTensorProto::next() {
  while (!positions_.empty()) {
    Position& pos = positions_.back();
    // loops over nodes
    bool break_loop = false;
    if (pos.graph->ref_node().size() > 0) {
      while (pos.node_index < static_cast<int>(pos.graph->ref_node().size())) {
        NodeProto* node = &(pos.graph->ref_node()[pos.node_index]);
        while (pos.attr_index < static_cast<int>(node->ref_attribute().size())) {
          AttributeProto& att = node->ref_attribute()[pos.attr_index];
          if (att.has_t()) {
            tp_ = &(att.ref_t());
            ++pos.attr_index;
            return true;
          } else if (att.has_g()) {
            GraphProto* subgraph = &(att.ref_g());
            // Do not swich this line with the next one, if vector needs to be resized,
            // the address of (Position&) pos changes as well.
            ++pos.attr_index;
            positions_.emplace_back(Position{subgraph});
            break_loop = true;
            break;
          }
          EXT_ENFORCE(!att.has_tensors(), "not implemented yet for attribute with tensors");
          EXT_ENFORCE(!att.has_graphs(), "not implemented yet for attribute with graphs");
          ++pos.attr_index;
        }
        if (break_loop)
          break;
        ++pos.node_index;
        pos.attr_index = 0;
      }
    }
    if (break_loop)
      continue;
    // loop over initializers
    if (pos.graph->ref_initializer().size() > 0) {
      if (pos.node_initializer_index < static_cast<int64_t>(pos.graph->ref_initializer().size())) {
        tp_ = &(pos.graph->ref_initializer()[pos.node_initializer_index]);
        ++pos.node_initializer_index;
        return true;
      }
    }
    positions_.pop_back();
  }
  return false;
}

void PopulateExternalData(ModelProto& model, size_t threshold, const std::string& external_data_location) {
  offset_t offset = 0;
  IteratorTensorProto it(&model.ref_graph());
  while (it.next()) {
    if (it->has_raw_data() && it->ref_raw_data().size() >= threshold) {
      EXT_ENFORCE(!it->has_external_data(), "External data should not be set already.");
      EXT_ENFORCE(
          !it->has_data_location() || it->ref_data_location() == TensorProto::DataLocation::DEFAULT,
          "External data should not be set already.");
      it->ref_data_location() = TensorProto::DataLocation::EXTERNAL;
      StringStringEntryProto& loc = it->add_external_data();
      loc.set_key("location");
      loc.set_value(external_data_location);
      StringStringEntryProto& off = it->add_external_data();
      off.set_key("offset");
      off.set_value(common_helpers::MakeString(offset));
      StringStringEntryProto& size = it->add_external_data();
      size.set_key("length");
      size.set_value(std::to_string(it->ref_raw_data().size()));
      offset += it->ref_raw_data().size();
    }
  }
}

void ClearExternalData(ModelProto& model) {
  IteratorTensorProto it(&model.ref_graph());
  while (it.next()) {
    if (it->has_external_data()) {
      EXT_ENFORCE(!it->ref_raw_data().empty(), "raw_data is empty, external data should not be removed.");
      it->clr_external_data();
      it->reset_data_location();
    }
  }
}

void SerializeModelProtoToStream(
    ModelProto& model,
    utils::BinaryWriteStream& stream,
    SerializeOptions& options,
    bool clear_external_data) {
  if (stream.ExternalWeights()) {
    utils::TwoFilesWriteStream& two_stream = dynamic_cast<utils::TwoFilesWriteStream&>(stream);
    std::filesystem::path parent_path = two_stream.file_path();
    parent_path = parent_path.parent_path();
    std::filesystem::path weight_path = two_stream.weights_file_path();
    weight_path = std::filesystem::relative(weight_path, parent_path);
    if (weight_path.empty()) {
      // If the relative path is empty, it means the weight file is in the same directory as the model.
      weight_path = two_stream.weights_file_path();
    }
    PopulateExternalData(model, options.raw_data_threshold, weight_path.string());
  }
  model.SerializeToStream(stream, options);
  if (stream.ExternalWeights() && clear_external_data)
    ClearExternalData(model);
}

void ParseModelProtoFromStream(
    ModelProto& model,
    utils::BinaryStream& stream,
    ParseOptions& options,
    bool clear_external_data) {
  if (stream.ExternalWeights()) {
    utils::TwoFilesStream& two_stream = dynamic_cast<utils::TwoFilesStream&>(stream);
    std::filesystem::path parent_path = two_stream.file_path();
    parent_path = parent_path.parent_path();
    std::filesystem::path weight_path = two_stream.weights_file_path();
    weight_path = std::filesystem::relative(weight_path, parent_path);
    if (weight_path.empty()) {
      // If the relative path is empty, it means the weight file is in the same directory as the model.
      weight_path = two_stream.weights_file_path();
    }
  }
  model.ParseFromStream(stream, options);
  if (stream.ExternalWeights() && clear_external_data)
    ClearExternalData(model);
}

} // namespace onnx2
