// Adapter for ArgMax/ArgMin in default domain from version 11 to 12

#pragma once

namespace ONNX_NAMESPACE { namespace version_conversion {

class ArgMaxArgMin_11_12 final : public Adapter {
  public:
    explicit ArgMaxArgMin_11_12(const std::string& op_name)
      : Adapter(op_name, OpSetID(11), OpSetID(12)) {}

    void adapt_argmax_argmin_11_12(std::shared_ptr<Graph> graph, Node* node) const {
      Symbol select_last_index = Symbol("select_last_index");
      node->i_(select_last_index, 0);
    }

    void adapt(std::shared_ptr<Graph> graph, Node* node) const override {
	    adapt_argmax_argmin_11_12(graph, node);
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
