// Interface for Op Version Adapters

#pragma once

#include "onnx/common/ir.h"
#include "onnx/onnx_pb.h"

namespace ONNX_NAMESPACE { namespace version_conversion {

class Adapter {
  private:
    std::string name_;
    OpSetID initial_version_;
    OpSetID target_version_;

  protected:
    bool c2_broadcastable(int64_t axis,
        const std::vector<Dimension>& input1_sizes,
        const std::vector<Dimension>& input2_sizes) const {
      return axis != (int) (input1_sizes.size() - input2_sizes.size());
    }

    bool numpy_broadcastable(const std::vector<Dimension>& input1_sizes,
        const std::vector<Dimension>& input2_sizes) const {
      for (int j = 1; j < (int) input1_sizes.size(); j++) {
        ONNX_ASSERTM(input1_sizes[j].dim == input2_sizes[j].dim, "A: %d, B: %d",
            input1_sizes[j].dim, input2_sizes[j].dim);
      }
      return input1_sizes.size() >= input2_sizes.size();
    }

    bool numpy_unibroadcastable(const std::vector<Dimension>& input_sizes) const {
      return input_sizes.size() == 1 || (input_sizes.size() == 2 &&
          (input_sizes[0].dim == 1 || input_sizes[1].dim == 1));
    }

  public:
    virtual ~Adapter() noexcept = default;

    explicit Adapter(const std::string& name, const OpSetID& initial_version,
        const OpSetID& target_version)
      : name_(name), initial_version_(initial_version),
        target_version_(target_version) {
    }

    virtual void adapt(std::shared_ptr<Graph> /*graph*/, Node* node) const = 0;

    const std::string& name() const {
      return name_;
    }

    const OpSetID& initial_version() const {
      return initial_version_;
    }

    const OpSetID& target_version() const {
      return target_version_;
    }
};

}} // namespace ONNX_NAMESPACE::version_conversion
