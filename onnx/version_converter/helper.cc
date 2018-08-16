// Helper Methods for Adapters

#include "onnx/version_converter/helper.h"

namespace ONNX_NAMESPACE { namespace version_conversion {
    bool onnx_opset7_broadcastable(int64_t axis,
        const std::vector<Dimension>& input1_sizes,
        const std::vector<Dimension>& input2_sizes) {
      // Assert that broadcasting semantics are correct
      for (int i = 0; i < (int) input2_sizes.size(); i++) {
        ONNX_ASSERTM(input2_sizes[i].dim == input1_sizes[axis + i].dim ||
            input2_sizes[i].dim == 1, "Dimension %d of input 2 does not match "
            "dimension %d of input 1 or the value 1", i, axis + i);
      }
      // Return false if a Reshape is necessary for forward compatibility
      return axis != (int) (input1_sizes.size() - input2_sizes.size());
    }

    bool numpy_unibroadcastable(const std::vector<Dimension>& input1_sizes,
        const std::vector<Dimension>& input2_sizes) {
      // Assert that equal of input1 larger
      ONNX_ASSERTM(input1_sizes.size() >= input2_sizes.size(),
          "ONNX Opset 6 can only broadcast input 2 to input 1");
      // Assert that axis is input1_sizes.size()-input2_sizes.size()
      bool broadcast = false;
      int axis = (int) (input1_sizes.size() - input2_sizes.size());
      for (int i = 0; i < (int) input2_sizes.size(); i++) {
        ONNX_ASSERTM(input2_sizes[i].dim == input1_sizes[axis + i].dim ||
            input2_sizes[i].dim == 1, "Dimension %d of input 2 does not match "
            "dimension %d of input 1 or the value 1", i, axis + i);
        if (input2_sizes[i].dim != input1_sizes[axis + i].dim) broadcast = true;
      }
      // Return true if broadcasting is required
      return input1_sizes.size() > input2_sizes.size() || broadcast;
    }

    void numpy_multibroadcastable(const std::vector<Dimension>& input1_sizes,
        const std::vector<Dimension>& input2_sizes) {
      // Generalize above for multibroadcastable case
      if (input1_sizes.size() < input2_sizes.size()) {
        numpy_unibroadcastable(input2_sizes, input1_sizes);
      } else {
        numpy_unibroadcastable(input1_sizes, input2_sizes);
      }
    }
}}
