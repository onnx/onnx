// Helper Methods for Adapters

#pragma once

#include "onnx/common/ir.h"

namespace ONNX_NAMESPACE { namespace version_conversion {
    // Return -1 if non-broadcastable, 0 if no broadcasting, 1 if requires
    // broadcasting
    int check_numpy_unibroadcastable_and_require_broadcast(
        const std::vector<Dimension>& input1_sizes,
        const std::vector<Dimension>& input2_sizes);

    void assert_numpy_multibroadcastable(const std::vector<Dimension>& input1_sizes,
        const std::vector<Dimension>& input2_sizes);

    void assertNotParams(const std::vector<Dimension>& sizes);

    void assertInputsAvailable(const ArrayRef<Value*>& inputs, const char* name, uint64_t num_inputs);
}}
