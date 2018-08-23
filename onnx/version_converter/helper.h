// Helper Methods for Adapters

#pragma once

#include "onnx/common/ir.h"

namespace ONNX_NAMESPACE { namespace version_conversion {
    bool numpy_unibroadcastable(const std::vector<Dimension>& input1_sizes,
        const std::vector<Dimension>& input2_sizes);

    void numpy_multibroadcastable(const std::vector<Dimension>& input1_sizes,
        const std::vector<Dimension>& input2_sizes);

    void assertNotParams(const std::vector<Dimension>& sizes);

    void assertInputsAvailable(const ArrayRef<Value*>& inputs, const char* name, uint64_t num_inputs);
}}
