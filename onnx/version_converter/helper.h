// Helper Methods for Adapters

#pragma once

#include "onnx/common/ir.h"

namespace ONNX_NAMESPACE { namespace version_conversion {
    bool onnx_opset7_requires_broadcasting(int64_t axis,
        const std::vector<Dimension>& input1_sizes,
        const std::vector<Dimension>& input2_sizes);

    bool numpy_unibroadcastable(const std::vector<Dimension>& input1_sizes,
        const std::vector<Dimension>& input2_sizes);

    void numpy_multibroadcastable(const std::vector<Dimension>& input1_sizes,
        const std::vector<Dimension>& input2_sizes);

    void assertNotParams(const std::vector<Dimension>& sizes);
}}
