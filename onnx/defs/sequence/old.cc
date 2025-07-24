/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <numeric>

#include "onnx/defs/function.h"
#include "onnx/defs/schema.h"
#include "onnx/defs/sequence/utils.h"

namespace ONNX_NAMESPACE {

ONNX_OPERATOR_SET_SCHEMA(
    SplitToSequence,
    11,
    OpSchema().FillUsing(
        defs::sequence::utils::SplitToSequenceOpGenerator(
            OpSchema::all_tensor_types(),
            OpSchema::all_tensor_sequence_types())));

}
