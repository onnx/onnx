# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
import unittest

import automatic_conversion_test_base
import numpy as np
import parameterized

import onnx
from onnx import helper

#####################################################################################
# Every test calls _test_op_conversion to downgrade a model from the most recent opset version
# to a early version and runs checker + shape inference on the downgraded model.
####################################################################################


class TestAutomaticDowngrade(automatic_conversion_test_base.TestAutomaticConversion):
    def _test_op_downgrade(self, op: str, *args, **kwargs):
        self._test_op_conversion(op, *args, **kwargs, is_upgrade=False)

    @parameterized.parameterized.expand(
        [
            "ReduceL1",
            "ReduceL2",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceMean",
            "ReduceMax",
            "ReduceMin",
            "ReduceProd",
            "ReduceSum",
            "ReduceSumSquare",
        ]
    )
    def test_reduce_ops(self, op) -> None:
        # TODO: need to add test cases for missing axes input which depends on this pr:
        # https://github.com/onnx/onnx/pull/5613
        axes = helper.make_tensor(
            "b", onnx.TensorProto.INT64, dims=[3], vals=np.array([0, 1, 2])
        )
        self._test_op_downgrade(
            op,
            from_opset=13,
            input_shapes=[[3, 4, 5], [3]],
            output_shapes=[[1, 1, 1]],
            input_types=[onnx.TensorProto.FLOAT, onnx.TensorProto.INT64],
            initializer=[axes],
        )


if __name__ == "__main__":
    unittest.main()
