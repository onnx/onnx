# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
import unittest

import numpy as np
from automatic_conversion_test_base import TestAutomaticConversion

from onnx import TensorProto, helper

#####################################################################################
# Every test calls _test_op_conversion to downgrade a model from the most recent opset version
# to a early version and runs checker + shape inference on the downgraded model.
####################################################################################


class TestAutomaticDowngrade(TestAutomaticConversion):
    def _test_op_downgrade(self, op, *args, **kwargs):
        self._test_op_conversion(op, *args, **kwargs, is_upgrade=False)

    def test_ReduceOps(self) -> None:
        axes = helper.make_tensor(
            "b", TensorProto.INT64, dims=[3], vals=np.array([0, 1, 2])
        )
        reduce_ops = [
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
        for reduce_op in reduce_ops:
            self._test_op_downgrade(
                reduce_op,
                13,
                [[3, 4, 5], [3]],
                [[1, 1, 1]],
                [TensorProto.FLOAT, TensorProto.INT64],
                initializer=[axes],
            )


if __name__ == "__main__":
    unittest.main()
