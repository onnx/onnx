# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import os
import platform
import unittest
from typing import Any

import numpy

import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx import ModelProto
from onnx.backend.base import Device, DeviceType
from onnx.reference import ReferenceEvaluator

# The following just executes a backend based on ReferenceEvaluator through the backend test


class ReferenceEvaluatorBackendRep(onnx.backend.base.BackendRep):
    def __init__(self, session):
        self._session = session

    def run(self, inputs, **kwargs):
        if isinstance(inputs, numpy.ndarray):
            inputs = [inputs]
        if isinstance(inputs, list):
            if len(inputs) == len(self._session.input_names):
                feeds = dict(zip(self._session.input_names, inputs))
            else:
                feeds = {}
                pos_inputs = 0
                for inp, tshape in zip(
                    self._session.input_names, self._session.input_types
                ):
                    shape = tuple(d.dim_value for d in tshape.tensor_type.shape.dim)
                    if shape == inputs[pos_inputs].shape:
                        feeds[inp] = inputs[pos_inputs]
                        pos_inputs += 1
                        if pos_inputs >= len(inputs):
                            break
        elif isinstance(inputs, dict):
            feeds = inputs
        else:
            raise TypeError(f"Unexpected input type {type(inputs)!r}.")
        outs = self._session.run(None, feeds)
        return outs


class ReferenceEvaluatorBackend(onnx.backend.base.Backend):
    @classmethod
    def is_opset_supported(cls, model):  # pylint: disable=unused-argument
        return True, ""

    @classmethod
    def supports_device(cls, device: str) -> bool:
        d = Device(device)
        return d.type == DeviceType.CPU  # type: ignore[no-any-return]

    @classmethod
    def create_inference_session(cls, model):
        return ReferenceEvaluator(model)

    @classmethod
    def prepare(
        cls, model: Any, device: str = "CPU", **kwargs: Any
    ) -> ReferenceEvaluatorBackendRep:
        # if isinstance(model, ReferenceEvaluatorBackendRep):
        #    return model
        if isinstance(model, ReferenceEvaluator):
            return ReferenceEvaluatorBackendRep(model)
        if isinstance(model, (str, bytes, ModelProto)):
            inf = cls.create_inference_session(model)
            return cls.prepare(inf, device, **kwargs)
        raise TypeError(f"Unexpected type {type(model)} for model.")

    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs):
        rep = cls.prepare(model, device, **kwargs)
        return rep.run(inputs, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        raise NotImplementedError("Unable to run the model node by node.")


backend_test = onnx.backend.test.BackendTest(ReferenceEvaluatorBackend, __name__)

if os.getenv("APPVEYOR"):
    backend_test.exclude("(test_vgg19|test_zfnet)")
if platform.architecture()[0] == "32bit":
    backend_test.exclude("(test_vgg19|test_zfnet|test_bvlc_alexnet)")
if platform.system() == "Windows":
    backend_test.exclude("test_sequence_model")

# The following tests are not supported.
backend_test.exclude(
    "(test_gradient"
    "|test_if_opt"
    "|test_loop16_seq_none"
    "|test_range_float_type_positive_delta_expanded"
    "|test_range_int32_type_negative_delta_expanded"
    "|test_scan_sum)"
)

# The following tests are about deprecated operators.
backend_test.exclude("(test_scatter_with_axis|test_scatter_without)")

# The following tests are using types not supported by numpy.
# They could be if method to_array is extended to support custom
# types the same as the reference implementation does
# (see onnx.reference.op_run.to_array_extended).
backend_test.exclude(
    "(test_cast_FLOAT_to_FLOAT8"
    "|test_cast_FLOAT16_to_FLOAT8"
    "|test_castlike_FLOAT_to_FLOAT8"
    "|test_castlike_FLOAT16_to_FLOAT8"
    "|test_cast_no_saturate_FLOAT_to_FLOAT8"
    "|test_cast_no_saturate_FLOAT16_to_FLOAT8"
    "|test_cast_BFLOAT16_to_FLOAT"
    "|test_castlike_BFLOAT16_to_FLOAT"
    "|test_quantizelinear_e4m3"
    "|test_quantizelinear_e5m2"
    ")"
)

# The following tests are using types not supported by NumPy.
# They could be if method to_array is extended to support custom
# types the same as the reference implementation does
# (see onnx.reference.op_run.to_array_extended).
backend_test.exclude(
    "(test_cast_FLOAT_to_BFLOAT16"
    "|test_castlike_FLOAT_to_BFLOAT16"
    "|test_castlike_FLOAT_to_BFLOAT16_expanded"
    ")"
)

# The following tests are too slow with the reference implementation (Conv).
backend_test.exclude(
    "(test_bvlc_alexnet"
    "|test_densenet121"
    "|test_inception_v1"
    "|test_inception_v2"
    "|test_resnet50"
    "|test_shufflenet"
    "|test_squeezenet"
    "|test_vgg19"
    "|test_zfnet512)"
)

# The following tests cannot pass because they consists in generating random number.
backend_test.exclude("(test_bernoulli)")

# The following tests fail due to a bug in the backend test comparison.
backend_test.exclude(
    "(test_cast_FLOAT_to_STRING|test_castlike_FLOAT_to_STRING|test_strnorm)"
)

# The following tests fail due to a shape mismatch.
backend_test.exclude(
    "(test_center_crop_pad_crop_axes_hwc_expanded"
    "|test_lppool_2d_dilations"
    "|test_averagepool_2d_dilations)"
)

# The following tests fail due to a type mismatch.
backend_test.exclude("(test_eyelike_without_dtype)")

# The following tests fail due to discrepancies (small but still higher than 1e-7).
backend_test.exclude("test_adam_multiple")  # 1e-2

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.test_cases)

if __name__ == "__main__":
    res = unittest.main(verbosity=2, exit=False)
    tests_run = res.result.testsRun
    errors = len(res.result.errors)
    skipped = len(res.result.skipped)
    unexpected_successes = len(res.result.unexpectedSuccesses)
    expected_failures = len(res.result.expectedFailures)
    print("---------------------------------")
    print(
        f"tests_run={tests_run} errors={errors} skipped={skipped} "
        f"unexpected_successes={unexpected_successes} "
        f"expected_failures={expected_failures}"
    )
