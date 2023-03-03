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

try:
    from onnxruntime import InferenceSession
except ImportError:
    # onnxruntime is not installed, all tests are skipped.
    InferenceSession = None

# The following just executes a backend based on InferenceSession through the backend test


class InferenceSessionBackendRep(onnx.backend.base.BackendRep):
    def __init__(self, session):
        self._session = session

    def run(self, inputs, **kwargs):
        if isinstance(inputs, numpy.ndarray):
            inputs = [inputs]
        if isinstance(inputs, list):
            input_names = [i.name for i in self._session.get_inputs()]
            input_shapes = [i.shape for i in self._session.get_inputs()]
            if len(inputs) == len(input_names):
                feeds = dict(zip(input_names, inputs))
            else:
                feeds = {}
                pos_inputs = 0
                for i, (inp, shape) in enumerate(zip(input_names, input_shapes)):
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


class InferenceSessionBackend(onnx.backend.base.Backend):
    @classmethod
    def is_opset_supported(cls, model):  # pylint: disable=unused-argument
        return True, ""

    @classmethod
    def supports_device(cls, device: str) -> bool:
        d = Device(device)
        return d.type == DeviceType.CPU  # type: ignore[no-any-return]

    @classmethod
    def create_inference_session(cls, model):
        return InferenceSession(model.SerializeToString())

    @classmethod
    def prepare(
        cls, model: Any, device: str = "CPU", **kwargs: Any
    ) -> InferenceSessionBackendRep:
        # if isinstance(model, InferenceSessionBackendRep):
        #    return model
        if isinstance(model, InferenceSession):
            return InferenceSessionBackendRep(model)
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


backend_test = onnx.backend.test.BackendTest(InferenceSessionBackend, __name__)

if os.getenv("APPVEYOR"):
    backend_test.exclude("(test_vgg19|test_zfnet)")
if platform.architecture()[0] == "32bit":
    backend_test.exclude("(test_vgg19|test_zfnet|test_bvlc_alexnet)")
if platform.system() == "Windows":
    backend_test.exclude("test_sequence_model")

# The following tests cannot pass because they consists in generating random number.
backend_test.exclude("(test_bernoulli)")

# import all test cases at global scope to make them visible to python.unittest
if InferenceSession is not None:
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
