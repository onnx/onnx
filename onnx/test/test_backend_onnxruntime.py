# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

import os
import platform
import sys
import unittest
from typing import Any

import numpy
from packaging.version import Version

import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx import ModelProto
from onnx.backend.base import Device, DeviceType

try:
    from onnxruntime import InferenceSession
    from onnxruntime import __version__ as ort_version
    from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
except ImportError:
    # onnxruntime is not installed, all tests are skipped.
    InferenceSession = None
    ort_version = None

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
                for inp, shape in zip(input_names, input_shapes):
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
    def convert_version_opset_before(cls, model):
        opsets = {d.domain: d.version for d in model.opset_import}
        if "" not in opsets:
            return None
        try:
            return onnx.version_converter.convert_version(model, opsets[""] - 1)
        except RuntimeError:
            # Let's try without any change.
            del model.opset_import[:]
            for k, v in opsets.items():
                d = model.opset_import.add()
                d.domain = k
                d.version = v if k != "" else v - 1
            return model

    @classmethod
    def create_inference_session(cls, model):
        try:
            return InferenceSession(model.SerializeToString())
        except InvalidArgument as e:
            if "Unsupported model IR version" in str(e):
                model.ir_version -= 1
                return cls.create_inference_session(model)
            if "Current official support for domain ai.onnx is till opset" in str(e):
                new_model = cls.convert_version_opset_before(model)
                if new_model is not None:
                    return cls.create_inference_session(new_model)
            raise e

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

# The following tests are not supported by onnxruntime.
backend_test.exclude(
    "("
    "test_adagrad"
    "|test_adam"
    "|test_add_uint8"
    "|bitshift_left_uint16"
    "|bitshift_right_uint16"
    "|cast_BFLOAT16_to_FLOAT"
    "|cast_FLOAT_to_BFLOAT16"
    "|castlike_BFLOAT16_to_FLOAT"
    "|castlike_FLOAT_to_BFLOAT16"
    "|clip_default_int8_min_expanded"
    "|clip_default_int8_max_expanded"
    "|div_uint8"
    "|gru_batchwise"  # Batchwise recurrent operations (layout == 1) are not supported.
    "|loop16_seq_none"  # The graph is missing type information needed to construct the ORT tensor.
    "|lstm_batchwise"  # Batchwise recurrent operations (layout == 1) are not supported.
    "|m(in|ax)_u?int(16|8)"
    "|momentum"
    "|mul_uint8"
    "|pow_types_float32_uint32"
    "|pow_types_float32_uint64"
    "|simple_rnn_batchwise"  # Batchwise recurrent operations (layout == 1) are not supported.
    "|sub_uint8"
    "|gradient_of_add"
    ")"
)

# The following tests fail due to small discrepancies.
backend_test.exclude("(cast_FLOAT_to_STRING|castlike_FLOAT_to_STRING|dft|stft)")

# The following tests fail due to huge discrepancies.
backend_test.exclude(
    "("
    "resize_downsample_scales_cubic_align_corners"
    "|resize_downsample_scales_linear_align_corners"
    "|training_dropout"
    ")"
)

# The following tests fail for no obvious reason.
backend_test.exclude(
    "("
    "maxunpool_export_with_output_shape"  # not the same expected output
    "|softplus_example_expanded"  # Could not find an implementation for Exp(1) node with name ''
    "|softplus_expanded"  # Could not find an implementation for Exp(1) node with name ''
    "|AvgPool[1-3]d"  # Could not find an implementation for AveragePool(1) node with name ''
    "|BatchNorm1d_3d_input_eval"  # Could not find an implementation for BatchNormalization(6) node with name ''
    "|BatchNorm[2-3]d_eval"  # Could not find an implementation for BatchNormalization(6) node with name ''
    "|GLU"  # Could not find an implementation for Mul(6) node with name ''
    "|Linear"  # Could not find an implementation for Gemm(6) node with name ''
    "|PReLU"  # Could not find an implementation for PRelu(6) node with name ''
    "|PoissonNLL"  # Could not find an implementation for Mul(6) node with name ''
    "|Softsign"  # Could not find an implementation for Gemm(6) node with name ''
    "|operator_add_broadcast"  # Could not find an implementation for Gemm(6) node with name ''
    "|operator_add_size1"  # Could not find an implementation for Gemm(6) node with name ''
    "|operator_addconstant"  # Could not find an implementation for Gemm(6) node with name ''
    "|operator_addmm"  # Could not find an implementation for Gemm(6) node with name ''
    "|operator_basic"  # Could not find an implementation for Add(6) node with name ''
    "|operator_mm"  # Could not find an implementation for Gemm(6) node with name ''
    "|operator_non_float_params"  # Could not find an implementation for Add(6) node with name ''
    "|operator_params"  # Could not find an implementation for Add(6) node with name ''
    "|operator_pow"  # Could not find an implementation for Pow(1) node with name ''
    ")"
)

# The following tests are new with opset 19.
if ort_version is not None and Version(ort_version) < Version("1.16"):
    # version should be 1.15 but there is no development version number.
    backend_test.exclude(
        "("
        "averagepool_2d_dilations"
        "|deform_conv"
        "|equal_string"
        "|optional_get_element_optional_sequence"
        "|identity_opt"
        "|half_pixel_symmetric"
        "|wrap_pad"
        ")"
    )

if sys.version_info[:2] < (3, 8) or Version(numpy.__version__) >= Version("1.23.5"):
    # Version 1.21.5 causes segmentation faults.
    # onnxruntime should be tested with the same numpy API
    # onnxruntime was compiled with.
    backend_test.exclude("")

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
