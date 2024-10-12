# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import platform
import sys
import unittest
from typing import Any

import numpy as np
import version_utils

import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx import ModelProto
from onnx.backend.base import Device, DeviceType
from onnx.reference import ReferenceEvaluator

# The following just executes a backend based on ReferenceEvaluator through the backend test


class TorchReferenceEvaluatorBackendRep(onnx.backend.base.BackendRep):
    def __init__(self, session):
        self._session = session
        try:
            import torch

            self.torch = torch
        except ImportError:
            self.torch = None

        self._type_mapping = {
            np.float16: torch.float16,
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.int8: torch.int8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.uint8: torch.uint8,
            np.uint16: torch.uint16,
            np.uint32: torch.uint32,
            np.uint64: torch.uint64,
            np.bool_: torch.bool,
            np.dtype("float16"): torch.float16,
            np.dtype("float32"): torch.float32,
            np.dtype("float64"): torch.float64,
            np.dtype("int8"): torch.int8,
            np.dtype("int16"): torch.int16,
            np.dtype("int32"): torch.int32,
            np.dtype("int64"): torch.int64,
            np.dtype("uint8"): torch.uint8,
            np.dtype("uint16"): torch.uint16,
            np.dtype("uint32"): torch.uint32,
            np.dtype("uint64"): torch.uint64,
            np.dtype("bool"): torch.bool,
        }

    def _fix_type(self, tensor, np_dtype):
        torch_dtype = self._type_mapping[np_dtype]
        if tensor.dtype == torch_dtype:
            return tensor
        return tensor.to(torch_dtype)

    def to_torch(self, feeds: dict[str, Any], device: str) -> dict[str, Any]:
        new_feeds = {}
        for k, v in feeds.items():
            if device == "CUDA":
                tensor = self.torch.Tensor(v).to("cuda")
            else:
                tensor = self.torch.Tensor(v.copy())
            new_feeds[k] = self._fix_type(tensor, v.dtype)
        return new_feeds

    def run(self, inputs, device=None, **kwargs):  # noqa: ARG002
        if isinstance(inputs, np.ndarray):
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
        feeds = self.to_torch(feeds, device)
        outs = self._session.run(None, feeds)
        np_outs = []
        for o in outs:
            if not isinstance(o, self.torch.Tensor):
                raise TypeError(f"One output is not a torch.Tensor but {type(o)}")
            # we need to convert to numpy as the backend is written for numpy
            np_outs.append(o.detach().numpy())
        return np_outs


class TorchReferenceEvaluatorBackend(onnx.backend.base.Backend):
    @classmethod
    def is_opset_supported(cls, model):  # noqa: ARG003
        return True, ""

    @classmethod
    def supports_device(cls, device: str) -> bool:
        try:
            import torch
        except ImportError:
            return False
        d = Device(device)
        if d.type == DeviceType.CPU:
            return True
        if d.type == DeviceType.CUDA and torch.cuda.is_available():
            return True
        return False

    @classmethod
    def create_inference_session(cls, model):
        return ReferenceEvaluator(model)

    @classmethod
    def prepare(
        cls, model: Any, device: str = "CPU", **kwargs: Any
    ) -> TorchReferenceEvaluatorBackendRep:
        # if isinstance(model, ReferenceEvaluatorBackendRep):
        #    return model
        if isinstance(model, ReferenceEvaluator):
            return TorchReferenceEvaluatorBackendRep(model)
        if isinstance(model, (str, bytes, ModelProto)):
            inf = cls.create_inference_session(model)
            return cls.prepare(inf, device, **kwargs)
        raise TypeError(f"Unexpected type {type(model)} for model.")

    @classmethod
    def run_model(cls, model, inputs, device=None, **kwargs):
        rep = cls.prepare(model, device, **kwargs)
        return rep.run(inputs, device=device, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        raise NotImplementedError("Unable to run the model node by node.")


dft_atol = 1e-3 if sys.platform != "linux" else 1e-6
backend_test = onnx.backend.test.BackendTest(
    TorchReferenceEvaluatorBackend,
    __name__,
    test_kwargs={
        "test_dft": {"atol": dft_atol},
        "test_dft_axis": {"atol": dft_atol},
        "test_dft_axis_opset19": {"atol": dft_atol},
        "test_dft_inverse": {"atol": dft_atol},
        "test_dft_inverse_opset19": {"atol": dft_atol},
        "test_dft_opset19": {"atol": dft_atol},
    },
)

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

# The following tests are using types not supported by np.
# They could be if method to_array is extended to support custom
# types the same as the reference implementation does
# (see onnx.reference.op_run.to_array_extended).
backend_test.exclude(
    "(test_quantizelinear_uint4"
    "|test_quantizelinear_int4"
    "|test_quantizelinear_float4e2m1"
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

# Currently google-re2/Pillow is not supported on Win32 and is required for the reference implementation of RegexFullMatch.
backend_test.exclude(
    "(test_regex_full_match_basic_cpu"
    "|test_regex_full_match_email_domain_cpu"
    "|test_regex_full_match_empty_cpu"
    "|test_image_decoder_decode_)"
)

if sys.version_info <= (3, 10):
    #  AttributeError: module 'np.typing' has no attribute 'NDArray'
    backend_test.exclude("test_image_decoder_decode_")

if sys.platform == "darwin":
    # FIXME: https://github.com/onnx/onnx/issues/5792
    backend_test.exclude("test_qlinearmatmul_3D_int8_float16_cpu")
    backend_test.exclude("test_qlinearmatmul_3D_int8_float32_cpu")

if version_utils.pillow_older_than("10.0"):
    backend_test.exclude("test_image_decoder_decode_webp_rgb")
    backend_test.exclude("test_image_decoder_decode_jpeg2k_rgb")

# still failing with pytorch
backend_test.exclude(
    "(test_adagrad"
    "|test_adam"
    "|affine"  # numpy transpose and torch transpose are different
    "|label_encoder_string"  # string not implemented
    "|label_encoder_tensor_mapping"
    "|label_encoder_tensor_value_only_mapping"
    "|tree_ensemble"  # no tree implementation
    ")"
)

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
