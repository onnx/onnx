# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import platform
import unittest
from typing import Any

import numpy
from packaging.version import Version

import onnx.backend.base
import onnx.backend.test
import onnx.shape_inference
import onnx.version_converter
from onnx.backend.base import Device, DeviceType

try:
    import onnxruntime as ort

    ort_version = Version(ort.__version__)
except ImportError:
    # onnxruntime is not installed, all tests are skipped.
    ort: Any = None  # type: ignore[no-redef]
    ort_version: Any = None  # type: ignore[no-redef]


# The following just executes a backend based on InferenceSession through the backend test


class InferenceSessionBackendRep(onnx.backend.base.BackendRep):
    def __init__(self, session):
        self._session = session

    def run(self, inputs, **kwargs):
        del kwargs  # Unused
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


def _create_inference_session(model: onnx.ModelProto, device: str):
    if device == "CPU":
        providers = ("CPUExecutionProvider",)
    elif device == "CUDA":
        providers = ("CUDAExecutionProvider",)
    else:
        raise ValueError(f"Unexpected device {device!r}.")
    try:
        session = ort.InferenceSession(model.SerializeToString(), providers=providers)
    except Exception as e:
        raise RuntimeError(
            f"Unable to create inference session. Model is:\n\n{onnx.printer.to_text(model)}"
        ) from e
    return session


class InferenceSessionBackend(onnx.backend.base.Backend):
    @classmethod
    def supports_device(cls, device: str) -> bool:
        providers = set(ort.get_available_providers())
        d = Device(device)
        if d.type == DeviceType.CPU and "CPUExecutionProvider" in providers:
            return True
        if d.type == DeviceType.CUDA and "CUDAExecutionProvider" in providers:
            return True
        return False

    @classmethod
    def prepare(
        cls, model: onnx.ModelProto, device: str = "CPU", **kwargs: Any
    ) -> InferenceSessionBackendRep:
        del kwargs  # Unused
        if not isinstance(model, (str, bytes, onnx.ModelProto)):
            raise TypeError(f"Unexpected type {type(model)} for model.")

        session = _create_inference_session(model, device)
        return InferenceSessionBackendRep(session)

    @classmethod
    def run_model(cls, model: onnx.ModelProto, inputs, device=None, **kwargs):
        return super().run_model(model, inputs, device=device, **kwargs)

    @classmethod
    def run_node(cls, node, inputs, device=None, outputs_info=None, **kwargs):
        raise NotImplementedError("Unable to run the model node by node.")


if ort is not None:
    backend_test = onnx.backend.test.BackendTest(InferenceSessionBackend, __name__)

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
        "|test_batchnorm_epsilon_training_mode"  # Training mode does not support BN opset 14 (or higher) yet.
        "|test_batchnorm_example_training_mode"  # Training mode does not support BN opset 14 (or higher) yet.
        "|_to_FLOAT8E4M3FN"  # No corresponding Numpy type for Tensor Type.
        "|_to_FLOAT8E5M2"  # No corresponding Numpy type for Tensor Type.
        "|cast_FLOAT8E"  # No corresponding Numpy type for Tensor Type.
        "|castlike_FLOAT8E"  # No corresponding Numpy type for Tensor Type.
        "|test_dequantizelinear_axis"  # y_scale must be a scalar or 1D tensor of size 1.
        "|test_dequantizelinear"  # No corresponding Numpy type for Tensor Type.
        "|test_quantizelinear_axis"  # y_scale must be a scalar or 1D tensor of size 1.
        "|test_quantizelinear"  # No corresponding Numpy type for Tensor Type.
        "|test_affine_grid_"  # new IR version 9 and opset version 20 not supported yet.
        "|test_quantizelinear_uint4"  # No corresponding Numpy type for Tensor Type.
        "|test_quantizelinear_int4"  # No corresponding Numpy type for Tensor Type.
        "|test_dequantizelinear_uint4"  # No corresponding Numpy type for Tensor Type.
        "|test_dequantizelinear_int4"  # No corresponding Numpy type for Tensor Type.
        "|test_cast_UINT4_to_FLOAT"  # No corresponding Numpy type for Tensor Type.
        "|test_cast_INT4_to_FLOAT"  # No corresponding Numpy type for Tensor Type.
        "|test_cast_UINT4_to_FLOAT16"  # No corresponding Numpy type for Tensor Type.
        "|test_cast_INT4_to_FLOAT16"  # No corresponding Numpy type for Tensor Type.
        "|test_maxpool_2d_ceil_output_size_reduce_by_one"  # TODO: remove after https://github.com/microsoft/onnxruntime/pull/18377 in Ort release.
        ")"
    )

    # Exclude all tests that require IR10 until onnxruntime aligns
    # TODO: Unwaive tests once onnxruntime supports Opset21/IR10 https://github.com/onnx/onnx/issues/5840
    backend_test.exclude(
        "("
        "test_cast_"
        "|test_castlike_"
        "|test_constant"
        "|test_edge_pad_cpu"
        "|test_flatten_"
        "|test_identity"
        "|test_reflect_pad"
        "|test_reshape_"
        "|test_shape_"
        "|test_size_"
        "|test_squeeze_"
        "|test_transpose_"
        "|test_unsqueeze_"
        "|test_wrap_pad_"
        ")"
    )

    # The following tests fail due to small discrepancies.
    backend_test.exclude("(cast_FLOAT_to_STRING|castlike_FLOAT_to_STRING|stft)")

    # The following tests fail due to huge discrepancies.
    backend_test.exclude(
        "("
        "resize_downsample_scales_cubic_align_corners"
        "|resize_downsample_scales_linear_align_corners"
        "|training_dropout"
        ")"
    )

    # The followiing tests fail due to a bug in onnxruntime in handling reduction
    # ops that perform reduction over an empty set of values.
    backend_test.exclude(
        "("
        "test_reduce_sum_empty_set"
        "|test_reduce_prod_empty_set"
        "|test_reduce_min_empty_set"
        "|test_reduce_max_empty_set"
        "|test_reduce_sum_square_empty_set"
        "|test_reduce_log_sum_empty_set"
        "|test_reduce_log_sum_exp_empty_set"
        "|test_reduce_l1_empty_set"
        "|test_reduce_l2_empty_set"
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

    # The following tests are new with opset 19 and 20, or ai.onnx.ml 4
    if ort_version is not None and ort_version < Version("1.16"):
        backend_test.exclude(
            "("
            "averagepool"
            "|_pad_"
            "|_resize_"
            "|_size_"
            "|cast"
            "|castlike"
            "|equal_string_broadcast"
            "|equal_string"
            "|equal"
            "|half_pixel_symmetric"
            "|identity"
            "|reshape"
            ")"
        )
    if ort_version is not None and ort_version < Version("1.17"):
        backend_test.exclude(
            "("
            "deform_conv"
            "|dequantizelinear_uint16"
            "|dequantizelinear_int16"
            "|quantizelinear_uint16"
            "|quantizelinear_int16"
            "|dft"
            "|gelu"
            "|gridsample"
            "|group_normalization"
            "|identity_opt"
            "|image_decoder"
            "|isinf_float16"
            "|label_encoder"
            "|optional_get_element_optional_sequence"
            "|qlinearmatmul_2D_int8"
            "|qlinearmatmul_2D_uint8_float16"
            "|qlinearmatmul_3D_int8"
            "|qlinearmatmul_3D_uint8_float16"
            "|reduce_max_bool_inputs"
            "|reduce_min_bool_inputs"
            "|regex_full_match"
            "|string_concat"
            "|string_split"
            "|constantofshape_float_ones"
            "|constantofshape_int_shape_zero"
            "|constantofshape_int_zeros"
            "|isinf"
            "|isinf_negative"
            "|isinf_positive"
            "|isnan"
            "|isnan_float16"
            "|qlinearmatmul_2D_uint8_float32"
            "|qlinearmatmul_3D_uint8_float32"
            ")"
        )
    if ort_version is not None and ort_version < Version("1.18"):
        # when adding new tests to the list, please add a comment with the reason for exclusion
        # for tests that "not supported by onnxruntime 1.17", it will be solved in the next
        # onnxruntime release with ONNX 1.16.0 integrated. The work is covered in ONNX integration procedure.
        backend_test.exclude(
            "("
            "deform_conv"  # deform_conv is not supported in onnxruntime
            "|dft"  # Max absolute difference > atol=1e-07. shall be able to set atol (https://github.com/onnx/onnx/issues/5897)
            "|group_normalization"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|identity_opt"  # fixed in ort 1.18 (https://github.com/microsoft/onnxruntime/pull/19273)
            "|image_decoder"  # image_decoder is not supported in onnxruntime
            "|optional_get_element_optional_sequence"  # fixed in ort 1.18 (https://github.com/microsoft/onnxruntime/pull/19273)
            "|qlinearmatmul_2D_int8"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|qlinearmatmul_2D_uint8_float16"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|qlinearmatmul_3D_int8"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|qlinearmatmul_3D_uint8_float16"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|qlinearmatmul_2D_uint8_float32"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|qlinearmatmul_3D_uint8_float32"  # new/updated test cases with opset and/or IR version not supported by onnxruntime 1.17
            "|tree_ensemble"  # tree_ensemble not yet implemented in ort
            ")"
        )

    # Import all test cases at global scope to make them visible to python.unittest
    globals().update(backend_test.test_cases)


if __name__ == "__main__":
    unittest.main()
