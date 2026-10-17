# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.model import expect


class Mod(Base):
    @staticmethod
    def export_fmod_0_signed_zero() -> None:
        for dtype in (np.float16, np.float32, np.float64):
            tensor_type = onnx.helper.np_dtype_to_tensor_dtype(np.dtype(dtype))
            mod = onnx.helper.make_node("Mod", ["x", "y"], ["remainder"], fmod=0)
            # Sign maps both signed zeros to zero; Reciprocal exposes them as signed infinities.
            reciprocal = onnx.helper.make_node("Reciprocal", ["remainder"], ["z"])
            graph = onnx.helper.make_graph(
                [mod, reciprocal],
                f"ModFmod0SignedZero{np.dtype(dtype).name}",
                [
                    onnx.helper.make_tensor_value_info("x", tensor_type, [4]),
                    onnx.helper.make_tensor_value_info("y", tensor_type, [4]),
                ],
                [
                    onnx.helper.make_tensor_value_info("z", tensor_type, [4]),
                ],
            )
            model = onnx.helper.make_model_gen_version(
                graph,
                producer_name="backend-test",
                opset_imports=[onnx.helper.make_opsetid("", 28)],
            )
            x = np.array([0.0, -0.0, 0.0, -0.0], dtype=dtype)
            y = np.array([-2.0, 2.0, 2.0, -2.0], dtype=dtype)
            z = np.array([-np.inf, np.inf, np.inf, -np.inf], dtype=dtype)
            expect(
                model,
                inputs=[x, y],
                outputs=[z],
                name=f"test_mod_fmod_0_signed_zero_{np.dtype(dtype).name}",
            )
