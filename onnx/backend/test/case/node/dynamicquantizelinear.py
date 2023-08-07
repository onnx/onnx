# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def estimation_quantization_scale(
    coef: np.array,
    to: int = onnx.TensorProto.FLOAT8E4M3FN,
    method: str = "naive",
    threshold: float = 0.99999,
) -> tuple[float, float]:
    """
    Estimates the scale parameter for the quantization to float 8 assuming
    the distribution of the coefficients is gaussian.
    """

    if to in (
        onnx.TensorProto.FLOAT8E4M3FN,
        onnx.TensorProto.FLOAT8E4M3FNUZ,
        onnx.TensorProto.FLOAT8E5M2,
        onnx.TensorProto.FLOAT8E5M2FNUZ,
    ):
        if to == onnx.TensorProto.FLOAT8E4M3FN:
            fct = onnx.numpy_helper.float8e4m3_to_float32
        elif to == onnx.TensorProto.FLOAT8E4M3FNUZ:

            def fct(x):
                return onnx.numpy_helper.float8e4m3_to_float32(x, uz=True)

        elif to == onnx.TensorProto.FLOAT8E5M2:
            fct = onnx.numpy_helper.float8e5m2_to_float32
        elif to == onnx.TensorProto.FLOAT8E5M2FNUZ:

            def fct(x):
                return onnx.numpy_helper.float8e5m2_to_float32(x, uz=True, fn=True)

        else:
            raise ValueError(f"Unexpected to={to!r}.")

        float8 = [fct(i) for i in range(0, 256)]
        quant_float = [f for f in float8 if not np.isnan(f)]
        cr = coef.ravel()
        ca = np.abs(cr)
        ca_den = ca.copy()
        ca_den[ca == 0] = 1
        std_coef = np.std(ca ** (1.0 / 3.0) * cr / ca_den)
        std_quant = np.std(np.array(quant_float, dtype=np.float32))
        zero = 0.0
        scale = std_quant / std_coef
    elif to == onnx.TensorProto.UINT8:
        qu = np.quantile(coef.ravel(), [1 - threshold, threshold])
        scale = 255 / (qu[1] - qu[0])
        zero = qu[0] * scale
    elif to == onnx.TensorProto.INT8:
        qu = np.quantile(coef.ravel(), [1 - threshold, threshold])
        scale = 254 / (qu[1] - qu[0])
        zero = (qu[0] + qu[1]) / 2 * scale
    else:
        raise ValueError(f"Unexpected quantization type for to={to}.")

    return 1.0 / scale, -zero


class DynamicQuantizeLinear(Base):
    @staticmethod
    def export_uint8() -> None:
        node = onnx.helper.make_node(
            "DynamicQuantizeLinear",
            inputs=["x"],
            outputs=["y", "y_scale", "y_zero_point"],
        )

        # expected scale 0.0196078438 and zero point 153
        X = np.array([0, 2, -3, -2.5, 1.34, 0.5]).astype(np.float32)
        x_min = np.minimum(0, np.min(X))
        x_max = np.maximum(0, np.max(X))
        Y_Scale = np.float32((x_max - x_min) / (255 - 0))  # uint8 -> [0, 255]
        Y_ZeroPoint = np.clip(round(0 - x_min / Y_Scale), 0, 255).astype(np.uint8)
        Y = np.clip(np.rint(X / Y_Scale) + Y_ZeroPoint, 0, 255).astype(np.uint8)

        expect(
            node,
            inputs=[X],
            outputs=[Y, Y_Scale, Y_ZeroPoint],
            name="test_dynamicquantizelinear",
        )

        # expected scale 0.0156862754 and zero point 255
        X = np.array([-1.0, -2.1, -1.3, -2.5, -3.34, -4.0]).astype(np.float32)
        x_min = np.minimum(0, np.min(X))
        x_max = np.maximum(0, np.max(X))
        Y_Scale = np.float32((x_max - x_min) / (255 - 0))  # uint8 -> [0, 255]
        Y_ZeroPoint = np.clip(round(0 - x_min / Y_Scale), 0, 255).astype(np.uint8)
        Y = np.clip(np.rint(X / Y_Scale) + Y_ZeroPoint, 0, 255).astype(np.uint8)

        expect(
            node,
            inputs=[X],
            outputs=[Y, Y_Scale, Y_ZeroPoint],
            name="test_dynamicquantizelinear_max_adjusted",
        )

        X = (
            np.array([1, 2.1, 1.3, 2.5, 3.34, 4.0, 1.5, 2.6, 3.9, 4.0, 3.0, 2.345])
            .astype(np.float32)
            .reshape((3, 4))
        )

        # expected scale 0.0156862754 and zero point 0
        x_min = np.minimum(0, np.min(X))
        x_max = np.maximum(0, np.max(X))
        Y_Scale = np.float32((x_max - x_min) / (255 - 0))  # uint8 -> [0, 255]
        Y_ZeroPoint = np.clip(round(0 - x_min / Y_Scale), 0, 255).astype(np.uint8)
        Y = np.clip(np.rint(X / Y_Scale) + Y_ZeroPoint, 0, 255).astype(np.uint8)

        expect(
            node,
            inputs=[X],
            outputs=[Y, Y_Scale, Y_ZeroPoint],
            name="test_dynamicquantizelinear_min_adjusted",
        )

    @staticmethod
    def export_int8() -> None:
        node = onnx.helper.make_node(
            "DynamicQuantizeLinear",
            inputs=["x"],
            outputs=["y", "y_scale", "y_zero_point"],
            to=3,
        )

        # expected scale 0.0196078438 and zero point 153
        X = np.array([0, 2, -3, -2.5, 1.34, 0.5]).astype(np.float32)
        x_min = np.minimum(0, np.min(X))
        x_max = np.maximum(0, np.max(X))
        Y_Scale = np.float32((x_max - x_min) / (127 + 127))  # int8 -> [-127, 127]
        Y_ZeroPoint = np.clip(round(-127 - x_min / Y_Scale), -127, 127).astype(np.int8)
        Y = np.clip(np.rint(X / Y_Scale) + Y_ZeroPoint, -127, 127).astype(np.int8)

        expect(
            node,
            inputs=[X],
            outputs=[Y, Y_Scale, Y_ZeroPoint],
            name="test_dynamicquantizelinear_int8",
        )

        # expected scale 0.0156862754 and zero point 255
        X = np.array([-1.0, -2.1, -1.3, -2.5, -3.34, -4.0]).astype(np.float32)
        x_min = np.minimum(0, np.min(X))
        x_max = np.maximum(0, np.max(X))
        Y_Scale = np.float32((x_max - x_min) / (127 + 127))  # int8 -> [-127, 127]
        Y_ZeroPoint = np.clip(round(-127 - x_min / Y_Scale), -127, 127).astype(np.int8)
        Y = np.clip(np.rint(X / Y_Scale) + Y_ZeroPoint, -127, 127).astype(np.int8)

        expect(
            node,
            inputs=[X],
            outputs=[Y, Y_Scale, Y_ZeroPoint],
            name="test_dynamicquantizelinear_max_adjusted_int8",
        )

        X = (
            np.array([1, 2.1, 1.3, 2.5, 3.34, 4.0, 1.5, 2.6, 3.9, 4.0, 3.0, 2.345])
            .astype(np.float32)
            .reshape((3, 4))
        )

        # expected scale 0.0156862754 and zero point 0
        x_min = np.minimum(0, np.min(X))
        x_max = np.maximum(0, np.max(X))
        Y_Scale = np.float32((x_max - x_min) / (127 + 127))  # uint8 -> [-127, 127]
        Y_ZeroPoint = np.clip(round(-127 - x_min / Y_Scale), -127, 127).astype(np.int8)
        Y = np.clip(np.rint(X / Y_Scale) + Y_ZeroPoint, -127, 127).astype(np.int8)

        expect(
            node,
            inputs=[X],
            outputs=[Y, Y_Scale, Y_ZeroPoint],
            name="test_dynamicquantizelinear_min_adjusted_int8",
        )

    @staticmethod
    def export_float16() -> None:
        node = onnx.helper.make_node(
            "DynamicQuantizeLinear",
            inputs=["x"],
            outputs=["y", "y_scale", "y_zero_point"],
        )

        X = np.array([0, 2, -3, -2.5, 1.34, 0.5]).astype(np.float16)
        x_min = np.minimum(0, np.min(X))
        x_max = np.maximum(0, np.max(X))
        Y_Scale = np.float16((x_max - x_min) / (255 - 0))  # uint8 -> [0, 255]
        Y_ZeroPoint = np.clip(round(0 - x_min / Y_Scale), 0, 255).astype(np.uint8)
        Y = np.clip(np.rint(X / Y_Scale) + Y_ZeroPoint, 0, 255).astype(np.uint8)

        expect(
            node,
            inputs=[X],
            outputs=[Y, Y_Scale, Y_ZeroPoint],
            name="test_dynamicquantizelinear_float16",
        )

        node = onnx.helper.make_node(
            "DynamicQuantizeLinear",
            inputs=["x"],
            outputs=["y", "y_scale", "y_zero_point"],
            to=onnx.TensorProto.INT8,
        )

        X = np.array([0, 2, -3, -2.5, 1.34, 0.5]).astype(np.float16)
        x_min = np.minimum(0, np.min(X))
        x_max = np.maximum(0, np.max(X))
        Y_Scale = np.float16((x_max - x_min) / (127 + 127))  # int8 -> [-127, 127]
        Y_ZeroPoint = np.clip(round(-127 - x_min / Y_Scale), -127, 127).astype(np.int8)
        Y = np.clip(np.rint(X / Y_Scale) + Y_ZeroPoint, -127, 127).astype(np.int8)

        expect(
            node,
            inputs=[X],
            outputs=[Y, Y_Scale, Y_ZeroPoint],
            name="test_dynamicquantizelinear_int8_float16",
        )

    @staticmethod
    def export_float8() -> None:
        for to in ["FLOAT8E4M3FN", "FLOAT8E4M3FNUZ", "FLOAT8E5M2", "FLOAT8E5M2FNUZ"]:
            node = onnx.helper.make_node(
                "DynamicQuantizeLinear",
                inputs=["x"],
                outputs=["y", "y_scale", "y_zero_point"],
                to=getattr(onnx.TensorProto, to),
            )

            # expected scale 0.0196078438 and zero point 153
            X = np.array([0, 2, -3, -2.5, 1.34, 0.5]).astype(np.float32)
            scale, zero = estimation_quantization_scale(X)
            Y_scaled = (X / scale).astype(X.dtype)
            Y8 = onnx.helper.make_tensor(
                "Y", getattr(onnx.TensorProto, to), [X.size], Y_scaled.tolist()
            )
            y_zero_point = onnx.helper.make_tensor(
                "y_zero_point", getattr(onnx.TensorProto, to), [], [0.0]
            )

            expect(
                node,
                inputs=[X],
                outputs=[Y8, np.array(scale, dtype=X.dtype), y_zero_point],
                name=f"test_dynamicquantizelinear_{to.lower()}",
            )
