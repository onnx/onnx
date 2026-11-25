# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io

import numpy as np
import pytest

from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator

Image = pytest.importorskip("PIL.Image")


def _decode(data: bytes, pixel_format: str) -> np.ndarray:
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "ImageDecoder", ["encoded"], ["image"], pixel_format=pixel_format
                )
            ],
            "image_decoder",
            [helper.make_tensor_value_info("encoded", TensorProto.UINT8, [None])],
            [helper.make_tensor_value_info("image", TensorProto.UINT8, None)],
        ),
        opset_imports=[helper.make_opsetid("", 20)],
    )
    encoded = np.frombuffer(data, dtype=np.uint8)
    return ReferenceEvaluator(model).run(None, {"encoded": encoded})[0]


def _gif() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (4, 4), (0, 0, 0)).save(output, format="GIF")
    return output.getvalue()


@pytest.mark.parametrize(
    "pixel_format,channels", [("RGB", 3), ("BGR", 3), ("Grayscale", 1)]
)
@pytest.mark.parametrize(
    "data", [_gif(), b"not an image"], ids=["undocumented-format", "garbage"]
)
def test_image_decoder_undecodable_returns_empty(data, pixel_format, channels):
    # GIF is a valid image to Pillow but not a format ImageDecoder supports.
    # Like any input it cannot decode, the operator returns an empty matrix.
    decoded = _decode(data, pixel_format)
    assert decoded.dtype == np.uint8
    assert decoded.shape == (0, 0, channels)


def test_gif_is_decodable_by_unrestricted_pillow():
    # Guards the test above: the GIF is rejected by the allow-list, not
    # because Pillow cannot read it.
    assert Image.open(io.BytesIO(_gif())).format == "GIF"
