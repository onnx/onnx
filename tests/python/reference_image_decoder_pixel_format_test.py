# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import io

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from onnx import TensorProto, checker, helper
from onnx.reference import ReferenceEvaluator

Image = pytest.importorskip("PIL.Image")
features = pytest.importorskip("PIL.features")


def _check_decode(data, expected_rgb, pixel_format):
    attributes = {} if pixel_format is None else {"pixel_format": pixel_format}
    expected = expected_rgb
    if pixel_format == "BGR":
        expected = expected[..., ::-1]
    elif pixel_format == "Grayscale":
        # These tests use only gray pixels when checking grayscale output.
        expected = expected[..., :1]
    encoded = np.frombuffer(data, dtype=np.uint8).copy()
    original = encoded.copy()
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("ImageDecoder", ["encoded"], ["image"], **attributes)],
            "image_decoder",
            [helper.make_tensor_value_info("encoded", TensorProto.UINT8, [None])],
            [helper.make_tensor_value_info("image", TensorProto.UINT8, expected.shape)],
        ),
        opset_imports=[helper.make_opsetid("", 20)],
    )
    checker.check_model(model, full_check=True)
    (actual,) = ReferenceEvaluator(model).run(None, {"encoded": encoded})
    assert actual.dtype == np.uint8
    assert actual.shape == expected.shape
    assert_array_equal(actual, expected)
    assert_array_equal(encoded, original)


@pytest.mark.parametrize("pixel_format", [None, "RGB", "BGR"])
@pytest.mark.parametrize("mode", ["1", "L", "LA", "P", "RGB", "RGBA"])
@pytest.mark.parametrize("container", ["PNG", "TIFF"])
def test_image_decoder_source_modes(container, mode, pixel_format):
    gray = np.array([[0, 51, 102], [153, 204, 255]], dtype=np.uint8)
    rgb = np.array(
        [
            [[11, 22, 33], [44, 55, 66], [77, 88, 99]],
            [[110, 121, 132], [143, 154, 165], [176, 187, 198]],
        ],
        dtype=np.uint8,
    )
    alpha = np.array([[0, 17, 128], [255, 0, 255]], dtype=np.uint8)
    expected = rgb
    if mode == "1":
        image = Image.fromarray(gray > 100)
        expected = np.repeat(((gray > 100) * 255)[..., None], 3, axis=2)
    elif mode in {"L", "LA"}:
        image = Image.fromarray(gray if mode == "L" else np.dstack([gray, alpha]))
        expected = np.repeat(gray[..., None], 3, axis=2)
    elif mode == "P":
        image = Image.new("P", (3, 2))
        image.putdata(range(6))
        image.putpalette(rgb.flatten().tolist() + [0] * (768 - rgb.size))
        if container == "PNG":
            image.info["transparency"] = 0
    else:
        image = Image.fromarray(rgb if mode == "RGB" else np.dstack([rgb, alpha]))
    output = io.BytesIO()
    image.save(output, format=container)
    _check_decode(output.getvalue(), expected, pixel_format)


@pytest.mark.parametrize("pixel_format", [None, "RGB", "BGR", "Grayscale"])
@pytest.mark.parametrize(
    "container", ["BMP", "JPEG", "JPEG2000", "PNG", "TIFF", "WEBP"]
)
def test_image_decoder_containers(container, pixel_format):
    codec = {"JPEG": "jpg", "JPEG2000": "jpg_2000", "WEBP": "webp"}.get(container)
    if codec is not None and not features.check(codec):
        pytest.skip(f"Pillow was built without {codec} support")
    gray = np.full((2, 3), 73, dtype=np.uint8)
    image = Image.fromarray(gray)
    output = io.BytesIO()
    image.save(output, format=container, lossless=True)
    _check_decode(
        output.getvalue(), np.repeat(gray[..., None], 3, axis=2), pixel_format
    )


@pytest.mark.parametrize("pixel_format", [None, "RGB", "BGR", "Grayscale"])
@pytest.mark.parametrize(
    "data,gray",
    [
        (b"P1\n3 2\n0 1 0\n1 0 1\n", [[255, 0, 255], [0, 255, 0]]),
        (b"P2\n3 2\n255\n0 51 102\n153 204 255\n", [[0, 51, 102], [153, 204, 255]]),
        (
            b"P3\n3 2\n255\n0 0 0 51 51 51 102 102 102\n153 153 153 204 204 204 255 255 255\n",
            [[0, 51, 102], [153, 204, 255]],
        ),
        (b"P4\n3 2\n\x40\xa0", [[255, 0, 255], [0, 255, 0]]),
        (b"P5\n3 2\n255\n\x00\x33\x66\x99\xcc\xff", [[0, 51, 102], [153, 204, 255]]),
        (
            b"P6\n3 2\n255\n"
            + bytes([0] * 3 + [51] * 3 + [102] * 3 + [153] * 3 + [204] * 3 + [255] * 3),
            [[0, 51, 102], [153, 204, 255]],
        ),
    ],
    ids=[
        "PBM-ascii",
        "PGM-ascii",
        "PPM-ascii",
        "PBM-binary",
        "PGM-binary",
        "PPM-binary",
    ],
)
def test_image_decoder_pnm(data, gray, pixel_format):
    expected = np.repeat(np.array(gray, dtype=np.uint8)[..., None], 3, axis=2)
    _check_decode(data, expected, pixel_format)


@pytest.mark.parametrize("pixel_format", [None, "RGB", "BGR"])
@pytest.mark.parametrize("container", ["BMP", "JPEG2000", "WEBP"])
def test_image_decoder_lossless_color(container, pixel_format):
    codec = {"JPEG2000": "jpg_2000", "WEBP": "webp"}.get(container)
    if codec is not None and not features.check(codec):
        pytest.skip(f"Pillow was built without {codec} support")
    rgb = np.array(
        [[[11, 22, 33], [44, 55, 66]], [[77, 88, 99], [110, 121, 132]]], dtype=np.uint8
    )
    image = Image.fromarray(rgb)
    output = io.BytesIO()
    image.save(output, format=container, lossless=True)
    _check_decode(output.getvalue(), rgb, pixel_format)
