# Copyright (c) ONNX Project Contributors
# SPDX-License-Identifier: Apache-2.0
# mypy: ignore-errors

from __future__ import annotations

import importlib.util
import io
import unittest

import numpy as np
import pytest

from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.reference import ReferenceEvaluator

skip_if_no_pillow = pytest.mark.skipif(
    importlib.util.find_spec("PIL") is None, reason="Pillow not installed"
)


def _image_decoder_model(pixel_format: str = "RGB"):
    node = make_node(
        "ImageDecoder", ["encoded"], ["decoded"], pixel_format=pixel_format
    )
    graph = make_graph(
        [node],
        "image_decoder",
        [make_tensor_value_info("encoded", TensorProto.UINT8, [None])],
        [make_tensor_value_info("decoded", TensorProto.UINT8, [None, None, None])],
    )
    return make_model(graph, opset_imports=[make_opsetid("", 20)])


def _run(encoded: bytes, pixel_format: str = "RGB"):
    evaluator = ReferenceEvaluator(_image_decoder_model(pixel_format))
    feeds = {"encoded": np.frombuffer(encoded, dtype=np.uint8)}
    return evaluator.run(None, feeds)[0]


@skip_if_no_pillow
class TestImageDecoder(unittest.TestCase):
    def test_decodes_png(self) -> None:
        import PIL.Image

        buf = io.BytesIO()
        PIL.Image.new("RGB", (3, 2), (10, 20, 30)).save(buf, format="PNG")
        decoded = _run(buf.getvalue())
        self.assertEqual(decoded.shape, (2, 3, 3))
        np.testing.assert_array_equal(decoded[0, 0], np.array([10, 20, 30], np.uint8))

    def test_rejects_undocumented_format(self) -> None:
        # GIF is a valid image to Pillow but is not among the formats the
        # ImageDecoder operator is specified to support. With the formats=
        # allow-list the decoder refuses it instead of decoding it.
        import PIL.Image

        buf = io.BytesIO()
        PIL.Image.new("RGB", (4, 4), (0, 0, 0)).save(buf, format="GIF")
        gif = buf.getvalue()
        # Sanity: unrestricted Pillow does decode these bytes as GIF.
        self.assertEqual(PIL.Image.open(io.BytesIO(gif)).format, "GIF")
        with self.assertRaises(PIL.UnidentifiedImageError):
            _run(gif)


if __name__ == "__main__":
    unittest.main(verbosity=2)
