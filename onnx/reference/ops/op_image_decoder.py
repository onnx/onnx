# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import io

import numpy as np

from onnx.reference.op_run import OpRun

# Image formats the ImageDecoder operator is specified to support
# (see onnx/defs/image/defs.cc). Restricting PIL.Image.open to this allow-list
# ensures `encoded` is only decoded as one of the documented formats instead of any
# plugin Pillow happens to have registered. Names are Pillow format identifiers.
_SUPPORTED_FORMATS = ("BMP", "JPEG", "JPEG2000", "TIFF", "PNG", "WEBP", "PPM")


class ImageDecoder(OpRun):
    def _run(self, encoded: np.ndarray, pixel_format="RGB") -> tuple[np.ndarray]:
        try:
            import PIL.Image  # noqa: PLC0415
        except ImportError as e:
            raise ImportError(
                "Pillow must be installed to use the reference implementation of the ImageDecoder operator"
            ) from e
        if pixel_format not in ("RGB", "BGR", "Grayscale"):
            raise ValueError(f"pixel_format={pixel_format!r} is not supported.")
        try:
            img = PIL.Image.open(
                io.BytesIO(encoded.tobytes()), formats=_SUPPORTED_FORMATS
            )
        except PIL.UnidentifiedImageError:
            # The operator returns an empty matrix for an invalid format
            # instead of failing.
            channels = 1 if pixel_format == "Grayscale" else 3
            return (np.empty((0, 0, channels), dtype=np.uint8),)
        if pixel_format == "BGR":
            decoded = np.array(img.convert("RGB"))[:, :, ::-1]
        elif pixel_format == "RGB":
            decoded = np.array(img.convert("RGB"))
        else:
            img = img.convert("L")
            decoded = np.array(img)
            decoded = np.expand_dims(decoded, axis=2)  # (H, W) to (H, W, 1)
        return (decoded,)
