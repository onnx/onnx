# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0

import cv2
import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


def generate_checkerboard(width, height, square_size):
    # Create an empty RGB image
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the number of squares in each dimension
    num_squares_x = width // square_size
    num_squares_y = height // square_size

    # Generate a random color for each square
    colors = np.random.randint(
        0, 256, size=(num_squares_y, num_squares_x, 3), dtype=np.uint8
    )

    # Iterate over each square
    for i in range(num_squares_y):
        for j in range(num_squares_x):
            # Calculate the position of the current square
            x = j * square_size
            y = i * square_size

            # Get the color for the current square
            color = colors[i, j]

            # Fill the square with the corresponding color
            image[y : y + square_size, x : x + square_size, :] = color

    return image


def generate_test_data(extension, pixel_format="RGB", h=40, w=40, tile_sz=5):
    data, output = None, None
    np.random.seed(12345)
    image = generate_checkerboard(h, w, tile_sz)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, encoded_image = cv2.imencode(extension, image_bgr)
    data = np.frombuffer(encoded_image, dtype=np.uint8)
    if pixel_format == "BGR":
        output = cv2.imdecode(data, cv2.IMREAD_COLOR)
    elif pixel_format == "RGB":
        output_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        output = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
    elif pixel_format == "Grayscale":
        output = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        output = np.expand_dims(output, axis=2)  # (H, W) to (H, W, 1)
    return data, output


class ImageDecoder(Base):
    @staticmethod
    def export_image_decoder_decode_jpeg_rgb() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="RGB",
        )

        data, output = generate_test_data(".jpg", "RGB")
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_jpeg_rgb",
        )

    @staticmethod
    def export_image_decoder_decode_jpeg_grayscale() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="Grayscale",
        )

        data, output = generate_test_data(".jpg", "Grayscale")
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_jpeg_grayscale",
        )

    @staticmethod
    def export_image_decoder_decode_jpeg_bgr() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="BGR",
        )

        data, output = generate_test_data(".jpg", "BGR")
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_jpeg_bgr",
        )

    @staticmethod
    def export_image_decoder_decode_jpeg2k_rgb() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="RGB",
        )

        data, output = generate_test_data(".jp2", "RGB")
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_jpeg2k_rgb",
        )

    @staticmethod
    def export_image_decoder_decode_bmp_rgb() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="RGB",
        )

        data, output = generate_test_data(".bmp", "RGB")
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_bmp_rgb",
        )

    @staticmethod
    def export_image_decoder_decode_png_rgb() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="RGB",
        )

        data, output = generate_test_data(".png", "RGB")
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_png_rgb",
        )

    @staticmethod
    def export_image_decoder_decode_tiff_rgb() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="RGB",
        )

        data, output = generate_test_data(".tiff", "RGB")
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_tiff_rgb",
        )

    @staticmethod
    def export_image_decoder_decode_webp_rgb() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="RGB",
        )

        data, output = generate_test_data(".webp", "RGB")
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_webp_rgb",
        )

    @staticmethod
    def export_image_decoder_decode_pnm_rgb() -> None:
        node = onnx.helper.make_node(
            "ImageDecoder",
            inputs=["data"],
            outputs=["output"],
            pixel_format="RGB",
        )

        data, output = generate_test_data(".pnm", "RGB")
        expect(
            node,
            inputs=[data],
            outputs=[output],
            name="test_image_decoder_decode_pnm_rgb",
        )
