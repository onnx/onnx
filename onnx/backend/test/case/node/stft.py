# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect


class STFT(Base):
    @staticmethod
    def export() -> None:
        signal = np.arange(0, 128, dtype=np.float32).reshape(1, 128, 1)
        length = np.array(16).astype(np.int64)
        onesided_length = (length >> 1) + 1
        step = np.array(8).astype(np.int64)

        no_window = ""  # optional input, not supplied
        node = onnx.helper.make_node(
            "STFT",
            inputs=["signal", "frame_step", no_window, "frame_length"],
            outputs=["output"],
        )

        nstfts = ((signal.shape[1] - length) // step) + 1
        # [batch_size][frames][frame_length][2]
        output = np.empty([1, nstfts, onesided_length, 2], dtype=np.float32)
        for i in range(nstfts):
            start = i * step
            stop = i * step + length
            complex_out = np.fft.fft(signal[0, start:stop, 0])[0:onesided_length]
            output[0, i] = np.stack((complex_out.real, complex_out.imag), axis=1)

        output = output.astype(signal.dtype)
        expect(
            node,
            inputs=[signal, step, length],
            outputs=[output],
            name="test_stft",
            opset_imports=[onnx.helper.make_opsetid("", 17)],
        )

        node = onnx.helper.make_node(
            "STFT",
            inputs=["signal", "frame_step", "window"],
            outputs=["output"],
        )

        # Test with window
        a0 = 0.5
        a1 = 0.5
        window = a0 + a1 * np.cos(
            2 * np.pi * np.arange(0, length, 1, dtype=np.float32) / length
        )
        nstfts = 1 + (signal.shape[1] - window.shape[0]) // step

        # [batch_size][frames][frame_length][2]
        output = np.empty([1, nstfts, onesided_length, 2], dtype=np.float32)
        for i in range(nstfts):
            start = i * step
            stop = i * step + length
            complex_out = np.fft.fft(signal[0, start:stop, 0] * window)[
                0:onesided_length
            ]
            output[0, i] = np.stack((complex_out.real, complex_out.imag), axis=1)
        window = window.astype(signal.dtype)
        output = output.astype(signal.dtype)
        expect(
            node,
            inputs=[signal, step, window],
            outputs=[output],
            name="test_stft_with_window",
            opset_imports=[onnx.helper.make_opsetid("", 17)],
        )

    @staticmethod
    def export_rank2() -> None:
        signal = np.arange(32, dtype=np.float32).reshape(2, 16)
        length = np.array(4, dtype=np.int64)
        step = np.array(2, dtype=np.int64)
        for onesided in (0, 1):
            for with_window in (False, True):
                window = np.hanning(4).astype(np.float32)
                node = onnx.helper.make_node(
                    "STFT",
                    inputs=[
                        "signal",
                        "frame_step",
                        "window" if with_window else "",
                        "frame_length",
                    ],
                    outputs=["output"],
                    onesided=onesided,
                )
                inputs = [signal, step]
                if with_window:
                    inputs.append(window)
                inputs.append(length)
                frames = np.stack(
                    [signal[:, start : start + 4] for start in range(0, 13, 2)],
                    axis=1,
                )
                if with_window:
                    frames = frames * window
                transformed = np.fft.fft(frames, axis=2)
                if onesided:
                    transformed = transformed[:, :, :3]
                output = np.stack((transformed.real, transformed.imag), axis=-1).astype(
                    np.float32
                )
                expect(
                    node,
                    inputs=inputs,
                    outputs=[output],
                    name=f"test_stft_rank2_onesided_{onesided}_window_{int(with_window)}",
                    opset_imports=[onnx.helper.make_opsetid("", 29)],
                )
