from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class FFT(Base):

    @staticmethod
    def export_dim_one():  # type: () -> None
        node = onnx.helper.make_node(
            'FFT',
            inputs=['X'],
            outputs=['Y'],
        )

        input_data = np.array([[[1., 0], [0, 0], [0, 0]]], dtype=np.float64)

        # Convert to complex
        input_data_complex = input_data.view(dtype=np.complex128)
        fft_result = np.fft.fft(input_data_complex)
        expected_output = np.stack([fft_result.real, fft_result.imag], axis=2)

        expect(node, inputs=[input_data], outputs=[expected_output],
               name='test_celu')
