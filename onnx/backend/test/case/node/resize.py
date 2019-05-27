from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class Resize(Base):

    @staticmethod
    def export_resize_upsample_nearest():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='nearest',
        )

        data = np.array([[[
            [1, 2],
            [3, 4],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)

        output = np.array([[[
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [3, 3, 3, 4, 4, 4],
            [3, 3, 3, 4, 4, 4],
        ]]], dtype=np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_upsample_nearest')

    @staticmethod
    def export_resize_downsample_nearest():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='nearest',
        )

        data = np.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

        output = np.array([[[
            [1, 3]
        ]]], dtype=np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_downsample_nearest')

    @staticmethod
    def export_resize_upsample_linear():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='linear',
            align_corners=False
        )

        data = np.array([[[
            [1, 2],
            [3, 4],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        output = np.array([[[
            [1.00, 1.25, 1.75, 2.00],
            [1.50, 1.75, 2.25, 2.50],
            [2.50, 2.75, 3.25, 3.50],
            [3.00, 3.25, 3.75, 4.00]
        ]]], dtype=np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_upsample_linear')

    @staticmethod
    def export_resize_upsample_linear_align_corners():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='linear',
            align_corners=1
        )

        data = np.array([[[
            [1, 2],
            [3, 4],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        output = np.array([[[
            [1.00000000, 1.33333333, 1.66666667, 2.00000000],
            [1.66666667, 2.00000000, 2.33333333, 2.66666667],
            [2.33333333, 2.66666667, 3.00000000, 3.33333333],
            [3.00000000, 3.33333333, 3.66666667, 4.00000000]
        ]]], dtype=np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_upsample_linear_align_corners')

    @staticmethod
    def export_resize_downsample_linear():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='linear',
        )

        data = np.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

        output = np.array([[[
            [1.5, 3.5]
        ]]], dtype=np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_downsample_linear')

    @staticmethod
    def export_resize_downsample_linear_align_corners():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='linear',
            align_corners=1
        )

        data = np.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

        output = np.array([[[
            [1, 4]
        ]]], dtype=np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_downsample_linear_align_corners')

    @staticmethod
    def export_resize_upsample_cubic():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='cubic',
            align_corners=False
        )

        data = np.array([[[
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        output = np.array([[[
            [ 0.47265625,  0.76953125,  1.24609375,  1.87500000,  2.28125000,
                2.91015625,  3.38671875,  3.68359375],
            [ 1.66015625,  1.95703125,  2.43359375,  3.06250000,  3.46875000,
                4.09765625,  4.57421875,  4.87109375],
            [ 3.56640625,  3.86328125,  4.33984375,  4.96875000,  5.37500000,
                6.00390625,  6.48046875,  6.77734375],
            [ 6.08203125,  6.37890625,  6.85546875,  7.48437500,  7.89062500,
                8.51953125,  8.99609375,  9.29296875],
            [ 7.70703125,  8.00390625,  8.48046875,  9.10937500,  9.51562500,
                10.14453125, 10.62109375, 10.91796875],
            [10.22265625, 10.51953125, 10.99609375, 11.62500000, 12.03125000,
                12.66015625, 13.13671875, 13.43359375],
            [12.12890625, 12.42578125, 12.90234375, 13.53125000, 13.93750000,
                14.56640625, 15.04296875, 15.33984375],
            [13.31640625, 13.61328125, 14.08984375, 14.71875000, 15.12500000,
                15.75390625, 16.23046875, 16.52734375]
        ]]], dtype=np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_upsample_cubic')

    @staticmethod
    def export_resize_upsample_cubic_align_corners():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='cubic',
            align_corners=True
        )

        data = np.array([[[
            [ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12],
            [13, 14, 15, 16],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        output = np.array([[[
            [ 1.00000000,  1.34110796,  1.80029082,  2.32944798,  2.67055368,
                3.19970822,  3.65889263,  4.00000000],
            [ 2.36443138,  2.70553946,  3.16472220,  3.69388032,  4.03498554,
                4.56413984,  5.02332497,  5.36443186],
            [ 4.20116329,  4.54227161,  5.00145435,  5.53061295,  5.87171793,
                6.40087080,  6.86005688,  7.20116329],
            [ 6.31779146,  6.65890026,  7.11808157,  7.64724159,  7.98834610,
                8.51749992,  8.97668552,  9.31779194],
            [ 7.68221235,  8.02332115,  8.48250198,  9.01166344,  9.35276890,
                9.88192177, 10.34110737, 10.68221474],
            [ 9.79883289, 10.13994312, 10.59912300, 11.12828350, 11.46938896,
                11.99853992, 12.45772648, 12.79883289],
            [11.63557053, 11.97667980, 12.43585873, 12.96502209, 13.30612659,
                13.83527756, 14.29446411, 14.63557053],
            [13.00000000, 13.34110928, 13.80028820, 14.32945251, 14.67055702,
                15.19970608, 15.65889359, 16.00000000]
        ]]], dtype=np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_upsample_cubic_align_corners')

    @staticmethod
    def export_resize_downsample_cubic():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='cubic',
            align_corners=False
        )

        data = np.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

        output = np.array([[[
            [ 1.63078713,  3.00462985,  4.37847328],
            [ 7.12615871,  8.50000000,  9.87384510],
            [12.62153149, 13.99537277, 15.36922169]
        ]]], dtype=np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_downsample_cubic')

    @staticmethod
    def export_resize_downsample_cubic_align_corners():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='cubic',
            align_corners=True
        )

        data = np.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        output = np.array([[[
            [ 1.00000000,  2.50000000,  4.00000000],
            [ 7.00000000,  8.50000000, 10.00000000],
            [13.00000000, 14.50000000, 16.00000000]
        ]]], dtype=np.float32)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_downsample_cubic_align_corners')
