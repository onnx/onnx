from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class ConvTranspose(Base):

    @staticmethod
    def export():  # type: () -> None
        x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                        [3., 4., 5.],
                        [6., 7., 8.]]]]).astype(np.float32)

        W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                        [1., 1., 1.],
                        [1., 1., 1.]],
                       [[1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(np.float32)

        node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"])

        y = np.array([[[[0., 1., 3., 3., 2.],  # (1, 2, 5, 5)
                        [3., 8., 15., 12., 7.],
                        [9., 21., 36., 27., 15.],
                        [9., 20., 33., 24., 13.],
                        [6., 13., 21., 15., 8.]],

                       [[0., 1., 3., 3., 2.],
                        [3., 8., 15., 12., 7.],
                        [9., 21., 36., 27., 15.],
                        [9., 20., 33., 24., 13.],
                        [6., 13., 21., 15., 8.]]]]).astype(np.float32)

        expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose')

    @staticmethod
    def export_convtranspose_1d():  # type: () -> None
        x = np.array([[[0., 1., 2.]]]).astype(np.float32)  # (1, 1, 3)

        W = np.array([[[1., 1., 1.],  # (1, 2, 3)
                       [1., 1., 1.]]]).astype(np.float32)

        node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"])

        y = np.array([[[0., 1., 3., 3., 2.],  # (1, 2, 5)
                       [0., 1., 3., 3., 2.]]]).astype(np.float32)

        expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_1d')

    @staticmethod
    def export_convtranspose_3d():  # type: () -> None
        x = np.array([[[[[0., 1., 2., 3., 4.],  # (1, 1, 3, 4, 5)
                         [5., 6., 7., 8., 9.],
                         [10., 11., 12., 13., 14.],
                         [15., 16., 17., 18., 19.]],
                        [[20., 21., 22., 23., 24.],
                         [25., 26., 27., 28., 29.],
                         [30., 31., 32., 33., 34.],
                         [35., 36., 37., 38., 39.]],
                        [[40., 41., 42., 43., 44.],
                         [45., 46., 47., 48., 49.],
                         [50., 51., 52., 53., 54.],
                         [55., 56., 57., 58., 59.]]]]]).astype(np.float32)

        W = np.array([[[[[1., 1., 1.],  # (1, 2, 3, 3, 3)
                         [1., 1., 1.],
                         [1., 1., 1.]],
                        [[1., 1., 1.],
                         [1., 1., 1.],
                         [1., 1., 1.]],
                        [[1., 1., 1.],
                         [1., 1., 1.],
                         [1., 1., 1.]]],
                       [[[1., 1., 1.],
                         [1., 1., 1.],
                         [1., 1., 1.]],
                        [[1., 1., 1.],
                         [1., 1., 1.],
                         [1., 1., 1.]],
                        [[1., 1., 1.],
                         [1., 1., 1.],
                         [1., 1., 1.]]]]]).astype(np.float32)

        node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"])

        y = np.array([[[[[0., 1., 3., 6., 9., 7., 4.],  # (1, 2, 5, 6, 7)
                         [5., 12., 21., 27., 33., 24., 13.],
                         [15., 33., 54., 63., 72., 51., 27.],
                         [30., 63., 99., 108., 117., 81., 42.],
                         [25., 52., 81., 87., 93., 64., 33.],
                         [15., 31., 48., 51., 54., 37., 19.]],

                        [[20., 42., 66., 72., 78., 54., 28.],
                         [50., 104., 162., 174., 186., 128., 66.],
                         [90., 186., 288., 306., 324., 222., 114.],
                         [120., 246., 378., 396., 414., 282., 144.],
                         [90., 184., 282., 294., 306., 208., 106.],
                         [50., 102., 156., 162., 168., 114., 58.]],

                        [[60., 123., 189., 198., 207., 141., 72.],
                         [135., 276., 423., 441., 459., 312., 159.],
                         [225., 459., 702., 729., 756., 513., 261.],
                         [270., 549., 837., 864., 891., 603., 306.],
                         [195., 396., 603., 621., 639., 432., 219.],
                         [105., 213., 324., 333., 342., 231., 117.]],

                        [[60., 122., 186., 192., 198., 134., 68.],
                         [130., 264., 402., 414., 426., 288., 146.],
                         [210., 426., 648., 666., 684., 462., 234.],
                         [240., 486., 738., 756., 774., 522., 264.],
                         [170., 344., 522., 534., 546., 368., 186.],
                         [90., 182., 276., 282., 288., 194., 98.]],

                        [[40., 81., 123., 126., 129., 87., 44.],
                         [85., 172., 261., 267., 273., 184., 93.],
                         [135., 273., 414., 423., 432., 291., 147.],
                         [150., 303., 459., 468., 477., 321., 162.],
                         [105., 212., 321., 327., 333., 224., 113.],
                         [55., 111., 168., 171., 174., 117., 59.]]],

                       [[[0., 1., 3., 6., 9., 7., 4.],
                         [5., 12., 21., 27., 33., 24., 13.],
                         [15., 33., 54., 63., 72., 51., 27.],
                         [30., 63., 99., 108., 117., 81., 42.],
                         [25., 52., 81., 87., 93., 64., 33.],
                         [15., 31., 48., 51., 54., 37., 19.]],

                        [[20., 42., 66., 72., 78., 54., 28.],
                         [50., 104., 162., 174., 186., 128., 66.],
                         [90., 186., 288., 306., 324., 222., 114.],
                         [120., 246., 378., 396., 414., 282., 144.],
                         [90., 184., 282., 294., 306., 208., 106.],
                         [50., 102., 156., 162., 168., 114., 58.]],

                        [[60., 123., 189., 198., 207., 141., 72.],
                         [135., 276., 423., 441., 459., 312., 159.],
                         [225., 459., 702., 729., 756., 513., 261.],
                         [270., 549., 837., 864., 891., 603., 306.],
                         [195., 396., 603., 621., 639., 432., 219.],
                         [105., 213., 324., 333., 342., 231., 117.]],

                        [[60., 122., 186., 192., 198., 134., 68.],
                         [130., 264., 402., 414., 426., 288., 146.],
                         [210., 426., 648., 666., 684., 462., 234.],
                         [240., 486., 738., 756., 774., 522., 264.],
                         [170., 344., 522., 534., 546., 368., 186.],
                         [90., 182., 276., 282., 288., 194., 98.]],

                        [[40., 81., 123., 126., 129., 87., 44.],
                         [85., 172., 261., 267., 273., 184., 93.],
                         [135., 273., 414., 423., 432., 291., 147.],
                         [150., 303., 459., 468., 477., 321., 162.],
                         [105., 212., 321., 327., 333., 224., 113.],
                         [55., 111., 168., 171., 174., 117., 59.]]]]]).astype(np.float32)

        expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_3d')

    @staticmethod
    def export_convtranspose_attributes():  # type: () -> None
        x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                        [3., 4., 5.],
                        [6., 7., 8.]]]]).astype(np.float32)

        W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                        [1., 1., 1.],
                        [1., 1., 1.]],
                       [[1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(np.float32)

        y = np.array([[[[0., 0., 1., 1., 3., 2., 2., 0.],  # (1, 2, 10, 8)
                        [0., 0., 1., 1., 3., 2., 2., 0.],
                        [0., 0., 1., 1., 3., 2., 2., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0.]],

                       [[0., 0., 1., 1., 3., 2., 2., 0.],
                        [0., 0., 1., 1., 3., 2., 2., 0.],
                        [0., 0., 1., 1., 3., 2., 2., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [3., 3., 7., 4., 9., 5., 5., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [6., 6., 13., 7., 15., 8., 8., 0.],
                        [0., 0., 0., 0., 0., 0., 0., 0.]]]]).astype(np.float32)

        node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                                     strides=[3, 2],
                                     output_shape=[1, 2, 10, 8])
        expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_output_shape')


        node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                                     strides=[3, 2],
                                     output_padding=[1, 1])
        expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_pad')

        node = onnx.helper.make_node(
            'ConvTranspose', ['X', 'W'], ['Y'],
            name='test',
            strides=[3, 2],
            output_shape=[10, 8],
            kernel_shape=[3, 3],
            output_padding=[1, 1]
        )
        expect(node, inputs=[x, W], outputs=[y],
               name='test_convtranspose_kernel_shape')

    @staticmethod
    def export_convtranspose_pads():  # type: () -> None
        x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                        [3., 4., 5.],
                        [6., 7., 8.]]]]).astype(np.float32)

        W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                        [1., 1., 1.],
                        [1., 1., 1.]],
                       [[1., 1., 1.],
                        [1., 1., 1.],
                        [1., 1., 1.]]]]).astype(np.float32)

        node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"],
                                     strides=[3, 2],
                                     pads=[1, 2, 1, 2])

        y = np.array([[[[1., 1., 3.],  # (1, 2, 7, 3)
                        [1., 1., 3.],
                        [7., 4., 9.],
                        [7., 4., 9.],
                        [7., 4., 9.],
                        [13., 7., 15.],
                        [13., 7., 15.]],

                       [[1., 1., 3.],
                        [1., 1., 3.],
                        [7., 4., 9.],
                        [7., 4., 9.],
                        [7., 4., 9.],
                        [13., 7., 15.],
                        [13., 7., 15.]]]]).astype(np.float32)

        expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_pads')
