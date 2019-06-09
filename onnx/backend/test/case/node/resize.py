from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def cartesian(arrays, out=None):
    """
    From https://stackoverflow.com/a/1235363

    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n // arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def interpolate_1d_with_x(data, factor, x, get_coeffs,
                          align_corners = False, exclude_outside=False):
    def get_neighbor(x, n, data):
        pad_width = np.ceil(n / 2).astype(np.int)
        padded = np.pad(data, pad_width, mode='edge')
        x += pad_width
        ret = padded[int(x - ((n + 1) // 2 - 1)): int(x + (n // 2)) + 1]
        return ret

    input_width = len(data)
    output_width = factor * input_width
    if align_corners:
        if output_width == 1:
            x_ori = 0
        else:
            x_ori = x * (input_width - 1) / (output_width - 1)
    else:
        x_ori = (x + 0.5) / factor - 0.5
    x_ori_int = np.floor(x_ori).astype(np.int).item()

    ratio = x_ori - x_ori_int
    coeffs = get_coeffs(ratio)
    n = len(coeffs)
    if exclude_outside:
        left = (n + 1) // 2 - 1
        for i in range(len(coeffs)):
            if x_ori_int + i - left < 0 or x_ori_int + i - left >= input_width:
                coeffs[i] = 0
        coeffs /= sum(coeffs)

    points = get_neighbor(x_ori, n, data)

    return np.dot(coeffs, points).item()


def interpolate_nd_with_x(data, n, factors, x, get_coeffs, align_corners = False, exclude_outside=False):
    if n == 1:
        return interpolate_1d_with_x(data, factors[0], x[0], get_coeffs, align_corners, exclude_outside)
    return interpolate_1d_with_x(
        [interpolate_nd_with_x(data[i], n - 1, factors[1:], x[1:], get_coeffs, align_corners, exclude_outside)
         for i in range(data.shape[0])], factors[0], x[0], get_coeffs, align_corners, exclude_outside)


def interpolate_nd(data, get_coeffs, output_size=None, factors=None, align_corners = False, exclude_outside=False):
    def get_all_coords(data):
        return cartesian([list(range(data.shape[i])) for i in range(len(data.shape))])

    assert output_size is not None or factors is not None
    if output_size is not None:
        factors = np.array(output_size) / np.array(data.shape)
    else:
        output_size = (factors * np.array(data.shape)).astype(np.int)

    ret = np.zeros(output_size)
    for x in get_all_coords(ret):
        ret[tuple(x)] = interpolate_nd_with_x(data, len(data.shape), factors, x, get_coeffs, align_corners,
                                              exclude_outside)
    return ret


def cubic_coeffs(ratio, A=-0.75):
    coeffs = [((A * (ratio + 1) - 5 * A) * (ratio + 1) + 8 * A) * (ratio + 1) - 4 * A,
              ((A + 2) * ratio - (A + 3)) * ratio * ratio + 1,
              ((A + 2) * (1 - ratio) - (A + 3)) * (1 - ratio) * (1 - ratio) + 1,
              ((A * ((1 - ratio) + 1) - 5 * A) * ((1 - ratio) + 1) + 8 * A) * ((1 - ratio) + 1) - 4 * A]

    return np.array(coeffs)


def linear_coeffs(ratio):
    return np.array([1 - ratio, ratio])

class Resize(Base):

    @staticmethod
    def export_resize_upsample_scales_nearest():  # type: () -> None
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
               name='test_resize_upsample_scales_nearest')

    @staticmethod
    def export_resize_downsample_scales_nearest():  # type: () -> None
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
               name='test_resize_downsample_scales_nearest')

    @staticmethod
    def export_resize_upsample_scales_linear():  # type: () -> None
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

        # [[[[1.   1.25 1.75 2.  ]
        #    [1.5  1.75 2.25 2.5 ]
        #    [2.5  2.75 3.25 3.5 ]
        #    [3.   3.25 3.75 4.  ]]]]
        output = interpolate_nd(data, linear_coeffs, factors=scales, align_corners=False)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_upsample_scales_linear')

    @staticmethod
    def export_resize_upsample_scales_linear_align_corners():  # type: () -> None
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

        # [[[[1.         1.33333333 1.66666667 2.        ]
        #    [1.66666667 2.         2.33333333 2.66666667]
        #    [2.33333333 2.66666667 3.         3.33333333]
        #    [3.         3.33333333 3.66666667 4.        ]]]]
        output = interpolate_nd(data, linear_coeffs, factors=scales, align_corners=True)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_upsample_scales_linear_align_corners')

    @staticmethod
    def export_resize_downsample_scales_linear():  # type: () -> None
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

        # [[[[2.6666665 4.3333331]]]]
        output = interpolate_nd(data, linear_coeffs, factors=scales, align_corners=False)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_downsample_scales_linear')

    @staticmethod
    def export_resize_downsample_scales_linear_align_corners():  # type: () -> None
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

        # [[[[1.       3.142857]]]]
        output = interpolate_nd(data, linear_coeffs, factors=scales, align_corners=True)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_downsample_scales_linear_align_corners')

    @staticmethod
    def export_resize_upsample_scales_cubic():  # type: () -> None
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

        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        # [[[[ 0.47265625  0.76953125  1.24609375  1.875       2.28125
        #      2.91015625  3.38671875  3.68359375]
        #    [ 1.66015625  1.95703125  2.43359375  3.0625      3.46875
        #      4.09765625  4.57421875  4.87109375]
        #    [ 3.56640625  3.86328125  4.33984375  4.96875     5.375
        #      6.00390625  6.48046875  6.77734375]
        #    [ 6.08203125  6.37890625  6.85546875  7.484375    7.890625
        #      8.51953125  8.99609375  9.29296875]
        #    [ 7.70703125  8.00390625  8.48046875  9.109375    9.515625
        #     10.14453125 10.62109375 10.91796875]
        #    [10.22265625 10.51953125 10.99609375 11.625      12.03125
        #     12.66015625 13.13671875 13.43359375]
        #    [12.12890625 12.42578125 12.90234375 13.53125    13.9375
        #     14.56640625 15.04296875 15.33984375]
        #    [13.31640625 13.61328125 14.08984375 14.71875    15.125
        #     15.75390625 16.23046875 16.52734375]]]]
        output = interpolate_nd(data, lambda x: cubic_coeffs(x), factors=scales, align_corners=False)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_upsample_scales_cubic')

    @staticmethod
    def export_resize_upsample_scales_cubic_align_corners():  # type: () -> None
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

        # [[[[ 1.          1.34110787  1.80029155  2.32944606  2.67055394
        #      3.19970845  3.65889213  4.        ]
        #    [ 2.36443149  2.70553936  3.16472303  3.69387755  4.03498542
        #      4.56413994  5.02332362  5.36443149]
        #    [ 4.20116618  4.54227405  5.00145773  5.53061224  5.87172012
        #      6.40087464  6.86005831  7.20116618]
        #    [ 6.31778426  6.65889213  7.1180758   7.64723032  7.98833819
        #      8.51749271  8.97667638  9.31778426]
        #    [ 7.68221574  8.02332362  8.48250729  9.01166181  9.35276968
        #      9.8819242  10.34110787 10.68221574]
        #    [ 9.79883382 10.13994169 10.59912536 11.12827988 11.46938776
        #     11.99854227 12.45772595 12.79883382]
        #    [11.63556851 11.97667638 12.43586006 12.96501458 13.30612245
        #     13.83527697 14.29446064 14.63556851]
        #    [13.         13.34110787 13.80029155 14.32944606 14.67055394
        #     15.19970845 15.65889213 16.        ]]]]
        output = interpolate_nd(data, lambda x: cubic_coeffs(x), factors=scales, align_corners=True)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_upsample_scales_cubic_align_corners')

    @staticmethod
    def export_resize_downsample_scales_cubic():  # type: () -> None
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

        # [[[[ 1.47119141  2.78125     4.08251953]
        #    [ 6.71142578  8.02148438  9.32275391]
        #    [11.91650391 13.2265625  14.52783203]]]]
        output = interpolate_nd(data, lambda x: cubic_coeffs(x), factors=scales, align_corners=False)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_downsample_scales_cubic')

    @staticmethod
    def export_resize_downsample_scales_cubic_align_corners():  # type: () -> None
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

        scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

        # [[[[ 1.          2.39519159  3.79038317]
        #    [ 6.58076634  7.97595793  9.37114951]
        #    [12.16153268 13.55672427 14.95191585]]]]
        output = interpolate_nd(data, lambda x: cubic_coeffs(x), factors=scales, align_corners=True)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_downsample_scales_cubic_align_corners')

    @staticmethod
    def export_resize_upsample_size_cubic():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'sizes'],
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

        sizes = np.array([1, 1, 9, 10], dtype=np.float32)

        # [[[[ 0.45507922  0.64057922  0.97157922  1.42257922  1.90732922
        #      2.22332922  2.70807922  3.15907922  3.49007922  3.67557922]
        #    [ 1.39437963  1.57987963  1.91087963  2.36187963  2.84662963
        #      3.16262963  3.64737963  4.09837963  4.42937963  4.61487963]
        #    [ 2.95130693  3.13680693  3.46780693  3.91880693  4.40355693
        #      4.71955693  5.20430693  5.65530693  5.98630693  6.17180693]
        #    [ 5.20525069  5.39075069  5.72175069  6.17275069  6.65750069
        #      6.97350069  7.45825069  7.90925069  8.24025069  8.42575069]
        #    [ 6.88975     7.07525     7.40625     7.85725     8.342
        #      8.658       9.14275     9.59375     9.92475    10.11025   ]
        #    [ 8.57424931  8.75974931  9.09074931  9.54174931 10.02649931
        #     10.34249931 10.82724931 11.27824931 11.60924931 11.79474931]
        #    [10.82819307 11.01369307 11.34469307 11.79569307 12.28044307
        #     12.59644307 13.08119307 13.53219307 13.86319307 14.04869307]
        #    [12.38512037 12.57062037 12.90162037 13.35262037 13.83737037
        #     14.15337037 14.63812037 15.08912037 15.42012037 15.60562037]
        #    [13.32442078 13.50992078 13.84092078 14.29192078 14.77667078
        #     15.09267078 15.57742078 16.02842078 16.35942078 16.54492078]]]]
        output = interpolate_nd(data, lambda x: cubic_coeffs(x), output_size=sizes, align_corners=False)

        expect(node, inputs=[data, sizes], outputs=[output],
               name='test_resize_upsample_sizes_cubic')

    @staticmethod
    def export_resize_downsample_size_cubic():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'sizes'],
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

        sizes = np.array([1, 1, 3, 3], dtype=np.float32)

        # [[[[ 1.63078704  3.00462963  4.37847222]
        #    [ 7.12615741  8.5         9.87384259]
        #    [12.62152778 13.99537037 15.36921296]]]]
        output = interpolate_nd(data, lambda x: cubic_coeffs(x), output_size=sizes, align_corners=False)

        expect(node, inputs=[data, sizes], outputs=[output],
               name='test_resize_downsample_sizes_cubic')

    @staticmethod
    def export_resize_upsample_scales_cubic_A_n0p5_exclude_outside():  # type: () -> None
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

        scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

        # [[[[ 0.55882353  0.81494204  1.35698249  1.89705882  2.39705882
        #      2.93713516  3.47917561  3.73529412]
        #    [ 1.58329755  1.83941606  2.38145651  2.92153285  3.42153285
        #      3.96160918  4.50364964  4.75976814]
        #    [ 3.75145936  4.00757787  4.54961832  5.08969466  5.58969466
        #      6.12977099  6.67181144  6.92792995]
        #    [ 5.91176471  6.16788321  6.70992366  7.25        7.75
        #      8.29007634  8.83211679  9.08823529]
        #    [ 7.91176471  8.16788321  8.70992366  9.25        9.75
        #     10.29007634 10.83211679 11.08823529]
        #    [10.07207005 10.32818856 10.87022901 11.41030534 11.91030534
        #     12.45038168 12.99242213 13.24854064]
        #    [12.24023186 12.49635036 13.03839082 13.57846715 14.07846715
        #     14.61854349 15.16058394 15.41670245]
        #    [13.26470588 13.52082439 14.06286484 14.60294118 15.10294118
        #     15.64301751 16.18505796 16.44117647]]]]
        output = interpolate_nd(data, lambda x: cubic_coeffs(x, A=-0.5), factors=scales, align_corners=False,
                                exclude_outside=True)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_upsample_sizes_cubic')

    @staticmethod
    def export_resize_downsample_scales_cubic_A_n0p5_exclude_outside():  # type: () -> None
        node = onnx.helper.make_node(
            'Resize',
            inputs=['X', 'scales'],
            outputs=['Y'],
            mode='cubic',
            align_corners=False,
            A=-0.5,
            exclude_outside=True
        )

        data = np.array([[[
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ]]], dtype=np.float32)

        scales = np.array([1.0, 1.0, 0.8, 0.8], dtype=np.float32)

        # [[[[ 1.36812675  2.6695014   4.0133367 ]
        #    [ 6.57362535  7.875       9.2188353 ]
        #    [11.94896657 13.25034122 14.59417652]]]]
        output = interpolate_nd(data, lambda x: cubic_coeffs(x, A=-0.5), factors=scales, align_corners=False,
                                exclude_outside=True)

        expect(node, inputs=[data, scales], outputs=[output],
               name='test_resize_downsample_sizes_cubic')
