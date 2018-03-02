from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools

import numpy as np

import onnx
from ..base import Base
from . import expect


class AveragePool(Base):

  @staticmethod
  def export():
    def _get_pad_shape(auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial, output_spatial_shape):
      pad_shape = [0] * len(input_spatial_shape)
      if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
          pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + kernel_spatial_shape[i] - \
                         input_spatial_shape[i]
      elif auto_pad == 'VALID':
        pass
      return pad_shape

    def _get_output_shape(auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial):
      out_shape = [0] * len(input_spatial_shape)
      if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
          out_shape[i] = int(np.ceil(input_spatial_shape[i] / strides_spatial[i]))
      elif auto_pad == 'VALID':
        for i in range(len(input_spatial_shape)):
          out_shape[i] = int(
            np.ceil((input_spatial_shape[i] - (kernel_spatial_shape[i] - 1)) / strides_spatial[i]))
      return out_shape

    def _pool(padded, spatial_size, kernel_shape, strides_shape, out_shape, pad_shape):
      y = np.zeros([x_shape[0], x_shape[1]] + list(out_shape))

      for shape in itertools.product(range(x_shape[0]),
                                     range(x_shape[1]),
                                     *[range(
                                       int((x_shape[i + 2] + pad_shape[i] - kernel_shape[i]) / strides_shape[i] + 1))
                                       for i in range(spatial_size)]):
        window = padded[shape[0], shape[1]]
        if spatial_size == 1:
          window = window[
                   strides_shape[0] * shape[2]:strides_shape[0] * shape[2] + kernel_shape[0]]
        elif spatial_size == 2:
          window = window[
                   strides_shape[0] * shape[2]:strides_shape[0] * shape[2] + kernel_shape[0],
                   strides_shape[1] * shape[3]:strides_shape[1] * shape[3] + kernel_shape[1]]
        elif spatial_size == 3:
          window = window[
                   strides_shape[0] * shape[2]:strides_shape[0] * shape[2] + kernel_shape[0],
                   strides_shape[1] * shape[3]:strides_shape[1] * shape[3] + kernel_shape[1],
                   strides_shape[2] * shape[4]:strides_shape[2] * shape[4] + kernel_shape[2]]
        average = np.average(window[np.where(~np.isnan(window))])
        if spatial_size == 1:
          y[shape[0], shape[1], shape[2]] = average
        elif spatial_size == 2:
          y[shape[0], shape[1], shape[2], shape[3]] = average
        elif spatial_size == 3:
          y[shape[0], shape[1], shape[2], shape[3], shape[4]] = average
      return y.astype(np.float32)

    node = onnx.helper.make_node(
      'AveragePool',
      inputs=['x'],
      outputs=['y'],
      kernel_shape=[2, 2],
      auto_pad='SAME_UPPER'
    )
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    spatial_size = np.ndim(x) - 2
    kernel_shape = (2, 2)
    strides = (1, 1)
    out_shape = _get_output_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides)
    pad_shape = _get_pad_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides, out_shape)
    pad_top = pad_shape[0] // 2
    pad_bottom = pad_shape[0] - pad_top
    pad_left = pad_shape[1] // 2
    pad_right = pad_shape[1] - pad_left
    padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                    constant_values=np.nan)
    y = _pool(padded, spatial_size, kernel_shape, strides, out_shape, pad_shape)

    expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_same_upper')

    node = onnx.helper.make_node(
      'AveragePool',
      inputs=['x'],
      outputs=['y'],
      kernel_shape=[2, 2],
      auto_pad='SAME_LOWER'
    )
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    spatial_size = np.ndim(x) - 2
    kernel_shape = (2, 2)
    strides = (1, 1)
    out_shape = _get_output_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides)
    pad_shape = _get_pad_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides, out_shape)
    pad_bottom = pad_shape[0] // 2
    pad_top = pad_shape[0] - pad_bottom
    pad_right = pad_shape[1] // 2
    pad_left = pad_shape[1] - pad_right
    padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                    constant_values=np.nan)
    y = _pool(padded, spatial_size, kernel_shape, strides, out_shape, pad_shape)

    expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_same_lower')

    node = onnx.helper.make_node(
      'AveragePool',
      inputs=['x'],
      outputs=['y'],
      kernel_shape=[3, 3],
      pads=[2, 2, 2, 2]
    )
    x = np.random.randn(1, 3, 28, 28).astype(np.float32)
    x_shape = np.shape(x)
    spatial_size = np.ndim(x) - 2
    kernel_shape = (3, 3)
    strides = (1, 1)
    pad_bottom = 2
    pad_top = 2
    pad_right = 2
    pad_left = 2
    pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
    out_shape = _get_output_shape('VALID', np.add(x_shape[2:], pad_shape), kernel_shape, strides)
    padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                    constant_values=np.nan)
    y = _pool(padded, spatial_size, kernel_shape, strides, out_shape, pad_shape)

    expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_pads')

    node = onnx.helper.make_node(
      'AveragePool',
      inputs=['x'],
      outputs=['y'],
      kernel_shape=[5, 5],
      strides=[3, 3]
    )
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    spatial_size = np.ndim(x) - 2
    kernel_shape = (5, 5)
    strides = (3, 3)
    out_shape = _get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
    padded = x
    y = _pool(padded, spatial_size, kernel_shape, strides, out_shape, (0, 0))

    expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_strides')


    node = onnx.helper.make_node(
      'AveragePool',
      inputs=['x'],
      outputs=['y'],
      kernel_shape=[2],
    )
    x = np.random.randn(1, 3, 32).astype(np.float32)
    x_shape = np.shape(x)
    spatial_size = np.ndim(x) - 2
    kernel_shape = [2]
    strides = [1]
    out_shape = _get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
    padded = x
    y = _pool(padded, spatial_size, kernel_shape, strides, out_shape, [0])

    expect(node, inputs=[x], outputs=[y], name='test_averagepool_1d_default')

    node = onnx.helper.make_node(
      'AveragePool',
      inputs=['x'],
      outputs=['y'],
      kernel_shape=[2, 2],
    )
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    spatial_size = np.ndim(x) - 2
    kernel_shape = (2, 2)
    strides = (1, 1)
    out_shape = _get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
    padded = x
    y = _pool(padded, spatial_size, kernel_shape, strides, out_shape, (0, 0))

    expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_default')

    node = onnx.helper.make_node(
      'AveragePool',
      inputs=['x'],
      outputs=['y'],
      kernel_shape=[2, 2, 2],
    )
    x = np.random.randn(1, 3, 32, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    spatial_size = np.ndim(x) - 2
    kernel_shape = [2, 2, 2]
    strides = [1, 1, 1]
    out_shape = _get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
    padded = x
    y = _pool(padded, spatial_size, kernel_shape, strides, out_shape, [0, 0, 0])

    expect(node, inputs=[x], outputs=[y], name='test_averagepool_3d_default')
