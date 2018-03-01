from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools

import numpy as np

import onnx
from ..base import Base
from . import expect


class MaxPool(Base):

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

    def _pool_2d(padded, x_shape, out_shape, kernel_shape, strides, pad_shape):
      k_h, k_w = kernel_shape
      s_h, s_w = strides
      y = np.zeros((x_shape[0], x_shape[1], out_shape[0], out_shape[1]))
      for n, c, h, w in itertools.product(range(x_shape[0]),
                                          range(x_shape[1]),
                                          range(int((x_shape[2] + pad_shape[0] - k_h) / s_h + 1)),
                                          range(int((x_shape[3] + pad_shape[1] - k_w) / s_w + 1))):
        window = padded[n, c, s_h * h:s_h * h + k_h, s_w * w:s_w * w + k_w]
        maximum = np.max(window[np.where(~np.isnan(window))])
        y[n, c, h, w] = maximum
      return y

    node = onnx.helper.make_node(
      'MaxPool',
      inputs=['x'],
      outputs=['y'],
      kernel_shape=[2, 2],
      auto_pad='SAME_UPPER'
    )
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    k_h, k_w = (2, 2)
    s_h, s_w = (1, 1)
    out_shape = _get_output_shape('SAME_UPPER', x_shape[2:4], (k_h, k_w), (s_h, s_w))
    pad_shape = _get_pad_shape('SAME_UPPER', x_shape[2:4], (k_h, k_w), (s_h, s_w), out_shape)
    pad_top = pad_shape[0] // 2
    pad_bottom = pad_shape[0] - pad_top
    pad_left = pad_shape[1] // 2
    pad_right = pad_shape[1] - pad_left
    padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                    constant_values=np.nan)
    y = _pool_2d(padded, x_shape, out_shape, (k_h, k_w), (s_h, s_w), pad_shape)

    expect(node, inputs=[x], outputs=[y], name='test_maxpool')

    node = onnx.helper.make_node(
      'MaxPool',
      inputs=['x'],
      outputs=['y'],
      kernel_shape=[2, 2],
      auto_pad='SAME_LOWER'
    )
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    k_h, k_w = (2, 2)
    s_h, s_w = (1, 1)
    out_shape = _get_output_shape('SAME_LOWER', x_shape[2:4], (k_h, k_w), (s_h, s_w))
    pad_shape = _get_pad_shape('SAME_LOWER', x_shape[2:4], (k_h, k_w), (s_h, s_w), out_shape)
    pad_bottom = pad_shape[0] // 2
    pad_top = pad_shape[0] - pad_bottom
    pad_right = pad_shape[1] // 2
    pad_left = pad_shape[1] - pad_right
    padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                    constant_values=np.nan)
    y = _pool_2d(padded, x_shape, out_shape, (k_h, k_w), (s_h, s_w), pad_shape)

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_same_lower')

    node = onnx.helper.make_node(
      'MaxPool',
      inputs=['x'],
      outputs=['y'],
      kernel_shape=[3, 3],
      pads=[2, 2, 2, 2]
    )
    x = np.random.randn(1, 3, 28, 28).astype(np.float32)
    x_shape = np.shape(x)
    k_h, k_w = (3, 3)
    s_h, s_w = (1, 1)
    pad_bottom = 2
    pad_top = 2
    pad_right = 2
    pad_left = 2
    pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
    out_shape = _get_output_shape('VALID', np.add(x_shape[2:4], pad_shape), (k_h, k_w), (s_h, s_w))
    padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                    constant_values=np.nan)
    y = _pool_2d(padded, x_shape, out_shape, (k_h, k_w), (s_h, s_w), pad_shape)

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_pads')

    node = onnx.helper.make_node(
      'MaxPool',
      inputs=['x'],
      outputs=['y'],
      kernel_shape=[5, 5],
      strides=[3, 3]
    )
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    k_h, k_w = (5, 5)
    s_h, s_w = (3, 3)
    out_shape = _get_output_shape('VALID', x_shape[2:4], (k_h, k_w), (s_h, s_w))
    padded = x
    y = _pool_2d(padded, x_shape, out_shape, (k_h, k_w), (s_h, s_w), (0, 0))

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_strides')

    node = onnx.helper.make_node(
      'MaxPool',
      inputs=['x'],
      outputs=['y'],
      kernel_shape=[2, 2],
    )
    x = np.random.randn(1, 3, 32, 32).astype(np.float32)
    x_shape = np.shape(x)
    k_h, k_w = (2, 2)
    s_h, s_w = (1, 1)
    out_shape = _get_output_shape('VALID', x_shape[2:4], (k_h, k_w), (s_h, s_w))
    padded = x
    y = _pool_2d(padded, x_shape, out_shape, (k_h, k_w), (s_h, s_w), (0, 0))

    expect(node, inputs=[x], outputs=[y], name='test_maxpool_default')
