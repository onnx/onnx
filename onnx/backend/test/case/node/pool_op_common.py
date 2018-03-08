import numpy as np
import itertools


def get_pad_shape(auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial, output_spatial_shape):
  pad_shape = [0] * len(input_spatial_shape)
  if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
    for i in range(len(input_spatial_shape)):
      pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + kernel_spatial_shape[i] - \
                     input_spatial_shape[i]
  elif auto_pad == 'VALID':
    pass
  return pad_shape

def get_output_shape(auto_pad, input_spatial_shape, kernel_spatial_shape, strides_spatial):
  out_shape = [0] * len(input_spatial_shape)
  if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
    for i in range(len(input_spatial_shape)):
      out_shape[i] = int(np.ceil(input_spatial_shape[i] / strides_spatial[i]))
  elif auto_pad == 'VALID':
    for i in range(len(input_spatial_shape)):
      out_shape[i] = int(
        np.ceil((input_spatial_shape[i] - (kernel_spatial_shape[i] - 1)) / strides_spatial[i]))
  return out_shape

def pool(padded, x_shape, kernel_shape, strides_shape, out_shape, pad_shape):
  spatial_size = len(x_shape) - 2
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
