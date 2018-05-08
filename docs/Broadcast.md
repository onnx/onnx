# Broadcast in ONNX

In ONNX, if necessary the smaller argument will be broadcasted to match the shape of
the larger one. In general, ONNX follows Numpy's
[general broadcasting rules](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules).

For example, the following tensor shapes are supported by Numpy-style broadcasting:

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar ==> shape(result) = (2, 3, 4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (5,), ==> shape(result) = (2, 3, 4, 5)
  shape(A) = (4, 5), shape(B) = (2, 3, 4, 5), ==> shape(result) = (2, 3, 4, 5)
  shape(A) = (1, 4, 5), shape(B) = (2, 3, 1, 1), ==> shape(result) = (2, 3, 4, 5)
  shape(A) = (3, 4, 5), shape(B) = (2, 1, 1, 1), ==> shape(result) = (2, 3, 4, 5)


