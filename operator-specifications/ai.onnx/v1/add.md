```toml
name = "Add"
domain = "ai.onnx"

[attributes.axis]
type = "INT"
desc = "If set, defines the broadcast dimensions. See doc for details."

[attributes.broadcast]
type = "INT"
desc = "Pass 1 to enable broadcasting"
default = 0

[attributes.consumed_inputs]
type = "INTS"
desc = "legacy optimization attribute."

[[inputs]]
name = "A"
type = "T"
desc = "First operand, should share the type with the second operand."

[[inputs]]
name = "B"
type = "T"
desc = "Second operand. With broadcasting can be of smaller size than A. If broadcasting is disabled it should be of the same size."

[[outputs]]
name = "C"
type = "T"
desc = "Result, has same dimensions and type as A"

[type_constraints.T]
types = ["tensor(float16)", "tensor(float)", "tensor(double)"]
desc = "Constrain input and output types to float tensors."
```

Performs element-wise binary addition (with limited broadcast support).

If necessary the right-hand-side argument will be broadcasted to match the
shape of left-hand-side argument. When broadcasting is specified, the second
tensor can either be of element size 1 (including a scalar tensor and any
tensor with rank equal to or smaller than the first tensor), or having its
shape as a contiguous subset of the first tensor's shape. The starting of the
mutually equal shape is specified by the argument "axis", and if it is not set,
suffix matching is assumed. 1-dim expansion doesn't work yet.

For example, the following tensor shapes are supported (with broadcast=1):

  shape(A) = (2, 3, 4, 5), shape(B) = (,), i.e. B is a scalar tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (1, 1), i.e. B is an 1-element tensor
  shape(A) = (2, 3, 4, 5), shape(B) = (5,)
  shape(A) = (2, 3, 4, 5), shape(B) = (4, 5)
  shape(A) = (2, 3, 4, 5), shape(B) = (3, 4), with axis=1
  shape(A) = (2, 3, 4, 5), shape(B) = (2), with axis=0

Attribute `broadcast=1` needs to be passed to enable broadcasting.
