```toml
name = "AffineGrid"
domain = "ai.onnx"
shape_inference = "defs::tensor::utils::AffineGridShapeInference_opset20"

[attributes.align_corners]
type = "INT"
desc = "if align_corners=1, consider -1 and 1 to refer to the centers of the corner pixels. if align_corners=0, consider -1 and 1 to refer to the outer edge the corner pixels."
default = 0

[[inputs]]
name = "theta"
type = "T1"
desc = "input batch of affine matrices with shape (N, 2, 3) for 2D or (N, 3, 4) for 3D"
differentiation = "non_differentiable"

[[inputs]]
name = "size"
type = "T2"
desc = "the target output image size (N, C, H, W) for 2D or (N, C, D, H, W) for 3D"
differentiation = "non_differentiable"

[[outputs]]
name = "grid"
type = "T1"
desc = "output tensor of shape (N, H, W, 2) of 2D sample coordinates or (N, D, H, W, 3) of 3D sample coordinates."
differentiation = "differentiable"

[type_constraints.T1]
types = "all_float_types_ir4"
desc = "Constrain grid types to float tensors."

[type_constraints.T2]
types = ["tensor(int64)"]
desc = "Constrain size's type to int64 tensors."

[function_bodies]
20 = """
{
   one = Constant <value_int: int = 1> ()
   two = Constant <value_int: int = 2> ()
   zero = Constant <value_int: int = 0> ()
   four = Constant <value_int: int = 4> ()
   one_1d = Constant <value_ints: ints = [1]> ()
   zero_1d = Constant <value_ints: ints = [0]> ()
   minus_one = Constant <value_int: int = -1> ()
   minus_one_f = CastLike (minus_one, theta)
   zero_f = CastLike (zero, theta)
   one_f = CastLike (one, theta)
   two_f = CastLike (two, theta)
   constant_align_corners = Constant <value_int: int = @align_corners> ()
   constant_align_corners_equal_zero = Equal (constant_align_corners, zero)
   size_ndim = Size (size)
   condition_is_2d = Equal (size_ndim, four)
   N, C, D, H, W = If (condition_is_2d) <then_branch: graph = g1 () => ( N_then,  C_then,  D_then,  H_then,  W_then) {
      N_then, C_then, H_then, W_then = Split <num_outputs: int = 4> (size)
      D_then = Identity (one_1d)
   }, else_branch: graph = g2 () => ( N_else,  C_else,  D_else,  H_else,  W_else) {
      N_else, C_else, D_else, H_else, W_else = Split <num_outputs: int = 5> (size)
   }>
   size_NCDHW = Concat <axis: int = 0> (N, C, D, H, W)
   theta_3d = If (condition_is_2d) <then_branch: graph = g3 () => ( theta_then) {
      gather_idx_6 = Constant <value_ints: ints = [0, 1, 2, 0, 1, 2]> ()
      shape_23 = Constant <value_ints: ints = [2, 3]> ()
      gather_idx_23 = Reshape (gather_idx_6, shape_23)
      shape_N23 = Concat <axis: int = 0> (N, shape_23)
      gather_idx_N23 = Expand (gather_idx_23, shape_N23)
      thetaN23 = GatherElements <axis: int = 2> (theta, gather_idx_N23)
      r1, r2 = Split <axis: int = 1, num_outputs: int = 2> (thetaN23)
      r1_ = Squeeze (r1)
      r2_ = Squeeze (r2)
      r11, r12, t1 = Split <axis: int = 1, num_outputs: int = 3> (r1_)
      r21, r22, t2 = Split <axis: int = 1, num_outputs: int = 3> (r2_)
      r11_shape = Shape (r21)
      float_zero_1d_ = ConstantOfShape (r11_shape)
      float_zero_1d = CastLike (float_zero_1d_, theta)
      float_one_1d = Add (float_zero_1d, one_f)
      R1 = Concat <axis: int = 1> (r11, r12, float_zero_1d, t1)
      R2 = Concat <axis: int = 1> (r21, r22, float_zero_1d, t2)
      R3 = Concat <axis: int = 1> (float_zero_1d, float_zero_1d, float_one_1d, float_zero_1d)
      R1_ = Unsqueeze (R1, one_1d)
      R2_ = Unsqueeze (R2, one_1d)
      R3_ = Unsqueeze (R3, one_1d)
      theta_then = Concat <axis: int = 1> (R1_, R2_, R3_)
   }, else_branch: graph = g4 () => ( theta_else) {
      theta_else = Identity (theta)
   }>
   two_1d = Constant <value_ints: ints = [2]> ()
   three_1d = Constant <value_ints: ints = [3]> ()
   five_1d = Constant <value_ints: ints = [5]> ()
   constant_D_H_W_shape = Slice (size_NCDHW, two_1d, five_1d)
   zeros_D_H_W_ = ConstantOfShape (constant_D_H_W_shape)
   zeros_D_H_W = CastLike (zeros_D_H_W_, theta)
   ones_D_H_W = Add (zeros_D_H_W, one_f)
   D_float = CastLike (D, zero_f)
   H_float = CastLike (H, zero_f)
   W_float = CastLike (W, zero_f)
   start_d, step_d, start_h, step_h, start_w, step_w = If (constant_align_corners_equal_zero) <then_branch: graph = h1 () => ( start_d_then,  step_d_then,  start_h_then,  step_h_then,  start_w_then,  step_w_then) {
      step_d_then = Div (two_f, D_float)
      step_h_then = Div (two_f, H_float)
      step_w_then = Div (two_f, W_float)
      step_d_half = Div (step_d_then, two_f)
      start_d_then = Add (minus_one_f, step_d_half)
      step_h_half = Div (step_h_then, two_f)
      start_h_then = Add (minus_one_f, step_h_half)
      step_w_half = Div (step_w_then, two_f)
      start_w_then = Add (minus_one_f, step_w_half)
   }, else_branch: graph = h2 () => ( start_d_else,  step_d_else,  start_h_else,  step_h_else,  start_w_else,  step_w_else) {
      D_float_nimus_one = Sub (D_float, one_f)
      H_float_nimus_one = Sub (H_float, one_f)
      W_float_nimus_one = Sub (W_float, one_f)
      D_equals_one = Equal (D, one)
      step_d_else = If (D_equals_one) <then_branch: graph = g5 () => ( step_d_else_then) {
         step_d_else_then = Identity (zero_f)
      }, else_branch: graph = g6 () => ( step_d_else_else) {
         step_d_else_else = Div (two_f, D_float_nimus_one)
      }>
      step_h_else = Div (two_f, H_float_nimus_one)
      step_w_else = Div (two_f, W_float_nimus_one)
      start_d_else = Identity (minus_one_f)
      start_h_else = Identity (minus_one_f)
      start_w_else = Identity (minus_one_f)
   }>
   grid_w_steps_int = Range (zero, W, one)
   grid_w_steps_float = CastLike (grid_w_steps_int, step_w)
   grid_w_steps = Mul (grid_w_steps_float, step_w)
   grid_w_0 = Add (start_w, grid_w_steps)
   grid_h_steps_int = Range (zero, H, one)
   grid_h_steps_float = CastLike (grid_h_steps_int, step_h)
   grid_h_steps = Mul (grid_h_steps_float, step_h)
   grid_h_0 = Add (start_h, grid_h_steps)
   grid_d_steps_int = Range (zero, D, one)
   grid_d_steps_float = CastLike (grid_d_steps_int, step_d)
   grid_d_steps = Mul (grid_d_steps_float, step_d)
   grid_d_0 = Add (start_d, grid_d_steps)
   zeros_H_W_D = Transpose <perm: ints = [1, 2, 0]> (zeros_D_H_W)
   grid_d_1 = Add (zeros_H_W_D, grid_d_0)
   grid_d = Transpose <perm: ints = [2, 0, 1]> (grid_d_1)
   zeros_D_W_H = Transpose <perm: ints = [0, 2, 1]> (zeros_D_H_W)
   grid_h_1 = Add (zeros_D_W_H, grid_h_0)
   grid_h = Transpose <perm: ints = [0, 2, 1]> (grid_h_1)
   grid_w = Add (grid_w_0, zeros_D_H_W)
   grid_w_usqzed = Unsqueeze (grid_w, minus_one)
   grid_h_usqzed = Unsqueeze (grid_h, minus_one)
   grid_d_usqzed = Unsqueeze (grid_d, minus_one)
   ones_D_H_W_usqzed = Unsqueeze (ones_D_H_W, minus_one)
   original_grid = Concat <axis: int = -1> (grid_w_usqzed, grid_h_usqzed, grid_d_usqzed, ones_D_H_W_usqzed)
   constant_shape_DHW_4 = Constant <value_ints: ints = [-1, 4]> ()
   original_grid_DHW_4 = Reshape (original_grid, constant_shape_DHW_4)
   original_grid_4_DHW_ = Transpose (original_grid_DHW_4)
   original_grid_4_DHW = CastLike (original_grid_4_DHW_, theta_3d)
   grid_N_3_DHW = MatMul (theta_3d, original_grid_4_DHW)
   grid_N_DHW_3 = Transpose <perm: ints = [0, 2, 1]> (grid_N_3_DHW)
   N_D_H_W_3 = Concat <axis: int = -1> (N, D, H, W, three_1d)
   grid_3d_else_ = Reshape (grid_N_DHW_3, N_D_H_W_3)
   grid_3d = CastLike (grid_3d_else_, theta_3d)
   grid = If (condition_is_2d) <then_branch: graph = g1 () => ( grid_then) {
      grid_squeezed = Squeeze (grid_3d, one_1d)
      grid_then = Slice (grid_squeezed, zero_1d, two_1d, three_1d)
   }, else_branch: graph = g2 () => ( grid_else) {
      grid_else = Identity (grid_3d)
   }>
}
"""
```

Generates a 2D or 3D flow field (sampling grid), given a batch of affine matrices theta
(https://pytorch.org/docs/stable/generated/torch.nn.functional.affine_grid.html).
An affine matrix `theta` is applied to a position tensor represented in its homogeneous expression. Here is an example in 3D:
```
[r00, r01, r02, t0]   [x]   [x']
[r10, r11, r12, t1] * [y] = [y']
[r20, r21, r22, t2]   [z]   [z']
[0,   0,   0,   1 ]   [1]   [1 ]
```
where `(x, y, z)` is the position in the original space, `(x', y', z')` is the position in the output space.
The last row is always `[0, 0, 0, 1]` and is not stored in the affine matrix. Therefore we have `theta` of shape `(N, 2, 3)` for 2D or `(N, 3, 4)` for 3D.

Input `size` is used to define grid of positions evenly spaced in the original 2D or 3D space, with dimensions ranging from `-1` to `1`.
The output `grid` contains positions in the output space.

When `align_corners=1`, consider `-1` and `1` to refer to the centers of the corner pixels (mark `v` in illustration).
```
v            v            v            v
|-------------------|------------------|
-1                  0                  1
```
When `align_corners=0`, consider `-1` and `1` to refer to the outer edge of the corner pixels.
```
    v        v         v         v
|------------------|-------------------|
-1                 0                   1
```
