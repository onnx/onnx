<!--- SPDX-License-Identifier: Apache-2.0 -->
# Test Coverage Report (ONNX Core Operators)
## Outlines
* [Node Test Coverage](#node-test-coverage)
* [Model Test Coverage](#model-test-coverage)
* [Overall Test Coverage](#overall-test-coverage)
# Node Test Coverage
## Summary
Node tests have covered 162/177 (91.53%, 5 generators excluded) common operators.

Node tests have covered 0/0 (N/A) experimental operators.

* [Covered Common Operators](#covered-common-operators)
* [No Cover Common Operators](#no-cover-common-operators)
* [Covered Experimental Operators](#covered-experimental-operators)
* [No Cover Experimental Operators](#no-cover-experimental-operators)

## &#x1F49A;Covered Common Operators
### Abs
There are 1 test cases, listed as following:
<details>
<summary>abs</summary>

```python
node = onnx.helper.make_node(
    'Abs',
    inputs=['x'],
    outputs=['y'],
)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = abs(x)

expect(node, inputs=[x], outputs=[y],
       name='test_abs')
```

</details>


### Acos
There are 1 test cases, listed as following:
<details>
<summary>acos</summary>

```python
node = onnx.helper.make_node(
    'Acos',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-0.5, 0, 0.5]).astype(np.float32)
y = np.arccos(x)
expect(node, inputs=[x], outputs=[y],
       name='test_acos_example')

x = np.random.rand(3, 4, 5).astype(np.float32)
y = np.arccos(x)
expect(node, inputs=[x], outputs=[y],
       name='test_acos')
```

</details>


### Acosh
There are 1 test cases, listed as following:
<details>
<summary>acosh</summary>

```python
node = onnx.helper.make_node(
    'Acosh',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([10, np.e, 1]).astype(np.float32)
y = np.arccosh(x)  # expected output [2.99322295,  1.65745449,  0.]
expect(node, inputs=[x], outputs=[y],
       name='test_acosh_example')

x = np.random.uniform(1.0, 10.0, (3, 4, 5)).astype(np.float32)
y = np.arccosh(x)
expect(node, inputs=[x], outputs=[y],
       name='test_acosh')
```

</details>


### Adagrad
There are 2 test cases, listed as following:
<details>
<summary>adagrad</summary>

```python
# Define operator attributes.
norm_coefficient = 0.001
epsilon = 1e-5
decay_factor = 0.1

# Create operator.
node = onnx.helper.make_node('Adagrad',
                             inputs=['R', 'T', 'X', 'G', 'H'],
                             outputs=['X_new', 'H_new'],
                             norm_coefficient=norm_coefficient,
                             epsilon=epsilon,
                             decay_factor=decay_factor,
                             domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN
                             )

# Define operator inputs.
r = np.array(0.1, dtype=np.float32)  # scalar
t = np.array(0, dtype=np.int64)  # scalar
x = np.array([1.0], dtype=np.float32)
g = np.array([-1.0], dtype=np.float32)
h = np.array([2.0], dtype=np.float32)

# Compute expected outputs of Adagrad.
x_new, h_new = apply_adagrad(r, t, x, g, h,
                             norm_coefficient, epsilon, decay_factor)

# Check results.
expect(node, inputs=[r, t, x, g, h],
       outputs=[x_new, h_new], name='test_adagrad',
       opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])
```

</details>
<details>
<summary>adagrad_multiple</summary>

```python
# Define operator attributes.
norm_coefficient = 0.001
epsilon = 1e-5
decay_factor = 0.1

node = onnx.helper.make_node('Adagrad',
                             inputs=['R', 'T', 'X1', 'X2',
                                     'G1', 'G2', 'H1', 'H2'],
                             outputs=['X1_new', 'X2_new',
                                      'H1_new', 'H2_new'],
                             norm_coefficient=norm_coefficient,
                             epsilon=epsilon,
                             decay_factor=decay_factor,
                             domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN
                             )

# Define operator inputs.
r = np.array(0.1, dtype=np.float32)  # scalar
t = np.array(0, dtype=np.int64)  # scalar

x1 = np.array([1.0], dtype=np.float32)
g1 = np.array([-1.0], dtype=np.float32)
h1 = np.array([2.0], dtype=np.float32)

x2 = np.array([1.0, 2.0], dtype=np.float32)
g2 = np.array([-1.0, -3.0], dtype=np.float32)
h2 = np.array([4.0, 1.0], dtype=np.float32)

# Compute expected outputs of Adagrad.
x1_new, h1_new = apply_adagrad(r, t, x1, g1, h1,
                               norm_coefficient, epsilon, decay_factor)
x2_new, h2_new = apply_adagrad(r, t, x2, g2, h2,
                               norm_coefficient, epsilon, decay_factor)

# Check results.
expect(node, inputs=[r, t, x1, x2, g1, g2, h1, h2],
       outputs=[x1_new, x2_new, h1_new, h2_new], name='test_adagrad_multiple',
       opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])
```

</details>


### Adam
There are 2 test cases, listed as following:
<details>
<summary>adam</summary>

```python
# Define operator attributes.
norm_coefficient = 0.001
alpha = 0.95
beta = 0.1
epsilon = 1e-7

# Create operator.
node = onnx.helper.make_node('Adam',
                             inputs=['R', 'T', 'X', 'G', 'V', 'H'],
                             outputs=['X_new', 'V_new', 'H_new'],
                             norm_coefficient=norm_coefficient,
                             alpha=alpha,
                             beta=beta,
                             epsilon=epsilon,
                             domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN
                             )

# Define operator inputs.
r = np.array(0.1, dtype=np.float32)  # scalar
t = np.array(0, dtype=np.int64)  # scalar
x = np.array([1.2, 2.8], dtype=np.float32)
g = np.array([-0.94, -2.5], dtype=np.float32)
v = np.array([1.7, 3.6], dtype=np.float32)
h = np.array([0.1, 0.1], dtype=np.float32)

# Compute expected outputs of Adam.
x_new, v_new, h_new = apply_adam(r, t, x, g, v, h,
                                 norm_coefficient, 0.0, alpha, beta,
                                 epsilon)

# Check results.
expect(node, inputs=[r, t, x, g, v, h],
       outputs=[x_new, v_new, h_new], name='test_adam',
       opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])
```

</details>
<details>
<summary>adam_multiple</summary>

```python
# Define operator attributes.
norm_coefficient = 0.001
alpha = 0.95
beta = 0.85
epsilon = 1e-2

node = onnx.helper.make_node('Adam',
                             inputs=['R', 'T', 'X1', 'X2',
                                     'G1', 'G2', 'V1', 'V2',
                                     'H1', 'H2'],
                             outputs=['X1_new', 'X2_new',
                                      'V1_new', 'V2_new',
                                      'H1_new', 'H2_new'],
                             norm_coefficient=norm_coefficient,
                             alpha=alpha,
                             beta=beta,
                             domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN
                             )

# Define operator inputs.
r = np.array(0.1, dtype=np.float32)  # scalar
t = np.array(0, dtype=np.int64)  # scalar

x1 = np.array([1.0], dtype=np.float32)
g1 = np.array([-1.0], dtype=np.float32)
v1 = np.array([2.0], dtype=np.float32)
h1 = np.array([0.5], dtype=np.float32)

x2 = np.array([1.0, 2.0], dtype=np.float32)
g2 = np.array([-1.0, -3.0], dtype=np.float32)
v2 = np.array([4.0, 1.0], dtype=np.float32)
h2 = np.array([1.0, 10.0], dtype=np.float32)

# Compute expected outputs of Adam.
x1_new, v1_new, h1_new = apply_adam(r, t, x1, g1, v1, h1,
                            norm_coefficient, 0.0, alpha, beta,
                            epsilon)
x2_new, v2_new, h2_new = apply_adam(r, t, x2, g2, v2, h2,
                            norm_coefficient, 0.0, alpha, beta,
                            epsilon)

# Check results.
expect(node, inputs=[r, t, x1, x2, g1, g2, v1, v2, h1, h2],
       outputs=[x1_new, x2_new, v1_new, v2_new, h1_new, h2_new],
       name='test_adam_multiple',
       opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])
```

</details>


### Add
There are 3 test cases, listed as following:
<details>
<summary>add</summary>

```python
node = onnx.helper.make_node(
    'Add',
    inputs=['x', 'y'],
    outputs=['sum'],
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(3, 4, 5).astype(np.float32)
expect(node, inputs=[x, y], outputs=[x + y],
       name='test_add')
```

</details>
<details>
<summary>add_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Add',
    inputs=['x', 'y'],
    outputs=['sum'],
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(5).astype(np.float32)
expect(node, inputs=[x, y], outputs=[x + y],
       name='test_add_bcast')
```

</details>
<details>
<summary>add_uint8</summary>

```python
node = onnx.helper.make_node(
    'Add',
    inputs=['x', 'y'],
    outputs=['sum'],
)

x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
expect(node, inputs=[x, y], outputs=[x + y],
       name='test_add_uint8')
```

</details>


### And
There are 2 test cases, listed as following:
<details>
<summary>and</summary>

```python
node = onnx.helper.make_node(
    'And',
    inputs=['x', 'y'],
    outputs=['and'],
)

# 2d
x = (np.random.randn(3, 4) > 0).astype(bool)
y = (np.random.randn(3, 4) > 0).astype(bool)
z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_and2d')

# 3d
x = (np.random.randn(3, 4, 5) > 0).astype(bool)
y = (np.random.randn(3, 4, 5) > 0).astype(bool)
z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_and3d')

# 4d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
y = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_and4d')
```

</details>
<details>
<summary>and_broadcast</summary>

```python
node = onnx.helper.make_node(
    'And',
    inputs=['x', 'y'],
    outputs=['and'],
)

# 3d vs 1d
x = (np.random.randn(3, 4, 5) > 0).astype(bool)
y = (np.random.randn(5) > 0).astype(bool)
z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_and_bcast3v1d')

# 3d vs 2d
x = (np.random.randn(3, 4, 5) > 0).astype(bool)
y = (np.random.randn(4, 5) > 0).astype(bool)
z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_and_bcast3v2d')

# 4d vs 2d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
y = (np.random.randn(5, 6) > 0).astype(bool)
z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_and_bcast4v2d')

# 4d vs 3d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
y = (np.random.randn(4, 5, 6) > 0).astype(bool)
z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_and_bcast4v3d')

# 4d vs 4d
x = (np.random.randn(1, 4, 1, 6) > 0).astype(bool)
y = (np.random.randn(3, 1, 5, 6) > 0).astype(bool)
z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_and_bcast4v4d')
```

</details>


### ArgMax
There are 8 test cases, listed as following:
<details>
<summary>default_axes_keepdims</summary>

```python
data = np.array([[2, 1], [3, 10]], dtype=np.float32)
keepdims = 1
node = onnx.helper.make_node(
    'ArgMax',
    inputs=['data'],
    outputs=['result'],
    keepdims=keepdims)

# result: [[1, 1]]
result = argmax_use_numpy(data, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_default_axis_example')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [1, 3, 4]
result = argmax_use_numpy(data, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_default_axis_random')
```

</details>
<details>
<summary>default_axes_keepdims_select_last_index</summary>

```python
data = np.array([[2, 2], [3, 10]], dtype=np.float32)
keepdims = 1
node = onnx.helper.make_node(
    'ArgMax',
    inputs=['data'],
    outputs=['result'],
    keepdims=keepdims,
    select_last_index=True)

# result: [[1, 1]]
result = argmax_use_numpy_select_last_index(data, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_default_axis_example_select_last_index')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [1, 3, 4]
result = argmax_use_numpy_select_last_index(data, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_default_axis_random_select_last_index')
```

</details>
<details>
<summary>keepdims</summary>

```python
data = np.array([[2, 1], [3, 10]], dtype=np.float32)
axis = 1
keepdims = 1
node = onnx.helper.make_node(
    'ArgMax',
    inputs=['data'],
    outputs=['result'],
    axis=axis,
    keepdims=keepdims)
# result: [[0], [1]]
result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_keepdims_example')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [2, 1, 4]
result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_keepdims_random')
```

</details>
<details>
<summary>keepdims_select_last_index</summary>

```python
data = np.array([[2, 2], [3, 10]], dtype=np.float32)
axis = 1
keepdims = 1
node = onnx.helper.make_node(
    'ArgMax',
    inputs=['data'],
    outputs=['result'],
    axis=axis,
    keepdims=keepdims,
    select_last_index=True)
# result: [[1], [1]]
result = argmax_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_keepdims_example_select_last_index')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [2, 1, 4]
result = argmax_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_keepdims_random_select_last_index')
```

</details>
<details>
<summary>negative_axis_keepdims</summary>

```python
data = np.array([[2, 1], [3, 10]], dtype=np.float32)
axis = -1
keepdims = 1
node = onnx.helper.make_node(
    'ArgMax',
    inputs=['data'],
    outputs=['result'],
    axis=axis,
    keepdims=keepdims)
# result: [[0], [1]]
result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_negative_axis_keepdims_example')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [2, 3, 1]
result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_negative_axis_keepdims_random')
```

</details>
<details>
<summary>negative_axis_keepdims_select_last_index</summary>

```python
data = np.array([[2, 2], [3, 10]], dtype=np.float32)
axis = -1
keepdims = 1
node = onnx.helper.make_node(
    'ArgMax',
    inputs=['data'],
    outputs=['result'],
    axis=axis,
    keepdims=keepdims,
    select_last_index=True)
# result: [[1], [1]]
result = argmax_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_negative_axis_keepdims_example_select_last_index')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [2, 3, 1]
result = argmax_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_negative_axis_keepdims_random_select_last_index')
```

</details>
<details>
<summary>no_keepdims</summary>

```python
data = np.array([[2, 1], [3, 10]], dtype=np.float32)
axis = 1
keepdims = 0
node = onnx.helper.make_node(
    'ArgMax',
    inputs=['data'],
    outputs=['result'],
    axis=axis,
    keepdims=keepdims)
# result: [0, 1]
result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_no_keepdims_example')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [2, 4]
result = argmax_use_numpy(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_no_keepdims_random')
```

</details>
<details>
<summary>no_keepdims_select_last_index</summary>

```python
data = np.array([[2, 2], [3, 10]], dtype=np.float32)
axis = 1
keepdims = 0
node = onnx.helper.make_node(
    'ArgMax',
    inputs=['data'],
    outputs=['result'],
    axis=axis,
    keepdims=keepdims,
    select_last_index=True)
# result: [1, 1]
result = argmax_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_no_keepdims_example_select_last_index')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [2, 4]
result = argmax_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmax_no_keepdims_random_select_last_index')
```

</details>


### ArgMin
There are 8 test cases, listed as following:
<details>
<summary>default_axes_keepdims</summary>

```python
data = np.array([[2, 1], [3, 10]], dtype=np.float32)
keepdims = 1
node = onnx.helper.make_node(
    'ArgMin',
    inputs=['data'],
    outputs=['result'],
    keepdims=keepdims)

# The content of result is : [[0], [0]]
result = argmin_use_numpy(data, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_default_axis_example')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [1, 3, 4]
result = argmin_use_numpy(data, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_default_axis_random')
```

</details>
<details>
<summary>default_axes_keepdims_select_last_index</summary>

```python
data = np.array([[2, 2], [3, 10]], dtype=np.float32)
keepdims = 1
node = onnx.helper.make_node(
    'ArgMin',
    inputs=['data'],
    outputs=['result'],
    keepdims=keepdims,
    select_last_index=True)

# result: [[0, 0]]
result = argmin_use_numpy_select_last_index(data, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_default_axis_example_select_last_index')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [1, 3, 4]
result = argmin_use_numpy_select_last_index(data, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_default_axis_random_select_last_index')
```

</details>
<details>
<summary>keepdims</summary>

```python
data = np.array([[2, 1], [3, 10]], dtype=np.float32)
axis = 1
keepdims = 1
node = onnx.helper.make_node(
    'ArgMin',
    inputs=['data'],
    outputs=['result'],
    axis=axis,
    keepdims=keepdims)
# The content of result is : [[1], [0]]
result = argmin_use_numpy(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_keepdims_example')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [2, 1, 4]
result = argmin_use_numpy(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_keepdims_random')
```

</details>
<details>
<summary>keepdims_select_last_index</summary>

```python
data = np.array([[2, 2], [3, 10]], dtype=np.float32)
axis = 1
keepdims = 1
node = onnx.helper.make_node(
    'ArgMin',
    inputs=['data'],
    outputs=['result'],
    axis=axis,
    keepdims=keepdims,
    select_last_index=True)
# result: [[1], [0]]
result = argmin_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_keepdims_example_select_last_index')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [2, 1, 4]
result = argmin_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_keepdims_random_select_last_index')
```

</details>
<details>
<summary>negative_axis_keepdims</summary>

```python
data = np.array([[2, 1], [3, 10]], dtype=np.float32)
axis = -1
keepdims = 1
node = onnx.helper.make_node(
    'ArgMin',
    inputs=['data'],
    outputs=['result'],
    axis=axis,
    keepdims=keepdims)
# The content of result is : [[1], [0]]
result = argmin_use_numpy(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_negative_axis_keepdims_example')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [2, 3, 1]
result = argmin_use_numpy(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_negative_axis_keepdims_random')
```

</details>
<details>
<summary>negative_axis_keepdims_select_last_index</summary>

```python
data = np.array([[2, 2], [3, 10]], dtype=np.float32)
axis = -1
keepdims = 1
node = onnx.helper.make_node(
    'ArgMin',
    inputs=['data'],
    outputs=['result'],
    axis=axis,
    keepdims=keepdims,
    select_last_index=True)
# result: [[1], [0]]
result = argmin_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_negative_axis_keepdims_example_select_last_index')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [2, 3, 1]
result = argmin_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_negative_axis_keepdims_random_select_last_index')
```

</details>
<details>
<summary>no_keepdims</summary>

```python
data = np.array([[2, 1], [3, 10]], dtype=np.float32)
axis = 1
keepdims = 0
node = onnx.helper.make_node(
    'ArgMin',
    inputs=['data'],
    outputs=['result'],
    axis=axis,
    keepdims=keepdims)
# The content of result is : [[1, 0]]
result = argmin_use_numpy(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_no_keepdims_example')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [2, 4]
result = argmin_use_numpy(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_no_keepdims_random')
```

</details>
<details>
<summary>no_keepdims_select_last_index</summary>

```python
data = np.array([[2, 2], [3, 10]], dtype=np.float32)
axis = 1
keepdims = 0
node = onnx.helper.make_node(
    'ArgMin',
    inputs=['data'],
    outputs=['result'],
    axis=axis,
    keepdims=keepdims,
    select_last_index=True)
# result: [[1, 0]]
result = argmin_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_no_keepdims_example_select_last_index')

data = np.random.uniform(-10, 10, [2, 3, 4]).astype(np.float32)
# result's shape: [2, 4]
result = argmin_use_numpy_select_last_index(data, axis=axis, keepdims=keepdims)
expect(node, inputs=[data], outputs=[result], name='test_argmin_no_keepdims_random_select_last_index')
```

</details>


### Asin
There are 1 test cases, listed as following:
<details>
<summary>asin</summary>

```python
node = onnx.helper.make_node(
    'Asin',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-0.5, 0, 0.5]).astype(np.float32)
y = np.arcsin(x)
expect(node, inputs=[x], outputs=[y],
       name='test_asin_example')

x = np.random.rand(3, 4, 5).astype(np.float32)
y = np.arcsin(x)
expect(node, inputs=[x], outputs=[y],
       name='test_asin')
```

</details>


### Asinh
There are 1 test cases, listed as following:
<details>
<summary>asinh</summary>

```python
node = onnx.helper.make_node(
    'Asinh',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.arcsinh(x)  # expected output [-0.88137358,  0.,  0.88137358]
expect(node, inputs=[x], outputs=[y],
       name='test_asinh_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.arcsinh(x)
expect(node, inputs=[x], outputs=[y],
       name='test_asinh')
```

</details>


### Atan
There are 1 test cases, listed as following:
<details>
<summary>atan</summary>

```python
node = onnx.helper.make_node(
    'Atan',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.arctan(x)
expect(node, inputs=[x], outputs=[y],
       name='test_atan_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.arctan(x)
expect(node, inputs=[x], outputs=[y],
       name='test_atan')
```

</details>


### Atanh
There are 1 test cases, listed as following:
<details>
<summary>atanh</summary>

```python
node = onnx.helper.make_node(
    'Atanh',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-0.5, 0, 0.5]).astype(np.float32)
y = np.arctanh(x)  # expected output [-0.54930615,  0.,  0.54930615]
expect(node, inputs=[x], outputs=[y],
       name='test_atanh_example')

x = np.random.uniform(0.0, 1.0, (3, 4, 5)).astype(np.float32)
y = np.arctanh(x)
expect(node, inputs=[x], outputs=[y],
       name='test_atanh')
```

</details>


### AveragePool
There are 13 test cases, listed as following:
<details>
<summary>averagepool_1d_default</summary>

```python
"""
input_shape: [1, 3, 32]
output_shape: [1, 3, 31]
"""
node = onnx.helper.make_node(
    'AveragePool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[2],
)
x = np.random.randn(1, 3, 32).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = [2]
strides = [1]
out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
padded = x
y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0], 'AVG')

expect(node, inputs=[x], outputs=[y], name='test_averagepool_1d_default')
```

</details>
<details>
<summary>averagepool_2d_ceil</summary>

```python
"""
input_shape: [1, 1, 4, 4]
output_shape: [1, 1, 2, 2]
"""
node = onnx.helper.make_node(
    'AveragePool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[3, 3],
    strides=[2, 2],
    ceil_mode=True
)
x = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]]).astype(np.float32)
y = np.array([[[
    [6, 7.5],
    [12, 13.5]]]]).astype(np.float32)

expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_ceil')
```

</details>
<details>
<summary>averagepool_2d_default</summary>

```python
"""
input_shape: [1, 3, 32, 32]
output_shape: [1, 3, 31, 31]
"""
node = onnx.helper.make_node(
    'AveragePool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[2, 2],
)
x = np.random.randn(1, 3, 32, 32).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = (2, 2)
strides = (1, 1)
out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
padded = x
y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'AVG')

expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_default')
```

</details>
<details>
<summary>averagepool_2d_pads</summary>

```python
"""
input_shape: [1, 3, 28, 28]
output_shape: [1, 3, 30, 30]
pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
"""
node = onnx.helper.make_node(
    'AveragePool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[3, 3],
    pads=[2, 2, 2, 2]
)
x = np.random.randn(1, 3, 28, 28).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = (3, 3)
strides = (1, 1)
pad_bottom = 2
pad_top = 2
pad_right = 2
pad_left = 2
pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
out_shape = get_output_shape('VALID', np.add(x_shape[2:], pad_shape), kernel_shape, strides)
padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                constant_values=np.nan)
y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG')

expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_pads')
```

</details>
<details>
<summary>averagepool_2d_pads_count_include_pad</summary>

```python
"""
input_shape: [1, 3, 28, 28]
output_shape: [1, 3, 30, 30]
pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
"""
node = onnx.helper.make_node(
    'AveragePool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[3, 3],
    pads=[2, 2, 2, 2],
    count_include_pad=1,
)
x = np.random.randn(1, 3, 28, 28).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = (3, 3)
strides = (1, 1)
pad_bottom = 2
pad_top = 2
pad_right = 2
pad_left = 2
pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
out_shape = get_output_shape('VALID', np.add(x_shape[2:], pad_shape), kernel_shape, strides)
padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                constant_values=0)
y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG', count_include_pad=1)

expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_pads_count_include_pad')
```

</details>
<details>
<summary>averagepool_2d_precomputed_pads</summary>

```python
"""
input_shape: [1, 1, 5, 5]
output_shape: [1, 1, 5, 5]
pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
"""
node = onnx.helper.make_node(
    'AveragePool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[5, 5],
    pads=[2, 2, 2, 2]

)
x = np.array([[[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
]]]).astype(np.float32)
y = np.array([[[[7, 7.5, 8, 8.5, 9],
                [9.5, 10, 10.5, 11, 11.5],
                [12, 12.5, 13, 13.5, 14],
                [14.5, 15, 15.5, 16, 16.5],
                [17, 17.5, 18, 18.5, 19]]]]).astype(np.float32)

expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_precomputed_pads')
```

</details>
<details>
<summary>averagepool_2d_precomputed_pads_count_include_pad</summary>

```python
"""
input_shape: [1, 1, 5, 5]
output_shape: [1, 1, 5, 5]
pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
"""
node = onnx.helper.make_node(
    'AveragePool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[5, 5],
    pads=[2, 2, 2, 2],
    count_include_pad=1
)
x = np.array([[[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
]]]).astype(np.float32)
y = np.array([[[[2.5200, 3.6000, 4.8000, 4.0800, 3.2400],
                [4.5600, 6.4000, 8.4000, 7.0400, 5.5200],
                [7.2000, 10.0000, 13.0000, 10.8000, 8.4000],
                [6.9600, 9.6000, 12.4000, 10.2400, 7.9200],
                [6.1200, 8.4000, 10.8000, 8.8800, 6.8400]]]]).astype(np.float32)

expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_precomputed_pads_count_include_pad')
```

</details>
<details>
<summary>averagepool_2d_precomputed_same_upper</summary>

```python
"""
input_shape: [1, 1, 5, 5]
output_shape: [1, 1, 3, 3]
pad_shape: [2, 2] -> [1, 1, 1, 1] by axis
"""
node = onnx.helper.make_node(
    'AveragePool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[3, 3],
    strides=[2, 2],
    auto_pad='SAME_UPPER'
)
x = np.array([[[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
]]]).astype(np.float32)
y = np.array([[[[4, 5.5, 7],
                [11.5, 13, 14.5],
                [19, 20.5, 22]]]]).astype(np.float32)

expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_precomputed_same_upper')
```

</details>
<details>
<summary>averagepool_2d_precomputed_strides</summary>

```python
"""
input_shape: [1, 1, 5, 5]
output_shape: [1, 1, 2, 2]
"""
node = onnx.helper.make_node(
    'AveragePool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[2, 2],
    strides=[2, 2]
)
x = np.array([[[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
]]]).astype(np.float32)
y = np.array([[[[4, 6],
                [14, 16]]]]).astype(np.float32)

expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_precomputed_strides')
```

</details>
<details>
<summary>averagepool_2d_same_lower</summary>

```python
"""
input_shape: [1, 3, 32, 32]
output_shape: [1, 3, 32, 32]
pad_shape: [1, 1] -> [1, 0, 1, 0] by axis
"""
node = onnx.helper.make_node(
    'AveragePool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[2, 2],
    auto_pad='SAME_LOWER'
)
x = np.random.randn(1, 3, 32, 32).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = (2, 2)
strides = (1, 1)
out_shape = get_output_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides)
pad_shape = get_pad_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides, out_shape)
pad_bottom = pad_shape[0] // 2
pad_top = pad_shape[0] - pad_bottom
pad_right = pad_shape[1] // 2
pad_left = pad_shape[1] - pad_right
padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                constant_values=np.nan)
y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG')

expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_same_lower')
```

</details>
<details>
<summary>averagepool_2d_same_upper</summary>

```python
"""
input_shape: [1, 3, 32, 32]
output_shape: [1, 3, 32, 32]
pad_shape: [1, 1] -> [0, 1, 0, 1] by axis
"""
node = onnx.helper.make_node(
    'AveragePool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[2, 2],
    auto_pad='SAME_UPPER'
)
x = np.random.randn(1, 3, 32, 32).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = (2, 2)
strides = (1, 1)
out_shape = get_output_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides)
pad_shape = get_pad_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides, out_shape)
pad_top = pad_shape[0] // 2
pad_bottom = pad_shape[0] - pad_top
pad_left = pad_shape[1] // 2
pad_right = pad_shape[1] - pad_left
padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                constant_values=np.nan)
y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'AVG')

expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_same_upper')
```

</details>
<details>
<summary>averagepool_2d_strides</summary>

```python
"""
input_shape: [1, 3, 32, 32]
output_shape: [1, 3, 10, 10]
"""
node = onnx.helper.make_node(
    'AveragePool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[5, 5],
    strides=[3, 3]
)
x = np.random.randn(1, 3, 32, 32).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = (5, 5)
strides = (3, 3)
out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
padded = x
y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'AVG')

expect(node, inputs=[x], outputs=[y], name='test_averagepool_2d_strides')
```

</details>
<details>
<summary>averagepool_3d_default</summary>

```python
"""
input_shape: [1, 3, 32, 32, 32]
output_shape: [1, 3, 31, 31, 31]
"""
node = onnx.helper.make_node(
    'AveragePool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[2, 2, 2],
)
x = np.random.randn(1, 3, 32, 32, 32).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = [2, 2, 2]
strides = [1, 1, 1]
out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
padded = x
y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0, 0, 0], 'AVG')

expect(node, inputs=[x], outputs=[y], name='test_averagepool_3d_default')
```

</details>


### BatchNormalization
There are 2 test cases, listed as following:
<details>
<summary>batchnormalization</summary>

```python
# input size: (2, 3, 4, 5)
x = np.random.randn(2, 3, 4, 5).astype(np.float32)
s = np.random.randn(3).astype(np.float32)
bias = np.random.randn(3).astype(np.float32)
mean = np.random.randn(3).astype(np.float32)
var = np.random.rand(3).astype(np.float32)
y = _batchnorm_test_mode(x, s, bias, mean, var).astype(np.float32)

node = onnx.helper.make_node(
    'BatchNormalization',
    inputs=['x', 's', 'bias', 'mean', 'var'],
    outputs=['y'],
)

# output size: (2, 3, 4, 5)
expect(node, inputs=[x, s, bias, mean, var], outputs=[y],
       name='test_batchnorm_example')

# input size: (2, 3, 4, 5)
x = np.random.randn(2, 3, 4, 5).astype(np.float32)
s = np.random.randn(3).astype(np.float32)
bias = np.random.randn(3).astype(np.float32)
mean = np.random.randn(3).astype(np.float32)
var = np.random.rand(3).astype(np.float32)
epsilon = 1e-2
y = _batchnorm_test_mode(x, s, bias, mean, var, epsilon).astype(np.float32)

node = onnx.helper.make_node(
    'BatchNormalization',
    inputs=['x', 's', 'bias', 'mean', 'var'],
    outputs=['y'],
    epsilon=epsilon,
)

# output size: (2, 3, 4, 5)
expect(node, inputs=[x, s, bias, mean, var], outputs=[y],
       name='test_batchnorm_epsilon')
```

</details>
<details>
<summary>train</summary>

```python
# input size: (2, 3, 4, 5)
x = np.random.randn(2, 3, 4, 5).astype(np.float32)
s = np.random.randn(3).astype(np.float32)
bias = np.random.randn(3).astype(np.float32)
mean = np.random.randn(3).astype(np.float32)
var = np.random.rand(3).astype(np.float32)
# using np.bool(1) while generating test data with "'bool' object has no attribute 'dtype'"
# working around by using np.byte(1).astype(bool)
training_mode = 1
y, output_mean, output_var = _batchnorm_training_mode(x, s, bias, mean, var)

node = onnx.helper.make_node(
    'BatchNormalization',
    inputs=['x', 's', 'bias', 'mean', 'var'],
    outputs=['y', 'output_mean', 'output_var'],
    training_mode=training_mode
)

# output size: (2, 3, 4, 5)
expect(node, inputs=[x, s, bias, mean, var],
       outputs=[y, output_mean, output_var],
       name='test_batchnorm_example_training_mode')

# input size: (2, 3, 4, 5)
x = np.random.randn(2, 3, 4, 5).astype(np.float32)
s = np.random.randn(3).astype(np.float32)
bias = np.random.randn(3).astype(np.float32)
mean = np.random.randn(3).astype(np.float32)
var = np.random.rand(3).astype(np.float32)
training_mode = 1
momentum = 0.9
epsilon = 1e-2
y, output_mean, output_var = _batchnorm_training_mode(x, s, bias, mean, var, momentum,
                                                      epsilon)

node = onnx.helper.make_node(
    'BatchNormalization',
    inputs=['x', 's', 'bias', 'mean', 'var'],
    outputs=['y', 'output_mean', 'output_var'],
    epsilon=epsilon,
    training_mode=training_mode
)

# output size: (2, 3, 4, 5)
expect(node, inputs=[x, s, bias, mean, var],
       outputs=[y, output_mean, output_var],
       name='test_batchnorm_epsilon_training_mode')
```

</details>


### Bernoulli
There are 3 test cases, listed as following:
<details>
<summary>bernoulli_with_dtype</summary>

```python
node = onnx.helper.make_node(
    'Bernoulli',
    inputs=['x'],
    outputs=['y'],
    dtype=onnx.TensorProto.DOUBLE,
)

x = np.random.uniform(0.0, 1.0, 10).astype(np.float32)
y = bernoulli_reference_implementation(x, np.float64)
expect(node, inputs=[x], outputs=[y], name='test_bernoulli_double')
```

</details>
<details>
<summary>bernoulli_with_seed</summary>

```python
seed = np.float(0)
node = onnx.helper.make_node(
    'Bernoulli',
    inputs=['x'],
    outputs=['y'],
    seed=seed,
)

x = np.random.uniform(0.0, 1.0, 10).astype(np.float32)
y = bernoulli_reference_implementation(x, np.float32)
expect(node, inputs=[x], outputs=[y], name='test_bernoulli_seed')
```

</details>
<details>
<summary>bernoulli_without_dtype</summary>

```python
node = onnx.helper.make_node(
    'Bernoulli',
    inputs=['x'],
    outputs=['y'],
)

x = np.random.uniform(0.0, 1.0, 10).astype(np.float)
y = bernoulli_reference_implementation(x, np.float)
expect(node, inputs=[x], outputs=[y], name='test_bernoulli')
```

</details>


### BitShift
There are 8 test cases, listed as following:
<details>
<summary>left_unit16</summary>

```python
node = onnx.helper.make_node(
    'BitShift',
    inputs=['x', 'y'],
    outputs=['z'],
    direction="LEFT"
)

x = np.array([16, 4, 1]).astype(np.uint16)
y = np.array([1, 2, 3]).astype(np.uint16)
z = x << y  # expected output [32, 16, 8]
expect(node, inputs=[x, y], outputs=[z],
       name='test_bitshift_left_uint16')
```

</details>
<details>
<summary>left_unit32</summary>

```python
node = onnx.helper.make_node(
    'BitShift',
    inputs=['x', 'y'],
    outputs=['z'],
    direction="LEFT"
)

x = np.array([16, 4, 1]).astype(np.uint32)
y = np.array([1, 2, 3]).astype(np.uint32)
z = x << y  # expected output [32, 16, 8]
expect(node, inputs=[x, y], outputs=[z],
       name='test_bitshift_left_uint32')
```

</details>
<details>
<summary>left_unit64</summary>

```python
node = onnx.helper.make_node(
    'BitShift',
    inputs=['x', 'y'],
    outputs=['z'],
    direction="LEFT"
)

x = np.array([16, 4, 1]).astype(np.uint64)
y = np.array([1, 2, 3]).astype(np.uint64)
z = x << y  # expected output [32, 16, 8]
expect(node, inputs=[x, y], outputs=[z],
       name='test_bitshift_left_uint64')
```

</details>
<details>
<summary>left_unit8</summary>

```python
node = onnx.helper.make_node(
    'BitShift',
    inputs=['x', 'y'],
    outputs=['z'],
    direction="LEFT"
)

x = np.array([16, 4, 1]).astype(np.uint8)
y = np.array([1, 2, 3]).astype(np.uint8)
z = x << y  # expected output [32, 16, 8]
expect(node, inputs=[x, y], outputs=[z],
       name='test_bitshift_left_uint8')
```

</details>
<details>
<summary>right_unit16</summary>

```python
node = onnx.helper.make_node(
    'BitShift',
    inputs=['x', 'y'],
    outputs=['z'],
    direction="RIGHT"
)

x = np.array([16, 4, 1]).astype(np.uint16)
y = np.array([1, 2, 3]).astype(np.uint16)
z = x >> y  # expected output [8, 1, 0]
expect(node, inputs=[x, y], outputs=[z],
       name='test_bitshift_right_uint16')
```

</details>
<details>
<summary>right_unit32</summary>

```python
node = onnx.helper.make_node(
    'BitShift',
    inputs=['x', 'y'],
    outputs=['z'],
    direction="RIGHT"
)

x = np.array([16, 4, 1]).astype(np.uint32)
y = np.array([1, 2, 3]).astype(np.uint32)
z = x >> y  # expected output [8, 1, 0]
expect(node, inputs=[x, y], outputs=[z],
       name='test_bitshift_right_uint32')
```

</details>
<details>
<summary>right_unit64</summary>

```python
node = onnx.helper.make_node(
    'BitShift',
    inputs=['x', 'y'],
    outputs=['z'],
    direction="RIGHT"
)

x = np.array([16, 4, 1]).astype(np.uint64)
y = np.array([1, 2, 3]).astype(np.uint64)
z = x >> y  # expected output [8, 1, 0]
expect(node, inputs=[x, y], outputs=[z],
       name='test_bitshift_right_uint64')
```

</details>
<details>
<summary>right_unit8</summary>

```python
node = onnx.helper.make_node(
    'BitShift',
    inputs=['x', 'y'],
    outputs=['z'],
    direction="RIGHT"
)

x = np.array([16, 4, 1]).astype(np.uint8)
y = np.array([1, 2, 3]).astype(np.uint8)
z = x >> y  # expected output [8, 1, 0]
expect(node, inputs=[x, y], outputs=[z],
       name='test_bitshift_right_uint8')
```

</details>


### BlackmanWindow
There are 1 test cases, listed as following:
<details>
<summary>blackmanwindow</summary>

```python
# Test periodic window
node = onnx.helper.make_node(
    'BlackmanWindow',
    inputs=['x'],
    outputs=['y'],
)
size = np.int32(10)
a0 = .42
a1 = -.5
a2 = .08
y = a0
y += a1 * np.cos(2 * 3.1415 * np.arange(0, size, 1, dtype=np.float32) / size)
y += a2 * np.cos(4 * 3.1415 * np.arange(0, size, 1, dtype=np.float32) / size)
expect(node, inputs=[size], outputs=[y],
       name='test_blackmanwindow')

# Test symmetric window
node = onnx.helper.make_node(
    'BlackmanWindow',
    inputs=['x'],
    outputs=['y'],
    periodic=0
)
size = np.int32(10)
a0 = .42
a1 = -.5
a2 = .08
y = a0
y += a1 * np.cos(2 * 3.1415 * np.arange(0, size, 1, dtype=np.float32) / (size - 1))
y += a2 * np.cos(4 * 3.1415 * np.arange(0, size, 1, dtype=np.float32) / (size - 1))
expect(node, inputs=[size], outputs=[y],
       name='test_blackmanwindow_symmetric')
```

</details>


### Cast
There are 1 test cases, listed as following:
<details>
<summary>cast</summary>

```python
shape = (3, 4)
test_cases = [
    ('FLOAT', 'FLOAT16'),
    ('FLOAT', 'DOUBLE'),
    ('FLOAT16', 'FLOAT'),
    ('FLOAT16', 'DOUBLE'),
    ('DOUBLE', 'FLOAT'),
    ('DOUBLE', 'FLOAT16'),
    ('FLOAT', 'STRING'),
    ('STRING', 'FLOAT'),
    ('FLOAT', 'BFLOAT16'),
    ('BFLOAT16', 'FLOAT'),
]

for from_type, to_type in test_cases:
    input_type_proto = None
    output_type_proto = None
    if 'BFLOAT16' == from_type or 'BFLOAT16' == to_type:
        np_fp32 = np.array(['0.47892547', '0.48033667', '0.49968487', '0.81910545',
            '0.47031248', '0.816468', '0.21087195', '0.7229038',
            'NaN', 'INF', '+INF', '-INF'], dtype=np.float32)
        little_endisan = sys.byteorder == 'little'
        np_uint16_view = np_fp32.view(dtype=np.uint16)
        np_bfp16 = np_uint16_view[1::2] if little_endisan else np_uint16_view[0::2]
        if 'BFLOAT16' == to_type:
            assert from_type == 'FLOAT'
            input = np_fp32.reshape([3, 4])
            output = np_bfp16.reshape([3, 4])
            input_type_proto = onnx.helper.make_tensor_type_proto(int(TensorProto.FLOAT), input.shape)
            output_type_proto = onnx.helper.make_tensor_type_proto(int(TensorProto.BFLOAT16), output.shape)
        else:
            assert to_type == 'FLOAT'
            input = np_bfp16.reshape([3, 4])
            #convert bfloat to FLOAT
            np_fp32_zeros = np.zeros((len(np_bfp16) * 2,), dtype=np.uint16)
            if little_endisan:
                np_fp32_zeros[1::2] = np_bfp16
            else:
                np_fp32_zeros[0::2] = np_bfp16
            np_fp32_from_bfloat = np_fp32_zeros.view(dtype=np.float32)
            output = np_fp32_from_bfloat.reshape([3, 4])
            input_type_proto = onnx.helper.make_tensor_type_proto(int(TensorProto.BFLOAT16), input.shape)
            output_type_proto = onnx.helper.make_tensor_type_proto(int(TensorProto.FLOAT), output.shape)
    elif 'STRING' != from_type:
        input = np.random.random_sample(shape).astype(
            TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, from_type)])
        if ('STRING' == to_type):
            # Converting input to str, then give it object dtype for generating script
            ss = []
            for i in input.flatten():
                s = str(i).encode('utf-8')
                su = s.decode('utf-8')
                ss.append(su)

            output = np.array(ss).astype(object).reshape([3, 4])
        else:
            output = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])
    else:
        input = np.array(['0.47892547', '0.48033667', '0.49968487', '0.81910545',
            '0.47031248', '0.816468', '0.21087195', '0.7229038',
            'NaN', 'INF', '+INF', '-INF'], dtype=np.dtype(object)).reshape([3, 4])
        output = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])
    node = onnx.helper.make_node(
        'Cast',
        inputs=['input'],
        outputs=['output'],
        to=getattr(TensorProto, to_type),
    )
    if input_type_proto and output_type_proto:
        expect(node, inputs=[input], outputs=[output],
                   name='test_cast_' + from_type + '_to_' + to_type,
                   input_type_protos=[input_type_proto],
                   output_type_protos=[output_type_proto])
    else:
        expect(node, inputs=[input], outputs=[output],
                   name='test_cast_' + from_type + '_to_' + to_type)
```

</details>


### CastLike
There are 1 test cases, listed as following:
<details>
<summary>castlike</summary>

```python
shape = (3, 4)
test_cases = [
    ('FLOAT', 'FLOAT16'),
    ('FLOAT', 'DOUBLE'),
    ('FLOAT16', 'FLOAT'),
    ('FLOAT16', 'DOUBLE'),
    ('DOUBLE', 'FLOAT'),
    ('DOUBLE', 'FLOAT16'),
    ('FLOAT', 'STRING'),
    ('STRING', 'FLOAT'),
    ('FLOAT', 'BFLOAT16'),
    ('BFLOAT16', 'FLOAT'),
]

for from_type, to_type in test_cases:
    input_type_proto = None
    output_type_proto = None
    if 'BFLOAT16' == from_type or 'BFLOAT16' == to_type:
        np_fp32 = np.array(['0.47892547', '0.48033667', '0.49968487', '0.81910545',
            '0.47031248', '0.816468', '0.21087195', '0.7229038',
            'NaN', 'INF', '+INF', '-INF'], dtype=np.float32)
        little_endisan = sys.byteorder == 'little'
        np_uint16_view = np_fp32.view(dtype=np.uint16)
        np_bfp16 = np_uint16_view[1::2] if little_endisan else np_uint16_view[0::2]
        if 'BFLOAT16' == to_type:
            assert from_type == 'FLOAT'
            input = np_fp32.reshape([3, 4])
            output = np_bfp16.reshape([3, 4])
            input_type_proto = onnx.helper.make_tensor_type_proto(int(TensorProto.FLOAT), input.shape)
            output_type_proto = onnx.helper.make_tensor_type_proto(int(TensorProto.BFLOAT16), output.shape)
        else:
            assert to_type == 'FLOAT'
            input = np_bfp16.reshape([3, 4])
            #convert bfloat to FLOAT
            np_fp32_zeros = np.zeros((len(np_bfp16) * 2,), dtype=np.uint16)
            if little_endisan:
                np_fp32_zeros[1::2] = np_bfp16
            else:
                np_fp32_zeros[0::2] = np_bfp16
            np_fp32_from_bfloat = np_fp32_zeros.view(dtype=np.float32)
            output = np_fp32_from_bfloat.reshape([3, 4])
            input_type_proto = onnx.helper.make_tensor_type_proto(int(TensorProto.BFLOAT16), input.shape)
            output_type_proto = onnx.helper.make_tensor_type_proto(int(TensorProto.FLOAT), output.shape)
    elif 'STRING' != from_type:
        input = np.random.random_sample(shape).astype(
            TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, from_type)])
        if ('STRING' == to_type):
            # Converting input to str, then give it np.object dtype for generating script
            ss = []
            for i in input.flatten():
                s = str(i).encode('utf-8')
                su = s.decode('utf-8')
                ss.append(su)

            output = np.array(ss).astype(np.object).reshape([3, 4])
        else:
            output = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])
    else:
        input = np.array(['0.47892547', '0.48033667', '0.49968487', '0.81910545',
            '0.47031248', '0.816468', '0.21087195', '0.7229038',
            'NaN', 'INF', '+INF', '-INF'], dtype=np.dtype(np.object)).reshape([3, 4])
        output = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])
    like = output.flatten()[0:1]
    node = onnx.helper.make_node(
        'CastLike',
        inputs=['input', 'like'],
        outputs=['output'],
    )
    if input_type_proto and output_type_proto:
        expect(node, inputs=[input, like], outputs=[output],
                   name='test_castlike_' + from_type + '_to_' + to_type,
                   input_type_protos=[input_type_proto, output_type_proto],
                   output_type_protos=[output_type_proto])
    else:
        expect(node, inputs=[input, like], outputs=[output],
                   name='test_castlike_' + from_type + '_to_' + to_type)
```

</details>


### Ceil
There are 1 test cases, listed as following:
<details>
<summary>ceil</summary>

```python
node = onnx.helper.make_node(
    'Ceil',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-1.5, 1.2]).astype(np.float32)
y = np.ceil(x)  # expected output [-1., 2.]
expect(node, inputs=[x], outputs=[y],
       name='test_ceil_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.ceil(x)
expect(node, inputs=[x], outputs=[y],
       name='test_ceil')
```

</details>


### Celu
There are 1 test cases, listed as following:
<details>
<summary>celu</summary>

```python
alpha = 2.0
node = onnx.helper.make_node(
    'Celu',
    inputs=['X'],
    outputs=['Y'],
    alpha=alpha,
)

input_data = np.array([[[[0.8439683], [0.5665144], [0.05836735]],
    [[0.02916367], [0.12964272], [0.5060197]],
    [[0.79538304], [0.9411346], [0.9546573]]],
    [[[0.17730942], [0.46192095], [0.26480448]],
    [[0.6746842], [0.01665257], [0.62473077]],
    [[0.9240844], [0.9722341], [0.11965699]]],
    [[[0.41356155], [0.9129373], [0.59330076]],
    [[0.81929934], [0.7862604], [0.11799799]],
    [[0.69248444], [0.54119414], [0.07513223]]]], dtype=np.float32)

# Calculate expected output data
positive_input = np.maximum(0, input_data)
negative_input = np.minimum(0, alpha * (np.exp(input_data / alpha) - 1))
expected_output = positive_input + negative_input

expect(node, inputs=[input_data], outputs=[expected_output],
       name='test_celu')
```

</details>


### Clip
There are 3 test cases, listed as following:
<details>
<summary>clip</summary>

```python
node = onnx.helper.make_node(
    'Clip',
    inputs=['x', 'min', 'max'],
    outputs=['y'],
)

x = np.array([-2, 0, 2]).astype(np.float32)
min_val = np.float32(-1)
max_val = np.float32(1)
y = np.clip(x, min_val, max_val)  # expected output [-1., 0., 1.]
expect(node, inputs=[x, min_val, max_val], outputs=[y],
       name='test_clip_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, min_val, max_val)
expect(node, inputs=[x, min_val, max_val], outputs=[y],
       name='test_clip')
node = onnx.helper.make_node(
    'Clip',
    inputs=['x', 'min', 'max'],
    outputs=['y'],
)

min_val = np.float32(-5)
max_val = np.float32(5)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.array([-1, 0, 1]).astype(np.float32)
expect(node, inputs=[x, min_val, max_val], outputs=[y],
       name='test_clip_inbounds')

x = np.array([-6, 0, 6]).astype(np.float32)
y = np.array([-5, 0, 5]).astype(np.float32)
expect(node, inputs=[x, min_val, max_val], outputs=[y],
       name='test_clip_outbounds')

x = np.array([-1, 0, 6]).astype(np.float32)
y = np.array([-1, 0, 5]).astype(np.float32)
expect(node, inputs=[x, min_val, max_val], outputs=[y],
       name='test_clip_splitbounds')
```

</details>
<details>
<summary>clip_default</summary>

```python
node = onnx.helper.make_node(
    'Clip',
    inputs=['x', 'min'],
    outputs=['y'],
)
min_val = np.float32(0)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, min_val, np.inf)
expect(node, inputs=[x, min_val], outputs=[y],
       name='test_clip_default_min')

no_min = ""  # optional input, not supplied
node = onnx.helper.make_node(
    'Clip',
    inputs=['x', no_min, 'max'],
    outputs=['y'],
)
max_val = np.float32(0)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, -np.inf, max_val)
expect(node, inputs=[x, max_val], outputs=[y],
       name='test_clip_default_max')

no_max = ""  # optional input, not supplied
node = onnx.helper.make_node(
    'Clip',
    inputs=['x', no_min, no_max],
    outputs=['y'],
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.array([-1, 0, 1]).astype(np.float32)
expect(node, inputs=[x], outputs=[y],
       name='test_clip_default_inbounds')
```

</details>
<details>
<summary>clip_default_int8</summary>

```python
node = onnx.helper.make_node(
    'Clip',
    inputs=['x', 'min'],
    outputs=['y'],
)
min_val = np.int8(0)
x = np.random.randn(3, 4, 5).astype(np.int8)
y = np.clip(x, min_val, np.iinfo(np.int8).max)
expect(node, inputs=[x, min_val], outputs=[y],
       name='test_clip_default_int8_min')

no_min = ""  # optional input, not supplied
node = onnx.helper.make_node(
    'Clip',
    inputs=['x', no_min, 'max'],
    outputs=['y'],
)
max_val = np.int8(0)
x = np.random.randn(3, 4, 5).astype(np.int8)
y = np.clip(x, np.iinfo(np.int8).min, max_val)
expect(node, inputs=[x, max_val], outputs=[y],
       name='test_clip_default_int8_max')

no_max = ""  # optional input, not supplied
node = onnx.helper.make_node(
    'Clip',
    inputs=['x', no_min, no_max],
    outputs=['y'],
)

x = np.array([-1, 0, 1]).astype(np.int8)
y = np.array([-1, 0, 1]).astype(np.int8)
expect(node, inputs=[x], outputs=[y],
       name='test_clip_default_int8_inbounds')
```

</details>


### Compress
There are 4 test cases, listed as following:
<details>
<summary>compress_0</summary>

```python
node = onnx.helper.make_node(
    'Compress',
    inputs=['input', 'condition'],
    outputs=['output'],
    axis=0,
)
input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
condition = np.array([0, 1, 1])
output = np.compress(condition, input, axis=0)
#print(output)
#[[ 3.  4.]
# [ 5.  6.]]

expect(node, inputs=[input, condition.astype(bool)], outputs=[output],
       name='test_compress_0')
```

</details>
<details>
<summary>compress_1</summary>

```python
node = onnx.helper.make_node(
    'Compress',
    inputs=['input', 'condition'],
    outputs=['output'],
    axis=1,
)
input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
condition = np.array([0, 1])
output = np.compress(condition, input, axis=1)
#print(output)
#[[ 2.]
# [ 4.]
# [ 6.]]

expect(node, inputs=[input, condition.astype(bool)], outputs=[output],
       name='test_compress_1')
```

</details>
<details>
<summary>compress_default_axis</summary>

```python
node = onnx.helper.make_node(
    'Compress',
    inputs=['input', 'condition'],
    outputs=['output'],
)
input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
condition = np.array([0, 1, 0, 0, 1])
output = np.compress(condition, input)
#print(output)
#[ 2., 5.]

expect(node, inputs=[input, condition.astype(bool)], outputs=[output],
       name='test_compress_default_axis')
```

</details>
<details>
<summary>compress_negative_axis</summary>

```python
node = onnx.helper.make_node(
    'Compress',
    inputs=['input', 'condition'],
    outputs=['output'],
    axis=-1,
)
input = np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32)
condition = np.array([0, 1])
output = np.compress(condition, input, axis=-1)
# print(output)
#[[ 2.]
# [ 4.]
# [ 6.]]
expect(node, inputs=[input, condition.astype(bool)], outputs=[output],
       name='test_compress_negative_axis')
```

</details>


### Concat
There are 1 test cases, listed as following:
<details>
<summary>concat</summary>

```python
test_cases: Dict[str, Sequence[Any]] = {
    '1d': ([1, 2],
           [3, 4]),
    '2d': ([[1, 2], [3, 4]],
           [[5, 6], [7, 8]]),
    '3d': ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
           [[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
}

for test_case, values_ in test_cases.items():
    values = [np.asarray(v, dtype=np.float32) for v in values_]
    for i in range(len(values[0].shape)):
        in_args = ['value' + str(k) for k in range(len(values))]
        node = onnx.helper.make_node(
            'Concat',
            inputs=[s for s in in_args],
            outputs=['output'],
            axis=i
        )
        output = np.concatenate(values, i)
        expect(node, inputs=[v for v in values], outputs=[output],
               name='test_concat_' + test_case + '_axis_' + str(i))

    for i in range(-len(values[0].shape), 0):
        in_args = ['value' + str(k) for k in range(len(values))]
        node = onnx.helper.make_node(
            'Concat',
            inputs=[s for s in in_args],
            outputs=['output'],
            axis=i
        )
        output = np.concatenate(values, i)
        expect(node, inputs=[v for v in values], outputs=[output],
               name='test_concat_' + test_case + '_axis_negative_' + str(abs(i)))
```

</details>


### Constant
There are 1 test cases, listed as following:
<details>
<summary>constant</summary>

```python
values = np.random.randn(5, 5).astype(np.float32)
node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['values'],
    value=onnx.helper.make_tensor(
        name='const_tensor',
        data_type=onnx.TensorProto.FLOAT,
        dims=values.shape,
        vals=values.flatten().astype(float),
    ),
)

expect(node, inputs=[], outputs=[values],
       name='test_constant')
```

</details>


### ConstantOfShape
There are 3 test cases, listed as following:
<details>
<summary>float_ones</summary>

```python
x = np.array([4, 3, 2]).astype(np.int64)
tensor_value = onnx.helper.make_tensor("value", onnx.TensorProto.FLOAT,
                                       [1], [1])
node = onnx.helper.make_node(
    'ConstantOfShape',
    inputs=['x'],
    outputs=['y'],
    value=tensor_value,
)

y = np.ones(x, dtype=np.float32)
expect(node, inputs=[x], outputs=[y],
       name='test_constantofshape_float_ones')
```

</details>
<details>
<summary>int32_shape_zero</summary>

```python
x = np.array([0, ]).astype(np.int64)
tensor_value = onnx.helper.make_tensor("value", onnx.TensorProto.INT32,
                                       [1], [0])
node = onnx.helper.make_node(
    'ConstantOfShape',
    inputs=['x'],
    outputs=['y'],
    value=tensor_value,
)
y = np.zeros(x, dtype=np.int32)
expect(node, inputs=[x], outputs=[y],
       name='test_constantofshape_int_shape_zero')
```

</details>
<details>
<summary>int32_zeros</summary>

```python
x = np.array([10, 6]).astype(np.int64)
tensor_value = onnx.helper.make_tensor("value", onnx.TensorProto.INT32,
                                       [1], [0])
node = onnx.helper.make_node(
    'ConstantOfShape',
    inputs=['x'],
    outputs=['y'],
    value=tensor_value,
)
y = np.zeros(x, dtype=np.int32)
expect(node, inputs=[x], outputs=[y],
       name='test_constantofshape_int_zeros')
```

</details>


### Conv
There are 3 test cases, listed as following:
<details>
<summary>conv</summary>

```python

x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                [5., 6., 7., 8., 9.],
                [10., 11., 12., 13., 14.],
                [15., 16., 17., 18., 19.],
                [20., 21., 22., 23., 24.]]]]).astype(np.float32)
W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                [1., 1., 1.],
                [1., 1., 1.]]]]).astype(np.float32)

# Convolution with padding
node_with_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
    pads=[1, 1, 1, 1],
)
y_with_padding = np.array([[[[12., 21., 27., 33., 24.],  # (1, 1, 5, 5) output tensor
                             [33., 54., 63., 72., 51.],
                             [63., 99., 108., 117., 81.],
                             [93., 144., 153., 162., 111.],
                             [72., 111., 117., 123., 84.]]]]).astype(np.float32)
expect(node_with_padding, inputs=[x, W], outputs=[y_with_padding],
       name='test_basic_conv_with_padding')

# Convolution without padding
node_without_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
    pads=[0, 0, 0, 0],
)
y_without_padding = np.array([[[[54., 63., 72.],  # (1, 1, 3, 3) output tensor
                                [99., 108., 117.],
                                [144., 153., 162.]]]]).astype(np.float32)
expect(node_without_padding, inputs=[x, W], outputs=[y_without_padding],
       name='test_basic_conv_without_padding')
```

</details>
<details>
<summary>conv_with_autopad_same</summary>

```python

x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 5, 5) input tensor
                [5., 6., 7., 8., 9.],
                [10., 11., 12., 13., 14.],
                [15., 16., 17., 18., 19.],
                [20., 21., 22., 23., 24.]]]]).astype(np.float32)
W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                [1., 1., 1.],
                [1., 1., 1.]]]]).astype(np.float32)

# Convolution with auto_pad='SAME_LOWER' and strides=2
node = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    auto_pad='SAME_LOWER',
    kernel_shape=[3, 3],
    strides=[2, 2],
)
y = np.array([[[[12., 27., 24.],
             [63., 108., 81.],
             [72., 117., 84.]]]]).astype(np.float32)
expect(node, inputs=[x, W], outputs=[y],
       name='test_conv_with_autopad_same')
```

</details>
<details>
<summary>conv_with_strides</summary>

```python

x = np.array([[[[0., 1., 2., 3., 4.],  # (1, 1, 7, 5) input tensor
                [5., 6., 7., 8., 9.],
                [10., 11., 12., 13., 14.],
                [15., 16., 17., 18., 19.],
                [20., 21., 22., 23., 24.],
                [25., 26., 27., 28., 29.],
                [30., 31., 32., 33., 34.]]]]).astype(np.float32)
W = np.array([[[[1., 1., 1.],  # (1, 1, 3, 3) tensor for convolution weights
                [1., 1., 1.],
                [1., 1., 1.]]]]).astype(np.float32)

# Convolution with strides=2 and padding
node_with_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    pads=[1, 1, 1, 1],
    strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
)
y_with_padding = np.array([[[[12., 27., 24.],  # (1, 1, 4, 3) output tensor
                             [63., 108., 81.],
                             [123., 198., 141.],
                             [112., 177., 124.]]]]).astype(np.float32)
expect(node_with_padding, inputs=[x, W], outputs=[y_with_padding],
       name='test_conv_with_strides_padding')

# Convolution with strides=2 and no padding
node_without_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    pads=[0, 0, 0, 0],
    strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
)
y_without_padding = np.array([[[[54., 72.],  # (1, 1, 3, 2) output tensor
                                [144., 162.],
                                [234., 252.]]]]).astype(np.float32)
expect(node_without_padding, inputs=[x, W], outputs=[y_without_padding],
       name='test_conv_with_strides_no_padding')

# Convolution with strides=2 and padding only along one dimension (the H dimension in NxCxHxW tensor)
node_with_asymmetric_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    pads=[1, 0, 1, 0],
    strides=[2, 2],  # Default values for other attributes: dilations=[1, 1], groups=1
)
y_with_asymmetric_padding = np.array([[[[21., 33.],  # (1, 1, 4, 2) output tensor
                                        [99., 117.],
                                        [189., 207.],
                                        [171., 183.]]]]).astype(np.float32)
expect(node_with_asymmetric_padding, inputs=[x, W], outputs=[y_with_asymmetric_padding],
       name='test_conv_with_strides_and_asymmetric_padding')
```

</details>


### ConvInteger
There are 2 test cases, listed as following:
<details>
<summary>with_padding</summary>

```python

x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.uint8).reshape((1, 1, 3, 3))
x_zero_point = np.uint8(1)
w = np.array([1, 1, 1, 1]).astype(np.uint8).reshape((1, 1, 2, 2))

y = np.array([1, 3, 5, 3, 5, 12, 16, 9, 11, 24, 28, 15, 7, 15, 17, 9]).astype(np.int32).reshape((1, 1, 4, 4))

# ConvInteger with padding
convinteger_node_with_padding = onnx.helper.make_node('ConvInteger',
    inputs=['x', 'w', 'x_zero_point'],
    outputs=['y'],
    pads=[1, 1, 1, 1],)

expect(convinteger_node_with_padding, inputs=[x, w, x_zero_point], outputs=[y],
       name='test_convinteger_with_padding')
```

</details>
<details>
<summary>without_padding</summary>

```python

x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]).astype(np.uint8).reshape((1, 1, 3, 3))
x_zero_point = np.uint8(1)
w = np.array([1, 1, 1, 1]).astype(np.uint8).reshape((1, 1, 2, 2))

y = np.array([12, 16, 24, 28]).astype(np.int32).reshape(1, 1, 2, 2)

# ConvInteger without padding
convinteger_node = onnx.helper.make_node('ConvInteger',
    inputs=['x', 'w', 'x_zero_point'],
    outputs=['y'])

expect(convinteger_node, inputs=[x, w, x_zero_point], outputs=[y],
       name='test_convinteger_without_padding')
```

</details>


### ConvTranspose
There are 7 test cases, listed as following:
<details>
<summary>convtranspose</summary>

```python
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
```

</details>
<details>
<summary>convtranspose_1d</summary>

```python
x = np.array([[[0., 1., 2.]]]).astype(np.float32)  # (1, 1, 3)

W = np.array([[[1., 1., 1.],  # (1, 2, 3)
               [1., 1., 1.]]]).astype(np.float32)

node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"])

y = np.array([[[0., 1., 3., 3., 2.],  # (1, 2, 5)
               [0., 1., 3., 3., 2.]]]).astype(np.float32)

expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_1d')
```

</details>
<details>
<summary>convtranspose_3d</summary>

```python
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
```

</details>
<details>
<summary>convtranspose_attributes</summary>

```python
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
                             output_shape=[10, 8])
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
```

</details>
<details>
<summary>convtranspose_autopad_same</summary>

```python
x = np.array([[[[0., 1., 2.],  # (1, 1, 3, 3)
                [3., 4., 5.],
                [6., 7., 8.]]]]).astype(np.float32)

W = np.array([[[[1., 1., 1.],  # (1, 2, 3, 3)
                [1., 1., 1.],
                [1., 1., 1.]],
               [[1., 1., 1.],
                [1., 1., 1.],
                [1., 1., 1.]]]]).astype(np.float32)

node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"], auto_pad="SAME_UPPER", strides=[2, 2])

y = np.array([[[[0., 0., 1., 1., 3., 2.],
                [0., 0., 1., 1., 3., 2.],
                [3., 3., 8., 5., 12., 7.],
                [3., 3., 7., 4., 9., 5.],
                [9., 9., 20., 11., 24., 13.],
                [6., 6., 13., 7., 15., 8.]],

               [[0., 0., 1., 1., 3., 2.],
                [0., 0., 1., 1., 3., 2.],
                [3., 3., 8., 5., 12., 7.],
                [3., 3., 7., 4., 9., 5.],
                [9., 9., 20., 11., 24., 13.],
                [6., 6., 13., 7., 15., 8.]]]]).astype(np.float32)

expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_autopad_same')
```

</details>
<details>
<summary>convtranspose_dilations</summary>

```python
x = np.array([[[[3., 8., 1.],  # (1, 1, 3, 3)
                [9., 5., 7.],
                [3., 2., 6.]]]]).astype(np.float32)
W = np.array([[[[7., 2.],  # (1, 1, 2, 2)
                [1., 9.]]]]).astype(np.float32)

node = onnx.helper.make_node("ConvTranspose", ["X", "W"], ["Y"], dilations=[2, 2])

y = np.array([[[[21., 56., 13., 16., 2.],  # [1, 1, 5, 5]
                [63., 35., 67., 10., 14.],
                [24., 22., 76., 76., 21.],
                [9., 5., 88., 45., 63.],
                [3., 2., 33., 18., 54.]]]]).astype(np.float32)

expect(node, inputs=[x, W], outputs=[y], name='test_convtranspose_dilations')
```

</details>
<details>
<summary>convtranspose_pads</summary>

```python
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
```

</details>


### Cos
There are 1 test cases, listed as following:
<details>
<summary>cos</summary>

```python
node = onnx.helper.make_node(
    'Cos',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.cos(x)
expect(node, inputs=[x], outputs=[y],
       name='test_cos_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.cos(x)
expect(node, inputs=[x], outputs=[y],
       name='test_cos')
```

</details>


### Cosh
There are 1 test cases, listed as following:
<details>
<summary>cosh</summary>

```python
node = onnx.helper.make_node(
    'Cosh',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.cosh(x)  # expected output [1.54308069,  1.,  1.54308069]
expect(node, inputs=[x], outputs=[y],
       name='test_cosh_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.cosh(x)
expect(node, inputs=[x], outputs=[y],
       name='test_cosh')
```

</details>


### CumSum
There are 7 test cases, listed as following:
<details>
<summary>cumsum_1d</summary>

```python
node = onnx.helper.make_node(
    'CumSum',
    inputs=['x', 'axis'],
    outputs=['y']
)
x = np.array([1., 2., 3., 4., 5.]).astype(np.float64)
axis = np.int32(0)
y = np.array([1., 3., 6., 10., 15.]).astype(np.float64)
expect(node, inputs=[x, axis], outputs=[y],
       name='test_cumsum_1d')
```

</details>
<details>
<summary>cumsum_1d_exclusive</summary>

```python
node = onnx.helper.make_node(
    'CumSum',
    inputs=['x', 'axis'],
    outputs=['y'],
    exclusive=1
)
x = np.array([1., 2., 3., 4., 5.]).astype(np.float64)
axis = np.int32(0)
y = np.array([0., 1., 3., 6., 10.]).astype(np.float64)
expect(node, inputs=[x, axis], outputs=[y],
       name='test_cumsum_1d_exclusive')
```

</details>
<details>
<summary>cumsum_1d_reverse</summary>

```python
node = onnx.helper.make_node(
    'CumSum',
    inputs=['x', 'axis'],
    outputs=['y'],
    reverse=1
)
x = np.array([1., 2., 3., 4., 5.]).astype(np.float64)
axis = np.int32(0)
y = np.array([15., 14., 12., 9., 5.]).astype(np.float64)
expect(node, inputs=[x, axis], outputs=[y],
       name='test_cumsum_1d_reverse')
```

</details>
<details>
<summary>cumsum_1d_reverse_exclusive</summary>

```python
node = onnx.helper.make_node(
    'CumSum',
    inputs=['x', 'axis'],
    outputs=['y'],
    reverse=1,
    exclusive=1
)
x = np.array([1., 2., 3., 4., 5.]).astype(np.float64)
axis = np.int32(0)
y = np.array([14., 12., 9., 5., 0.]).astype(np.float64)
expect(node, inputs=[x, axis], outputs=[y],
       name='test_cumsum_1d_reverse_exclusive')
```

</details>
<details>
<summary>cumsum_2d_axis_0</summary>

```python
node = onnx.helper.make_node(
    'CumSum',
    inputs=['x', 'axis'],
    outputs=['y'],
)
x = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float64).reshape((2, 3))
axis = np.int32(0)
y = np.array([1., 2., 3., 5., 7., 9.]).astype(np.float64).reshape((2, 3))
expect(node, inputs=[x, axis], outputs=[y],
       name='test_cumsum_2d_axis_0')
```

</details>
<details>
<summary>cumsum_2d_axis_1</summary>

```python
node = onnx.helper.make_node(
    'CumSum',
    inputs=['x', 'axis'],
    outputs=['y'],
)
x = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float64).reshape((2, 3))
axis = np.int32(1)
y = np.array([1., 3., 6., 4., 9., 15.]).astype(np.float64).reshape((2, 3))
expect(node, inputs=[x, axis], outputs=[y],
       name='test_cumsum_2d_axis_1')
```

</details>
<details>
<summary>cumsum_2d_negative_axis</summary>

```python
node = onnx.helper.make_node(
    'CumSum',
    inputs=['x', 'axis'],
    outputs=['y'],
)
x = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float64).reshape((2, 3))
axis = np.int32(-1)
y = np.array([1., 3., 6., 4., 9., 15.]).astype(np.float64).reshape((2, 3))
expect(node, inputs=[x, axis], outputs=[y],
       name='test_cumsum_2d_negative_axis')
```

</details>


### DFT
There are 1 test cases, listed as following:
<details>
<summary>dft</summary>

```python
node = onnx.helper.make_node(
    'DFT',
    inputs=['x'],
    outputs=['y'],
    axis=1
)
x = np.arange(0, 100).reshape(10, 10).astype(np.float32)
y = np.fft.fft(x, axis=0)

x = x.reshape(1, 10, 10, 1)
y = np.stack((y.real, y.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
expect(node, inputs=[x], outputs=[y],
       name='test_dft')

node = onnx.helper.make_node(
    'DFT',
    inputs=['x'],
    outputs=['y'],
    axis=2
)
x = np.arange(0, 100).reshape(10, 10).astype(np.float32)
y = np.fft.fft(x, axis=1)

x = x.reshape(1, 10, 10, 1)
y = np.stack((y.real, y.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
expect(node, inputs=[x], outputs=[y],
       name='test_dft_axis')

node = onnx.helper.make_node(
    'DFT',
    inputs=['x'],
    outputs=['y'],
    inverse=1,
    axis=1
)
x = np.arange(0, 100, dtype=np.complex64).reshape(10, 10,)
y = np.fft.ifft(x, axis=0)

x = np.stack((x.real, x.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
y = np.stack((y.real, y.imag), axis=2).astype(np.float32).reshape(1, 10, 10, 2)
expect(node, inputs=[x], outputs=[y],
       name='test_dft_inverse')
```

</details>


### DepthToSpace
There are 2 test cases, listed as following:
<details>
<summary>crd_mode_example</summary>

```python
node = onnx.helper.make_node(
    'DepthToSpace',
    inputs=['x'],
    outputs=['y'],
    blocksize=2,
    mode='CRD'
)

# (1, 8, 2, 3) input tensor
x = np.array([[[[0., 1., 2.],
                [3., 4., 5.]],
               [[9., 10., 11.],
                [12., 13., 14.]],
               [[18., 19., 20.],
                [21., 22., 23.]],
               [[27., 28., 29.],
                [30., 31., 32.]],
               [[36., 37., 38.],
                [39., 40., 41.]],
               [[45., 46., 47.],
                [48., 49., 50.]],
               [[54., 55., 56.],
                [57., 58., 59.]],
               [[63., 64., 65.],
                [66., 67., 68.]]]]).astype(np.float32)

# (1, 2, 4, 6) output tensor
y = np.array([[[[0., 9., 1., 10., 2., 11.],
                [18., 27., 19., 28., 20., 29.],
                [3., 12., 4., 13., 5., 14.],
                [21., 30., 22., 31., 23., 32.]],
               [[36., 45., 37., 46., 38., 47.],
                [54., 63., 55., 64., 56., 65.],
                [39., 48., 40., 49., 41., 50.],
                [57., 66., 58., 67., 59., 68.]]]]).astype(np.float32)
expect(node, inputs=[x], outputs=[y],
       name='test_depthtospace_crd_mode_example')
```

</details>
<details>
<summary>default_mode_example</summary>

```python
node = onnx.helper.make_node(
    'DepthToSpace',
    inputs=['x'],
    outputs=['y'],
    blocksize=2,
    mode='DCR'
)

# (1, 8, 2, 3) input tensor
x = np.array([[[[0., 1., 2.],
                [3., 4., 5.]],
               [[9., 10., 11.],
                [12., 13., 14.]],
               [[18., 19., 20.],
                [21., 22., 23.]],
               [[27., 28., 29.],
                [30., 31., 32.]],
               [[36., 37., 38.],
                [39., 40., 41.]],
               [[45., 46., 47.],
                [48., 49., 50.]],
               [[54., 55., 56.],
                [57., 58., 59.]],
               [[63., 64., 65.],
                [66., 67., 68.]]]]).astype(np.float32)

# (1, 2, 4, 6) output tensor
y = np.array([[[[0., 18., 1., 19., 2., 20.],
                [36., 54., 37., 55., 38., 56.],
                [3., 21., 4., 22., 5., 23.],
                [39., 57., 40., 58., 41., 59.]],
               [[9., 27., 10., 28., 11., 29.],
                [45., 63., 46., 64., 47., 65.],
                [12., 30., 13., 31., 14., 32.],
                [48., 66., 49., 67., 50., 68.]]]]).astype(np.float32)
expect(node, inputs=[x], outputs=[y],
       name='test_depthtospace_example')
```

</details>


### DequantizeLinear
There are 2 test cases, listed as following:
<details>
<summary>axis</summary>

```python
node = onnx.helper.make_node('DequantizeLinear',
                             inputs=['x', 'x_scale', 'x_zero_point'],
                             outputs=['y'],)

# 1-D tensor zero point and scale of size equal to axis 1 of the input tensor
x = np.array([[[[3, 89],
                [34, 200],
                [74, 59]],

               [[5, 24],
                [24, 87],
                [32, 13]],

               [[245, 99],
                [4, 142],
                [121, 102]], ], ], dtype=np.uint8)
x_scale = np.array([2, 4, 5], dtype=np.float32)
x_zero_point = np.array([84, 24, 196], dtype=np.uint8)
y = (x.astype(np.float32) - x_zero_point.reshape(1, 3, 1, 1).astype(np.float32)) * x_scale.reshape(1, 3, 1, 1)

expect(node, inputs=[x, x_scale, x_zero_point], outputs=[y],
       name='test_dequantizelinear_axis')
```

</details>
<details>
<summary>dequantizelinear</summary>

```python
node = onnx.helper.make_node('DequantizeLinear',
                             inputs=['x', 'x_scale', 'x_zero_point'],
                             outputs=['y'],)

# scalar zero point and scale
x = np.array([0, 3, 128, 255]).astype(np.uint8)
x_scale = np.float32(2)
x_zero_point = np.uint8(128)
y = np.array([-256, -250, 0, 254], dtype=np.float32)

expect(node, inputs=[x, x_scale, x_zero_point], outputs=[y],
       name='test_dequantizelinear')
```

</details>


### Det
There are 2 test cases, listed as following:
<details>
<summary>2d</summary>

```python
node = onnx.helper.make_node(
    'Det',
    inputs=['x'],
    outputs=['y'],
)

x = np.arange(4).reshape(2, 2).astype(np.float32)
y = np.linalg.det(x)  # expect -2
expect(node, inputs=[x], outputs=[y],
       name='test_det_2d')
```

</details>
<details>
<summary>nd</summary>

```python
node = onnx.helper.make_node(
    'Det',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([[[1, 2], [3, 4]], [[1, 2], [2, 1]], [[1, 3], [3, 1]]]).astype(np.float32)
y = np.linalg.det(x)  # expect array([-2., -3., -8.])
expect(node, inputs=[x], outputs=[y],
       name='test_det_nd')
```

</details>


### Div
There are 2 test cases, listed as following:
<details>
<summary>div</summary>

```python
node = onnx.helper.make_node(
    'Div',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.array([3, 4]).astype(np.float32)
y = np.array([1, 2]).astype(np.float32)
z = x / y  # expected output [3., 2.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_div_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.rand(3, 4, 5).astype(np.float32) + 1.0
z = x / y
expect(node, inputs=[x, y], outputs=[z],
       name='test_div')

x = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8) + 1
z = x // y
expect(node, inputs=[x, y], outputs=[z],
       name='test_div_uint8')
```

</details>
<details>
<summary>div_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Div',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.rand(5).astype(np.float32) + 1.0
z = x / y
expect(node, inputs=[x, y], outputs=[z],
       name='test_div_bcast')
```

</details>


### Dropout
There are 12 test cases, listed as following:
<details>
<summary>default</summary>

```python
seed = np.int64(0)
node = onnx.helper.make_node(
    'Dropout',
    inputs=['x'],
    outputs=['y'],
    seed=seed
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = dropout(x)
expect(node, inputs=[x], outputs=[y], name='test_dropout_default')
```

</details>
<details>
<summary>default_mask</summary>

```python
seed = np.int64(0)
node = onnx.helper.make_node(
    'Dropout',
    inputs=['x'],
    outputs=['y', 'z'],
    seed=seed
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y, z = dropout(x, return_mask=True)
expect(node, inputs=[x], outputs=[y, z], name='test_dropout_default_mask')
```

</details>
<details>
<summary>default_mask_ratio</summary>

```python
seed = np.int64(0)
node = onnx.helper.make_node(
    'Dropout',
    inputs=['x', 'r'],
    outputs=['y', 'z'],
    seed=seed
)

r = np.float32(0.1)
x = np.random.randn(3, 4, 5).astype(np.float32)
y, z = dropout(x, r, return_mask=True)
expect(node, inputs=[x, r], outputs=[y, z], name='test_dropout_default_mask_ratio')
```

</details>
<details>
<summary>default_old</summary>

```python
node = onnx.helper.make_node(
    'Dropout',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = x
expect(node, inputs=[x], outputs=[y],
       name='test_dropout_default_old', opset_imports=[helper.make_opsetid("", 11)])
```

</details>
<details>
<summary>default_ratio</summary>

```python
seed = np.int64(0)
node = onnx.helper.make_node(
    'Dropout',
    inputs=['x', 'r'],
    outputs=['y'],
    seed=seed
)

r = np.float32(0.1)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = dropout(x, r)
expect(node, inputs=[x, r], outputs=[y], name='test_dropout_default_ratio')
```

</details>
<details>
<summary>random_old</summary>

```python
node = onnx.helper.make_node(
    'Dropout',
    inputs=['x'],
    outputs=['y'],
    ratio=.2,
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = x
expect(node, inputs=[x], outputs=[y],
       name='test_dropout_random_old', opset_imports=[helper.make_opsetid("", 11)])
```

</details>
<details>
<summary>training</summary>

```python
seed = np.int64(0)
node = onnx.helper.make_node(
    'Dropout',
    inputs=['x', 'r', 't'],
    outputs=['y'],
    seed=seed
)

x = np.random.randn(3, 4, 5).astype(np.float32)
r = np.float32(0.75)
t = np.bool_(True)
y = dropout(x, r, training_mode=t)
expect(node, inputs=[x, r, t], outputs=[y], name='test_training_dropout')
```

</details>
<details>
<summary>training_default</summary>

```python
seed = np.int64(0)
node = onnx.helper.make_node(
    'Dropout',
    inputs=['x', 'r', 't'],
    outputs=['y'],
    seed=seed
)

x = np.random.randn(3, 4, 5).astype(np.float32)
r = np.float32(0.5)
t = np.bool_(True)
y = dropout(x, r, training_mode=t)
expect(node, inputs=[x, r, t], outputs=[y], name='test_training_dropout_default')
```

</details>
<details>
<summary>training_default_ratio_mask</summary>

```python
seed = np.int64(0)
node = onnx.helper.make_node(
    'Dropout',
    inputs=['x', 'r', 't'],
    outputs=['y', 'z'],
    seed=seed
)

x = np.random.randn(3, 4, 5).astype(np.float32)
r = np.float32(0.5)
t = np.bool_(True)
y, z = dropout(x, r, training_mode=t, return_mask=True)
expect(node, inputs=[x, r, t], outputs=[y, z], name='test_training_dropout_default_mask')
```

</details>
<details>
<summary>training_default_zero_ratio</summary>

```python
seed = np.int64(0)
node = onnx.helper.make_node(
    'Dropout',
    inputs=['x', 'r', 't'],
    outputs=['y'],
    seed=seed
)

x = np.random.randn(3, 4, 5).astype(np.float32)
r = np.float32(0.0)
t = np.bool_(True)
y = dropout(x, r, training_mode=t)
expect(node, inputs=[x, r, t], outputs=[y], name='test_training_dropout_zero_ratio')
```

</details>
<details>
<summary>training_default_zero_ratio_mask</summary>

```python
seed = np.int64(0)
node = onnx.helper.make_node(
    'Dropout',
    inputs=['x', 'r', 't'],
    outputs=['y', 'z'],
    seed=seed
)

x = np.random.randn(3, 4, 5).astype(np.float32)
r = np.float32(0.0)
t = np.bool_(True)
y, z = dropout(x, r, training_mode=t, return_mask=True)
expect(node, inputs=[x, r, t], outputs=[y, z], name='test_training_dropout_zero_ratio_mask')
```

</details>
<details>
<summary>training_ratio_mask</summary>

```python
seed = np.int64(0)
node = onnx.helper.make_node(
    'Dropout',
    inputs=['x', 'r', 't'],
    outputs=['y', 'z'],
    seed=seed
)

x = np.random.randn(3, 4, 5).astype(np.float32)
r = np.float32(0.75)
t = np.bool_(True)
y, z = dropout(x, r, training_mode=t, return_mask=True)
expect(node, inputs=[x, r, t], outputs=[y, z], name='test_training_dropout_mask')
```

</details>


### DynamicQuantizeLinear
There are 1 test cases, listed as following:
<details>
<summary>dynamicquantizelinear</summary>

```python
node = onnx.helper.make_node('DynamicQuantizeLinear',
    inputs=['x'],
    outputs=['y', 'y_scale', 'y_zero_point'],
)

# expected scale 0.0196078438 and zero point 153
X = np.array([0, 2, -3, -2.5, 1.34, 0.5]).astype(np.float32)
x_min = np.minimum(0, np.min(X))
x_max = np.maximum(0, np.max(X))
Y_Scale = np.float32((x_max - x_min) / (255 - 0))  # uint8 -> [0, 255]
Y_ZeroPoint = np.clip(round((0 - x_min) / Y_Scale), 0, 255).astype(np.uint8)
Y = np.clip(np.round(X / Y_Scale) + Y_ZeroPoint, 0, 255).astype(np.uint8)

expect(node, inputs=[X], outputs=[Y, Y_Scale, Y_ZeroPoint],
       name='test_dynamicquantizelinear')

# expected scale 0.0156862754 and zero point 255
X = np.array([-1.0, -2.1, -1.3, -2.5, -3.34, -4.0]).astype(np.float32)
x_min = np.minimum(0, np.min(X))
x_max = np.maximum(0, np.max(X))
Y_Scale = np.float32((x_max - x_min) / (255 - 0))  # uint8 -> [0, 255]
Y_ZeroPoint = np.clip(round((0 - x_min) / Y_Scale), 0, 255).astype(np.uint8)
Y = np.clip(np.round(X / Y_Scale) + Y_ZeroPoint, 0, 255).astype(np.uint8)

expect(node, inputs=[X], outputs=[Y, Y_Scale, Y_ZeroPoint],
       name='test_dynamicquantizelinear_max_adjusted')

X = np.array([1, 2.1, 1.3, 2.5,
              3.34, 4.0, 1.5, 2.6,
              3.9, 4.0, 3.0, 2.345]).astype(np.float32).reshape((3, 4))

# expected scale 0.0156862754 and zero point 0
x_min = np.minimum(0, np.min(X))
x_max = np.maximum(0, np.max(X))
Y_Scale = np.float32((x_max - x_min) / (255 - 0))  # uint8 -> [0, 255]
Y_ZeroPoint = np.clip(round((0 - x_min) / Y_Scale), 0, 255).astype(np.uint8)
Y = np.clip(np.round(X / Y_Scale) + Y_ZeroPoint, 0, 255).astype(np.uint8)

expect(node, inputs=[X], outputs=[Y, Y_Scale, Y_ZeroPoint],
       name='test_dynamicquantizelinear_min_adjusted')
```

</details>


### Einsum
There are 5 test cases, listed as following:
<details>
<summary>einsum_batch_diagonal</summary>

```python
Eqn = '...ii ->...i'
node = onnx.helper.make_node(
    'Einsum',
    inputs=['x'],
    outputs=['y'],
    equation=Eqn
)

X = np.random.randn(3, 5, 5)
Z = einsum_reference_implementation(Eqn, (X,))

expect(node, inputs=[X], outputs=[Z], name='test_einsum_batch_diagonal')
```

</details>
<details>
<summary>einsum_batch_matmul</summary>

```python
Eqn = 'bij, bjk -> bik'
node = onnx.helper.make_node(
    'Einsum',
    inputs=['x', 'y'],
    outputs=['z'],
    equation=Eqn
)

X = np.random.randn(5, 2, 3)
Y = np.random.randn(5, 3, 4)
Z = einsum_reference_implementation(Eqn, (X, Y))

expect(node, inputs=[X, Y], outputs=[Z], name='test_einsum_batch_matmul')
```

</details>
<details>
<summary>einsum_inner_prod</summary>

```python
Eqn = 'i,i'
node = onnx.helper.make_node(
    'Einsum',
    inputs=['x', 'y'],
    outputs=['z'],
    equation=Eqn
)

X = np.random.randn(5)
Y = np.random.randn(5)
Z = einsum_reference_implementation(Eqn, (X, Y))

expect(node, inputs=[X, Y], outputs=[Z], name='test_einsum_inner_prod')
```

</details>
<details>
<summary>einsum_sum</summary>

```python
Eqn = 'ij->i'
node = onnx.helper.make_node(
    'Einsum',
    inputs=['x'],
    outputs=['y'],
    equation=Eqn
)

X = np.random.randn(3, 4)
Z = einsum_reference_implementation(Eqn, (X,))

expect(node, inputs=[X], outputs=[Z], name='test_einsum_sum')
```

</details>
<details>
<summary>einsum_transpose</summary>

```python
Eqn = 'ij->ji'
node = onnx.helper.make_node(
    'Einsum',
    inputs=['x'],
    outputs=['y'],
    equation=Eqn
)

X = np.random.randn(3, 4)
Y = einsum_reference_implementation(Eqn, (X,))

expect(node, inputs=[X], outputs=[Y], name='test_einsum_transpose')
```

</details>


### Elu
There are 2 test cases, listed as following:
<details>
<summary>elu</summary>

```python
node = onnx.helper.make_node(
    'Elu',
    inputs=['x'],
    outputs=['y'],
    alpha=2.0
)

x = np.array([-1, 0, 1]).astype(np.float32)
# expected output [-1.2642411, 0., 1.]
y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0
expect(node, inputs=[x], outputs=[y],
       name='test_elu_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0
expect(node, inputs=[x], outputs=[y],
       name='test_elu')
```

</details>
<details>
<summary>elu_default</summary>

```python
default_alpha = 1.0
node = onnx.helper.make_node(
    'Elu',
    inputs=['x'],
    outputs=['y'],
)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, 0, np.inf) + (np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha
expect(node, inputs=[x], outputs=[y],
       name='test_elu_default')
```

</details>


### Equal
There are 2 test cases, listed as following:
<details>
<summary>equal</summary>

```python
node = onnx.helper.make_node(
    'Equal',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
y = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
z = np.equal(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_equal')
```

</details>
<details>
<summary>equal_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Equal',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
y = (np.random.randn(5) * 10).astype(np.int32)
z = np.equal(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_equal_bcast')
```

</details>


### Erf
There are 1 test cases, listed as following:
<details>
<summary>erf</summary>

```python
node = onnx.helper.make_node(
    'Erf',
    inputs=['x'],
    outputs=['y'],
)

x = np.random.randn(1, 3, 32, 32).astype(np.float32)
y = np.vectorize(math.erf)(x).astype(np.float32)
expect(node, inputs=[x], outputs=[y],
       name='test_erf')
```

</details>


### Exp
There are 1 test cases, listed as following:
<details>
<summary>exp</summary>

```python
node = onnx.helper.make_node(
    'Exp',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.exp(x)  # expected output [0.36787945, 1., 2.71828175]
expect(node, inputs=[x], outputs=[y],
       name='test_exp_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.exp(x)
expect(node, inputs=[x], outputs=[y],
       name='test_exp')
```

</details>


### Expand
There are 2 test cases, listed as following:
<details>
<summary>dim_changed</summary>

```python
node = onnx.helper.make_node(
    'Expand',
    inputs=['data', 'new_shape'],
    outputs=['expanded'],
)
shape = [3, 1]
data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
#print(data)
#[[1.], [2.], [3.]]
new_shape = [2, 1, 6]
expanded = data * np.ones(new_shape, dtype=np.float32)
#print(expanded)
#[[[1., 1., 1., 1., 1., 1.],
#  [2., 2., 2., 2., 2., 2.],
#  [3., 3., 3., 3., 3., 3.]],
#
# [[1., 1., 1., 1., 1., 1.],
#  [2., 2., 2., 2., 2., 2.],
#  [3., 3., 3., 3., 3., 3.]]]
new_shape = np.array(new_shape, dtype=np.int64)
expect(node, inputs=[data, new_shape], outputs=[expanded],
       name='test_expand_dim_changed')
```

</details>
<details>
<summary>dim_unchanged</summary>

```python
node = onnx.helper.make_node(
    'Expand',
    inputs=['data', 'new_shape'],
    outputs=['expanded'],
)
shape = [3, 1]
new_shape = [3, 4]
data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
#print(data)
#[[1.], [2.], [3.]]
expanded = np.tile(data, 4)
#print(expanded)
#[[1., 1., 1., 1.],
# [2., 2., 2., 2.],
# [3., 3., 3., 3.]]
new_shape = np.array(new_shape, dtype=np.int64)
expect(node, inputs=[data, new_shape], outputs=[expanded],
       name='test_expand_dim_unchanged')
```

</details>


### EyeLike
There are 3 test cases, listed as following:
<details>
<summary>populate_off_main_diagonal</summary>

```python
shape = (4, 5)
off_diagonal_offset = 1
node = onnx.helper.make_node(
    'EyeLike',
    inputs=['x'],
    outputs=['y'],
    k=off_diagonal_offset,
    dtype=onnx.TensorProto.FLOAT,
)

x = np.random.randint(0, 100, size=shape, dtype=np.int32)
y = np.eye(shape[0], shape[1], k=off_diagonal_offset, dtype=np.float32)
expect(node, inputs=[x], outputs=[y], name='test_eyelike_populate_off_main_diagonal')
```

</details>
<details>
<summary>with_dtype</summary>

```python
shape = (3, 4)
node = onnx.helper.make_node(
    'EyeLike',
    inputs=['x'],
    outputs=['y'],
    dtype=onnx.TensorProto.DOUBLE,
)

x = np.random.randint(0, 100, size=shape, dtype=np.int32)
y = np.eye(shape[0], shape[1], dtype=np.float64)
expect(node, inputs=[x], outputs=[y], name='test_eyelike_with_dtype')
```

</details>
<details>
<summary>without_dtype</summary>

```python
shape = (4, 4)
node = onnx.helper.make_node(
    'EyeLike',
    inputs=['x'],
    outputs=['y'],
)

x = np.random.randint(0, 100, size=shape, dtype=np.int32)
y = np.eye(shape[0], shape[1], dtype=np.int32)
expect(node, inputs=[x], outputs=[y], name='test_eyelike_without_dtype')
```

</details>


### Flatten
There are 3 test cases, listed as following:
<details>
<summary>flatten</summary>

```python
shape = (2, 3, 4, 5)
a = np.random.random_sample(shape).astype(np.float32)

for i in range(len(shape)):
    node = onnx.helper.make_node(
        'Flatten',
        inputs=['a'],
        outputs=['b'],
        axis=i,
    )

    new_shape = (1, -1) if i == 0 else (np.prod(shape[0:i]).astype(int), -1)
    b = np.reshape(a, new_shape)
    expect(node, inputs=[a], outputs=[b],
           name='test_flatten_axis' + str(i))
```

</details>
<details>
<summary>flatten_negative_axis</summary>

```python
shape = (2, 3, 4, 5)
a = np.random.random_sample(shape).astype(np.float32)

for i in range(-len(shape), 0):
    node = onnx.helper.make_node(
        'Flatten',
        inputs=['a'],
        outputs=['b'],
        axis=i,
    )

    new_shape = (np.prod(shape[0:i]).astype(int), -1)
    b = np.reshape(a, new_shape)
    expect(node, inputs=[a], outputs=[b],
           name='test_flatten_negative_axis' + str(abs(i)))
```

</details>
<details>
<summary>flatten_with_default_axis</summary>

```python
node = onnx.helper.make_node(
    'Flatten',
    inputs=['a'],
    outputs=['b'],  # Default value for axis: axis=1
)

shape = (5, 4, 3, 2)
a = np.random.random_sample(shape).astype(np.float32)
new_shape = (5, 24)
b = np.reshape(a, new_shape)
expect(node, inputs=[a], outputs=[b],
       name='test_flatten_default_axis')
```

</details>


### Floor
There are 1 test cases, listed as following:
<details>
<summary>floor</summary>

```python
node = onnx.helper.make_node(
    'Floor',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-1.5, 1.2, 2]).astype(np.float32)
y = np.floor(x)  # expected output [-2., 1., 2.]
expect(node, inputs=[x], outputs=[y],
       name='test_floor_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.floor(x)
expect(node, inputs=[x], outputs=[y],
       name='test_floor')
```

</details>


### GRU
There are 4 test cases, listed as following:
<details>
<summary>batchwise</summary>

```python
input = np.array([[[1., 2.]], [[3., 4.]], [[5., 6.]]]).astype(np.float32)

input_size = 2
hidden_size = 6
number_of_gates = 3
weight_scale = 0.2
layout = 1

node = onnx.helper.make_node(
    'GRU',
    inputs=['X', 'W', 'R'],
    outputs=['Y', 'Y_h'],
    hidden_size=hidden_size,
    layout=layout
)

W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

gru = GRU_Helper(X=input, W=W, R=R, layout=layout)
Y, Y_h = gru.step()
expect(node, inputs=[input, W, R], outputs=[Y.astype(np.float32), Y_h.astype(np.float32)], name='test_gru_batchwise')
```

</details>
<details>
<summary>defaults</summary>

```python
input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

input_size = 2
hidden_size = 5
weight_scale = 0.1
number_of_gates = 3

node = onnx.helper.make_node(
    'GRU',
    inputs=['X', 'W', 'R'],
    outputs=['', 'Y_h'],
    hidden_size=hidden_size
)

W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

gru = GRU_Helper(X=input, W=W, R=R)
_, Y_h = gru.step()
expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_gru_defaults')
```

</details>
<details>
<summary>initial_bias</summary>

```python
input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]).astype(np.float32)

input_size = 3
hidden_size = 3
weight_scale = 0.1
custom_bias = 0.1
number_of_gates = 3

node = onnx.helper.make_node(
    'GRU',
    inputs=['X', 'W', 'R', 'B'],
    outputs=['', 'Y_h'],
    hidden_size=hidden_size
)

W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

# Adding custom bias
W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(np.float32)
R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
B = np.concatenate((W_B, R_B), axis=1)

gru = GRU_Helper(X=input, W=W, R=R, B=B)
_, Y_h = gru.step()
expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_gru_with_initial_bias')
```

</details>
<details>
<summary>seq_length</summary>

```python
input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                  [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]]).astype(np.float32)

input_size = 3
hidden_size = 5
number_of_gates = 3

node = onnx.helper.make_node(
    'GRU',
    inputs=['X', 'W', 'R', 'B'],
    outputs=['', 'Y_h'],
    hidden_size=hidden_size
)

W = np.random.randn(1, number_of_gates * hidden_size, input_size).astype(np.float32)
R = np.random.randn(1, number_of_gates * hidden_size, hidden_size).astype(np.float32)

# Adding custom bias
W_B = np.random.randn(1, number_of_gates * hidden_size).astype(np.float32)
R_B = np.random.randn(1, number_of_gates * hidden_size).astype(np.float32)
B = np.concatenate((W_B, R_B), axis=1)

gru = GRU_Helper(X=input, W=W, R=R, B=B)
_, Y_h = gru.step()
expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_gru_seq_length')
```

</details>


### Gather
There are 4 test cases, listed as following:
<details>
<summary>gather_0</summary>

```python
node = onnx.helper.make_node(
    'Gather',
    inputs=['data', 'indices'],
    outputs=['y'],
    axis=0,
)
data = np.random.randn(5, 4, 3, 2).astype(np.float32)
indices = np.array([0, 1, 3])
y = np.take(data, indices, axis=0)

expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y],
       name='test_gather_0')
```

</details>
<details>
<summary>gather_1</summary>

```python
node = onnx.helper.make_node(
    'Gather',
    inputs=['data', 'indices'],
    outputs=['y'],
    axis=1,
)
data = np.random.randn(5, 4, 3, 2).astype(np.float32)
indices = np.array([0, 1, 3])
y = np.take(data, indices, axis=1)

expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y],
       name='test_gather_1')
```

</details>
<details>
<summary>gather_2d_indices</summary>

```python
node = onnx.helper.make_node(
    'Gather',
    inputs=['data', 'indices'],
    outputs=['y'],
    axis=1,
)
data = np.random.randn(3, 3).astype(np.float32)
indices = np.array([[0, 2]])
y = np.take(data, indices, axis=1)

expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y],
       name='test_gather_2d_indices')
```

</details>
<details>
<summary>gather_negative_indices</summary>

```python
node = onnx.helper.make_node(
    'Gather',
    inputs=['data', 'indices'],
    outputs=['y'],
    axis=0,
)
data = np.arange(10).astype(np.float32)
indices = np.array([0, -9, -10])
y = np.take(data, indices, axis=0)

# print(y)
# [0. 1. 0.]

expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y],
       name='test_gather_negative_indices')
```

</details>


### GatherElements
There are 3 test cases, listed as following:
<details>
<summary>gather_elements_0</summary>

```python
axis = 1
node = onnx.helper.make_node(
    'GatherElements',
    inputs=['data', 'indices'],
    outputs=['y'],
    axis=axis,
)
data = np.array([[1, 2],
                 [3, 4]], dtype=np.float32)
indices = np.array([[0, 0],
                    [1, 0]], dtype=np.int32)

y = gather_elements(data, indices, axis)
# print(y) produces
# [[1, 1],
#  [4, 3]]

expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y],
       name='test_gather_elements_0')
```

</details>
<details>
<summary>gather_elements_1</summary>

```python
axis = 0
node = onnx.helper.make_node(
    'GatherElements',
    inputs=['data', 'indices'],
    outputs=['y'],
    axis=axis,
)
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]], dtype=np.float32)
indices = np.array([[1, 2, 0],
                    [2, 0, 0]], dtype=np.int32)

y = gather_elements(data, indices, axis)
# print(y) produces
# [[4, 8, 3],
#  [7, 2, 3]]

expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y],
       name='test_gather_elements_1')
```

</details>
<details>
<summary>gather_elements_negative_indices</summary>

```python
axis = 0
node = onnx.helper.make_node(
    'GatherElements',
    inputs=['data', 'indices'],
    outputs=['y'],
    axis=axis,
)
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]], dtype=np.float32)
indices = np.array([[-1, -2, 0],
                    [-2, 0, 0]], dtype=np.int32)

y = gather_elements(data, indices, axis)
# print(y) produces
# [[7, 5, 3],
#  [4, 2, 3]]

expect(node, inputs=[data, indices.astype(np.int64)], outputs=[y],
       name='test_gather_elements_negative_indices')
```

</details>


### GatherND
There are 3 test cases, listed as following:
<details>
<summary>float32</summary>

```python
node = onnx.helper.make_node(
    'GatherND',
    inputs=['data', 'indices'],
    outputs=['output'],
)

data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.float32)
indices = np.array([[[0, 1]], [[1, 0]]], dtype=np.int64)
output = gather_nd_impl(data, indices, 0)
expected_output = np.array([[[2, 3]], [[4, 5]]], dtype=np.float32)
assert (np.array_equal(output, expected_output))
expect(node, inputs=[data, indices], outputs=[output],
       name='test_gathernd_example_float32')
```

</details>
<details>
<summary>int32</summary>

```python
node = onnx.helper.make_node(
    'GatherND',
    inputs=['data', 'indices'],
    outputs=['output'],
)

data = np.array([[0, 1], [2, 3]], dtype=np.int32)
indices = np.array([[0, 0], [1, 1]], dtype=np.int64)
output = gather_nd_impl(data, indices, 0)
expected_output = np.array([0, 3], dtype=np.int32)
assert (np.array_equal(output, expected_output))
expect(node, inputs=[data, indices], outputs=[output],
       name='test_gathernd_example_int32')
```

</details>
<details>
<summary>int32_batchdim_1</summary>

```python
node = onnx.helper.make_node(
    'GatherND',
    inputs=['data', 'indices'],
    outputs=['output'],
    batch_dims=1,
)

data = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=np.int32)
indices = np.array([[1], [0]], dtype=np.int64)
output = gather_nd_impl(data, indices, 1)
expected_output = np.array([[2, 3], [4, 5]], dtype=np.int32)
assert (np.array_equal(output, expected_output))
expect(node, inputs=[data, indices], outputs=[output],
       name='test_gathernd_example_int32_batch_dim1')
```

</details>


### Gemm
There are 11 test cases, listed as following:
<details>
<summary>all_attributes</summary>

```python
node = onnx.helper.make_node(
    'Gemm',
    inputs=['a', 'b', 'c'],
    outputs=['y'],
    alpha=0.25,
    beta=0.35,
    transA=1,
    transB=1
)
a = np.random.ranf([4, 3]).astype(np.float32)
b = np.random.ranf([5, 4]).astype(np.float32)
c = np.random.ranf([1, 5]).astype(np.float32)
y = gemm_reference_implementation(a, b, c, transA=1, transB=1, alpha=0.25, beta=0.35)
expect(node, inputs=[a, b, c], outputs=[y],
       name='test_gemm_all_attributes')
```

</details>
<details>
<summary>alpha</summary>

```python
node = onnx.helper.make_node(
    'Gemm',
    inputs=['a', 'b', 'c'],
    outputs=['y'],
    alpha=0.5
)
a = np.random.ranf([3, 5]).astype(np.float32)
b = np.random.ranf([5, 4]).astype(np.float32)
c = np.zeros([1, 4]).astype(np.float32)
y = gemm_reference_implementation(a, b, c, alpha=0.5)
expect(node, inputs=[a, b, c], outputs=[y],
       name='test_gemm_alpha')
```

</details>
<details>
<summary>beta</summary>

```python
node = onnx.helper.make_node(
    'Gemm',
    inputs=['a', 'b', 'c'],
    outputs=['y'],
    beta=0.5
)
a = np.random.ranf([2, 7]).astype(np.float32)
b = np.random.ranf([7, 4]).astype(np.float32)
c = np.random.ranf([1, 4]).astype(np.float32)
y = gemm_reference_implementation(a, b, c, beta=0.5)
expect(node, inputs=[a, b, c], outputs=[y],
       name='test_gemm_beta')
```

</details>
<details>
<summary>default_matrix_bias</summary>

```python
node = onnx.helper.make_node(
    'Gemm',
    inputs=['a', 'b', 'c'],
    outputs=['y']
)
a = np.random.ranf([3, 6]).astype(np.float32)
b = np.random.ranf([6, 4]).astype(np.float32)
c = np.random.ranf([3, 4]).astype(np.float32)
y = gemm_reference_implementation(a, b, c)
expect(node, inputs=[a, b, c], outputs=[y],
       name='test_gemm_default_matrix_bias')
```

</details>
<details>
<summary>default_no_bias</summary>

```python
node = onnx.helper.make_node(
    'Gemm',
    inputs=['a', 'b'],
    outputs=['y']
)
a = np.random.ranf([2, 10]).astype(np.float32)
b = np.random.ranf([10, 3]).astype(np.float32)
y = gemm_reference_implementation(a, b)
expect(node, inputs=[a, b], outputs=[y],
       name='test_gemm_default_no_bias')
```

</details>
<details>
<summary>default_scalar_bias</summary>

```python
node = onnx.helper.make_node(
    'Gemm',
    inputs=['a', 'b', 'c'],
    outputs=['y']
)
a = np.random.ranf([2, 3]).astype(np.float32)
b = np.random.ranf([3, 4]).astype(np.float32)
c = np.array(3.14).astype(np.float32)
y = gemm_reference_implementation(a, b, c)
expect(node, inputs=[a, b, c], outputs=[y],
       name='test_gemm_default_scalar_bias')
```

</details>
<details>
<summary>default_single_elem_vector_bias</summary>

```python
node = onnx.helper.make_node(
    'Gemm',
    inputs=['a', 'b', 'c'],
    outputs=['y']
)
a = np.random.ranf([3, 7]).astype(np.float32)
b = np.random.ranf([7, 3]).astype(np.float32)
c = np.random.ranf([1]).astype(np.float32)
y = gemm_reference_implementation(a, b, c)
expect(node, inputs=[a, b, c], outputs=[y],
       name='test_gemm_default_single_elem_vector_bias')
```

</details>
<details>
<summary>default_vector_bias</summary>

```python
node = onnx.helper.make_node(
    'Gemm',
    inputs=['a', 'b', 'c'],
    outputs=['y']
)
a = np.random.ranf([2, 7]).astype(np.float32)
b = np.random.ranf([7, 4]).astype(np.float32)
c = np.random.ranf([1, 4]).astype(np.float32)
y = gemm_reference_implementation(a, b, c)
expect(node, inputs=[a, b, c], outputs=[y],
       name='test_gemm_default_vector_bias')
```

</details>
<details>
<summary>default_zero_bias</summary>

```python
node = onnx.helper.make_node(
    'Gemm',
    inputs=['a', 'b', 'c'],
    outputs=['y']
)
a = np.random.ranf([3, 5]).astype(np.float32)
b = np.random.ranf([5, 4]).astype(np.float32)
c = np.zeros([1, 4]).astype(np.float32)
y = gemm_reference_implementation(a, b, c)
expect(node, inputs=[a, b, c], outputs=[y],
       name='test_gemm_default_zero_bias')
```

</details>
<details>
<summary>transposeA</summary>

```python
node = onnx.helper.make_node(
    'Gemm',
    inputs=['a', 'b', 'c'],
    outputs=['y'],
    transA=1
)
a = np.random.ranf([6, 3]).astype(np.float32)
b = np.random.ranf([6, 4]).astype(np.float32)
c = np.zeros([1, 4]).astype(np.float32)
y = gemm_reference_implementation(a, b, c, transA=1)
expect(node, inputs=[a, b, c], outputs=[y],
       name='test_gemm_transposeA')
```

</details>
<details>
<summary>transposeB</summary>

```python
node = onnx.helper.make_node(
    'Gemm',
    inputs=['a', 'b', 'c'],
    outputs=['y'],
    transB=1
)
a = np.random.ranf([3, 6]).astype(np.float32)
b = np.random.ranf([4, 6]).astype(np.float32)
c = np.zeros([1, 4]).astype(np.float32)
y = gemm_reference_implementation(a, b, c, transB=1)
expect(node, inputs=[a, b, c], outputs=[y],
       name='test_gemm_transposeB')
```

</details>


### GlobalAveragePool
There are 2 test cases, listed as following:
<details>
<summary>globalaveragepool</summary>

```python
node = onnx.helper.make_node(
    'GlobalAveragePool',
    inputs=['x'],
    outputs=['y'],
)
x = np.random.randn(1, 3, 5, 5).astype(np.float32)
y = np.mean(x, axis=tuple(range(2, np.ndim(x))), keepdims=True)
expect(node, inputs=[x], outputs=[y], name='test_globalaveragepool')
```

</details>
<details>
<summary>globalaveragepool_precomputed</summary>

```python

node = onnx.helper.make_node(
    'GlobalAveragePool',
    inputs=['x'],
    outputs=['y'],
)
x = np.array([[[
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]]]).astype(np.float32)
y = np.array([[[[5]]]]).astype(np.float32)
expect(node, inputs=[x], outputs=[y], name='test_globalaveragepool_precomputed')
```

</details>


### GlobalMaxPool
There are 2 test cases, listed as following:
<details>
<summary>globalmaxpool</summary>

```python

node = onnx.helper.make_node(
    'GlobalMaxPool',
    inputs=['x'],
    outputs=['y'],
)
x = np.random.randn(1, 3, 5, 5).astype(np.float32)
y = np.max(x, axis=tuple(range(2, np.ndim(x))), keepdims=True)
expect(node, inputs=[x], outputs=[y], name='test_globalmaxpool')
```

</details>
<details>
<summary>globalmaxpool_precomputed</summary>

```python

node = onnx.helper.make_node(
    'GlobalMaxPool',
    inputs=['x'],
    outputs=['y'],
)
x = np.array([[[
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]]]).astype(np.float32)
y = np.array([[[[9]]]]).astype(np.float32)
expect(node, inputs=[x], outputs=[y], name='test_globalmaxpool_precomputed')
```

</details>


### Gradient
There are 2 test cases, listed as following:
<details>
<summary>gradient_scalar_add</summary>

```python
add_node = onnx.helper.make_node('Add',
                                 ['a', 'b'], ['c'], name='my_add')
gradient_node = onnx.helper.make_node(
    'Gradient', ['a', 'b'],
    ['dc_da', 'dc_db'], name='my_gradient',
    domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
    xs=['a', 'b'], y='c')

a = np.array(1.0).astype(np.float32)
b = np.array(2.0).astype(np.float32)
c = a + b
# dc / da = d(a+b) / da = 1
dc_da = np.array(1).astype(np.float32)
# db / db = d(a+b) / db = 1
dc_db = np.array(1).astype(np.float32)

graph = onnx.helper.make_graph(
    nodes=[add_node, gradient_node],
    name='GradientOfAdd',
    inputs=[
        onnx.helper.make_tensor_value_info('a', onnx.TensorProto.FLOAT,
                                           []),
        onnx.helper.make_tensor_value_info('b', onnx.TensorProto.FLOAT,
                                           [])],
    outputs=[
        onnx.helper.make_tensor_value_info('c', onnx.TensorProto.FLOAT,
                                           []),
        onnx.helper.make_tensor_value_info('dc_da',
                                           onnx.TensorProto.FLOAT, []),
        onnx.helper.make_tensor_value_info('dc_db',
                                           onnx.TensorProto.FLOAT, [])])
opsets = [
    onnx.helper.make_operatorsetid(ONNX_DOMAIN, 12),
    onnx.helper.make_operatorsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)]
model = onnx.helper.make_model_gen_version(
    graph,
    producer_name='backend-test',
    opset_imports=opsets)
expect(model, inputs=[a, b], outputs=[c, dc_da, dc_db],
       name='test_gradient_of_add')
```

</details>
<details>
<summary>gradient_scalar_add_and_mul</summary>

```python
add_node = onnx.helper.make_node('Add',
                                 ['a', 'b'], ['c'], name='my_add')
mul_node = onnx.helper.make_node('Mul',
                                 ['c', 'a'], ['d'], name='my_mul')
gradient_node = onnx.helper.make_node(
    'Gradient', ['a', 'b'],
    ['dd_da', 'dd_db'], name='my_gradient',
    domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN,
    xs=['a', 'b'], y='d')

a = np.array(1.0).astype(np.float32)
b = np.array(2.0).astype(np.float32)
c = a + b
# d = a * c = a * (a + b)
d = a * c
# dd / da = d(a*a+a*b) / da = 2 * a + b
dd_da = (2 * a + b).astype(np.float32)
# dd / db = d(a*a+a*b) / db = a
dd_db = a

graph = onnx.helper.make_graph(
    nodes=[add_node, mul_node, gradient_node],
    name='GradientOfTwoOperators',
    inputs=[
        onnx.helper.make_tensor_value_info('a', onnx.TensorProto.FLOAT,
                                           []),
        onnx.helper.make_tensor_value_info('b', onnx.TensorProto.FLOAT,
                                           [])],
    outputs=[
        onnx.helper.make_tensor_value_info('d', onnx.TensorProto.FLOAT,
                                           []),
        onnx.helper.make_tensor_value_info('dd_da',
                                           onnx.TensorProto.FLOAT, []),
        onnx.helper.make_tensor_value_info('dd_db',
                                           onnx.TensorProto.FLOAT, [])])

opsets = [
    onnx.helper.make_operatorsetid(ONNX_DOMAIN, 12),
    onnx.helper.make_operatorsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)]
model = onnx.helper.make_model_gen_version(graph,
    producer_name='backend-test',
    opset_imports=opsets)
expect(model, inputs=[a, b], outputs=[d, dd_da, dd_db],
       name='test_gradient_of_add_and_mul')
```

</details>


### Greater
There are 4 test cases, listed as following:
<details>
<summary>greater</summary>

```python
node = onnx.helper.make_node(
    'Greater',
    inputs=['x', 'y'],
    outputs=['greater'],
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(3, 4, 5).astype(np.float32)
z = np.greater(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_greater')
```

</details>
<details>
<summary>greater</summary>

```python
node = onnx.helper.make_node(
    'GreaterOrEqual',
    inputs=['x', 'y'],
    outputs=['greater_equal'],
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(3, 4, 5).astype(np.float32)
z = np.greater_equal(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_greater_equal')
```

</details>
<details>
<summary>greater_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Greater',
    inputs=['x', 'y'],
    outputs=['greater'],
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(5).astype(np.float32)
z = np.greater(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_greater_bcast')
```

</details>
<details>
<summary>greater_broadcast</summary>

```python
node = onnx.helper.make_node(
    'GreaterOrEqual',
    inputs=['x', 'y'],
    outputs=['greater_equal'],
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(5).astype(np.float32)
z = np.greater_equal(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_greater_equal_bcast')
```

</details>


### GridSample
There are 3 test cases, listed as following:
<details>
<summary>gridsample</summary>

```python
node = onnx.helper.make_node(
    'GridSample',
    inputs=['X', 'Grid'],
    outputs=['Y'],
    mode='bilinear',
    padding_mode='zeros',
    align_corners=0,
)
# X shape, [N, C, H, W] - [1, 1, 4, 4]
X = np.array(
    [
        [
            [
                [0., 1., 2., 3.],
                [4., 5., 6., 7.],
                [8., 9., 10., 11.],
                [12., 13., 14., 15.]
            ]
        ]
    ],
    dtype=np.float32,
)
# Grid shape, [N, H_out, W_out, 2] - [1, 6, 6, 2]
Grid = np.array(
    [
        [
            [
                [-1.0000, -1.0000],
                [-0.6000, -1.0000],
                [-0.2000, -1.0000],
                [0.2000, -1.0000],
                [0.6000, -1.0000],
                [1.0000, -1.0000]
            ],
            [
                [-1.0000, -0.6000],
                [-0.6000, -0.6000],
                [-0.2000, -0.6000],
                [0.2000, -0.6000],
                [0.6000, -0.6000],
                [1.0000, -0.6000]
            ],
            [
                [-1.0000, -0.2000],
                [-0.6000, -0.2000],
                [-0.2000, -0.2000],
                [0.2000, -0.2000],
                [0.6000, -0.2000],
                [1.0000, -0.2000]
            ],
            [
                [-1.0000, 0.2000],
                [-0.6000, 0.2000],
                [-0.2000, 0.2000],
                [0.2000, 0.2000],
                [0.6000, 0.2000],
                [1.0000, 0.2000]
            ],
            [
                [-1.0000, 0.6000],
                [-0.6000, 0.6000],
                [-0.2000, 0.6000],
                [0.2000, 0.6000],
                [0.6000, 0.6000],
                [1.0000, 0.6000]
            ],
            [
                [-1.0000, 1.0000],
                [-0.6000, 1.0000],
                [-0.2000, 1.0000],
                [0.2000, 1.0000],
                [0.6000, 1.0000],
                [1.0000, 1.0000]
            ]
        ]
    ],
    dtype=np.float32,
)
# Y shape, [N, C, H_out, W_out] - [1, 1, 6, 6]
Y = np.array(
    [
        [
            [
                [0.0000, 0.1500, 0.5500, 0.9500, 1.3500, 0.7500],
                [0.6000, 1.5000, 2.3000, 3.1000, 3.9000, 2.1000],
                [2.2000, 4.7000, 5.5000, 6.3000, 7.1000, 3.7000],
                [3.8000, 7.9000, 8.7000, 9.5000, 10.3000, 5.3000],
                [5.4000, 11.1000, 11.9000, 12.7000, 13.5000, 6.9000],
                [3.0000, 6.1500, 6.5500, 6.9500, 7.3500, 3.7500]
            ]
        ]
    ],
    dtype=np.float32,
)
expect(node, inputs=[X, Grid], outputs=[Y],
       name='test_gridsample')
```

</details>
<details>
<summary>gridsample_mode_aligncorners</summary>

```python
# X shape, [N, C, H, W] - [1, 1, 3, 2]
X = np.array(
    [
        [
            [
                [0., 1.],
                [2., 3.],
                [4., 5.]
            ]
        ]
    ],
    dtype=np.float32,
)
# Grid shape, [N, H_out, W_out, 2] - [1, 2, 4, 2]
Grid = np.array(
    [
        [
            [
                [-1.0000, -1.0000],
                [-0.5000, -0.5000],
                [-0.2000, -0.2000],
                [0.0000, 0.0000]
            ],

            [
                [0.0000, 0.0000],
                [-0.2000, -0.2000],
                [0.5000, 0.5000],
                [1.0000, 1.0000]
            ]
        ]
    ],
    dtype=np.float32,
)

# setting mode = 'bilinear', default align_corners = 0
node = onnx.helper.make_node(
    'GridSample',
    inputs=['X', 'Grid'],
    outputs=['Y'],
    mode='bilinear',
)
# Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
Y_bilinear = np.array(
    [
        [
            [
                [0.0000, 0.5000, 1.7000, 2.5000],
                [2.5000, 1.7000, 4.5000, 1.2500]
            ]
        ]
    ],
    dtype=np.float32,
)

expect(node, inputs=[X, Grid], outputs=[Y_bilinear],
       name='test_gridsample_bilinear')

# setting mode = 'bilinear', align_corners = 1
node = onnx.helper.make_node(
    'GridSample',
    inputs=['X', 'Grid'],
    outputs=['Y'],
    mode='bilinear',
    align_corners=1,
)
# Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
Y_align_corners = np.array(
    [
        [
            [
                [0.0000, 1.2500, 2.0000, 2.5000],
                [2.5000, 2.0000, 3.7500, 5.0000]
            ]
        ]
    ],
    dtype=np.float32,
)

expect(node, inputs=[X, Grid], outputs=[Y_align_corners],
       name='test_gridsample_aligncorners_true')

# setting mode = 'nearest'
node = onnx.helper.make_node(
    'GridSample',
    inputs=['X', 'Grid'],
    outputs=['Y'],
    mode='nearest',
)
# Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
Y_nearest = np.array(
    [
        [
            [
                [0., 0., 2., 2.],
                [2., 2., 5., 0.]
            ]
        ]
    ],
    dtype=np.float32,
)

expect(node, inputs=[X, Grid], outputs=[Y_nearest],
       name='test_gridsample_nearest')

# setting mode = 'bicubic'
node = onnx.helper.make_node(
    'GridSample',
    inputs=['X', 'Grid'],
    outputs=['Y'],
    mode='bicubic',
)
# Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
Y_bicubic = np.array(
    [
        [
            [
                [-0.1406, 0.3828, 1.7556, 2.9688],
                [2.9688, 1.7556, 5.1445, 1.3906]
            ]
        ]
    ],
    dtype=np.float32,
)

expect(node, inputs=[X, Grid], outputs=[Y_bicubic],
       name='test_gridsample_bicubic')
```

</details>
<details>
<summary>gridsample_paddingmode</summary>

```python
# X shape, [N, C, H, W] - [1, 1, 3, 2]
X = np.array(
    [
        [
            [
                [0., 1.],
                [2., 3.],
                [4., 5.]
            ]
        ]
    ],
    dtype=np.float32,
)
# Grid shape, [N, H_out, W_out, 2] - [1, 2, 4, 2]
Grid = np.array(
    [
        [
            [
                [-10.0000, -10.0000],
                [-5.0000, -5.0000],
                [-0.2000, -0.2000],
                [10.0000, 10.0000]
            ],

            [
                [10.0000, 10.0000],
                [-0.2000, -0.2000],
                [5.0000, 5.0000],
                [10.0000, 10.0000]
            ]
        ]
    ],
    dtype=np.float32,
)

# setting padding_mode = 'zeros'
node = onnx.helper.make_node(
    'GridSample',
    inputs=['X', 'Grid'],
    outputs=['Y'],
    padding_mode='zeros',
)
# Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
Y_zeros = np.array(
    [
        [
            [
                [0.0000, 0.0000, 1.7000, 0.0000],
                [0.0000, 1.7000, 0.0000, 0.0000]
            ]
        ]
    ],
    dtype=np.float32,
)

expect(node, inputs=[X, Grid], outputs=[Y_zeros],
       name='test_gridsample_zeros_padding')

# setting padding_mode = 'border'
node = onnx.helper.make_node(
    'GridSample',
    inputs=['X', 'Grid'],
    outputs=['Y'],
    padding_mode='border',
)
# Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
Y_border = np.array(
    [
        [
            [
                [0.0000, 0.0000, 1.7000, 5.0000],
                [5.0000, 1.7000, 5.0000, 5.0000]
            ]
        ]
    ],
    dtype=np.float32,
)

expect(node, inputs=[X, Grid], outputs=[Y_border],
       name='test_gridsample_border_padding')

# setting padding_mode = 'reflection'
node = onnx.helper.make_node(
    'GridSample',
    inputs=['X', 'Grid'],
    outputs=['Y'],
    padding_mode='reflection',
)
# Y shape, [N, C, H_out, W_out] - [1, 1, 2, 4]
Y_reflection = np.array(
    [
        [
            [
                [2.5000, 0.0000, 1.7000, 2.5000],
                [2.5000, 1.7000, 5.0000, 2.5000]
            ]
        ]
    ],
    dtype=np.float32,
)

expect(node, inputs=[X, Grid], outputs=[Y_reflection],
       name='test_gridsample_reflection_padding')
```

</details>


### HammingWindow
There are 1 test cases, listed as following:
<details>
<summary>hammingwindow</summary>

```python
# Test periodic window
node = onnx.helper.make_node(
    'HammingWindow',
    inputs=['x'],
    outputs=['y'],
)
size = np.int32(10)
a0 = 25 / 46
a1 = 1 - a0
y = a0 - a1 * np.cos(2 * 3.1415 * np.arange(0, size, 1, dtype=np.float32) / size)
expect(node, inputs=[size], outputs=[y],
       name='test_hammingwindow')

# Test symmetric window
node = onnx.helper.make_node(
    'HammingWindow',
    inputs=['x'],
    outputs=['y'],
    periodic=0
)
size = np.int32(10)
a0 = 25 / 46
a1 = 1 - a0
y = a0 - a1 * np.cos(2 * 3.1415 * np.arange(0, size, 1, dtype=np.float32) / (size - 1))
expect(node, inputs=[size], outputs=[y],
       name='test_hammingwindow_symmetric')
```

</details>


### HannWindow
There are 1 test cases, listed as following:
<details>
<summary>hannwindow</summary>

```python
# Test periodic window
node = onnx.helper.make_node(
    'HannWindow',
    inputs=['x'],
    outputs=['y'],
)
size = np.int32(10)
a0 = .5
a1 = .5
y = a0 - a1 * np.cos(2 * 3.1415 * np.arange(0, size, 1, dtype=np.float32) / size)
expect(node, inputs=[size], outputs=[y],
       name='test_hannwindow')

# Test symmetric window
node = onnx.helper.make_node(
    'HannWindow',
    inputs=['x'],
    outputs=['y'],
    periodic=0
)
size = np.int32(10)
a0 = .5
a1 = .5
y = a0 - a1 * np.cos(2 * 3.1415 * np.arange(0, size, 1, dtype=np.float32) / (size - 1))
expect(node, inputs=[size], outputs=[y],
       name='test_hannwindow_symmetric')
```

</details>


### HardSigmoid
There are 2 test cases, listed as following:
<details>
<summary>hardsigmoid</summary>

```python
node = onnx.helper.make_node(
    'HardSigmoid',
    inputs=['x'],
    outputs=['y'],
    alpha=0.5,
    beta=0.6
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.clip(x * 0.5 + 0.6, 0, 1)  # expected output [0.1, 0.6, 1.]
expect(node, inputs=[x], outputs=[y],
       name='test_hardsigmoid_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x * 0.5 + 0.6, 0, 1)
expect(node, inputs=[x], outputs=[y],
       name='test_hardsigmoid')
```

</details>
<details>
<summary>hardsigmoid_default</summary>

```python
default_alpha = 0.2
default_beta = 0.5
node = onnx.helper.make_node(
    'HardSigmoid',
    inputs=['x'],
    outputs=['y'],
)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x * default_alpha + default_beta, 0, 1)
expect(node, inputs=[x], outputs=[y],
       name='test_hardsigmoid_default')
```

</details>


### HardSwish
There are 1 test cases, listed as following:
<details>
<summary>hardswish</summary>

```python
node = onnx.helper.make_node(
    'HardSwish',
    inputs=['x'],
    outputs=['y'],
)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = hardswish(x)

expect(node, inputs=[x], outputs=[y],
       name='test_hardswish')
```

</details>


### Hardmax
There are 2 test cases, listed as following:
<details>
<summary>hardmax</summary>

```python
node = onnx.helper.make_node(
    'Hardmax',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([[3, 0, 1, 2], [2, 5, 1, 0], [0, 1, 3, 2],
              [0, 1, 2, 3]]).astype(np.float32)
# expect result:
# [[1. 0. 0. 0.]
# [0. 1. 0. 0.]
# [0. 0. 1. 0.]
# [0. 0. 0. 1.]]
y = hardmax(x)
expect(node, inputs=[x], outputs=[y],
       name='test_hardmax_example')

# For multiple occurrences of the maximal values, the first occurrence is selected for one-hot output
x = np.array([[3, 3, 3, 1]]).astype(np.float32)
# expect result:
# [[1, 0, 0, 0]]
y = hardmax(x)
expect(node, inputs=[x], outputs=[y],
       name='test_hardmax_one_hot')
```

</details>
<details>
<summary>hardmax_axis</summary>

```python
x = np.random.randn(3, 4, 5).astype(np.float32)
node = onnx.helper.make_node(
    'Hardmax',
    inputs=['x'],
    outputs=['y'],
    axis=0,
)
y = hardmax(x, axis=0)
expect(node, inputs=[x], outputs=[y],
       name='test_hardmax_axis_0')

node = onnx.helper.make_node(
    'Hardmax',
    inputs=['x'],
    outputs=['y'],
    axis=1,
)
y = hardmax(x, axis=1)
expect(node, inputs=[x], outputs=[y],
       name='test_hardmax_axis_1')

node = onnx.helper.make_node(
    'Hardmax',
    inputs=['x'],
    outputs=['y'],
    axis=2,
)
y = hardmax(x, axis=2)
expect(node, inputs=[x], outputs=[y],
       name='test_hardmax_axis_2')

node = onnx.helper.make_node(
    'Hardmax',
    inputs=['x'],
    outputs=['y'],
    axis=-1,
)
y = hardmax(x, axis=-1)
expect(node, inputs=[x], outputs=[y],
       name='test_hardmax_negative_axis')

# default axis is -1
node = onnx.helper.make_node(
    'Hardmax',
    inputs=['x'],
    outputs=['y'],
)
expect(node, inputs=[x], outputs=[y],
       name='test_hardmax_default_axis')
```

</details>


### Identity
There are 3 test cases, listed as following:
<details>
<summary>identity</summary>

```python
node = onnx.helper.make_node(
    'Identity',
    inputs=['x'],
    outputs=['y'],
)

data = np.array([[[
    [1, 2],
    [3, 4],
]]], dtype=np.float32)

expect(node, inputs=[data], outputs=[data],
       name='test_identity')
```

</details>
<details>
<summary>identity_opt</summary>

```python
ten_in_tp = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, shape=[5])
seq_in_tp = onnx.helper.make_sequence_type_proto(ten_in_tp)
opt_in_tp = onnx.helper.make_optional_type_proto(seq_in_tp)

identity_node = onnx.helper.make_node(
    'Identity',
    inputs=['opt_in'],
    outputs=['opt_out']
)

x = [np.array([1, 2, 3, 4, 5]).astype(np.float32)]

expect(identity_node, inputs=[x], outputs=[x], name='test_identity_opt',
       opset_imports=[onnx.helper.make_opsetid("", 16)],
       input_type_protos=[opt_in_tp],
       output_type_protos=[opt_in_tp])
```

</details>
<details>
<summary>sequence</summary>

```python
node = onnx.helper.make_node(
    'Identity',
    inputs=['x'],
    outputs=['y'],
)

data = [
    np.array([[[
        [1, 2],
        [3, 4],
    ]]], dtype=np.float32),
    np.array([[[
        [2, 3],
        [1, 5],
    ]]], dtype=np.float32)]

expect(node, inputs=[data], outputs=[data], name='test_identity_sequence')
```

</details>


### If
There are 3 test cases, listed as following:
<details>
<summary>if</summary>

```python
# Given a bool scalar input cond.
# return constant tensor x if cond is True, otherwise return constant tensor y.

then_out = onnx.helper.make_tensor_value_info('then_out', onnx.TensorProto.FLOAT, [5])
else_out = onnx.helper.make_tensor_value_info('else_out', onnx.TensorProto.FLOAT, [5])

x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
y = np.array([5, 4, 3, 2, 1]).astype(np.float32)

then_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['then_out'],
    value=onnx.numpy_helper.from_array(x)
)

else_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['else_out'],
    value=onnx.numpy_helper.from_array(y)
)

then_body = onnx.helper.make_graph(
    [then_const_node],
    'then_body',
    [],
    [then_out]
)

else_body = onnx.helper.make_graph(
    [else_const_node],
    'else_body',
    [],
    [else_out]
)

if_node = onnx.helper.make_node(
    'If',
    inputs=['cond'],
    outputs=['res'],
    then_branch=then_body,
    else_branch=else_body
)

cond = np.array(1).astype(bool)
res = x if cond else y
expect(if_node, inputs=[cond], outputs=[res], name='test_if',
       opset_imports=[onnx.helper.make_opsetid("", 11)])
```

</details>
<details>
<summary>if_optional</summary>

```python
# Given a bool scalar input cond, return an empty optional sequence of
# tensor if True, return an optional sequence with value x
# (the input optional sequence) otherwise.

ten_in_tp = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, shape=[5])
seq_in_tp = onnx.helper.make_sequence_type_proto(ten_in_tp)

then_out_tensor_tp = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, shape=[5])
then_out_seq_tp = onnx.helper.make_sequence_type_proto(then_out_tensor_tp)
then_out_opt_tp = onnx.helper.make_optional_type_proto(then_out_seq_tp)
then_out = onnx.helper.make_value_info('optional_empty', then_out_opt_tp)

else_out_tensor_tp = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, shape=[5])
else_out_seq_tp = onnx.helper.make_sequence_type_proto(else_out_tensor_tp)
else_out_opt_tp = onnx.helper.make_optional_type_proto(else_out_seq_tp)
else_out = onnx.helper.make_value_info('else_opt', else_out_opt_tp)

x = [np.array([1, 2, 3, 4, 5]).astype(np.float32)]
cond = np.array(0).astype(bool)
res = compute_if_outputs(x, cond)

opt_empty_in = onnx.helper.make_node(
    'Optional',
    inputs=[],
    outputs=['optional_empty'],
    type=seq_in_tp
)

then_body = onnx.helper.make_graph(
    [opt_empty_in],
    'then_body',
    [],
    [then_out]
)

else_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['x'],
    value=onnx.numpy_helper.from_array(x[0])
)

else_seq_node = onnx.helper.make_node(
    'SequenceConstruct',
    inputs=['x'],
    outputs=['else_seq']
)

else_optional_seq_node = onnx.helper.make_node(
    'Optional',
    inputs=['else_seq'],
    outputs=['else_opt']
)

else_body = onnx.helper.make_graph(
    [else_const_node, else_seq_node, else_optional_seq_node],
    'else_body',
    [],
    [else_out]
)

if_node = onnx.helper.make_node(
    'If',
    inputs=['cond'],
    outputs=['sequence'],
    then_branch=then_body,
    else_branch=else_body
)

expect(if_node, inputs=[cond], outputs=[res], name='test_if_opt',
       output_type_protos=[else_out_opt_tp],
       opset_imports=[onnx.helper.make_opsetid("", 16)])
```

</details>
<details>
<summary>if_seq</summary>

```python
# Given a bool scalar input cond.
# return constant sequence x if cond is True, otherwise return constant sequence y.

then_out = onnx.helper.make_tensor_sequence_value_info('then_out', onnx.TensorProto.FLOAT, shape=[5])
else_out = onnx.helper.make_tensor_sequence_value_info('else_out', onnx.TensorProto.FLOAT, shape=[5])

x = [np.array([1, 2, 3, 4, 5]).astype(np.float32)]
y = [np.array([5, 4, 3, 2, 1]).astype(np.float32)]

then_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['x'],
    value=onnx.numpy_helper.from_array(x[0])
)

then_seq_node = onnx.helper.make_node(
    'SequenceConstruct',
    inputs=['x'],
    outputs=['then_out']
)

else_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['y'],
    value=onnx.numpy_helper.from_array(y[0])
)

else_seq_node = onnx.helper.make_node(
    'SequenceConstruct',
    inputs=['y'],
    outputs=['else_out']
)

then_body = onnx.helper.make_graph(
    [then_const_node, then_seq_node],
    'then_body',
    [],
    [then_out]
)

else_body = onnx.helper.make_graph(
    [else_const_node, else_seq_node],
    'else_body',
    [],
    [else_out]
)

if_node = onnx.helper.make_node(
    'If',
    inputs=['cond'],
    outputs=['res'],
    then_branch=then_body,
    else_branch=else_body
)

cond = np.array(1).astype(bool)
res = x if cond else y
expect(if_node, inputs=[cond], outputs=[res], name='test_if_seq',
       opset_imports=[onnx.helper.make_opsetid("", 13)])
```

</details>


### InstanceNormalization
There are 1 test cases, listed as following:
<details>
<summary>instancenormalization</summary>

```python
def _instancenorm_test_mode(x, s, bias, epsilon=1e-5):  # type: ignore
    dims_x = len(x.shape)
    axis = tuple(range(2, dims_x))
    mean = np.mean(x, axis=axis, keepdims=True)
    var = np.var(x, axis=axis, keepdims=True)
    dim_ones = (1,) * (dims_x - 2)
    s = s.reshape(-1, *dim_ones)
    bias = bias.reshape(-1, *dim_ones)
    return s * (x - mean) / np.sqrt(var + epsilon) + bias

# input size: (1, 2, 1, 3)
x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
s = np.array([1.0, 1.5]).astype(np.float32)
bias = np.array([0, 1]).astype(np.float32)
y = _instancenorm_test_mode(x, s, bias).astype(np.float32)

node = onnx.helper.make_node(
    'InstanceNormalization',
    inputs=['x', 's', 'bias'],
    outputs=['y'],
)

# output size: (1, 2, 1, 3)
expect(node, inputs=[x, s, bias], outputs=[y],
       name='test_instancenorm_example')

# input size: (2, 3, 4, 5)
x = np.random.randn(2, 3, 4, 5).astype(np.float32)
s = np.random.randn(3).astype(np.float32)
bias = np.random.randn(3).astype(np.float32)
epsilon = 1e-2
y = _instancenorm_test_mode(x, s, bias, epsilon).astype(np.float32)

node = onnx.helper.make_node(
    'InstanceNormalization',
    inputs=['x', 's', 'bias'],
    outputs=['y'],
    epsilon=epsilon,
)

# output size: (2, 3, 4, 5)
expect(node, inputs=[x, s, bias], outputs=[y],
       name='test_instancenorm_epsilon')
```

</details>


### IsInf
There are 3 test cases, listed as following:
<details>
<summary>infinity</summary>

```python
node = onnx.helper.make_node('IsInf',
                             inputs=['x'],
                             outputs=['y'],
                             )

x = np.array([-1.2, np.nan, np.inf, 2.8, np.NINF, np.inf],
             dtype=np.float32)
y = np.isinf(x)
expect(node, inputs=[x], outputs=[y], name='test_isinf')
```

</details>
<details>
<summary>negative_infinity_only</summary>

```python
node = onnx.helper.make_node('IsInf',
                             inputs=['x'],
                             outputs=['y'],
                             detect_positive=0
                             )

x = np.array([-1.7, np.nan, np.inf, -3.6, np.NINF, np.inf],
             dtype=np.float32)
y = np.isneginf(x)
expect(node, inputs=[x], outputs=[y], name='test_isinf_negative')
```

</details>
<details>
<summary>positive_infinity_only</summary>

```python
node = onnx.helper.make_node('IsInf',
                             inputs=['x'],
                             outputs=['y'],
                             detect_negative=0
                             )

x = np.array([-1.7, np.nan, np.inf, 3.6, np.NINF, np.inf],
             dtype=np.float32)
y = np.isposinf(x)
expect(node, inputs=[x], outputs=[y], name='test_isinf_positive')
```

</details>


### IsNaN
There are 1 test cases, listed as following:
<details>
<summary>isnan</summary>

```python
node = onnx.helper.make_node(
    'IsNaN',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([3.0, np.nan, 4.0, np.nan], dtype=np.float32)
y = np.isnan(x)
expect(node, inputs=[x], outputs=[y], name='test_isnan')
```

</details>


### LRN
There are 2 test cases, listed as following:
<details>
<summary>default</summary>

```python
alpha = 0.0001
beta = 0.75
bias = 1.0
nsize = 3
node = onnx.helper.make_node(
    'LRN',
    inputs=['x'],
    outputs=['y'],
    size=3
)
x = np.random.randn(5, 5, 5, 5).astype(np.float32)
square_sum = np.zeros((5, 5, 5, 5)).astype(np.float32)
for n, c, h, w in np.ndindex(x.shape):
    square_sum[n, c, h, w] = sum(x[n,
                                   max(0, c - int(math.floor((nsize - 1) / 2))):min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
                                   h,
                                   w] ** 2)
y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
expect(node, inputs=[x], outputs=[y],
       name='test_lrn_default')
```

</details>
<details>
<summary>lrn</summary>

```python
alpha = 0.0002
beta = 0.5
bias = 2.0
nsize = 3
node = onnx.helper.make_node(
    'LRN',
    inputs=['x'],
    outputs=['y'],
    alpha=alpha,
    beta=beta,
    bias=bias,
    size=nsize
)
x = np.random.randn(5, 5, 5, 5).astype(np.float32)
square_sum = np.zeros((5, 5, 5, 5)).astype(np.float32)
for n, c, h, w in np.ndindex(x.shape):
    square_sum[n, c, h, w] = sum(x[n,
                                   max(0, c - int(math.floor((nsize - 1) / 2))):min(5, c + int(math.ceil((nsize - 1) / 2)) + 1),
                                   h,
                                   w] ** 2)
y = x / ((bias + (alpha / nsize) * square_sum) ** beta)
expect(node, inputs=[x], outputs=[y],
       name='test_lrn')
```

</details>


### LSTM
There are 4 test cases, listed as following:
<details>
<summary>batchwise</summary>

```python
input = np.array([[[1., 2.]], [[3., 4.]], [[5., 6.]]]).astype(np.float32)

input_size = 2
hidden_size = 7
weight_scale = 0.3
number_of_gates = 4
layout = 1

node = onnx.helper.make_node(
    'LSTM',
    inputs=['X', 'W', 'R'],
    outputs=['Y', 'Y_h'],
    hidden_size=hidden_size,
    layout=layout
)

W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

lstm = LSTM_Helper(X=input, W=W, R=R, layout=layout)
Y, Y_h = lstm.step()
expect(node, inputs=[input, W, R], outputs=[Y.astype(np.float32), Y_h.astype(np.float32)], name='test_lstm_batchwise')
```

</details>
<details>
<summary>defaults</summary>

```python
input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

input_size = 2
hidden_size = 3
weight_scale = 0.1
number_of_gates = 4

node = onnx.helper.make_node(
    'LSTM',
    inputs=['X', 'W', 'R'],
    outputs=['', 'Y_h'],
    hidden_size=hidden_size
)

W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

lstm = LSTM_Helper(X=input, W=W, R=R)
_, Y_h = lstm.step()
expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_lstm_defaults')
```

</details>
<details>
<summary>initial_bias</summary>

```python
input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]).astype(np.float32)

input_size = 3
hidden_size = 4
weight_scale = 0.1
custom_bias = 0.1
number_of_gates = 4

node = onnx.helper.make_node(
    'LSTM',
    inputs=['X', 'W', 'R', 'B'],
    outputs=['', 'Y_h'],
    hidden_size=hidden_size
)

W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)

# Adding custom bias
W_B = custom_bias * np.ones((1, number_of_gates * hidden_size)).astype(np.float32)
R_B = np.zeros((1, number_of_gates * hidden_size)).astype(np.float32)
B = np.concatenate((W_B, R_B), 1)

lstm = LSTM_Helper(X=input, W=W, R=R, B=B)
_, Y_h = lstm.step()
expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_lstm_with_initial_bias')
```

</details>
<details>
<summary>peepholes</summary>

```python
input = np.array([[[1., 2., 3., 4.], [5., 6., 7., 8.]]]).astype(np.float32)

input_size = 4
hidden_size = 3
weight_scale = 0.1
number_of_gates = 4
number_of_peepholes = 3

node = onnx.helper.make_node(
    'LSTM',
    inputs=['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h', 'initial_c', 'P'],
    outputs=['', 'Y_h'],
    hidden_size=hidden_size
)

# Initializing Inputs
W = weight_scale * np.ones((1, number_of_gates * hidden_size, input_size)).astype(np.float32)
R = weight_scale * np.ones((1, number_of_gates * hidden_size, hidden_size)).astype(np.float32)
B = np.zeros((1, 2 * number_of_gates * hidden_size)).astype(np.float32)
seq_lens = np.repeat(input.shape[0], input.shape[1]).astype(np.int32)
init_h = np.zeros((1, input.shape[1], hidden_size)).astype(np.float32)
init_c = np.zeros((1, input.shape[1], hidden_size)).astype(np.float32)
P = weight_scale * np.ones((1, number_of_peepholes * hidden_size)).astype(np.float32)

lstm = LSTM_Helper(X=input, W=W, R=R, B=B, P=P, initial_c=init_c, initial_h=init_h)
_, Y_h = lstm.step()
expect(node, inputs=[input, W, R, B, seq_lens, init_h, init_c, P], outputs=[Y_h.astype(np.float32)],
       name='test_lstm_with_peepholes')
```

</details>


### LayerNormalization
There are 4 test cases, listed as following:
<details>
<summary>d</summary>

```python
X = np.random.randn(3, 4).astype(np.float32)

def case(axis: int) -> None:
    normalized_shape = calculate_normalized_shape(X.shape, axis)
    W = np.random.randn(*normalized_shape).astype(np.float32)
    B = np.random.randn(*normalized_shape).astype(np.float32)
    Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis=axis)

    node = onnx.helper.make_node(
        'LayerNormalization',
        inputs=['X', 'W', 'B'],
        outputs=['Y', 'Mean', 'InvStdDev'],
        axis=axis,
    )

    if axis < 0:
        name = f'test_layer_normalization_2d_axis_negative_{-axis}'
    else:
        name = f'test_layer_normalization_2d_axis{axis}'

    expect(node, inputs=[X, W, B], outputs=[Y, mean, inv_std_dev],
           name=name)

for i in range(len(X.shape)):
    case(i)
    case(i - len(X.shape))
```

</details>
<details>
<summary>d_epsilon</summary>

```python
epsilon = 1e-1
X = np.random.randn(2, 3, 5).astype(np.float32)

def case(axis: int) -> None:
    normalized_shape = calculate_normalized_shape(X.shape, axis)
    W = np.random.randn(*normalized_shape).astype(np.float32)
    B = np.random.randn(*normalized_shape).astype(np.float32)
    Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis, epsilon)
    node = onnx.helper.make_node(
        'LayerNormalization',
        inputs=['X', 'W', 'B'],
        outputs=['Y', 'Mean', 'InvStdDev'],
        axis=axis,
        epsilon=epsilon
    )

    if axis < 0:
        name = f'test_layer_normalization_3d_axis_negative_{-axis}_epsilon'
    else:
        name = f'test_layer_normalization_3d_axis{axis}_epsilon'

    expect(node, inputs=[X, W, B], outputs=[Y, mean, inv_std_dev],
           name=name)

for i in range(len(X.shape)):
    case(i)
    case(i - len(X.shape))
```

</details>
<details>
<summary>default_axis</summary>

```python
X = np.random.randn(2, 3, 4, 5).astype(np.float32)

# Default axis in LayerNormalization is -1.
normalized_shape = calculate_normalized_shape(X.shape, -1)
W = np.random.randn(*normalized_shape).astype(np.float32)
B = np.random.randn(*normalized_shape).astype(np.float32)
# Axis is default to -1 in the reference implementation.
Y, mean, inv_std_dev = _layer_normalization(X, W, B)

# Not specifying axis attribute means -1.
node = onnx.helper.make_node(
    'LayerNormalization',
    inputs=['X', 'W', 'B'],
    outputs=['Y', 'Mean', 'InvStdDev']
)

expect(node, inputs=[X, W, B], outputs=[Y, mean, inv_std_dev],
       name='test_layer_normalization_default_axis')
```

</details>
<details>
<summary>layernormalization</summary>

```python
X = np.random.randn(2, 3, 4, 5).astype(np.float32)

def case(axis: int) -> None:
    normalized_shape = calculate_normalized_shape(X.shape, axis)
    W = np.random.randn(*normalized_shape).astype(np.float32)
    B = np.random.randn(*normalized_shape).astype(np.float32)
    Y, mean, inv_std_dev = _layer_normalization(X, W, B, axis)

    node = onnx.helper.make_node(
        'LayerNormalization',
        inputs=['X', 'W', 'B'],
        outputs=['Y', 'Mean', 'InvStdDev'],
        axis=axis,
    )

    if axis < 0:
        name = f'test_layer_normalization_4d_axis_negative_{-axis}'
    else:
        name = f'test_layer_normalization_4d_axis{axis}'

    expect(node, inputs=[X, W, B], outputs=[Y, mean, inv_std_dev],
           name=name)

for i in range(len(X.shape)):
    case(i)
    case(i - len(X.shape))
```

</details>


### LeakyRelu
There are 2 test cases, listed as following:
<details>
<summary>leakyrelu</summary>

```python
node = onnx.helper.make_node(
    'LeakyRelu',
    inputs=['x'],
    outputs=['y'],
    alpha=0.1
)

x = np.array([-1, 0, 1]).astype(np.float32)
# expected output [-0.1, 0., 1.]
y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
expect(node, inputs=[x], outputs=[y],
       name='test_leakyrelu_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * 0.1
expect(node, inputs=[x], outputs=[y],
       name='test_leakyrelu')
```

</details>
<details>
<summary>leakyrelu_default</summary>

```python
default_alpha = 0.01
node = onnx.helper.make_node(
    'LeakyRelu',
    inputs=['x'],
    outputs=['y'],
)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * default_alpha
expect(node, inputs=[x], outputs=[y],
       name='test_leakyrelu_default')
```

</details>


### Less
There are 4 test cases, listed as following:
<details>
<summary>less</summary>

```python
node = onnx.helper.make_node(
    'Less',
    inputs=['x', 'y'],
    outputs=['less'],
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(3, 4, 5).astype(np.float32)
z = np.less(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_less')
```

</details>
<details>
<summary>less</summary>

```python
node = onnx.helper.make_node(
    'LessOrEqual',
    inputs=['x', 'y'],
    outputs=['less_equal'],
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(3, 4, 5).astype(np.float32)
z = np.less_equal(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_less_equal')
```

</details>
<details>
<summary>less_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Less',
    inputs=['x', 'y'],
    outputs=['less'],
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(5).astype(np.float32)
z = np.less(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_less_bcast')
```

</details>
<details>
<summary>less_broadcast</summary>

```python
node = onnx.helper.make_node(
    'LessOrEqual',
    inputs=['x', 'y'],
    outputs=['less_equal'],
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(5).astype(np.float32)
z = np.less_equal(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_less_equal_bcast')
```

</details>


### Log
There are 1 test cases, listed as following:
<details>
<summary>log</summary>

```python
node = onnx.helper.make_node(
    'Log',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([1, 10]).astype(np.float32)
y = np.log(x)  # expected output [0., 2.30258512]
expect(node, inputs=[x], outputs=[y],
       name='test_log_example')

x = np.exp(np.random.randn(3, 4, 5).astype(np.float32))
y = np.log(x)
expect(node, inputs=[x], outputs=[y],
       name='test_log')
```

</details>


### LogSoftmax
There are 2 test cases, listed as following:
<details>
<summary>logsoftmax</summary>

```python
node = onnx.helper.make_node(
    'LogSoftmax',
    inputs=['x'],
    outputs=['y'],
)
x = np.array([[-1, 0, 1]]).astype(np.float32)
# expected output
# [[-2.4076061 -1.407606  -0.407606 ]]
y = logsoftmax(x)
expect(node, inputs=[x], outputs=[y],
       name='test_logsoftmax_example_1')
```

</details>
<details>
<summary>logsoftmax_axis</summary>

```python
x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]
             ).astype(np.float32)
# expected output
# [[-3.4401896  -2.4401896  -1.4401896  -0.44018966]
# [-3.4401896  -2.4401896  -1.4401896  -0.44018966]]
y = logsoftmax(x)

node = onnx.helper.make_node(
    'LogSoftmax',
    inputs=['x'],
    outputs=['y'],
)
expect(node, inputs=[x], outputs=[y],
       name='test_logsoftmax_large_number')

x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
node = onnx.helper.make_node(
    'LogSoftmax',
    inputs=['x'],
    outputs=['y'],
    axis=0,
)
y = logsoftmax(x, axis=0)
expect(node, inputs=[x], outputs=[y],
       name='test_logsoftmax_axis_0')

node = onnx.helper.make_node(
    'LogSoftmax',
    inputs=['x'],
    outputs=['y'],
    axis=1,
)
y = logsoftmax(x, axis=1)
expect(node, inputs=[x], outputs=[y],
       name='test_logsoftmax_axis_1')

node = onnx.helper.make_node(
    'LogSoftmax',
    inputs=['x'],
    outputs=['y'],
    axis=2,
)
y = logsoftmax(x, axis=2)
expect(node, inputs=[x], outputs=[y],
       name='test_logsoftmax_axis_2')

node = onnx.helper.make_node(
    'LogSoftmax',
    inputs=['x'],
    outputs=['y'],
    axis=-1,
)
y = logsoftmax(x, axis=-1)
expect(node, inputs=[x], outputs=[y],
       name='test_logsoftmax_negative_axis')

# default axis is -1
node = onnx.helper.make_node(
    'LogSoftmax',
    inputs=['x'],
    outputs=['y'],
)
expect(node, inputs=[x], outputs=[y],
       name='test_logsoftmax_default_axis')
```

</details>


### Loop
There are 3 test cases, listed as following:
<details>
<summary>loop_11</summary>

```python
# Given a tensor x of values [x1, ..., xN], and initial tensor y
# sum up its elements using a scan
# returning the final state (y+x1+x2+...+xN) as well the scan_output
# [y+x1, y+x1+x2, ..., y+x1+x2+...+xN]

y_in = onnx.helper.make_tensor_value_info('y_in', onnx.TensorProto.FLOAT, [1])
y_out = onnx.helper.make_tensor_value_info('y_out', onnx.TensorProto.FLOAT, [1])
scan_out = onnx.helper.make_tensor_value_info('scan_out', onnx.TensorProto.FLOAT, [1])
cond_in = onnx.helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
cond_out = onnx.helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
iter_count = onnx.helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])

x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
y = np.array([-2]).astype(np.float32)

x_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['x'],
    value=onnx.helper.make_tensor(
        name='const_tensor_x',
        data_type=onnx.TensorProto.FLOAT,
        dims=x.shape,
        vals=x.flatten().astype(float),
    )
)

one_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['one'],
    value=onnx.helper.make_tensor(
        name='const_tensor_one',
        data_type=onnx.TensorProto.INT64,
        dims=(),
        vals=[1]
    )
)

i_add_node = onnx.helper.make_node(
    'Add',
    inputs=['iter_count', 'one'],
    outputs=['end']
)

start_unsqueeze_node = onnx.helper.make_node(
    'Unsqueeze',
    inputs=['iter_count'],
    outputs=['slice_start'],
    axes=[0]
)

end_unsqueeze_node = onnx.helper.make_node(
    'Unsqueeze',
    inputs=['end'],
    outputs=['slice_end'],
    axes=[0]
)

slice_node = onnx.helper.make_node(
    'Slice',
    inputs=['x', 'slice_start', 'slice_end'],
    outputs=['slice_out']
)

y_add_node = onnx.helper.make_node(
    'Add',
    inputs=['y_in', 'slice_out'],
    outputs=['y_out']
)

identity_node = onnx.helper.make_node(
    'Identity',
    inputs=['cond_in'],
    outputs=['cond_out']
)

scan_identity_node = onnx.helper.make_node(
    'Identity',
    inputs=['y_out'],
    outputs=['scan_out']
)

loop_body = onnx.helper.make_graph(
    [identity_node, x_const_node, one_const_node, i_add_node,
     start_unsqueeze_node, end_unsqueeze_node, slice_node, y_add_node,
     scan_identity_node],
    'loop_body',
    [iter_count, cond_in, y_in],
    [cond_out, y_out, scan_out]
)

node = onnx.helper.make_node(
    'Loop',
    inputs=['trip_count', 'cond', 'y'],
    outputs=['res_y', 'res_scan'],
    body=loop_body
)

trip_count = np.array(5).astype(np.int64)
res_y = np.array([13]).astype(np.float32)
cond = np.array(1).astype(bool)
res_scan = np.array([-1, 1, 4, 8, 13]).astype(np.float32).reshape((5, 1))
expect(node, inputs=[trip_count, cond, y], outputs=[res_y, res_scan],
       name='test_loop11', opset_imports=[onnx.helper.make_opsetid("", 11)])
```

</details>
<details>
<summary>loop_13</summary>

```python
# Given a tensor x of values [x1, ..., xN],
# Return a sequence of tensors of
#   [[x1], [x1, x2], ..., [x1, ..., xN]]

seq_in = onnx.helper.make_tensor_sequence_value_info('seq_in', onnx.TensorProto.FLOAT, None)
seq_out = onnx.helper.make_tensor_sequence_value_info('seq_out', onnx.TensorProto.FLOAT, None)
cond_in = onnx.helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
cond_out = onnx.helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
iter_count = onnx.helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])

x = np.array([1, 2, 3, 4, 5]).astype(np.float32)

x_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['x'],
    value=onnx.helper.make_tensor(
        name='const_tensor_x',
        data_type=onnx.TensorProto.FLOAT,
        dims=x.shape,
        vals=x.flatten().astype(float),
    )
)

one_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['one'],
    value=onnx.helper.make_tensor(
        name='const_tensor_one',
        data_type=onnx.TensorProto.INT64,
        dims=(),
        vals=[1]
    )
)

zero_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['slice_start'],
    value=onnx.helper.make_tensor(
        name='const_tensor_zero',
        data_type=onnx.TensorProto.INT64,
        dims=(1,),
        vals=[0]
    )
)

axes_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['axes'],
    value=onnx.helper.make_tensor(
        name='const_tensor_axes',
        data_type=onnx.TensorProto.INT64,
        dims=(),
        vals=[0]
    )
)

add_node = onnx.helper.make_node(
    'Add',
    inputs=['iter_count', 'one'],
    outputs=['end']
)

end_unsqueeze_node = onnx.helper.make_node(
    'Unsqueeze',
    inputs=['end', 'axes'],
    outputs=['slice_end']
)

slice_node = onnx.helper.make_node(
    'Slice',
    inputs=['x', 'slice_start', 'slice_end'],
    outputs=['slice_out']
)

insert_node = onnx.helper.make_node(
    'SequenceInsert',
    inputs=['seq_in', 'slice_out'],
    outputs=['seq_out']
)

identity_node = onnx.helper.make_node(
    'Identity',
    inputs=['cond_in'],
    outputs=['cond_out']
)

loop_body = onnx.helper.make_graph(
    [identity_node, x_const_node, one_const_node, zero_const_node, add_node,
     axes_node, end_unsqueeze_node, slice_node, insert_node],
    'loop_body',
    [iter_count, cond_in, seq_in],
    [cond_out, seq_out]
)

node = onnx.helper.make_node(
    'Loop',
    inputs=['trip_count', 'cond', 'seq_empty'],
    outputs=['seq_res'],
    body=loop_body
)

trip_count = np.array(5).astype(np.int64)
seq_empty: List[Any] = []
seq_res = [x[:int(i)] for i in x]
cond = np.array(1).astype(bool)
expect(node, inputs=[trip_count, cond, seq_empty], outputs=[seq_res],
       name='test_loop13_seq', opset_imports=[onnx.helper.make_opsetid("", 13)],
       input_type_protos=[onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT64, trip_count.shape),
                          onnx.helper.make_tensor_type_proto(onnx.TensorProto.BOOL, cond.shape),
                          onnx.helper.make_sequence_type_proto(
                              onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, []))])
```

</details>
<details>
<summary>loop_16_none</summary>

```python
# Given a tensor sequence of values [x1, ..., xN], and an initial optional sequence of tensors [x0],
# Return a concatenated sequence of tensors of
#   [x0, [x1], [x1, x2], ..., [x1, ..., xN]]

ten_in_tp = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [])
seq_in_tp = onnx.helper.make_sequence_type_proto(ten_in_tp)
opt_in_tp = onnx.helper.make_optional_type_proto(seq_in_tp)
opt_in = onnx.helper.make_value_info('opt_seq_in', opt_in_tp)
seq_out = onnx.helper.make_tensor_sequence_value_info('seq_out', onnx.TensorProto.FLOAT, [])
cond_in = onnx.helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
cond_out = onnx.helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
iter_count = onnx.helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])

x0 = np.array(0).astype(np.float32)
x = np.array([1, 2, 3, 4, 5]).astype(np.float32)

optional_has_elem_node = onnx.helper.make_node(
    'OptionalHasElement',
    inputs=['opt_seq_in'],
    outputs=['optional_has_elem']
)

optional_is_none = onnx.helper.make_node(
    'Not',
    inputs=['optional_has_elem'],
    outputs=['optional_is_none']
)

optional_get_elem = onnx.helper.make_node(
    'OptionalGetElement',
    inputs=['opt_seq_in'],
    outputs=['seq_in']
)

constant_in = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['constant_in'],
    value=onnx.helper.make_tensor(
        name='const_tensor',
        data_type=onnx.TensorProto.FLOAT,
        dims=(),
        vals=[0]
    )
)

seq_const_in = onnx.helper.make_node(
    'SequenceConstruct',
    inputs=['constant_in'],
    outputs=['init_seq_in']
)

then_seq_out = onnx.helper.make_tensor_sequence_value_info('init_seq_in', onnx.TensorProto.FLOAT, [])
then_body = onnx.helper.make_graph(
    [constant_in, seq_const_in],
    'then_body',
    [],
    [then_seq_out]
)

else_seq_out = onnx.helper.make_tensor_sequence_value_info('seq_in', onnx.TensorProto.FLOAT, [])
else_body = onnx.helper.make_graph(
    [optional_get_elem],
    'else_body',
    [],
    [else_seq_out]
)

if_node = onnx.helper.make_node(
    'If',
    inputs=['optional_is_none'],
    outputs=['sequence'],
    then_branch=then_body,
    else_branch=else_body
)

x_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['x'],
    value=onnx.helper.make_tensor(
        name='const_tensor_x',
        data_type=onnx.TensorProto.FLOAT,
        dims=x.shape,
        vals=x.flatten().astype(float),
    )
)

one_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['one'],
    value=onnx.helper.make_tensor(
        name='const_tensor_one',
        data_type=onnx.TensorProto.INT64,
        dims=(),
        vals=[1]
    )
)

zero_const_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['slice_start'],
    value=onnx.helper.make_tensor(
        name='const_tensor_zero',
        data_type=onnx.TensorProto.INT64,
        dims=(1,),
        vals=[0]
    )
)

axes_node = onnx.helper.make_node(
    'Constant',
    inputs=[],
    outputs=['axes'],
    value=onnx.helper.make_tensor(
        name='const_tensor_axes',
        data_type=onnx.TensorProto.INT64,
        dims=(),
        vals=[0]
    )
)

add_node = onnx.helper.make_node(
    'Add',
    inputs=['iter_count', 'one'],
    outputs=['end']
)

end_unsqueeze_node = onnx.helper.make_node(
    'Unsqueeze',
    inputs=['end', 'axes'],
    outputs=['slice_end']
)

slice_node = onnx.helper.make_node(
    'Slice',
    inputs=['x', 'slice_start', 'slice_end'],
    outputs=['slice_out']
)

insert_node = onnx.helper.make_node(
    'SequenceInsert',
    inputs=['sequence', 'slice_out'],
    outputs=['seq_out']
)

identity_node = onnx.helper.make_node(
    'Identity',
    inputs=['cond_in'],
    outputs=['cond_out']
)

loop_body = onnx.helper.make_graph(
    [identity_node, optional_has_elem_node, optional_is_none, if_node, x_const_node, one_const_node,
     zero_const_node, add_node, axes_node, end_unsqueeze_node, slice_node, insert_node],
    'loop_body',
    [iter_count, cond_in, opt_in],
    [cond_out, seq_out]
)

node = onnx.helper.make_node(
    'Loop',
    inputs=['trip_count', 'cond', 'opt_seq'],
    outputs=['seq_res'],
    body=loop_body
)

trip_count = np.array(5).astype(np.int64)
cond = np.array(1).astype(bool)
seq_res = compute_loop_outputs(x, [x0], trip_count)
opt_seq_in: List[Any] = [x0]
expect(node, inputs=[trip_count, cond, opt_seq_in], outputs=[seq_res],
       name='test_loop16_seq_none', opset_imports=[onnx.helper.make_opsetid("", 16)],
       input_type_protos=[onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT64, trip_count.shape),
                          onnx.helper.make_tensor_type_proto(onnx.TensorProto.BOOL, cond.shape),
                          opt_in_tp])
```

</details>


### MatMul
There are 1 test cases, listed as following:
<details>
<summary>matmul</summary>

```python
node = onnx.helper.make_node(
    'MatMul',
    inputs=['a', 'b'],
    outputs=['c'],
)

# 2d
a = np.random.randn(3, 4).astype(np.float32)
b = np.random.randn(4, 3).astype(np.float32)
c = np.matmul(a, b)
expect(node, inputs=[a, b], outputs=[c],
       name='test_matmul_2d')

# 3d
a = np.random.randn(2, 3, 4).astype(np.float32)
b = np.random.randn(2, 4, 3).astype(np.float32)
c = np.matmul(a, b)
expect(node, inputs=[a, b], outputs=[c],
       name='test_matmul_3d')

# 4d
a = np.random.randn(1, 2, 3, 4).astype(np.float32)
b = np.random.randn(1, 2, 4, 3).astype(np.float32)
c = np.matmul(a, b)
expect(node, inputs=[a, b], outputs=[c],
       name='test_matmul_4d')
```

</details>


### MatMulInteger
There are 1 test cases, listed as following:
<details>
<summary>matmulinteger</summary>

```python
node = onnx.helper.make_node('MatMulInteger',
    inputs=['A', 'B', 'a_zero_point', 'b_zero_point'],
    outputs=['Y'],)

A = np.array([[11, 7, 3],
    [10, 6, 2],
    [9, 5, 1],
    [8, 4, 0], ], dtype=np.uint8)

a_zero_point = np.array([12], dtype=np.uint8)

B = np.array([[1, 4],
    [2, 5],
    [3, 6], ], dtype=np.uint8)

b_zero_point = np.array([0], dtype=np.uint8)

output = np.array([[-38, -83],
    [-44, -98],
    [-50, -113],
    [-56, -128], ], dtype=np.int32)

expect(node, inputs=[A, B, a_zero_point, b_zero_point], outputs=[output],
       name='test_matmulinteger')
```

</details>


### Max
There are 2 test cases, listed as following:
<details>
<summary>max</summary>

```python
data_0 = np.array([3, 2, 1]).astype(np.float32)
data_1 = np.array([1, 4, 4]).astype(np.float32)
data_2 = np.array([2, 5, 3]).astype(np.float32)
result = np.array([3, 5, 4]).astype(np.float32)
node = onnx.helper.make_node(
    'Max',
    inputs=['data_0', 'data_1', 'data_2'],
    outputs=['result'],
)
expect(node, inputs=[data_0, data_1, data_2], outputs=[result],
       name='test_max_example')

node = onnx.helper.make_node(
    'Max',
    inputs=['data_0'],
    outputs=['result'],
)
expect(node, inputs=[data_0], outputs=[data_0],
       name='test_max_one_input')

result = np.maximum(data_0, data_1)
node = onnx.helper.make_node(
    'Max',
    inputs=['data_0', 'data_1'],
    outputs=['result'],
)
expect(node, inputs=[data_0, data_1], outputs=[result],
       name='test_max_two_inputs')
```

</details>
<details>
<summary>max_all_numeric_types</summary>

```python
for op_dtype in all_numeric_dtypes:
    data_0 = np.array([3, 2, 1]).astype(op_dtype)
    data_1 = np.array([1, 4, 4]).astype(op_dtype)
    result = np.array([3, 4, 4]).astype(op_dtype)
    node = onnx.helper.make_node(
        'Max',
        inputs=['data_0', 'data_1'],
        outputs=['result'],
    )
    expect(node, inputs=[data_0, data_1], outputs=[result],
           name=f'test_max_{np.dtype(op_dtype).name}')
```

</details>


### MaxPool
There are 15 test cases, listed as following:
<details>
<summary>maxpool_1d_default</summary>

```python
"""
input_shape: [1, 3, 32]
output_shape: [1, 3, 31]
"""
node = onnx.helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[2],
)
x = np.random.randn(1, 3, 32).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = [2]
strides = [1]
out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
padded = x
y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0], 'MAX')

expect(node, inputs=[x], outputs=[y], name='test_maxpool_1d_default')
```

</details>
<details>
<summary>maxpool_2d_ceil</summary>

```python
"""
input_shape: [1, 1, 4, 4]
output_shape: [1, 1, 2, 2]
"""
node = onnx.helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[3, 3],
    strides=[2, 2],
    ceil_mode=True
)
x = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]]).astype(np.float32)
y = np.array([[[
    [11, 12],
    [15, 16]]]]).astype(np.float32)

expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_ceil')
```

</details>
<details>
<summary>maxpool_2d_default</summary>

```python
"""
input_shape: [1, 3, 32, 32]
output_shape: [1, 3, 31, 31]
"""
node = onnx.helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[2, 2],
)
x = np.random.randn(1, 3, 32, 32).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = (2, 2)
strides = (1, 1)
out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
padded = x
y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'MAX')

expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_default')
```

</details>
<details>
<summary>maxpool_2d_dilations</summary>

```python
"""
input_shape: [1, 1, 4, 4]
output_shape: [1, 1, 2, 2]
"""
node = onnx.helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[2, 2],
    strides=[1, 1],
    dilations=[2, 2]
)
x = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]]).astype(np.float32)
y = np.array([[[
    [11, 12],
    [15, 16]]]]).astype(np.float32)

expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_dilations')
```

</details>
<details>
<summary>maxpool_2d_pads</summary>

```python
"""
input_shape: [1, 3, 28, 28]
output_shape: [1, 3, 30, 30]
pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
"""
node = onnx.helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[3, 3],
    pads=[2, 2, 2, 2]
)
x = np.random.randn(1, 3, 28, 28).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = (3, 3)
strides = (1, 1)
pad_bottom = pad_top = pad_right = pad_left = 2
pad_shape = [pad_top + pad_bottom, pad_left + pad_right]
out_shape = get_output_shape('VALID', np.add(x_shape[2:], pad_shape), kernel_shape, strides)
padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                constant_values=np.nan)
y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'MAX')

expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_pads')
```

</details>
<details>
<summary>maxpool_2d_precomputed_pads</summary>

```python
"""
input_shape: [1, 1, 5, 5]
output_shape: [1, 1, 5, 5]
pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
"""
node = onnx.helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[5, 5],
    pads=[2, 2, 2, 2]

)
x = np.array([[[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
]]]).astype(np.float32)
y = np.array([[[
    [13, 14, 15, 15, 15],
    [18, 19, 20, 20, 20],
    [23, 24, 25, 25, 25],
    [23, 24, 25, 25, 25],
    [23, 24, 25, 25, 25]]]]).astype(np.float32)

expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_precomputed_pads')
```

</details>
<details>
<summary>maxpool_2d_precomputed_same_upper</summary>

```python
"""
input_shape: [1, 1, 5, 5]
output_shape: [1, 1, 3, 3]
pad_shape: [2, 2] -> [1, 1, 1, 1] by axis
"""
node = onnx.helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[3, 3],
    strides=[2, 2],
    auto_pad='SAME_UPPER'
)
x = np.array([[[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
]]]).astype(np.float32)
y = np.array([[[[7, 9, 10],
                [17, 19, 20],
                [22, 24, 25]]]]).astype(np.float32)

expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_precomputed_same_upper')
```

</details>
<details>
<summary>maxpool_2d_precomputed_strides</summary>

```python
"""
input_shape: [1, 1, 5, 5]
output_shape: [1, 1, 2, 2]
"""
node = onnx.helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[2, 2],
    strides=[2, 2]
)
x = np.array([[[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
]]]).astype(np.float32)
y = np.array([[[[7, 9],
                [17, 19]]]]).astype(np.float32)

expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_precomputed_strides')
```

</details>
<details>
<summary>maxpool_2d_same_lower</summary>

```python
"""
input_shape: [1, 3, 32, 32]
output_shape: [1, 3, 32, 32]
pad_shape: [1, 1] -> [1, 0, 1, 0] by axis
"""
node = onnx.helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[2, 2],
    auto_pad='SAME_LOWER'
)
x = np.random.randn(1, 3, 32, 32).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = (2, 2)
strides = (1, 1)
out_shape = get_output_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides)
pad_shape = get_pad_shape('SAME_LOWER', x_shape[2:], kernel_shape, strides, out_shape)
pad_bottom = pad_shape[0] // 2
pad_top = pad_shape[0] - pad_bottom
pad_right = pad_shape[1] // 2
pad_left = pad_shape[1] - pad_right
padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                constant_values=np.nan)
y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'MAX')

expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_same_lower')
```

</details>
<details>
<summary>maxpool_2d_same_upper</summary>

```python
"""
input_shape: [1, 3, 32, 32]
output_shape: [1, 3, 32, 32]
pad_shape: [1, 1] -> [0, 1, 0, 1] by axis
"""
node = onnx.helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[2, 2],
    auto_pad='SAME_UPPER'
)
x = np.random.randn(1, 3, 32, 32).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = (2, 2)
strides = (1, 1)
out_shape = get_output_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides)
pad_shape = get_pad_shape('SAME_UPPER', x_shape[2:], kernel_shape, strides, out_shape)
pad_top = pad_shape[0] // 2
pad_bottom = pad_shape[0] - pad_top
pad_left = pad_shape[1] // 2
pad_right = pad_shape[1] - pad_left
padded = np.pad(x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                constant_values=np.nan)
y = pool(padded, x_shape, kernel_shape, strides, out_shape, pad_shape, 'MAX')

expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_same_upper')
```

</details>
<details>
<summary>maxpool_2d_strides</summary>

```python
"""
input_shape: [1, 3, 32, 32]
output_shape: [1, 3, 10, 10]
"""
node = onnx.helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[5, 5],
    strides=[3, 3]
)
x = np.random.randn(1, 3, 32, 32).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = (5, 5)
strides = (3, 3)
out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
padded = x
y = pool(padded, x_shape, kernel_shape, strides, out_shape, (0, 0), 'MAX')

expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_strides')
```

</details>
<details>
<summary>maxpool_2d_uint8</summary>

```python
"""
input_shape: [1, 1, 5, 5]
output_shape: [1, 1, 5, 5]
pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
"""
node = onnx.helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[5, 5],
    pads=[2, 2, 2, 2]
)
x = np.array([[[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
]]]).astype(np.uint8)
y = np.array([[[
    [13, 14, 15, 15, 15],
    [18, 19, 20, 20, 20],
    [23, 24, 25, 25, 25],
    [23, 24, 25, 25, 25],
    [23, 24, 25, 25, 25]]]]).astype(np.uint8)

expect(node, inputs=[x], outputs=[y], name='test_maxpool_2d_uint8')
```

</details>
<details>
<summary>maxpool_3d_default</summary>

```python
"""
input_shape: [1, 3, 32, 32, 32]
output_shape: [1, 3, 31, 31, 31]
"""
node = onnx.helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y'],
    kernel_shape=[2, 2, 2],
)
x = np.random.randn(1, 3, 32, 32, 32).astype(np.float32)
x_shape = np.shape(x)
kernel_shape = [2, 2, 2]
strides = [1, 1, 1]
out_shape = get_output_shape('VALID', x_shape[2:], kernel_shape, strides)
padded = x
y = pool(padded, x_shape, kernel_shape, strides, out_shape, [0, 0, 0], 'MAX')

expect(node, inputs=[x], outputs=[y], name='test_maxpool_3d_default')
```

</details>
<details>
<summary>maxpool_with_argmax_2d_precomputed_pads</summary>

```python
"""
input_shape: [1, 1, 5, 5]
output_shape: [1, 1, 5, 5]
pad_shape: [4, 4] -> [2, 2, 2, 2] by axis
"""
node = onnx.helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y', 'z'],
    kernel_shape=[5, 5],
    pads=[2, 2, 2, 2]
)
x = np.array([[[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
]]]).astype(np.float32)
y = np.array([[[
    [13, 14, 15, 15, 15],
    [18, 19, 20, 20, 20],
    [23, 24, 25, 25, 25],
    [23, 24, 25, 25, 25],
    [23, 24, 25, 25, 25]]]]).astype(np.float32)
z = np.array([[[
    [12, 13, 14, 14, 14],
    [17, 18, 19, 19, 19],
    [22, 23, 24, 24, 24],
    [22, 23, 24, 24, 24],
    [22, 23, 24, 24, 24]]]]).astype(np.int64)

expect(node, inputs=[x], outputs=[y, z], name='test_maxpool_with_argmax_2d_precomputed_pads')
```

</details>
<details>
<summary>maxpool_with_argmax_2d_precomputed_strides</summary>

```python
"""
input_shape: [1, 1, 5, 5]
output_shape: [1, 1, 2, 2]
"""
node = onnx.helper.make_node(
    'MaxPool',
    inputs=['x'],
    outputs=['y', 'z'],
    kernel_shape=[2, 2],
    strides=[2, 2],
    storage_order=1
)
x = np.array([[[
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25],
]]]).astype(np.float32)
y = np.array([[[[7, 9],
                [17, 19]]]]).astype(np.float32)
z = np.array([[[[6, 16],
                [8, 18]]]]).astype(np.int64)

expect(node, inputs=[x], outputs=[y, z], name='test_maxpool_with_argmax_2d_precomputed_strides')
```

</details>


### MaxUnpool
There are 2 test cases, listed as following:
<details>
<summary>with_output_shape</summary>

```python
node = onnx.helper.make_node(
    'MaxUnpool',
    inputs=['xT', 'xI', 'output_shape'],
    outputs=['y'],
    kernel_shape=[2, 2],
    strides=[2, 2]
)
xT = np.array([[[[5, 6],
                 [7, 8]]]], dtype=np.float32)
xI = np.array([[[[5, 7],
                 [13, 15]]]], dtype=np.int64)
output_shape = np.array((1, 1, 5, 5), dtype=np.int64)
y = np.array([[[[0, 0, 0, 0, 0],
                [0, 5, 0, 6, 0],
                [0, 0, 0, 0, 0],
                [0, 7, 0, 8, 0],
                [0, 0, 0, 0, 0]]]], dtype=np.float32)
expect(node, inputs=[xT, xI, output_shape], outputs=[y], name='test_maxunpool_export_with_output_shape')
```

</details>
<details>
<summary>without_output_shape</summary>

```python
node = onnx.helper.make_node(
    'MaxUnpool',
    inputs=['xT', 'xI'],
    outputs=['y'],
    kernel_shape=[2, 2],
    strides=[2, 2]
)
xT = np.array([[[[1, 2],
                 [3, 4]]]], dtype=np.float32)
xI = np.array([[[[5, 7],
                 [13, 15]]]], dtype=np.int64)
y = np.array([[[[0, 0, 0, 0],
                [0, 1, 0, 2],
                [0, 0, 0, 0],
                [0, 3, 0, 4]]]], dtype=np.float32)
expect(node, inputs=[xT, xI], outputs=[y], name='test_maxunpool_export_without_output_shape')
```

</details>


### Mean
There are 1 test cases, listed as following:
<details>
<summary>mean</summary>

```python
data_0 = np.array([3, 0, 2]).astype(np.float32)
data_1 = np.array([1, 3, 4]).astype(np.float32)
data_2 = np.array([2, 6, 6]).astype(np.float32)
result = np.array([2, 3, 4]).astype(np.float32)
node = onnx.helper.make_node(
    'Mean',
    inputs=['data_0', 'data_1', 'data_2'],
    outputs=['result'],
)
expect(node, inputs=[data_0, data_1, data_2], outputs=[result],
       name='test_mean_example')

node = onnx.helper.make_node(
    'Mean',
    inputs=['data_0'],
    outputs=['result'],
)
expect(node, inputs=[data_0], outputs=[data_0],
       name='test_mean_one_input')

result = np.divide(np.add(data_0, data_1), 2.)
node = onnx.helper.make_node(
    'Mean',
    inputs=['data_0', 'data_1'],
    outputs=['result'],
)
expect(node, inputs=[data_0, data_1], outputs=[result],
       name='test_mean_two_inputs')
```

</details>


### MeanVarianceNormalization
There are 1 test cases, listed as following:
<details>
<summary>meanvariancenormalization</summary>

```python
node = onnx.helper.make_node(
    'MeanVarianceNormalization',
    inputs=['X'],
    outputs=['Y']
)

input_data = np.array([[[[0.8439683], [0.5665144], [0.05836735]],
    [[0.02916367], [0.12964272], [0.5060197]],
    [[0.79538304], [0.9411346], [0.9546573]]],
    [[[0.17730942], [0.46192095], [0.26480448]],
    [[0.6746842], [0.01665257], [0.62473077]],
    [[0.9240844], [0.9722341], [0.11965699]]],
    [[[0.41356155], [0.9129373], [0.59330076]],
    [[0.81929934], [0.7862604], [0.11799799]],
    [[0.69248444], [0.54119414], [0.07513223]]]], dtype=np.float32)

# Calculate expected output data
data_mean = np.mean(input_data, axis=(0, 2, 3), keepdims=1)
data_mean_squared = np.power(data_mean, 2)
data_squared = np.power(input_data, 2)
data_squared_mean = np.mean(data_squared, axis=(0, 2, 3), keepdims=1)
std = np.sqrt(data_squared_mean - data_mean_squared)
expected_output = (input_data - data_mean) / (std + 1e-9)

expect(node, inputs=[input_data], outputs=[expected_output],
       name='test_mvn')
```

</details>


### MelWeightMatrix
There are 1 test cases, listed as following:
<details>
<summary>melweightmatrix</summary>

```python
node = onnx.helper.make_node(
    "MelWeightMatrix",
    inputs=['num_mel_bins', 'dft_length', 'sample_rate', 'lower_edge_hertz', 'upper_edge_hertz'],
    outputs=['output'],
)

num_mel_bins = np.int32(8)
dft_length = np.int32(16)
sample_rate = np.int32(8192)
lower_edge_hertz = np.float32(0)
upper_edge_hertz = np.float32(8192 / 2)

num_spectrogram_bins = dft_length // 2 + 1
frequency_bins = np.arange(0, num_mel_bins + 2)

low_frequency_mel = 2595 * np.log10(1 + lower_edge_hertz / 700)
high_frequency_mel = 2595 * np.log10(1 + upper_edge_hertz / 700)
mel_step = (high_frequency_mel - low_frequency_mel) / frequency_bins.shape[0]

frequency_bins = frequency_bins * mel_step + low_frequency_mel
frequency_bins = 700 * (np.power(10, (frequency_bins / 2595)) - 1)
frequency_bins = ((dft_length + 1) * frequency_bins) // sample_rate
frequency_bins = frequency_bins.astype(int)

output = np.zeros((num_spectrogram_bins, num_mel_bins))
output.flags.writeable = True

for i in range(num_mel_bins):
    lower_frequency_value = frequency_bins[i]     # left
    center_frequency_point = frequency_bins[i + 1]  # center
    higher_frequency_point = frequency_bins[i + 2]  # right
    low_to_center = center_frequency_point - lower_frequency_value
    if low_to_center == 0:
        output[center_frequency_point, i] = 1
    else:
        for j in range(lower_frequency_value, center_frequency_point + 1):
            output[j, i] = float(j - lower_frequency_value) / float(low_to_center)
    center_to_high = higher_frequency_point - center_frequency_point
    if center_to_high > 0:
        for j in range(center_frequency_point, higher_frequency_point):
            output[j, i] = float(higher_frequency_point - j) / float(center_to_high)

# Expected output
# 1.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
# 0.000000, 0.000000, 1.000000, 1.000000, 0.000000, 0.000000, 0.000000, 0.000000,
# 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000, 0.000000,
# 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000, 0.000000,
# 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000,
# 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 1.000000,
# 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
# 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
# 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000,
output = output.astype(np.float32)
expect(node, inputs=[num_mel_bins, dft_length, sample_rate, lower_edge_hertz, upper_edge_hertz], outputs=[output],
        name='test_melweightmatrix')
```

</details>


### Min
There are 2 test cases, listed as following:
<details>
<summary>min</summary>

```python
data_0 = np.array([3, 2, 1]).astype(np.float32)
data_1 = np.array([1, 4, 4]).astype(np.float32)
data_2 = np.array([2, 5, 0]).astype(np.float32)
result = np.array([1, 2, 0]).astype(np.float32)
node = onnx.helper.make_node(
    'Min',
    inputs=['data_0', 'data_1', 'data_2'],
    outputs=['result'],
)
expect(node, inputs=[data_0, data_1, data_2], outputs=[result],
       name='test_min_example')

node = onnx.helper.make_node(
    'Min',
    inputs=['data_0'],
    outputs=['result'],
)
expect(node, inputs=[data_0], outputs=[data_0],
       name='test_min_one_input')

result = np.minimum(data_0, data_1)
node = onnx.helper.make_node(
    'Min',
    inputs=['data_0', 'data_1'],
    outputs=['result'],
)
expect(node, inputs=[data_0, data_1], outputs=[result],
       name='test_min_two_inputs')
```

</details>
<details>
<summary>min_all_numeric_types</summary>

```python
for op_dtype in all_numeric_dtypes:
    data_0 = np.array([3, 2, 1]).astype(op_dtype)
    data_1 = np.array([1, 4, 4]).astype(op_dtype)
    result = np.array([1, 2, 1]).astype(op_dtype)
    node = onnx.helper.make_node(
        'Min',
        inputs=['data_0', 'data_1'],
        outputs=['result'],
    )
    expect(node, inputs=[data_0, data_1], outputs=[result],
           name=f'test_min_{np.dtype(op_dtype).name}')
```

</details>


### Mod
There are 13 test cases, listed as following:
<details>
<summary>mod_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Mod',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.arange(0, 30).reshape([3, 2, 5]).astype(np.int32)
y = np.array([7]).astype(np.int32)
z = np.mod(x, y)
#   array([[[0, 1, 2, 3, 4],
#     [5, 6, 0, 1, 2]],

#    [[3, 4, 5, 6, 0],
#     [1, 2, 3, 4, 5]],

#    [[6, 0, 1, 2, 3],
#     [4, 5, 6, 0, 1]]], dtype=int32)
expect(node, inputs=[x, y], outputs=[z],
       name='test_mod_broadcast')
```

</details>
<details>
<summary>mod_int64_fmod</summary>

```python
node = onnx.helper.make_node(
    'Mod',
    inputs=['x', 'y'],
    outputs=['z'],
    fmod=1
)

x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int64)
y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int64)
z = np.fmod(x, y)  # expected output [ 0,  1,  5,  0, -1,  3]
expect(node, inputs=[x, y], outputs=[z],
       name='test_mod_int64_fmod')
```

</details>
<details>
<summary>mod_mixed_sign_float16</summary>

```python
node = onnx.helper.make_node(
    'Mod',
    inputs=['x', 'y'],
    outputs=['z'],
    fmod=1
)

x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float16)
y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float16)
z = np.fmod(x, y)  # expected output [-0.10156, 0.3984 , 5. , 0.10156, -0.3984 ,  3.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_mod_mixed_sign_float16')
```

</details>
<details>
<summary>mod_mixed_sign_float32</summary>

```python
node = onnx.helper.make_node(
    'Mod',
    inputs=['x', 'y'],
    outputs=['z'],
    fmod=1
)

x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float32)
y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float32)
z = np.fmod(x, y)  # expected output [-0.10000038, 0.39999962, 5. , 0.10000038, -0.39999962, 3.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_mod_mixed_sign_float32')
```

</details>
<details>
<summary>mod_mixed_sign_float64</summary>

```python
node = onnx.helper.make_node(
    'Mod',
    inputs=['x', 'y'],
    outputs=['z'],
    fmod=1
)

x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0]).astype(np.float64)
y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0]).astype(np.float64)
z = np.fmod(x, y)  # expected output [-0.1,  0.4,  5. ,  0.1, -0.4,  3.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_mod_mixed_sign_float64')
```

</details>
<details>
<summary>mod_mixed_sign_int16</summary>

```python
node = onnx.helper.make_node(
    'Mod',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int16)
y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int16)
z = np.mod(x, y)  # expected output [ 0, -2,  5,  0,  2,  3]
expect(node, inputs=[x, y], outputs=[z],
       name='test_mod_mixed_sign_int16')
```

</details>
<details>
<summary>mod_mixed_sign_int32</summary>

```python
node = onnx.helper.make_node(
    'Mod',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int32)
y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int32)
z = np.mod(x, y)  # expected output [ 0, -2,  5,  0,  2,  3]
expect(node, inputs=[x, y], outputs=[z],
       name='test_mod_mixed_sign_int32')
```

</details>
<details>
<summary>mod_mixed_sign_int64</summary>

```python
node = onnx.helper.make_node(
    'Mod',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int64)
y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int64)
z = np.mod(x, y)  # expected output [ 0, -2,  5,  0,  2,  3]
expect(node, inputs=[x, y], outputs=[z],
       name='test_mod_mixed_sign_int64')
```

</details>
<details>
<summary>mod_mixed_sign_int8</summary>

```python
node = onnx.helper.make_node(
    'Mod',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.array([-4, 7, 5, 4, -7, 8]).astype(np.int8)
y = np.array([2, -3, 8, -2, 3, 5]).astype(np.int8)
z = np.mod(x, y)  # expected output [ 0, -2,  5,  0,  2,  3]
expect(node, inputs=[x, y], outputs=[z],
       name='test_mod_mixed_sign_int8')
```

</details>
<details>
<summary>mod_uint16</summary>

```python
node = onnx.helper.make_node(
    'Mod',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.array([4, 7, 5]).astype(np.uint16)
y = np.array([2, 3, 8]).astype(np.uint16)
z = np.mod(x, y)  # expected output [0, 1, 5]
expect(node, inputs=[x, y], outputs=[z],
       name='test_mod_uint16')
```

</details>
<details>
<summary>mod_uint32</summary>

```python
node = onnx.helper.make_node(
    'Mod',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.array([4, 7, 5]).astype(np.uint32)
y = np.array([2, 3, 8]).astype(np.uint32)
z = np.mod(x, y)  # expected output [0, 1, 5]
expect(node, inputs=[x, y], outputs=[z],
       name='test_mod_uint32')
```

</details>
<details>
<summary>mod_uint64</summary>

```python
node = onnx.helper.make_node(
    'Mod',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.array([4, 7, 5]).astype(np.uint64)
y = np.array([2, 3, 8]).astype(np.uint64)
z = np.mod(x, y)  # expected output [0, 1, 5]
expect(node, inputs=[x, y], outputs=[z],
       name='test_mod_uint64')
```

</details>
<details>
<summary>mod_uint8</summary>

```python
node = onnx.helper.make_node(
    'Mod',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.array([4, 7, 5]).astype(np.uint8)
y = np.array([2, 3, 8]).astype(np.uint8)
z = np.mod(x, y)  # expected output [0, 1, 5]
expect(node, inputs=[x, y], outputs=[z],
       name='test_mod_uint8')
```

</details>


### Momentum
There are 3 test cases, listed as following:
<details>
<summary>momentum</summary>

```python
# Define operator attributes.
norm_coefficient = 0.001
alpha = 0.95
beta = 0.1

# Create operator.
node = onnx.helper.make_node('Momentum',
                             inputs=['R', 'T', 'X', 'G', 'V'],
                             outputs=['X_new', 'V_new'],
                             norm_coefficient=norm_coefficient,
                             alpha=alpha,
                             beta=beta,
                             mode='standard',
                             domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN
                             )

# Define operator inputs.
r = np.array(0.1, dtype=np.float32)  # scalar
t = np.array(0, dtype=np.int64)  # scalar
x = np.array([1.2, 2.8], dtype=np.float32)
g = np.array([-0.94, -2.5], dtype=np.float32)
v = np.array([1.7, 3.6], dtype=np.float32)

# Compute expected outputs of Momentum.
x_new, v_new = apply_momentum(r, t, x, g, v,
                              norm_coefficient, alpha, beta)

# Check results.
expect(node, inputs=[r, t, x, g, v],
       outputs=[x_new, v_new], name='test_momentum',
       opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])
```

</details>
<details>
<summary>momentum_multiple</summary>

```python
# Define operator attributes.
norm_coefficient = 0.001
alpha = 0.95
beta = 0.85

node = onnx.helper.make_node('Momentum',
                             inputs=['R', 'T', 'X1', 'X2',
                                     'G1', 'G2', 'H1', 'H2'],
                             outputs=['X1_new', 'X2_new',
                                      'V1_new', 'V2_new'],
                             norm_coefficient=norm_coefficient,
                             alpha=alpha,
                             beta=beta,
                             mode='standard',
                             domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN
                             )

# Define operator inputs.
r = np.array(0.1, dtype=np.float32)  # scalar
t = np.array(0, dtype=np.int64)  # scalar

x1 = np.array([1.0], dtype=np.float32)
g1 = np.array([-1.0], dtype=np.float32)
v1 = np.array([2.0], dtype=np.float32)

x2 = np.array([1.0, 2.0], dtype=np.float32)
g2 = np.array([-1.0, -3.0], dtype=np.float32)
v2 = np.array([4.0, 1.0], dtype=np.float32)

# Compute expected outputs of Momentum.
x1_new, v1_new = apply_momentum(r, t, x1, g1, v1,
                                norm_coefficient, alpha, beta)
x2_new, v2_new = apply_momentum(r, t, x2, g2, v2,
                                norm_coefficient, alpha, beta)

# Check results.
expect(node, inputs=[r, t, x1, x2, g1, g2, v1, v2],
       outputs=[x1_new, x2_new, v1_new, v2_new], name='test_momentum_multiple',
       opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])
```

</details>
<details>
<summary>nesterov_momentum</summary>

```python
# Define operator attributes.
norm_coefficient = 0.01
alpha = 0.95
beta = 1.0

# Create operator.
node = onnx.helper.make_node('Momentum',
                             inputs=['R', 'T', 'X', 'G', 'V'],
                             outputs=['X_new', 'V_new'],
                             norm_coefficient=norm_coefficient,
                             alpha=alpha,
                             beta=beta,
                             mode='nesterov',
                             domain=AI_ONNX_PREVIEW_TRAINING_DOMAIN
                             )

# Define operator inputs.
r = np.array(0.1, dtype=np.float32)  # scalar
t = np.array(0, dtype=np.int64)  # scalar
x = np.array([1.2, 2.8], dtype=np.float32)
g = np.array([-0.94, -2.5], dtype=np.float32)
v = np.array([1.7, 3.6], dtype=np.float32)

# Compute expected outputs of Momentum.
x_new, v_new = apply_nesterov(r, t, x, g, v,
                              norm_coefficient, alpha, beta)

# Check results.
expect(node, inputs=[r, t, x, g, v],
       outputs=[x_new, v_new], name='test_nesterov_momentum',
       opset_imports=[onnx.helper.make_opsetid(AI_ONNX_PREVIEW_TRAINING_DOMAIN, 1)])
```

</details>


### Mul
There are 2 test cases, listed as following:
<details>
<summary>mul</summary>

```python
node = onnx.helper.make_node(
    'Mul',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.array([1, 2, 3]).astype(np.float32)
y = np.array([4, 5, 6]).astype(np.float32)
z = x * y  # expected output [4., 10., 18.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_mul_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(3, 4, 5).astype(np.float32)
z = x * y
expect(node, inputs=[x, y], outputs=[z],
       name='test_mul')

x = np.random.randint(4, size=(3, 4, 5), dtype=np.uint8)
y = np.random.randint(24, size=(3, 4, 5), dtype=np.uint8)
z = x * y
expect(node, inputs=[x, y], outputs=[z],
       name='test_mul_uint8')
```

</details>
<details>
<summary>mul_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Mul',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(5).astype(np.float32)
z = x * y
expect(node, inputs=[x, y], outputs=[z],
       name='test_mul_bcast')
```

</details>


### Neg
There are 1 test cases, listed as following:
<details>
<summary>neg</summary>

```python
node = onnx.helper.make_node(
    'Neg',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-4, 2]).astype(np.float32)
y = np.negative(x)  # expected output [4., -2.],
expect(node, inputs=[x], outputs=[y],
       name='test_neg_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.negative(x)
expect(node, inputs=[x], outputs=[y],
       name='test_neg')
```

</details>


### NegativeLogLikelihoodLoss
There are 18 test cases, listed as following:
<details>
<summary>input_shape_is_NC</summary>

```python
reduction = 'none'
node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target'],
    outputs=['loss'],
    reduction=reduction
)

N, C = 3, 5
np.random.seed(0)
input = np.random.rand(N, C).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, )).astype(np.int64)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction)

expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NC')
```

</details>
<details>
<summary>input_shape_is_NCd1</summary>

```python
reduction = 'mean'
node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target'],
    outputs=['loss'],
    reduction=reduction
)

N, C, d1 = 3, 5, 2
np.random.seed(0)
input = np.random.rand(N, C, d1).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, d1)).astype(np.int64)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction)

expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1')
```

</details>
<details>
<summary>input_shape_is_NCd1_ii</summary>

```python
reduction = 'mean'
ignore_index = np.int64(1)
node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target'],
    outputs=['loss'],
    reduction=reduction,
    ignore_index=ignore_index
)

N, C, d1 = 3, 5, 2
np.random.seed(0)
input = np.random.rand(N, C, d1).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, d1)).astype(np.int64)
target[0][0] = np.int64(1)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction, ignore_index=ignore_index)

expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1_ii')
```

</details>
<details>
<summary>input_shape_is_NCd1_mean_weight_negative_ii</summary>

```python
reduction = 'mean'
ignore_index = np.int64(-1)

node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target', 'weight'],
    outputs=['loss'],
    reduction=reduction,
    ignore_index=ignore_index)

N, C, dim1 = 3, 5, 6
np.random.seed(0)
input = np.random.rand(N, C, dim1).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, dim1)).astype(np.int64)
target[0][0] = -1
weight = np.random.rand(C).astype(np.float32)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input,
                                                                    target,
                                                                    weight=weight,
                                                                    reduction=reduction,
                                                                    ignore_index=ignore_index)

expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1_mean_weight_negative_ii')
```

</details>
<details>
<summary>input_shape_is_NCd1_weight</summary>

```python
reduction = 'mean'
node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target', 'weight'],
    outputs=['loss'],
    reduction=reduction
)

N, C, d1 = 3, 5, 2
np.random.seed(0)
input = np.random.rand(N, C, d1).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, d1)).astype(np.int64)
weight = np.random.rand(C).astype(np.float32)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction)

expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1_weight')
```

</details>
<details>
<summary>input_shape_is_NCd1_weight_ii</summary>

```python
reduction = 'mean'
ignore_index = np.int64(1)
node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target', 'weight'],
    outputs=['loss'],
    reduction=reduction,
    ignore_index=ignore_index
)

N, C, d1 = 3, 5, 2
np.random.seed(0)
input = np.random.rand(N, C, d1).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, d1)).astype(np.int64)
target[0][0] = np.int64(1)
weight = np.random.rand(C).astype(np.float32)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction, ignore_index=ignore_index)

expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1_weight_ii')
```

</details>
<details>
<summary>input_shape_is_NCd1d2</summary>

```python
reduction = 'none'
node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target'],
    outputs=['loss'],
    reduction=reduction
)

N, C, dim1, dim2 = 3, 5, 6, 6
np.random.seed(0)
input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction)

expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1d2')
```

</details>
<details>
<summary>input_shape_is_NCd1d2_no_weight_reduction_mean_ii</summary>

```python
reduction = 'mean'
ignore_index = np.int64(1)
node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target'],
    outputs=['loss'],
    reduction=reduction,
    ignore_index=ignore_index
)

N, C, dim1, dim2 = 3, 5, 6, 6
np.random.seed(0)
input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
target[0][0][0] = np.int64(1)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, reduction=reduction, ignore_index=ignore_index)

expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1d2_no_weight_reduction_mean_ii')
```

</details>
<details>
<summary>input_shape_is_NCd1d2_reduction_mean</summary>

```python
reduction = 'mean'
node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target'],
    outputs=['loss'],
    reduction=reduction
)

N, C, dim1, dim2 = 3, 5, 6, 6
np.random.seed(0)
input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction)

expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1d2_reduction_mean')
```

</details>
<details>
<summary>input_shape_is_NCd1d2_reduction_sum</summary>

```python
reduction = 'sum'
node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target'],
    outputs=['loss'],
    reduction=reduction
)

N, C, dim1, dim2 = 3, 5, 6, 6
np.random.seed(0)
input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=None, reduction=reduction)

expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1d2_reduction_sum')
```

</details>
<details>
<summary>input_shape_is_NCd1d2_with_weight</summary>

```python
reduction = 'none'
node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target', 'weight'],
    outputs=['loss'],
    reduction=reduction
)

N, C, dim1, dim2 = 3, 5, 6, 6
np.random.seed(0)
input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
weight = np.random.rand(C).astype(np.float32)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction)

expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1d2_with_weight')
```

</details>
<details>
<summary>input_shape_is_NCd1d2_with_weight_reduction_mean</summary>

```python
reduction = 'mean'
node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target', 'weight'],
    outputs=['loss'],
    reduction=reduction
)

N, C, dim1, dim2 = 3, 5, 6, 6
np.random.seed(0)
input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
weight = np.random.rand(C).astype(np.float32)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction)

expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1d2_with_weight_reduction_mean')
```

</details>
<details>
<summary>input_shape_is_NCd1d2_with_weight_reduction_sum</summary>

```python
reduction = 'sum'
node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target', 'weight'],
    outputs=['loss'],
    reduction=reduction
)

N, C, dim1, dim2 = 3, 5, 6, 6
np.random.seed(0)
input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
weight = np.random.rand(C).astype(np.float32)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction)

expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1d2_with_weight_reduction_sum')
```

</details>
<details>
<summary>input_shape_is_NCd1d2_with_weight_reduction_sum_ii</summary>

```python
reduction = 'sum'
ignore_index = np.int64(0)
node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target', 'weight'],
    outputs=['loss'],
    reduction=reduction,
    ignore_index=ignore_index
)

N, C, dim1, dim2 = 3, 5, 6, 6
np.random.seed(0)
input = np.random.rand(N, C, dim1, dim2).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, dim1, dim2)).astype(np.int64)
target[0][0][0] = np.int64(0)
weight = np.random.rand(C).astype(np.float32)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input, target, weight=weight, reduction=reduction, ignore_index=ignore_index)

expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1d2_with_weight_reduction_sum_ii')
```

</details>
<details>
<summary>input_shape_is_NCd1d2d3_none_no_weight_negative_ii</summary>

```python
reduction = 'none'
ignore_index = np.int64(-5)

node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target'],
    outputs=['loss'],
    reduction=reduction,
    ignore_index=ignore_index)

N, C, dim1, dim2, dim3 = 3, 5, 6, 6, 5
np.random.seed(0)
input = np.random.rand(N, C, dim1, dim2, dim3).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3)).astype(np.int64)
target[0][0][0][0] = -5

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input,
                                                                    target,
                                                                    reduction=reduction,
                                                                    ignore_index=ignore_index)

expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1d2d3_none_no_weight_negative_ii')
```

</details>
<details>
<summary>input_shape_is_NCd1d2d3_sum_weight_high_ii</summary>

```python
reduction = 'sum'
ignore_index = np.int64(10)

node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target', 'weight'],
    outputs=['loss'],
    reduction=reduction,
    ignore_index=ignore_index)

N, C = 3, 5
np.random.seed(0)
input = np.random.rand(N, C).astype(np.float32)
target = np.random.randint(0, high=C, size=(N)).astype(np.int64)
target[0] = 10
weight = np.random.rand(C).astype(np.float32)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input,
                                                                    target,
                                                                    weight=weight,
                                                                    reduction=reduction,
                                                                    ignore_index=ignore_index)

expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1d2d3_sum_weight_high_ii')
```

</details>
<details>
<summary>input_shape_is_NCd1d2d3d4d5_mean_weight</summary>

```python
reduction = 'mean'

node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target', 'weight'],
    outputs=['loss'],
    reduction=reduction)

N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
np.random.seed(0)
input = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)).astype(np.int64)
weight = np.random.rand(C).astype(np.float32)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input,
                                                                    target,
                                                                    weight=weight,
                                                                    reduction=reduction)

expect(node, inputs=[input, target, weight], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1d2d3d4d5_mean_weight')
```

</details>
<details>
<summary>input_shape_is_NCd1d2d3d4d5_none_no_weight</summary>

```python
reduction = 'none'

node = onnx.helper.make_node(
    'NegativeLogLikelihoodLoss',
    inputs=['input', 'target'],
    outputs=['loss'],
    reduction=reduction)

N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
np.random.seed(0)
input = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
target = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)).astype(np.int64)

negative_log_likelihood_loss = compute_negative_log_likelihood_loss(input,
                                                                    target,
                                                                    reduction=reduction)

expect(node, inputs=[input, target], outputs=[negative_log_likelihood_loss],
    name='test_nllloss_NCd1d2d3d4d5_none_no_weight')
```

</details>


### NonMaxSuppression
There are 9 test cases, listed as following:
<details>
<summary>nonmaxsuppression_center_point_box_format</summary>

```python
node = onnx.helper.make_node(
    'NonMaxSuppression',
    inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
    outputs=['selected_indices'],
    center_point_box=1
)
boxes = np.array([[
    [0.5, 0.5, 1.0, 1.0],
    [0.5, 0.6, 1.0, 1.0],
    [0.5, 0.4, 1.0, 1.0],
    [0.5, 10.5, 1.0, 1.0],
    [0.5, 10.6, 1.0, 1.0],
    [0.5, 100.5, 1.0, 1.0]
]]).astype(np.float32)
scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
max_output_boxes_per_class = np.array([3]).astype(np.int64)
iou_threshold = np.array([0.5]).astype(np.float32)
score_threshold = np.array([0.0]).astype(np.float32)
selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)

expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_center_point_box_format')
```

</details>
<details>
<summary>nonmaxsuppression_flipped_coordinates</summary>

```python
node = onnx.helper.make_node(
    'NonMaxSuppression',
    inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
    outputs=['selected_indices']
)
boxes = np.array([[
    [1.0, 1.0, 0.0, 0.0],
    [0.0, 0.1, 1.0, 1.1],
    [0.0, 0.9, 1.0, -0.1],
    [0.0, 10.0, 1.0, 11.0],
    [1.0, 10.1, 0.0, 11.1],
    [1.0, 101.0, 0.0, 100.0]
]]).astype(np.float32)
scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
max_output_boxes_per_class = np.array([3]).astype(np.int64)
iou_threshold = np.array([0.5]).astype(np.float32)
score_threshold = np.array([0.0]).astype(np.float32)
selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)

expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_flipped_coordinates')
```

</details>
<details>
<summary>nonmaxsuppression_identical_boxes</summary>

```python
node = onnx.helper.make_node(
    'NonMaxSuppression',
    inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
    outputs=['selected_indices']
)
boxes = np.array([[
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],

    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0]
]]).astype(np.float32)
scores = np.array([[[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]]]).astype(np.float32)
max_output_boxes_per_class = np.array([3]).astype(np.int64)
iou_threshold = np.array([0.5]).astype(np.float32)
score_threshold = np.array([0.0]).astype(np.float32)
selected_indices = np.array([[0, 0, 0]]).astype(np.int64)

expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_identical_boxes')
```

</details>
<details>
<summary>nonmaxsuppression_limit_output_size</summary>

```python
node = onnx.helper.make_node(
    'NonMaxSuppression',
    inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
    outputs=['selected_indices']
)
boxes = np.array([[
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.1, 1.0, 1.1],
    [0.0, -0.1, 1.0, 0.9],
    [0.0, 10.0, 1.0, 11.0],
    [0.0, 10.1, 1.0, 11.1],
    [0.0, 100.0, 1.0, 101.0]
]]).astype(np.float32)
scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
max_output_boxes_per_class = np.array([2]).astype(np.int64)
iou_threshold = np.array([0.5]).astype(np.float32)
score_threshold = np.array([0.0]).astype(np.float32)
selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)

expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_limit_output_size')
```

</details>
<details>
<summary>nonmaxsuppression_single_box</summary>

```python
node = onnx.helper.make_node(
    'NonMaxSuppression',
    inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
    outputs=['selected_indices']
)
boxes = np.array([[
    [0.0, 0.0, 1.0, 1.0]
]]).astype(np.float32)
scores = np.array([[[0.9]]]).astype(np.float32)
max_output_boxes_per_class = np.array([3]).astype(np.int64)
iou_threshold = np.array([0.5]).astype(np.float32)
score_threshold = np.array([0.0]).astype(np.float32)
selected_indices = np.array([[0, 0, 0]]).astype(np.int64)

expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_single_box')
```

</details>
<details>
<summary>nonmaxsuppression_suppress_by_IOU</summary>

```python
node = onnx.helper.make_node(
    'NonMaxSuppression',
    inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
    outputs=['selected_indices']
)
boxes = np.array([[
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.1, 1.0, 1.1],
    [0.0, -0.1, 1.0, 0.9],
    [0.0, 10.0, 1.0, 11.0],
    [0.0, 10.1, 1.0, 11.1],
    [0.0, 100.0, 1.0, 101.0]
]]).astype(np.float32)
scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
max_output_boxes_per_class = np.array([3]).astype(np.int64)
iou_threshold = np.array([0.5]).astype(np.float32)
score_threshold = np.array([0.0]).astype(np.float32)
selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 0, 5]]).astype(np.int64)

expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_suppress_by_IOU')
```

</details>
<details>
<summary>nonmaxsuppression_suppress_by_IOU_and_scores</summary>

```python
node = onnx.helper.make_node(
    'NonMaxSuppression',
    inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
    outputs=['selected_indices']
)
boxes = np.array([[
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.1, 1.0, 1.1],
    [0.0, -0.1, 1.0, 0.9],
    [0.0, 10.0, 1.0, 11.0],
    [0.0, 10.1, 1.0, 11.1],
    [0.0, 100.0, 1.0, 101.0]
]]).astype(np.float32)
scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
max_output_boxes_per_class = np.array([3]).astype(np.int64)
iou_threshold = np.array([0.5]).astype(np.float32)
score_threshold = np.array([0.4]).astype(np.float32)
selected_indices = np.array([[0, 0, 3], [0, 0, 0]]).astype(np.int64)

expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_suppress_by_IOU_and_scores')
```

</details>
<details>
<summary>nonmaxsuppression_two_batches</summary>

```python
node = onnx.helper.make_node(
    'NonMaxSuppression',
    inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
    outputs=['selected_indices']
)
boxes = np.array([[[0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9],
                   [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1],
                   [0.0, 100.0, 1.0, 101.0]],
                  [[0.0, 0.0, 1.0, 1.0],
                   [0.0, 0.1, 1.0, 1.1],
                   [0.0, -0.1, 1.0, 0.9],
                   [0.0, 10.0, 1.0, 11.0],
                   [0.0, 10.1, 1.0, 11.1],
                   [0.0, 100.0, 1.0, 101.0]]]).astype(np.float32)
scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]],
                   [[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
max_output_boxes_per_class = np.array([2]).astype(np.int64)
iou_threshold = np.array([0.5]).astype(np.float32)
score_threshold = np.array([0.0]).astype(np.float32)
selected_indices = np.array([[0, 0, 3], [0, 0, 0], [1, 0, 3], [1, 0, 0]]).astype(np.int64)

expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_two_batches')
```

</details>
<details>
<summary>nonmaxsuppression_two_classes</summary>

```python
node = onnx.helper.make_node(
    'NonMaxSuppression',
    inputs=['boxes', 'scores', 'max_output_boxes_per_class', 'iou_threshold', 'score_threshold'],
    outputs=['selected_indices']
)
boxes = np.array([[
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.1, 1.0, 1.1],
    [0.0, -0.1, 1.0, 0.9],
    [0.0, 10.0, 1.0, 11.0],
    [0.0, 10.1, 1.0, 11.1],
    [0.0, 100.0, 1.0, 101.0]
]]).astype(np.float32)
scores = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3],
                    [0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]]).astype(np.float32)
max_output_boxes_per_class = np.array([2]).astype(np.int64)
iou_threshold = np.array([0.5]).astype(np.float32)
score_threshold = np.array([0.0]).astype(np.float32)
selected_indices = np.array([[0, 0, 3], [0, 0, 0], [0, 1, 3], [0, 1, 0]]).astype(np.int64)

expect(node, inputs=[boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold], outputs=[selected_indices], name='test_nonmaxsuppression_two_classes')
```

</details>


### NonZero
There are 1 test cases, listed as following:
<details>
<summary>nonzero</summary>

```python
node = onnx.helper.make_node(
    'NonZero',
    inputs=['condition'],
    outputs=['result'],
)

condition = np.array([[1, 0], [1, 1]], dtype=bool)
result = np.array(np.nonzero(condition), dtype=np.int64)  # expected output [[0, 1, 1], [0, 0, 1]]
expect(node, inputs=[condition], outputs=[result],
       name='test_nonzero_example')
```

</details>


### Not
There are 1 test cases, listed as following:
<details>
<summary>not</summary>

```python
node = onnx.helper.make_node(
    'Not',
    inputs=['x'],
    outputs=['not'],
)

# 2d
x = (np.random.randn(3, 4) > 0).astype(bool)
expect(node, inputs=[x], outputs=[np.logical_not(x)],
       name='test_not_2d')

# 3d
x = (np.random.randn(3, 4, 5) > 0).astype(bool)
expect(node, inputs=[x], outputs=[np.logical_not(x)],
       name='test_not_3d')

# 4d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
expect(node, inputs=[x], outputs=[np.logical_not(x)],
       name='test_not_4d')
```

</details>


### OneHot
There are 4 test cases, listed as following:
<details>
<summary>with_axis</summary>

```python
axisValue = 1
on_value = 3
off_value = 1
output_type = np.float32
node = onnx.helper.make_node(
    'OneHot',
    inputs=['indices', 'depth', 'values'],
    outputs=['y'],
    axis=axisValue
)
indices = np.array([[1, 9],
                    [2, 4]], dtype=np.float32)
depth = np.float32(10)
values = np.array([off_value, on_value], dtype=output_type)
y = one_hot(indices, depth, axis=axisValue, dtype=output_type)
y = y * (on_value - off_value) + off_value
expect(node, inputs=[indices, depth, values], outputs=[y], name='test_onehot_with_axis')
```

</details>
<details>
<summary>with_negative_axis</summary>

```python
axisValue = -2
on_value = 3
off_value = 1
output_type = np.float32
node = onnx.helper.make_node(
    'OneHot',
    inputs=['indices', 'depth', 'values'],
    outputs=['y'],
    axis=axisValue
)
indices = np.array([[1, 9],
                    [2, 4]], dtype=np.float32)
depth = np.float32(10)
values = np.array([off_value, on_value], dtype=output_type)
y = one_hot(indices, depth, axis=axisValue, dtype=output_type)
y = y * (on_value - off_value) + off_value
expect(node, inputs=[indices, depth, values], outputs=[y], name='test_onehot_with_negative_axis')
```

</details>
<details>
<summary>with_negative_indices</summary>

```python
axisValue = 1
on_value = 3
off_value = 1
output_type = np.float32
node = onnx.helper.make_node(
    'OneHot',
    inputs=['indices', 'depth', 'values'],
    outputs=['y'],
    axis=axisValue
)
indices = np.array([0, -7, -8], dtype=np.int64)

# print(y)
# [[3. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
#  [1. 1. 1. 3. 1. 1. 1. 1. 1. 1.]
#  [1. 1. 3. 1. 1. 1. 1. 1. 1. 1.]]

depth = np.float32(10)
values = np.array([off_value, on_value], dtype=output_type)
y = one_hot(indices, depth, axis=axisValue, dtype=output_type)
y = y * (on_value - off_value) + off_value
expect(node, inputs=[indices, depth, values], outputs=[y], name='test_onehot_negative_indices')
```

</details>
<details>
<summary>without_axis</summary>

```python
on_value = 5
off_value = 2
output_type = np.int32
node = onnx.helper.make_node(
    'OneHot',
    inputs=['indices', 'depth', 'values'],
    outputs=['y']
)
indices = np.array([0, 7, 8], dtype=np.int64)
depth = np.float32(12)
values = np.array([off_value, on_value], dtype=output_type)
y = one_hot(indices, depth, dtype=output_type)
y = y * (on_value - off_value) + off_value
expect(node, inputs=[indices, depth, values], outputs=[y], name='test_onehot_without_axis')
```

</details>


### OptionalHasElement
There are 4 test cases, listed as following:
<details>
<summary>empty</summary>

```python
optional = None
tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=onnx.TensorProto.INT32, shape=[])
input_type_proto = onnx.helper.make_optional_type_proto(tensor_type_proto)
node = onnx.helper.make_node(
    'OptionalHasElement',
    inputs=['optional_input'],
    outputs=['output']
)
output = optional_has_element_reference_implementation(optional)
expect(node, inputs=[optional], outputs=[output],
       input_type_protos=[input_type_proto],
       name='test_optional_has_element_empty')
```

</details>
<details>
<summary>get_element_sequence</summary>

```python
optional = [np.array([1, 2, 3, 4]).astype(np.int32)]
tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=onnx.TensorProto.INT32, shape=[4, ])
seq_type_proto = onnx.helper.make_sequence_type_proto(tensor_type_proto)
input_type_proto = onnx.helper.make_optional_type_proto(seq_type_proto)

node = onnx.helper.make_node(
    'OptionalGetElement',
    inputs=['optional_input'],
    outputs=['output']
)
output = optional_get_element_reference_implementation(optional)
expect(node, inputs=[optional], outputs=[output],
       input_type_protos=[input_type_proto],
       name='test_optional_get_element_sequence')
```

</details>
<details>
<summary>get_element_tensor</summary>

```python
optional = np.array([1, 2, 3, 4]).astype(np.float32)
tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=onnx.TensorProto.FLOAT, shape=[4, ])
input_type_proto = onnx.helper.make_optional_type_proto(tensor_type_proto)

node = onnx.helper.make_node(
    'OptionalGetElement',
    inputs=['optional_input'],
    outputs=['output']
)
output = optional_get_element_reference_implementation(optional)
expect(node, inputs=[optional], outputs=[output],
       input_type_protos=[input_type_proto],
       name='test_optional_get_element')
```

</details>
<details>
<summary>optionalhaselement</summary>

```python
optional = np.array([1, 2, 3, 4]).astype(np.float32)
tensor_type_proto = onnx.helper.make_tensor_type_proto(elem_type=onnx.TensorProto.FLOAT, shape=[4, ])
input_type_proto = onnx.helper.make_optional_type_proto(tensor_type_proto)
node = onnx.helper.make_node(
    'OptionalHasElement',
    inputs=['optional_input'],
    outputs=['output']
)
output = optional_has_element_reference_implementation(optional)
expect(node, inputs=[optional], outputs=[output],
       input_type_protos=[input_type_proto],
       name='test_optional_has_element')
```

</details>


### Or
There are 2 test cases, listed as following:
<details>
<summary>or</summary>

```python
node = onnx.helper.make_node(
    'Or',
    inputs=['x', 'y'],
    outputs=['or'],
)

# 2d
x = (np.random.randn(3, 4) > 0).astype(bool)
y = (np.random.randn(3, 4) > 0).astype(bool)
z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or2d')

# 3d
x = (np.random.randn(3, 4, 5) > 0).astype(bool)
y = (np.random.randn(3, 4, 5) > 0).astype(bool)
z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or3d')

# 4d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
y = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or4d')
```

</details>
<details>
<summary>or_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Or',
    inputs=['x', 'y'],
    outputs=['or'],
)

# 3d vs 1d
x = (np.random.randn(3, 4, 5) > 0).astype(bool)
y = (np.random.randn(5) > 0).astype(bool)
z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_bcast3v1d')

# 3d vs 2d
x = (np.random.randn(3, 4, 5) > 0).astype(bool)
y = (np.random.randn(4, 5) > 0).astype(bool)
z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_bcast3v2d')

# 4d vs 2d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
y = (np.random.randn(5, 6) > 0).astype(bool)
z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_bcast4v2d')

# 4d vs 3d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
y = (np.random.randn(4, 5, 6) > 0).astype(bool)
z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_bcast4v3d')

# 4d vs 4d
x = (np.random.randn(1, 4, 1, 6) > 0).astype(bool)
y = (np.random.randn(3, 1, 5, 6) > 0).astype(bool)
z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_bcast4v4d')
```

</details>


### PRelu
There are 2 test cases, listed as following:
<details>
<summary>prelu</summary>

```python
node = onnx.helper.make_node(
    'PRelu',
    inputs=['x', 'slope'],
    outputs=['y'],
)

x = np.random.randn(3, 4, 5).astype(np.float32)
slope = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

expect(node, inputs=[x, slope], outputs=[y],
       name='test_prelu_example')
```

</details>
<details>
<summary>prelu_broadcast</summary>

```python
node = onnx.helper.make_node(
    'PRelu',
    inputs=['x', 'slope'],
    outputs=['y'],
)

x = np.random.randn(3, 4, 5).astype(np.float32)
slope = np.random.randn(5).astype(np.float32)
y = np.clip(x, 0, np.inf) + np.clip(x, -np.inf, 0) * slope

expect(node, inputs=[x, slope], outputs=[y],
       name='test_prelu_broadcast')
```

</details>


### Pad
There are 2 test cases, listed as following:
<details>
<summary>constant_pad</summary>

```python
node = onnx.helper.make_node(
    'Pad',
    inputs=['x', 'pads', 'value'],
    outputs=['y'],
    mode='constant'
)
x = np.random.randn(1, 3, 4, 5).astype(np.float32)
pads = np.array([0, 0, 1, 3, 0, 0, 2, 4]).astype(np.int64)  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
value = np.float32(1.2)
y = pad_impl(
    x,
    pads,
    'constant',
    1.2
)

expect(node, inputs=[x, pads, value], outputs=[y],
       name='test_constant_pad')
```

</details>
<details>
<summary>reflection_and_edge_pad</summary>

```python
for mode in ['edge', 'reflect']:
    node = onnx.helper.make_node(
        'Pad',
        inputs=['x', 'pads'],
        outputs=['y'],
        mode=mode
    )
    x = np.random.randn(1, 3, 4, 5).astype(np.int32)
    pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)  # pad order [x1_begin, x2_begin, ..., x1_end, x2_end, ...]
    y = pad_impl(
        x,
        pads,
        mode
    )

    expect(node, inputs=[x, pads], outputs=[y],
           name=f'test_{mode}_pad')
```

</details>


### Pow
There are 3 test cases, listed as following:
<details>
<summary>pow</summary>

```python
node = onnx.helper.make_node(
    'Pow',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.array([1, 2, 3]).astype(np.float32)
y = np.array([4, 5, 6]).astype(np.float32)
z = pow(x, y)  # expected output [1., 32., 729.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_pow_example')

x = np.arange(60).reshape(3, 4, 5).astype(np.float32)
y = np.random.randn(3, 4, 5).astype(np.float32)
z = pow(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_pow')
```

</details>
<details>
<summary>pow_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Pow',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.array([1, 2, 3]).astype(np.float32)
y = np.array(2).astype(np.float32)
z = pow(x, y)  # expected output [1., 4., 9.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_pow_bcast_scalar')

node = onnx.helper.make_node(
    'Pow',
    inputs=['x', 'y'],
    outputs=['z'],
)
x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
y = np.array([1, 2, 3]).astype(np.float32)
# expected output [[1, 4, 27], [4, 25, 216]]
z = pow(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_pow_bcast_array')
```

</details>
<details>
<summary>types</summary>

```python
node = onnx.helper.make_node(
    'Pow',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.array([1, 2, 3]).astype(np.float32)
y = np.array([4, 5, 6]).astype(np.int64)
z = pow(x, y)  # expected output [1., 32., 729.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_pow_types_float32_int64')

x = np.array([1, 2, 3]).astype(np.int64)
y = np.array([4, 5, 6]).astype(np.float32)
z = pow(x, y)  # expected output [1, 32, 729]
expect(node, inputs=[x, y], outputs=[z],
       name='test_pow_types_int64_float32')

x = np.array([1, 2, 3]).astype(np.float32)
y = np.array([4, 5, 6]).astype(np.int32)
z = pow(x, y)  # expected output [1., 32., 729.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_pow_types_float32_int32')

x = np.array([1, 2, 3]).astype(np.int32)
y = np.array([4, 5, 6]).astype(np.float32)
z = pow(x, y)  # expected output [1, 32, 729]
expect(node, inputs=[x, y], outputs=[z],
       name='test_pow_types_int32_float32')

x = np.array([1, 2, 3]).astype(np.float32)
y = np.array([4, 5, 6]).astype(np.uint64)
z = pow(x, y)  # expected output [1., 32., 729.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_pow_types_float32_uint64')

x = np.array([1, 2, 3]).astype(np.float32)
y = np.array([4, 5, 6]).astype(np.uint32)
z = pow(x, y)  # expected output [1., 32., 729.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_pow_types_float32_uint32')

x = np.array([1, 2, 3]).astype(np.int64)
y = np.array([4, 5, 6]).astype(np.int64)
z = pow(x, y)  # expected output [1, 32, 729]
expect(node, inputs=[x, y], outputs=[z],
       name='test_pow_types_int64_int64')

x = np.array([1, 2, 3]).astype(np.int32)
y = np.array([4, 5, 6]).astype(np.int32)
z = pow(x, y)  # expected output [1, 32, 729]
expect(node, inputs=[x, y], outputs=[z],
       name='test_pow_types_int32_int32')
```

</details>


### QLinearConv
There are 1 test cases, listed as following:
<details>
<summary>qlinearconv</summary>

```python
node = onnx.helper.make_node('QLinearConv',
    inputs=['x', 'x_scale', 'x_zero_point', 'w', 'w_scale', 'w_zero_point', 'y_scale', 'y_zero_point'],
    outputs=['y'],)

x = np.array([[255, 174, 162, 25, 203, 168, 58],
    [15, 59, 237, 95, 129, 0, 64],
    [56, 242, 153, 221, 168, 12, 166],
    [232, 178, 186, 195, 237, 162, 237],
    [188, 39, 124, 77, 80, 102, 43],
    [127, 230, 21, 83, 41, 40, 134],
    [255, 154, 92, 141, 42, 148, 247], ], dtype=np.uint8).reshape((1, 1, 7, 7))

x_scale = np.float32(0.00369204697)
x_zero_point = np.uint8(132)

w = np.array([0], dtype=np.uint8).reshape((1, 1, 1, 1))

w_scale = np.array([0.00172794575], dtype=np.float32)
w_zero_point = np.array([255], dtype=np.uint8)

y_scale = np.float32(0.00162681262)
y_zero_point = np.uint8(123)

output = np.array([[0, 81, 93, 230, 52, 87, 197],
    [240, 196, 18, 160, 126, 255, 191],
    [199, 13, 102, 34, 87, 243, 89],
    [23, 77, 69, 60, 18, 93, 18],
    [67, 216, 131, 178, 175, 153, 212],
    [128, 25, 234, 172, 214, 215, 121],
    [0, 101, 163, 114, 213, 107, 8], ], dtype=np.uint8).reshape((1, 1, 7, 7))

expect(node, inputs=[x, x_scale, x_zero_point, w, w_scale, w_zero_point, y_scale, y_zero_point], outputs=[output],
       name='test_qlinearconv')
```

</details>


### QLinearMatMul
There are 1 test cases, listed as following:
<details>
<summary>qlinearmatmul</summary>

```python
node = onnx.helper.make_node('QLinearMatMul',
    inputs=['a', 'a_scale', 'a_zero_point', 'b', 'b_scale', 'b_zero_point', 'y_scale', 'y_zero_point'],
    outputs=['y'],)

#2D
a = np.array([[208, 236, 0, 238],
    [3, 214, 255, 29], ], dtype=np.uint8)

a_scale = np.array([0.0066], dtype=np.float32)
a_zero_point = np.array([113], dtype=np.uint8)

b = np.array([[152, 51, 244],
    [60, 26, 255],
    [0, 127, 246],
    [127, 254, 247]], dtype=np.uint8)

b_scale = np.array([0.00705], dtype=np.float32)
b_zero_point = np.array([114], dtype=np.uint8)

y_scale = np.array([0.0107], dtype=np.float32)
y_zero_point = np.array([118], dtype=np.uint8)

output = np.array([[168, 115, 255],
    [1, 66, 151], ], dtype=np.uint8)

expect(node, inputs=[a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point], outputs=[output],
       name='test_qlinearmatmul_2D')

#3D
a = np.array([[[208, 236, 0, 238],
    [3, 214, 255, 29]],
    [[208, 236, 0, 238],
    [3, 214, 255, 29]]], dtype=np.uint8)

a_scale = np.array([0.0066], dtype=np.float32)
a_zero_point = np.array([113], dtype=np.uint8)

b = np.array([[[152, 51, 244],
    [60, 26, 255],
    [0, 127, 246],
    [127, 254, 247]],
    [[152, 51, 244],
    [60, 26, 255],
    [0, 127, 246],
    [127, 254, 247]]], dtype=np.uint8)

b_scale = np.array([0.00705], dtype=np.float32)
b_zero_point = np.array([114], dtype=np.uint8)

y_scale = np.array([0.0107], dtype=np.float32)
y_zero_point = np.array([118], dtype=np.uint8)

output = np.array([[[168, 115, 255],
    [1, 66, 151]],
    [[168, 115, 255],
    [1, 66, 151]]], dtype=np.uint8)

expect(node, inputs=[a, a_scale, a_zero_point, b, b_scale, b_zero_point, y_scale, y_zero_point], outputs=[output],
       name='test_qlinearmatmul_3D')
```

</details>


### QuantizeLinear
There are 2 test cases, listed as following:
<details>
<summary>axis</summary>

```python
node = onnx.helper.make_node('QuantizeLinear',
                             inputs=['x', 'y_scale', 'y_zero_point'],
                             outputs=['y'],)

x = np.array([[[[-162, 10],
                [-100, 232],
                [-20, -50]],

               [[-76, 0],
                [0, 252],
                [32, -44]],

               [[245, -485],
                [-960, -270],
                [-375, -470]], ], ], dtype=np.float32)
y_scale = np.array([2, 4, 5], dtype=np.float32)
y_zero_point = np.array([84, 24, 196], dtype=np.uint8)
y = (x / y_scale.reshape(1, 3, 1, 1) + y_zero_point.reshape(1, 3, 1, 1)).astype(np.uint8)

expect(node, inputs=[x, y_scale, y_zero_point], outputs=[y],
       name='test_quantizelinear_axis')
```

</details>
<details>
<summary>quantizelinear</summary>

```python
node = onnx.helper.make_node('QuantizeLinear',
                             inputs=['x', 'y_scale', 'y_zero_point'],
                             outputs=['y'],)

x = np.array([0, 2, 3, 1000, -254, -1000]).astype(np.float32)
y_scale = np.float32(2)
y_zero_point = np.uint8(128)
y = np.array([128, 129, 130, 255, 1, 0]).astype(np.uint8)

expect(node, inputs=[x, y_scale, y_zero_point], outputs=[y],
       name='test_quantizelinear')
```

</details>


### RNN
There are 4 test cases, listed as following:
<details>
<summary>batchwise</summary>

```python
input = np.array([[[1., 2.]], [[3., 4.]], [[5., 6.]]]).astype(np.float32)

input_size = 2
hidden_size = 4
weight_scale = 0.5
layout = 1

node = onnx.helper.make_node(
    'RNN',
    inputs=['X', 'W', 'R'],
    outputs=['Y', 'Y_h'],
    hidden_size=hidden_size,
    layout=layout
)

W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

rnn = RNN_Helper(X=input, W=W, R=R, layout=layout)
Y, Y_h = rnn.step()
expect(node, inputs=[input, W, R], outputs=[Y.astype(np.float32), Y_h.astype(np.float32)], name='test_simple_rnn_batchwise')
```

</details>
<details>
<summary>defaults</summary>

```python
input = np.array([[[1., 2.], [3., 4.], [5., 6.]]]).astype(np.float32)

input_size = 2
hidden_size = 4
weight_scale = 0.1

node = onnx.helper.make_node(
    'RNN',
    inputs=['X', 'W', 'R'],
    outputs=['', 'Y_h'],
    hidden_size=hidden_size
)

W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

rnn = RNN_Helper(X=input, W=W, R=R)
_, Y_h = rnn.step()
expect(node, inputs=[input, W, R], outputs=[Y_h.astype(np.float32)], name='test_simple_rnn_defaults')
```

</details>
<details>
<summary>initial_bias</summary>

```python
input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]).astype(np.float32)

input_size = 3
hidden_size = 5
custom_bias = 0.1
weight_scale = 0.1

node = onnx.helper.make_node(
    'RNN',
    inputs=['X', 'W', 'R', 'B'],
    outputs=['', 'Y_h'],
    hidden_size=hidden_size
)

W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)

# Adding custom bias
W_B = custom_bias * np.ones((1, hidden_size)).astype(np.float32)
R_B = np.zeros((1, hidden_size)).astype(np.float32)
B = np.concatenate((W_B, R_B), axis=1)

rnn = RNN_Helper(X=input, W=W, R=R, B=B)
_, Y_h = rnn.step()
expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)],
       name='test_simple_rnn_with_initial_bias')
```

</details>
<details>
<summary>seq_length</summary>

```python
input = np.array([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                  [[10., 11., 12.], [13., 14., 15.], [16., 17., 18.]]]).astype(np.float32)

input_size = 3
hidden_size = 5

node = onnx.helper.make_node(
    'RNN',
    inputs=['X', 'W', 'R', 'B'],
    outputs=['', 'Y_h'],
    hidden_size=hidden_size
)

W = np.random.randn(1, hidden_size, input_size).astype(np.float32)
R = np.random.randn(1, hidden_size, hidden_size).astype(np.float32)

# Adding custom bias
W_B = np.random.randn(1, hidden_size).astype(np.float32)
R_B = np.random.randn(1, hidden_size).astype(np.float32)
B = np.concatenate((W_B, R_B), axis=1)

rnn = RNN_Helper(X=input, W=W, R=R, B=B)
_, Y_h = rnn.step()
expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_rnn_seq_length')
```

</details>


### Range
There are 2 test cases, listed as following:
<details>
<summary>range_float_type_positive_delta</summary>

```python
node = onnx.helper.make_node(
    'Range',
    inputs=['start', 'limit', 'delta'],
    outputs=['output'],
)

start = np.float32(1)
limit = np.float32(5)
delta = np.float32(2)

output = np.arange(start, limit, delta, dtype=np.float32)  # expected output [1.0, 3.0]
expect(node, inputs=[start, limit, delta], outputs=[output],
       name='test_range_float_type_positive_delta')
```

</details>
<details>
<summary>range_int32_type_negative_delta</summary>

```python
node = onnx.helper.make_node(
    'Range',
    inputs=['start', 'limit', 'delta'],
    outputs=['output'],
)

start = np.int32(10)
limit = np.int32(6)
delta = np.int32(-3)

output = np.arange(start, limit, delta, dtype=np.int32)  # expected output [10, 7]
expect(node, inputs=[start, limit, delta], outputs=[output],
       name='test_range_int32_type_negative_delta')
```

</details>


### Reciprocal
There are 1 test cases, listed as following:
<details>
<summary>reciprocal</summary>

```python
node = onnx.helper.make_node(
    'Reciprocal',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-4, 2]).astype(np.float32)
y = np.reciprocal(x)  # expected output [-0.25, 0.5],
expect(node, inputs=[x], outputs=[y],
       name='test_reciprocal_example')

x = np.random.rand(3, 4, 5).astype(np.float32) + 0.5
y = np.reciprocal(x)
expect(node, inputs=[x], outputs=[y],
       name='test_reciprocal')
```

</details>


### ReduceL1
There are 4 test cases, listed as following:
<details>
<summary>default_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = None
keepdims = 1

node = onnx.helper.make_node(
    'ReduceL1',
    inputs=['data'],
    outputs=['reduced'],
    keepdims=keepdims
)

data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
#print(data)
#[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

reduced = np.sum(a=np.abs(data), axis=axes, keepdims=keepdims == 1)
#print(reduced)
#[[[78.]]]

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_l1_default_axes_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sum(a=np.abs(data), axis=axes, keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_l1_default_axes_keepdims_random')
```

</details>
<details>
<summary>do_not_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [2]
keepdims = 0

node = onnx.helper.make_node(
    'ReduceL1',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims
)

data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
#print(data)
#[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)
#print(reduced)
#[[3., 7.], [11., 15.], [19., 23.]]

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_l1_do_not_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_l1_do_not_keepdims_random')
```

</details>
<details>
<summary>keepdims</summary>

```python
shape = [3, 2, 2]
axes = [2]
keepdims = 1

node = onnx.helper.make_node(
    'ReduceL1',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims
)

data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
#print(data)
#[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)
#print(reduced)
#[[[3.], [7.]], [[11.], [15.]], [[19.], [23.]]]

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_l1_keep_dims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_l1_keep_dims_random')
```

</details>
<details>
<summary>negative_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [-1]
keepdims = 1

node = onnx.helper.make_node(
    'ReduceL1',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims
)

data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
# print(data)
#[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)
# print(reduced)
#[[[3.], [7.]], [[11.], [15.]], [[19.], [23.]]]

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_l1_negative_axes_keep_dims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sum(a=np.abs(data), axis=tuple(axes), keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_l1_negative_axes_keep_dims_random')
```

</details>


### ReduceL2
There are 4 test cases, listed as following:
<details>
<summary>default_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = None
keepdims = 1

node = onnx.helper.make_node(
    'ReduceL2',
    inputs=['data'],
    outputs=['reduced'],
    keepdims=keepdims
)

data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
#print(data)
#[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

reduced = np.sqrt(np.sum(
    a=np.square(data), axis=axes, keepdims=keepdims == 1))
#print(reduced)
#[[[25.49509757]]]

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_l2_default_axes_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sqrt(np.sum(
    a=np.square(data), axis=axes, keepdims=keepdims == 1))

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_l2_default_axes_keepdims_random')
```

</details>
<details>
<summary>do_not_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [2]
keepdims = 0

node = onnx.helper.make_node(
    'ReduceL2',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims
)

data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
#print(data)
#[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

reduced = np.sqrt(np.sum(
    a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))
#print(reduced)
#[[2.23606798, 5.],
# [7.81024968, 10.63014581],
# [13.45362405, 16.2788206]]

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_l2_do_not_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sqrt(np.sum(
    a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_l2_do_not_keepdims_random')
```

</details>
<details>
<summary>keepdims</summary>

```python
shape = [3, 2, 2]
axes = [2]
keepdims = 1

node = onnx.helper.make_node(
    'ReduceL2',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims
)

data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
#print(data)
#[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

reduced = np.sqrt(np.sum(
    a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))
#print(reduced)
#[[[2.23606798], [5.]]
# [[7.81024968], [10.63014581]]
# [[13.45362405], [16.2788206 ]]]

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_l2_keep_dims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sqrt(np.sum(
    a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_l2_keep_dims_random')
```

</details>
<details>
<summary>negative_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [-1]
keepdims = 1

node = onnx.helper.make_node(
    'ReduceL2',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims
)

data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
# print(data)
#[[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

reduced = np.sqrt(np.sum(
    a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))
# print(reduced)
#[[[2.23606798], [5.]]
# [[7.81024968], [10.63014581]]
# [[13.45362405], [16.2788206 ]]]

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_l2_negative_axes_keep_dims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sqrt(np.sum(
    a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1))

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_l2_negative_axes_keep_dims_random')
```

</details>


### ReduceLogSum
There are 3 test cases, listed as following:
<details>
<summary>keepdims</summary>

```python
node = onnx.helper.make_node(
    'ReduceLogSum',
    inputs=['data'],
    outputs=["reduced"]
)
data = np.random.ranf([3, 4, 5]).astype(np.float32)
reduced = np.log(np.sum(data, keepdims=True))
expect(node, inputs=[data], outputs=[reduced],
       name='test_reduce_log_sum_default')
```

</details>
<details>
<summary>negative_axes_keepdims</summary>

```python
node = onnx.helper.make_node(
    'ReduceLogSum',
    inputs=['data'],
    outputs=["reduced"],
    axes=[-2]
)
data = np.random.ranf([3, 4, 5]).astype(np.float32)
reduced = np.log(np.sum(data, axis=(-2), keepdims=True))
# print(reduced)
expect(node, inputs=[data], outputs=[reduced],
       name='test_reduce_log_sum_negative_axes')
```

</details>
<details>
<summary>nokeepdims</summary>

```python
node = onnx.helper.make_node(
    'ReduceLogSum',
    inputs=['data'],
    outputs=["reduced"],
    axes=[2, 1],
    keepdims=0
)
data = np.random.ranf([3, 4, 5]).astype(np.float32)
reduced = np.log(np.sum(data, axis=(2, 1), keepdims=False))
expect(node, inputs=[data], outputs=[reduced],
       name='test_reduce_log_sum_desc_axes')

node = onnx.helper.make_node(
    'ReduceLogSum',
    inputs=['data'],
    outputs=["reduced"],
    axes=[0, 1],
    keepdims=0
)
data = np.random.ranf([3, 4, 5]).astype(np.float32)
reduced = np.log(np.sum(data, axis=(0, 1), keepdims=False))
expect(node, inputs=[data], outputs=[reduced],
       name='test_reduce_log_sum_asc_axes')
```

</details>


### ReduceLogSumExp
There are 4 test cases, listed as following:
<details>
<summary>default_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = None
keepdims = 1

node = onnx.helper.make_node(
    'ReduceLogSumExp',
    inputs=['data'],
    outputs=['reduced'],
    keepdims=keepdims
)

data = np.array(
    [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
    dtype=np.double)
reduced = np.log(np.sum(np.exp(data),
                        axis=axes,
                        keepdims=keepdims == 1))
# print(reduced)
# [[[60.00671387]]]

expect(node, inputs=[data], outputs=[reduced],
      name='test_reduce_log_sum_exp_default_axes_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.double)
reduced = np.log(np.sum(np.exp(data),
                        axis=axes,
                        keepdims=keepdims == 1))
expect(node, inputs=[data], outputs=[reduced],
      name='test_reduce_log_sum_exp_default_axes_keepdims_random')
```

</details>
<details>
<summary>do_not_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [1]
keepdims = 0
node = onnx.helper.make_node(
    'ReduceLogSumExp',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims
)

data = np.array(
    [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
    dtype=np.double)
reduced = np.log(np.sum(
    np.exp(data), axis=tuple(axes), keepdims=keepdims == 1))
# print(reduced)
#[[20., 2.31326175]
# [40.00004578, 2.31326175]
# [60.00671387, 2.31326175]]

expect(node, inputs=[data], outputs=[reduced],
      name='test_reduce_log_sum_exp_do_not_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.double)
reduced = np.log(np.sum(
    np.exp(data), axis=tuple(axes), keepdims=keepdims == 1))

expect(node, inputs=[data], outputs=[reduced],
    name='test_reduce_log_sum_exp_do_not_keepdims_random')
```

</details>
<details>
<summary>keepdims</summary>

```python
shape = [3, 2, 2]
axes = [1]
keepdims = 1
node = onnx.helper.make_node(
    'ReduceLogSumExp',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims
)

data = np.array(
    [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
    dtype=np.double)
reduced = np.log(np.sum(np.exp(data),
                        axis=tuple(axes),
                        keepdims=keepdims == 1))
# print(reduced)
# [[[20., 2.31326175]]
# [[40.00004578, 2.31326175]]
# [[60.00671387, 2.31326175]]]

expect(node, inputs=[data], outputs=[reduced],
      name='test_reduce_log_sum_exp_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.double)
reduced = np.log(np.sum(np.exp(data),
                        axis=tuple(axes),
                        keepdims=keepdims == 1))

expect(node, inputs=[data], outputs=[reduced],
      name='test_reduce_log_sum_exp_keepdims_random')
```

</details>
<details>
<summary>negative_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [-2]
keepdims = 1
node = onnx.helper.make_node(
    'ReduceLogSumExp',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims
)

data = np.array(
    [[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]],
    dtype=np.double)
reduced = np.log(np.sum(np.exp(data),
                        axis=tuple(axes),
                        keepdims=keepdims == 1))
# print(reduced)
# [[[20., 2.31326175]]
# [[40.00004578, 2.31326175]]
# [[60.00671387, 2.31326175]]]

expect(node, inputs=[data], outputs=[reduced],
      name='test_reduce_log_sum_exp_negative_axes_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.double)
reduced = np.log(np.sum(np.exp(data),
                        axis=tuple(axes),
                        keepdims=keepdims == 1))

expect(node, inputs=[data], outputs=[reduced],
      name='test_reduce_log_sum_exp_negative_axes_keepdims_random')
```

</details>


### ReduceMax
There are 4 test cases, listed as following:
<details>
<summary>default_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = None
keepdims = 1
node = onnx.helper.make_node(
    'ReduceMax',
    inputs=['data'],
    outputs=['reduced'],
    keepdims=keepdims)

data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
reduced = np.maximum.reduce(data, axis=axes, keepdims=keepdims == 1)
#print(reduced)
[[[60.]]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_default_axes_keepdim_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.maximum.reduce(data, axis=axes, keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_default_axes_keepdims_random')
```

</details>
<details>
<summary>do_not_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [1]
keepdims = 0

node = onnx.helper.make_node(
    'ReduceMax',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
#print(reduced)
#[[20., 2.]
# [40., 2.]
# [60., 2.]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_do_not_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_do_not_keepdims_random')
```

</details>
<details>
<summary>keepdims</summary>

```python
shape = [3, 2, 2]
axes = [1]
keepdims = 1

node = onnx.helper.make_node(
    'ReduceMax',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
#print(reduced)
#[[[20., 2.]]
# [[40., 2.]]
# [[60., 2.]]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_keepdims_random')
```

</details>
<details>
<summary>negative_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [-2]
keepdims = 1

node = onnx.helper.make_node(
    'ReduceMax',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
# print(reduced)
#[[[20., 2.]]
# [[40., 2.]]
# [[60., 2.]]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_negative_axes_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.maximum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_max_negative_axes_keepdims_random')
```

</details>


### ReduceMean
There are 4 test cases, listed as following:
<details>
<summary>default_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = None
keepdims = 1

node = onnx.helper.make_node(
    'ReduceMean',
    inputs=['data'],
    outputs=['reduced'],
    keepdims=keepdims)

data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
reduced = np.mean(data, axis=axes, keepdims=keepdims == 1)
#print(reduced)
#[[[18.25]]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_mean_default_axes_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.mean(data, axis=axes, keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_mean_default_axes_keepdims_random')
```

</details>
<details>
<summary>do_not_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [1]
keepdims = 0

node = onnx.helper.make_node(
    'ReduceMean',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)
#print(reduced)
#[[12.5, 1.5]
# [35., 1.5]
# [57.5, 1.5]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_mean_do_not_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_mean_do_not_keepdims_random')
```

</details>
<details>
<summary>keepdims</summary>

```python
shape = [3, 2, 2]
axes = [1]
keepdims = 1

node = onnx.helper.make_node(
    'ReduceMean',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)
#print(reduced)
#[[[12.5, 1.5]]
# [[35., 1.5]]
# [[57.5, 1.5]]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_mean_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_mean_keepdims_random')
```

</details>
<details>
<summary>negative_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [-2]
keepdims = 1

node = onnx.helper.make_node(
    'ReduceMean',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)
# print(reduced)
# [[[12.5, 1.5]]
# [[35., 1.5]]
# [[57.5, 1.5]]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_mean_negative_axes_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.mean(data, axis=tuple(axes), keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_mean_negative_axes_keepdims_random')
```

</details>


### ReduceMin
There are 4 test cases, listed as following:
<details>
<summary>default_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = None
keepdims = 1

node = onnx.helper.make_node(
    'ReduceMin',
    inputs=['data'],
    outputs=['reduced'],
    keepdims=keepdims)

data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
reduced = np.minimum.reduce(data, axis=axes, keepdims=keepdims == 1)
#print(reduced)
#[[[1.]]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_min_default_axes_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.minimum.reduce(data, axis=axes, keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_min_default_axes_keepdims_random')
```

</details>
<details>
<summary>do_not_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [1]
keepdims = 0

node = onnx.helper.make_node(
    'ReduceMin',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
#print(reduced)
#[[5., 1.]
# [30., 1.]
# [55., 1.]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_min_do_not_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_min_do_not_keepdims_random')
```

</details>
<details>
<summary>keepdims</summary>

```python
shape = [3, 2, 2]
axes = [1]
keepdims = 1

node = onnx.helper.make_node(
    'ReduceMin', inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
#print(reduced)
#[[[5., 1.]]
# [[30., 1.]]
# [[55., 1.]]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_min_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_min_keepdims_random')
```

</details>
<details>
<summary>negative_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [-2]
keepdims = 1

node = onnx.helper.make_node(
    'ReduceMin', inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[5, 1], [20, 2]], [[30, 1], [40, 2]], [[55, 1], [60, 2]]], dtype=np.float32)
reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)
# print(reduced)
#[[[5., 1.]]
# [[30., 1.]]
# [[55., 1.]]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_min_negative_axes_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.minimum.reduce(data, axis=tuple(axes), keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_min_negative_axes_keepdims_random')
```

</details>


### ReduceProd
There are 4 test cases, listed as following:
<details>
<summary>default_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = None
keepdims = 1

node = onnx.helper.make_node(
    'ReduceProd',
    inputs=['data'],
    outputs=['reduced'],
    keepdims=keepdims)

data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
reduced = np.prod(data, axis=axes, keepdims=keepdims == 1)
#print(reduced)
#[[[4.790016e+08]]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_default_axes_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.prod(data, axis=axes, keepdims=keepdims == 1)
expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_default_axes_keepdims_random')
```

</details>
<details>
<summary>do_not_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [1]
keepdims = 0

node = onnx.helper.make_node(
    'ReduceProd',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
#print(reduced)
#[[3., 8.]
# [35., 48.]
# [99., 120.]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_do_not_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_do_not_keepdims_random')
```

</details>
<details>
<summary>keepdims</summary>

```python
shape = [3, 2, 2]
axes = [1]
keepdims = 1

node = onnx.helper.make_node(
    'ReduceProd',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
#print(reduced)
#[[[3., 8.]]
# [[35., 48.]]
# [[99., 120.]]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_keepdims_random')
```

</details>
<details>
<summary>negative_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [-2]
keepdims = 1

node = onnx.helper.make_node(
    'ReduceProd',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
# print(reduced)
#[[[3., 8.]]
# [[35., 48.]]
# [[99., 120.]]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_negative_axes_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.prod(data, axis=tuple(axes), keepdims=keepdims == 1)
expect(node, inputs=[data], outputs=[reduced], name='test_reduce_prod_negative_axes_keepdims_random')
```

</details>


### ReduceSum
There are 5 test cases, listed as following:
<details>
<summary>default_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = np.array([], dtype=np.int64)
keepdims = 1

node = onnx.helper.make_node(
    'ReduceSum',
    inputs=['data', 'axes'],
    outputs=['reduced'],
    keepdims=keepdims)

data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
reduced = np.sum(data, axis=None, keepdims=keepdims == 1)
#print(reduced)
#[[[78.]]]

expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_default_axes_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sum(data, axis=None, keepdims=keepdims == 1)

expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_default_axes_keepdims_random')
```

</details>
<details>
<summary>do_not_keepdims</summary>

```python
shape = [3, 2, 2]
axes = np.array([1], dtype=np.int64)
keepdims = 0

node = onnx.helper.make_node(
    'ReduceSum',
    inputs=['data', 'axes'],
    outputs=['reduced'],
    keepdims=keepdims)

data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)
#print(reduced)
#[[4., 6.]
# [12., 14.]
# [20., 22.]]

expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_do_not_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)

expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_do_not_keepdims_random')
```

</details>
<details>
<summary>empty_axes_input_noop</summary>

```python
shape = [3, 2, 2]
keepdims = 1

node = onnx.helper.make_node(
    'ReduceSum',
    inputs=['data', 'axes'],
    outputs=['reduced'],
    keepdims=keepdims,
    noop_with_empty_axes=True)

data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
axes = np.array([], dtype=np.int64)
reduced = np.array(data)
#print(reduced)
#[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]

expect(node, inputs=[data, axes], outputs=[reduced],
       name='test_reduce_sum_empty_axes_input_noop_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.array(data)

expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_negative_axes_keepdims_random')
```

</details>
<details>
<summary>keepdims</summary>

```python
shape = [3, 2, 2]
axes = np.array([1], dtype=np.int64)
keepdims = 1

node = onnx.helper.make_node(
    'ReduceSum',
    inputs=['data', 'axes'],
    outputs=['reduced'],
    keepdims=keepdims)

data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)
#print(reduced)
#[[[4., 6.]]
# [[12., 14.]]
# [[20., 22.]]]

expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)

expect(node, inputs=[data, axes], outputs=[reduced], name='test_reduce_sum_keepdims_random')
```

</details>
<details>
<summary>negative_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = np.array([-2], dtype=np.int64)
keepdims = 1

node = onnx.helper.make_node(
    'ReduceSum',
    inputs=['data', 'axes'],
    outputs=['reduced'],
    keepdims=keepdims)

data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
reduced = np.sum(data, axis=tuple(axes.tolist()), keepdims=keepdims == 1)
# print(reduced)
#[[[4., 6.]]
# [[12., 14.]]
# [[20., 22.]]]

expect(node, inputs=[data, axes], outputs=[reduced],
       name='test_reduce_sum_negative_axes_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sum(data, axis=tuple(
    axes.tolist()), keepdims=keepdims == 1)

expect(node, inputs=[data, axes], outputs=[reduced],
       name='test_reduce_sum_negative_axes_keepdims_random')
```

</details>


### ReduceSumSquare
There are 4 test cases, listed as following:
<details>
<summary>default_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = None
keepdims = 1

node = onnx.helper.make_node(
    'ReduceSumSquare',
    inputs=['data'],
    outputs=['reduced'],
    keepdims=keepdims)

data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
reduced = np.sum(np.square(data), axis=axes, keepdims=keepdims == 1)
#print(reduced)
#[[[650.]]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_square_default_axes_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sum(np.square(data), axis=axes, keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_square_default_axes_keepdims_random')
```

</details>
<details>
<summary>do_not_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [1]
keepdims = 0

node = onnx.helper.make_node(
    'ReduceSumSquare',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
reduced = np.sum(np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
#print(reduced)
#[[10., 20.]
# [74., 100.]
# [202., 244.]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_square_do_not_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sum(np.square(data), axis=tuple(axes), keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_square_do_not_keepdims_random')
```

</details>
<details>
<summary>keepdims</summary>

```python
shape = [3, 2, 2]
axes = [1]
keepdims = 1

node = onnx.helper.make_node(
    'ReduceSumSquare',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
reduced = np.sum(np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
#print(reduced)
#[[[10., 20.]]
# [[74., 100.]]
# [[202., 244.]]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_square_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sum(np.square(data), axis=tuple(axes), keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_square_keepdims_random')
```

</details>
<details>
<summary>negative_axes_keepdims</summary>

```python
shape = [3, 2, 2]
axes = [-2]
keepdims = 1

node = onnx.helper.make_node(
    'ReduceSumSquare',
    inputs=['data'],
    outputs=['reduced'],
    axes=axes,
    keepdims=keepdims)

data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=np.float32)
reduced = np.sum(np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
# print(reduced)
#[[[10., 20.s]]
# [[74., 100.]]
# [[202., 244.]]]

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_square_negative_axes_keepdims_example')

np.random.seed(0)
data = np.random.uniform(-10, 10, shape).astype(np.float32)
reduced = np.sum(np.square(data), axis=tuple(axes), keepdims=keepdims == 1)

expect(node, inputs=[data], outputs=[reduced], name='test_reduce_sum_square_negative_axes_keepdims_random')
```

</details>


### Relu
There are 1 test cases, listed as following:
<details>
<summary>relu</summary>

```python
node = onnx.helper.make_node(
    'Relu',
    inputs=['x'],
    outputs=['y'],
)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, 0, np.inf)

expect(node, inputs=[x], outputs=[y],
       name='test_relu')
```

</details>


### Reshape
There are 2 test cases, listed as following:
<details>
<summary>allowzero</summary>

```python
original_shape = [0, 3, 4]
test_cases = {
    'allowzero_reordered': np.array([3, 4, 0], dtype=np.int64),
}
data = np.random.random_sample(original_shape).astype(np.float32)

for test_name, shape in test_cases.items():
    node = onnx.helper.make_node(
        'Reshape',
        inputs=['data', 'shape'],
        outputs=['reshaped'],
        allowzero=1,  # if allowzero=1, final shape = (3, 4, 0)
                      # if allowzero=0, final shape = (3, 4, 4)
    )

    reshaped = reshape_reference_implementation(data, shape, allowzero=1)

    expect(node, inputs=[data, shape], outputs=[reshaped],
           name='test_reshape_' + test_name)
```

</details>
<details>
<summary>reshape</summary>

```python
original_shape = [2, 3, 4]
test_cases = {
    'reordered_all_dims': np.array([4, 2, 3], dtype=np.int64),
    'reordered_last_dims': np.array([2, 4, 3], dtype=np.int64),
    'reduced_dims': np.array([2, 12], dtype=np.int64),
    'extended_dims': np.array([2, 3, 2, 2], dtype=np.int64),
    'one_dim': np.array([24], dtype=np.int64),
    'negative_dim': np.array([2, -1, 2], dtype=np.int64),
    'negative_extended_dims': np.array([-1, 2, 3, 4], dtype=np.int64),
    'zero_dim': np.array([2, 0, 4, 1], dtype=np.int64),
    'zero_and_negative_dim': np.array([2, 0, 1, -1], dtype=np.int64),
}
data = np.random.random_sample(original_shape).astype(np.float32)

for test_name, shape in test_cases.items():
    node = onnx.helper.make_node(
        'Reshape',
        inputs=['data', 'shape'],
        outputs=['reshaped'],
    )

    reshaped = reshape_reference_implementation(data, shape)

    expect(node, inputs=[data, shape], outputs=[reshaped],
           name='test_reshape_' + test_name)
```

</details>


### Resize
There are 23 test cases, listed as following:
<details>
<summary>resize_downsample_scales_cubic</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', 'scales'],
    outputs=['Y'],
    mode='cubic',
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
output = interpolate_nd(
    data, cubic_coeffs, scale_factors=scales).astype(np.float32)

expect(node, inputs=[data, scales], outputs=[output],
       name='test_resize_downsample_scales_cubic')
```

</details>
<details>
<summary>resize_downsample_scales_cubic_A_n0p5_exclude_outside</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', 'scales'],
    outputs=['Y'],
    mode='cubic',
    cubic_coeff_a=-0.5,
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
output = interpolate_nd(data, lambda x: cubic_coeffs(x, A=-0.5), scale_factors=scales,
                        exclude_outside=True).astype(np.float32)

expect(node, inputs=[data, scales], outputs=[output],
       name='test_resize_downsample_scales_cubic_A_n0p5_exclude_outside')
```

</details>
<details>
<summary>resize_downsample_scales_cubic_align_corners</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', 'scales'],
    outputs=['Y'],
    mode='cubic',
    coordinate_transformation_mode='align_corners'
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
output = interpolate_nd(
    data, cubic_coeffs, scale_factors=scales, coordinate_transformation_mode='align_corners').astype(np.float32)

expect(node, inputs=[data, scales], outputs=[output],
       name='test_resize_downsample_scales_cubic_align_corners')
```

</details>
<details>
<summary>resize_downsample_scales_linear</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', 'scales'],
    outputs=['Y'],
    mode='linear',
)

data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]]], dtype=np.float32)

scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

# [[[[2.6666665 4.3333331]]]]
output = interpolate_nd(
    data, linear_coeffs, scale_factors=scales).astype(np.float32)

expect(node, inputs=[data, scales], outputs=[output],
       name='test_resize_downsample_scales_linear')
```

</details>
<details>
<summary>resize_downsample_scales_linear_align_corners</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', 'scales'],
    outputs=['Y'],
    mode='linear',
    coordinate_transformation_mode='align_corners'
)

data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]]], dtype=np.float32)

scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

# [[[[1.       3.142857]]]]
output = interpolate_nd(
    data, linear_coeffs, scale_factors=scales, coordinate_transformation_mode='align_corners').astype(np.float32)

expect(node, inputs=[data, scales], outputs=[output],
       name='test_resize_downsample_scales_linear_align_corners')
```

</details>
<details>
<summary>resize_downsample_scales_nearest</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', 'scales'],
    outputs=['Y'],
    mode='nearest',
)

data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]]], dtype=np.float32)

scales = np.array([1.0, 1.0, 0.6, 0.6], dtype=np.float32)

# [[[[1. 3.]]]]
output = interpolate_nd(
    data, nearest_coeffs, scale_factors=scales).astype(np.float32)

expect(node, inputs=[data, scales], outputs=[output],
       name='test_resize_downsample_scales_nearest')
```

</details>
<details>
<summary>resize_downsample_sizes_cubic</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', '', 'sizes'],
    outputs=['Y'],
    mode='cubic',
)

data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]], dtype=np.float32)

sizes = np.array([1, 1, 3, 3], dtype=np.int64)

# [[[[ 1.63078704  3.00462963  4.37847222]
#    [ 7.12615741  8.5         9.87384259]
#    [12.62152778 13.99537037 15.36921296]]]]
output = interpolate_nd(
    data, cubic_coeffs, output_size=sizes).astype(np.float32)

expect(node, inputs=[data, sizes], outputs=[output],
       name='test_resize_downsample_sizes_cubic')
```

</details>
<details>
<summary>resize_downsample_sizes_linear_pytorch_half_pixel</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', '', 'sizes'],
    outputs=['Y'],
    mode='linear',
    coordinate_transformation_mode='pytorch_half_pixel'
)

data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]], dtype=np.float32)

sizes = np.array([1, 1, 3, 1], dtype=np.int64)

# [[[[ 1.6666666]
#    [ 7.       ]
#    [12.333333 ]]]]
output = interpolate_nd(
    data, linear_coeffs, output_size=sizes, coordinate_transformation_mode='pytorch_half_pixel').astype(np.float32)

expect(node, inputs=[data, sizes], outputs=[output],
       name='test_resize_downsample_sizes_linear_pytorch_half_pixel')
```

</details>
<details>
<summary>resize_downsample_sizes_nearest</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', '', 'sizes'],
    outputs=['Y'],
    mode='nearest',
)

data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
]]], dtype=np.float32)

sizes = np.array([1, 1, 1, 3], dtype=np.int64)

# [[[[1. 3.]]]]
output = interpolate_nd(
    data, nearest_coeffs, output_size=sizes).astype(np.float32)

expect(node, inputs=[data, sizes], outputs=[output],
       name='test_resize_downsample_sizes_nearest')
```

</details>
<details>
<summary>resize_tf_crop_and_resize</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', 'roi', '', 'sizes'],
    outputs=['Y'],
    mode='linear',
    coordinate_transformation_mode='tf_crop_and_resize'
)

data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]], dtype=np.float32)

# Note: for some rois, the result may be different with that of TF for inaccurate floating point
roi = np.array([0, 0, 0.4, 0.6, 1, 1, 0.6, 0.8], dtype=np.float32)
sizes = np.array([1, 1, 3, 3], dtype=np.int64)

# [[[[ 7.6000004  7.9        8.2      ]
#    [ 8.8        9.1        9.400001 ]
#    [10.        10.3       10.6      ]]]]
output = interpolate_nd(data, linear_coeffs, output_size=sizes, roi=roi,
                        coordinate_transformation_mode='tf_crop_and_resize').astype(np.float32)

expect(node, inputs=[data, roi, sizes], outputs=[output],
       name='test_resize_tf_crop_and_resize')
```

</details>
<details>
<summary>resize_tf_crop_and_resize_extrapolation_value</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', 'roi', '', 'sizes'],
    outputs=['Y'],
    mode='linear',
    coordinate_transformation_mode='tf_crop_and_resize',
    extrapolation_value=10.0
)

data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]], dtype=np.float32)

# Note: for some rois, the result may be different with that of TF for inaccurate floating point
roi = np.array([0, 0, 0.4, 0.6, 1, 1, 1.2, 1.7], dtype=np.float32)
sizes = np.array([1, 1, 3, 3], dtype=np.int64)

# [[[[ 7.6000004 10.        10.       ]
#    [12.400001  10.        10.       ]
#    [10.        10.        10.       ]]]]
output = interpolate_nd(data, linear_coeffs, output_size=sizes, roi=roi,
                        coordinate_transformation_mode='tf_crop_and_resize', extrapolation_value=10.0).astype(np.float32)

expect(node, inputs=[data, roi, sizes], outputs=[output],
       name='test_resize_tf_crop_and_resize')
```

</details>
<details>
<summary>resize_upsample_scales_cubic</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', 'scales'],
    outputs=['Y'],
    mode='cubic',
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
output = interpolate_nd(
    data, cubic_coeffs, scale_factors=scales).astype(np.float32)

expect(node, inputs=[data, scales], outputs=[output],
       name='test_resize_upsample_scales_cubic')
```

</details>
<details>
<summary>resize_upsample_scales_cubic_A_n0p5_exclude_outside</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', 'scales'],
    outputs=['Y'],
    mode='cubic',
    cubic_coeff_a=-0.5,
    exclude_outside=True
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
output = interpolate_nd(data, lambda x: cubic_coeffs(x, A=-0.5), scale_factors=scales,
                        exclude_outside=True).astype(np.float32)

expect(node, inputs=[data, scales], outputs=[output],
       name='test_resize_upsample_scales_cubic_A_n0p5_exclude_outside')
```

</details>
<details>
<summary>resize_upsample_scales_cubic_align_corners</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', 'scales'],
    outputs=['Y'],
    mode='cubic',
    coordinate_transformation_mode='align_corners'
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
output = interpolate_nd(
    data, cubic_coeffs, scale_factors=scales, coordinate_transformation_mode='align_corners').astype(np.float32)

expect(node, inputs=[data, scales], outputs=[output],
       name='test_resize_upsample_scales_cubic_align_corners')
```

</details>
<details>
<summary>resize_upsample_scales_cubic_asymmetric</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', 'scales'],
    outputs=['Y'],
    mode='cubic',
    coordinate_transformation_mode='asymmetric'
)

data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]], dtype=np.float32)

scales = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)

# [[[[ 1.       1.40625  2.       2.5      3.       3.59375  4.
#      4.09375]
#    [ 2.625    3.03125  3.625    4.125    4.625    5.21875  5.625
#      5.71875]
#    [ 5.       5.40625  6.       6.5      7.       7.59375  8.
#      8.09375]
#    [ 7.       7.40625  8.       8.5      9.       9.59375 10.
#     10.09375]
#    [ 9.       9.40625 10.      10.5     11.      11.59375 12.
#     12.09375]
#    [11.375   11.78125 12.375   12.875   13.375   13.96875 14.375
#     14.46875]
#    [13.      13.40625 14.      14.5     15.      15.59375 16.
#     16.09375]
#    [13.375   13.78125 14.375   14.875   15.375   15.96875 16.375
#     16.46875]]]]
output = interpolate_nd(data, lambda x: cubic_coeffs(x, A=-0.75), scale_factors=scales,
                        coordinate_transformation_mode='asymmetric').astype(np.float32)

expect(node, inputs=[data, scales], outputs=[output],
       name='test_resize_upsample_scales_cubic_asymmetric')
```

</details>
<details>
<summary>resize_upsample_scales_linear</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', 'scales'],
    outputs=['Y'],
    mode='linear',
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
output = interpolate_nd(
    data, linear_coeffs, scale_factors=scales).astype(np.float32)

expect(node, inputs=[data, scales], outputs=[output],
       name='test_resize_upsample_scales_linear')
```

</details>
<details>
<summary>resize_upsample_scales_linear_align_corners</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', 'scales'],
    outputs=['Y'],
    mode='linear',
    coordinate_transformation_mode='align_corners'
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
output = interpolate_nd(
    data, linear_coeffs, scale_factors=scales, coordinate_transformation_mode='align_corners').astype(np.float32)

expect(node, inputs=[data, scales], outputs=[output],
       name='test_resize_upsample_scales_linear_align_corners')
```

</details>
<details>
<summary>resize_upsample_scales_nearest</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', 'scales'],
    outputs=['Y'],
    mode='nearest',
)

data = np.array([[[
    [1, 2],
    [3, 4],
]]], dtype=np.float32)

scales = np.array([1.0, 1.0, 2.0, 3.0], dtype=np.float32)

# [[[[1. 1. 1. 2. 2. 2.]
#    [1. 1. 1. 2. 2. 2.]
#    [3. 3. 3. 4. 4. 4.]
#    [3. 3. 3. 4. 4. 4.]]]]
output = interpolate_nd(
    data, nearest_coeffs, scale_factors=scales).astype(np.float32)

expect(node, inputs=[data, scales], outputs=[output],
       name='test_resize_upsample_scales_nearest')
```

</details>
<details>
<summary>resize_upsample_sizes_cubic</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', '', 'sizes'],
    outputs=['Y'],
    mode='cubic',
)

data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]], dtype=np.float32)

sizes = np.array([1, 1, 9, 10], dtype=np.int64)

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
output = interpolate_nd(
    data, cubic_coeffs, output_size=sizes).astype(np.float32)

expect(node, inputs=[data, sizes], outputs=[output],
       name='test_resize_upsample_sizes_cubic')
```

</details>
<details>
<summary>resize_upsample_sizes_nearest</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', '', 'sizes'],
    outputs=['Y'],
    mode='nearest',
)

data = np.array([[[
    [1, 2],
    [3, 4],
]]], dtype=np.float32)

sizes = np.array([1, 1, 7, 8], dtype=np.int64)

# [[[[1. 1. 1. 1. 2. 2. 2. 2.]
#    [1. 1. 1. 1. 2. 2. 2. 2.]
#    [1. 1. 1. 1. 2. 2. 2. 2.]
#    [1. 1. 1. 1. 2. 2. 2. 2.]
#    [3. 3. 3. 3. 4. 4. 4. 4.]
#    [3. 3. 3. 3. 4. 4. 4. 4.]
#    [3. 3. 3. 3. 4. 4. 4. 4.]]]]
output = interpolate_nd(
    data, nearest_coeffs, output_size=sizes).astype(np.float32)

expect(node, inputs=[data, sizes], outputs=[output],
       name='test_resize_upsample_sizes_nearest')
```

</details>
<details>
<summary>resize_upsample_sizes_nearest_ceil_half_pixel</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', '', 'sizes'],
    outputs=['Y'],
    mode='nearest',
    coordinate_transformation_mode='half_pixel',
    nearest_mode='ceil'
)

data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]], dtype=np.float32)

sizes = np.array([1, 1, 8, 8], dtype=np.int64)

# [[[[ 1.  2.  2.  3.  3.  4.  4.  4.]
#    [ 5.  6.  6.  7.  7.  8.  8.  8.]
#    [ 5.  6.  6.  7.  7.  8.  8.  8.]
#    [ 9. 10. 10. 11. 11. 12. 12. 12.]
#    [ 9. 10. 10. 11. 11. 12. 12. 12.]
#    [13. 14. 14. 15. 15. 16. 16. 16.]
#    [13. 14. 14. 15. 15. 16. 16. 16.]
#    [13. 14. 14. 15. 15. 16. 16. 16.]]]]
output = interpolate_nd(
    data, lambda x: nearest_coeffs(x, mode='ceil'), output_size=sizes).astype(np.float32)

expect(node, inputs=[data, sizes], outputs=[output],
       name='test_resize_upsample_sizes_nearest_ceil_half_pixel')
```

</details>
<details>
<summary>resize_upsample_sizes_nearest_floor_align_corners</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', '', 'sizes'],
    outputs=['Y'],
    mode='nearest',
    coordinate_transformation_mode='align_corners',
    nearest_mode='floor'
)

data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]], dtype=np.float32)

sizes = np.array([1, 1, 8, 8], dtype=np.int64)

# [[[[ 1.  1.  1.  2.  2.  3.  3.  4.]
#    [ 1.  1.  1.  2.  2.  3.  3.  4.]
#    [ 1.  1.  1.  2.  2.  3.  3.  4.]
#    [ 5.  5.  5.  6.  6.  7.  7.  8.]
#    [ 5.  5.  5.  6.  6.  7.  7.  8.]
#    [ 9.  9.  9. 10. 10. 11. 11. 12.]
#    [ 9.  9.  9. 10. 10. 11. 11. 12.]
#    [13. 13. 13. 14. 14. 15. 15. 16.]]]]
output = interpolate_nd(
    data, lambda x: nearest_coeffs(x, mode='floor'), output_size=sizes, coordinate_transformation_mode='align_corners').astype(np.float32)

expect(node, inputs=[data, sizes], outputs=[output],
       name='test_resize_upsample_sizes_nearest_floor_align_corners')
```

</details>
<details>
<summary>resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric</summary>

```python
node = onnx.helper.make_node(
    'Resize',
    inputs=['X', '', '', 'sizes'],
    outputs=['Y'],
    mode='nearest',
    coordinate_transformation_mode='asymmetric',
    nearest_mode='round_prefer_ceil'
)

data = np.array([[[
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
]]], dtype=np.float32)

sizes = np.array([1, 1, 8, 8], dtype=np.int64)

# [[[[ 1.  2.  2.  3.  3.  4.  4.  4.]
#    [ 5.  6.  6.  7.  7.  8.  8.  8.]
#    [ 5.  6.  6.  7.  7.  8.  8.  8.]
#    [ 9. 10. 10. 11. 11. 12. 12. 12.]
#    [ 9. 10. 10. 11. 11. 12. 12. 12.]
#    [13. 14. 14. 15. 15. 16. 16. 16.]
#    [13. 14. 14. 15. 15. 16. 16. 16.]
#    [13. 14. 14. 15. 15. 16. 16. 16.]]]]
output = interpolate_nd(
    data, lambda x: nearest_coeffs(x, mode='round_prefer_ceil'),
    output_size=sizes, coordinate_transformation_mode='asymmetric').astype(np.float32)

expect(node, inputs=[data, sizes], outputs=[output],
       name='test_resize_upsample_sizes_nearest_round_prefer_ceil_asymmetric')
```

</details>


### ReverseSequence
There are 2 test cases, listed as following:
<details>
<summary>reversesequence_batch</summary>

```python
node = onnx.helper.make_node(
    'ReverseSequence',
    inputs=['x', 'sequence_lens'],
    outputs=['y'],
    time_axis=1,
    batch_axis=0,
)
x = np.array([[0.0, 1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0, 7.0],
              [8.0, 9.0, 10.0, 11.0],
              [12.0, 13.0, 14.0, 15.0]], dtype=np.float32)
sequence_lens = np.array([1, 2, 3, 4], dtype=np.int64)

y = np.array([[0.0, 1.0, 2.0, 3.0],
              [5.0, 4.0, 6.0, 7.0],
              [10.0, 9.0, 8.0, 11.0],
              [15.0, 14.0, 13.0, 12.0]], dtype=np.float32)

expect(node, inputs=[x, sequence_lens], outputs=[y],
       name='test_reversesequence_batch')
```

</details>
<details>
<summary>reversesequence_time</summary>

```python
node = onnx.helper.make_node(
    'ReverseSequence',
    inputs=['x', 'sequence_lens'],
    outputs=['y'],
    time_axis=0,
    batch_axis=1,
)
x = np.array([[0.0, 4.0, 8.0, 12.0],
              [1.0, 5.0, 9.0, 13.0],
              [2.0, 6.0, 10.0, 14.0],
              [3.0, 7.0, 11.0, 15.0]], dtype=np.float32)
sequence_lens = np.array([4, 3, 2, 1], dtype=np.int64)

y = np.array([[3.0, 6.0, 9.0, 12.0],
              [2.0, 5.0, 8.0, 13.0],
              [1.0, 4.0, 10.0, 14.0],
              [0.0, 7.0, 11.0, 15.0]], dtype=np.float32)

expect(node, inputs=[x, sequence_lens], outputs=[y],
       name='test_reversesequence_time')
```

</details>


### RoiAlign
There are 2 test cases, listed as following:
<details>
<summary>roialign_aligned_false</summary>

```python
node = onnx.helper.make_node(
    "RoiAlign",
    inputs=["X", "rois", "batch_indices"],
    outputs=["Y"],
    spatial_scale=1.0,
    output_height=5,
    output_width=5,
    sampling_ratio=2,
    coordinate_transformation_mode="output_half_pixel",
)

X, batch_indices, rois = get_roi_align_input_values()
# (num_rois, C, output_height, output_width)
Y = np.array(
    [
        [
            [
                [0.4664, 0.4466, 0.3405, 0.5688, 0.6068],
                [0.3714, 0.4296, 0.3835, 0.5562, 0.3510],
                [0.2768, 0.4883, 0.5222, 0.5528, 0.4171],
                [0.4713, 0.4844, 0.6904, 0.4920, 0.8774],
                [0.6239, 0.7125, 0.6289, 0.3355, 0.3495],
            ]
        ],
        [
            [
                [0.3022, 0.4305, 0.4696, 0.3978, 0.5423],
                [0.3656, 0.7050, 0.5165, 0.3172, 0.7015],
                [0.2912, 0.5059, 0.6476, 0.6235, 0.8299],
                [0.5916, 0.7389, 0.7048, 0.8372, 0.8893],
                [0.6227, 0.6153, 0.7097, 0.6154, 0.4585],
            ]
        ],
        [
            [
                [0.2384, 0.3379, 0.3717, 0.6100, 0.7601],
                [0.3767, 0.3785, 0.7147, 0.9243, 0.9727],
                [0.5749, 0.5826, 0.5709, 0.7619, 0.8770],
                [0.5355, 0.2566, 0.2141, 0.2796, 0.3600],
                [0.4365, 0.3504, 0.2887, 0.3661, 0.2349],
            ]
        ],
    ],
    dtype=np.float32,
)

expect(node, inputs=[X, rois, batch_indices], outputs=[Y], name="test_roialign_aligned_false")
```

</details>
<details>
<summary>roialign_aligned_true</summary>

```python
node = onnx.helper.make_node(
    "RoiAlign",
    inputs=["X", "rois", "batch_indices"],
    outputs=["Y"],
    spatial_scale=1.0,
    output_height=5,
    output_width=5,
    sampling_ratio=2,
    coordinate_transformation_mode="half_pixel",
)

X, batch_indices, rois = get_roi_align_input_values()
# (num_rois, C, output_height, output_width)
Y = np.array(
    [
        [
            [
                [0.5178, 0.3434, 0.3229, 0.4474, 0.6344],
                [0.4031, 0.5366, 0.4428, 0.4861, 0.4023],
                [0.2512, 0.4002, 0.5155, 0.6954, 0.3465],
                [0.3350, 0.4601, 0.5881, 0.3439, 0.6849],
                [0.4932, 0.7141, 0.8217, 0.4719, 0.4039],
            ]
        ],
        [
            [
                [0.3070, 0.2187, 0.3337, 0.4880, 0.4870],
                [0.1871, 0.4914, 0.5561, 0.4192, 0.3686],
                [0.1433, 0.4608, 0.5971, 0.5310, 0.4982],
                [0.2788, 0.4386, 0.6022, 0.7000, 0.7524],
                [0.5774, 0.7024, 0.7251, 0.7338, 0.8163],
            ]
        ],
        [
            [
                [0.2393, 0.4075, 0.3379, 0.2525, 0.4743],
                [0.3671, 0.2702, 0.4105, 0.6419, 0.8308],
                [0.5556, 0.4543, 0.5564, 0.7502, 0.9300],
                [0.6626, 0.5617, 0.4813, 0.4954, 0.6663],
                [0.6636, 0.3721, 0.2056, 0.1928, 0.2478],
            ]
        ],
    ],
    dtype=np.float32,
)

expect(node, inputs=[X, rois, batch_indices], outputs=[Y], name="test_roialign_aligned_true")
```

</details>


### Round
There are 1 test cases, listed as following:
<details>
<summary>round</summary>

```python
node = onnx.helper.make_node(
    'Round',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([0.1, 0.5, 0.9, 1.2, 1.5,
            1.8, 2.3, 2.5, 2.7, -1.1,
            -1.5, -1.9, -2.2, -2.5, -2.8]).astype(np.float32)
y = np.array([0., 0., 1., 1., 2.,
            2., 2., 2., 3., -1.,
            -2., -2., -2., -2., -3.]).astype(np.float32)  # expected output
expect(node, inputs=[x], outputs=[y],
       name='test_round')
```

</details>


### STFT
There are 1 test cases, listed as following:
<details>
<summary>stft</summary>

```python
signal = np.arange(0, 128, dtype=np.float32).reshape(1, 128, 1)
length = np.array(16).astype(np.int64)
onesided_length = (length >> 1) + 1
step = np.array(8).astype(np.int64)

no_window = ""  # optional input, not supplied
node = onnx.helper.make_node(
    'STFT',
    inputs=['signal', 'frame_step', no_window, 'frame_length'],
    outputs=['output'],
)

nstfts = ((signal.shape[1] - length) // step) + 1
# [batch_size][frames][frame_length][2]
output = np.empty([1, nstfts, onesided_length, 2], dtype=np.float32)
for i in range(nstfts):
    start = i * step
    stop = i * step + length
    complex_out = np.fft.fft(signal[0, start:stop, 0])[0:onesided_length]
    output[0, i] = np.stack((complex_out.real, complex_out.imag), axis=1)

expect(node, inputs=[signal, step, length], outputs=[output],
       name='test_stft')

node = onnx.helper.make_node(
    'STFT',
    inputs=['signal', 'frame_step', 'window'],
    outputs=['output'],
)

# Test with window
a0 = .5
a1 = .5
window = a0 + a1 * np.cos(2 * 3.1415 * np.arange(0, length, 1, dtype=np.float32) / length)
nstfts = 1 + (signal.shape[1] - window.shape[0]) // step

# [batch_size][frames][frame_length][2]
output = np.empty([1, nstfts, onesided_length, 2], dtype=np.float32)
for i in range(nstfts):
    start = i * step
    stop = i * step + length
    complex_out = np.fft.fft(signal[0, start:stop, 0] * window)[0:onesided_length]
    output[0, i] = np.stack((complex_out.real, complex_out.imag), axis=1)
expect(node, inputs=[signal, step, window], outputs=[output],
       name='test_stft_with_window')
```

</details>


### Scan
There are 2 test cases, listed as following:
<details>
<summary>scan_8</summary>

```python
# Given an input sequence [x1, ..., xN], sum up its elements using a scan
# returning the final state (x1+x2+...+xN) as well the scan_output
# [x1, x1+x2, ..., x1+x2+...+xN]
#
# create graph to represent scan body
sum_in = onnx.helper.make_tensor_value_info('sum_in', onnx.TensorProto.FLOAT, [2])
next = onnx.helper.make_tensor_value_info('next', onnx.TensorProto.FLOAT, [2])
sum_out = onnx.helper.make_tensor_value_info('sum_out', onnx.TensorProto.FLOAT, [2])
scan_out = onnx.helper.make_tensor_value_info('scan_out', onnx.TensorProto.FLOAT, [2])
add_node = onnx.helper.make_node(
    'Add',
    inputs=['sum_in', 'next'],
    outputs=['sum_out']
)
id_node = onnx.helper.make_node(
    'Identity',
    inputs=['sum_out'],
    outputs=['scan_out']
)
scan_body = onnx.helper.make_graph(
    [add_node, id_node],
    'scan_body',
    [sum_in, next],
    [sum_out, scan_out]
)
# create scan op node
no_sequence_lens = ''   # optional input, not supplied
node = onnx.helper.make_node(
    'Scan',
    inputs=[no_sequence_lens, 'initial', 'x'],
    outputs=['y', 'z'],
    num_scan_inputs=1,
    body=scan_body
)
# create inputs for batch-size 1, sequence-length 3, inner dimension 2
initial = np.array([0, 0]).astype(np.float32).reshape((1, 2))
x = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32).reshape((1, 3, 2))
# final state computed = [1 + 3 + 5, 2 + 4 + 6]
y = np.array([9, 12]).astype(np.float32).reshape((1, 2))
# scan-output computed
z = np.array([1, 2, 4, 6, 9, 12]).astype(np.float32).reshape((1, 3, 2))

expect(node, inputs=[initial, x], outputs=[y, z],
       name='test_scan_sum', opset_imports=[onnx.helper.make_opsetid("", 8)])
```

</details>
<details>
<summary>scan_9</summary>

```python
# Given an input sequence [x1, ..., xN], sum up its elements using a scan
# returning the final state (x1+x2+...+xN) as well the scan_output
# [x1, x1+x2, ..., x1+x2+...+xN]
#
# create graph to represent scan body
sum_in = onnx.helper.make_tensor_value_info('sum_in', onnx.TensorProto.FLOAT, [2])
next = onnx.helper.make_tensor_value_info('next', onnx.TensorProto.FLOAT, [2])
sum_out = onnx.helper.make_tensor_value_info('sum_out', onnx.TensorProto.FLOAT, [2])
scan_out = onnx.helper.make_tensor_value_info('scan_out', onnx.TensorProto.FLOAT, [2])
add_node = onnx.helper.make_node(
    'Add',
    inputs=['sum_in', 'next'],
    outputs=['sum_out']
)
id_node = onnx.helper.make_node(
    'Identity',
    inputs=['sum_out'],
    outputs=['scan_out']
)
scan_body = onnx.helper.make_graph(
    [add_node, id_node],
    'scan_body',
    [sum_in, next],
    [sum_out, scan_out]
)
# create scan op node
node = onnx.helper.make_node(
    'Scan',
    inputs=['initial', 'x'],
    outputs=['y', 'z'],
    num_scan_inputs=1,
    body=scan_body
)
# create inputs for sequence-length 3, inner dimension 2
initial = np.array([0, 0]).astype(np.float32).reshape((2,))
x = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32).reshape((3, 2))
# final state computed = [1 + 3 + 5, 2 + 4 + 6]
y = np.array([9, 12]).astype(np.float32).reshape((2,))
# scan-output computed
z = np.array([1, 2, 4, 6, 9, 12]).astype(np.float32).reshape((3, 2))

expect(node, inputs=[initial, x], outputs=[y, z],
       name='test_scan9_sum', opset_imports=[onnx.helper.make_opsetid("", 9)])
```

</details>


### Scatter
There are 2 test cases, listed as following:
<details>
<summary>scatter_with_axis</summary>

```python
axis = 1
node = onnx.helper.make_node(
    'Scatter',
    inputs=['data', 'indices', 'updates'],
    outputs=['y'],
    axis=axis,
)
data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
indices = np.array([[1, 3]], dtype=np.int64)
updates = np.array([[1.1, 2.1]], dtype=np.float32)

y = scatter(data, indices, updates, axis=axis)
# print(y) produces
# [[1.0, 1.1, 3.0, 2.1, 5.0]]

expect(node, inputs=[data, indices, updates], outputs=[y],
       name='test_scatter_with_axis', opset_imports=[helper.make_opsetid("", 10)])
```

</details>
<details>
<summary>scatter_without_axis</summary>

```python
node = onnx.helper.make_node(
    'Scatter',
    inputs=['data', 'indices', 'updates'],
    outputs=['y'],
)
data = np.zeros((3, 3), dtype=np.float32)
indices = np.array([[1, 0, 2], [0, 2, 1]], dtype=np.int64)
updates = np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=np.float32)

y = scatter(data, indices, updates)
# print(y) produces
# [[2.0, 1.1, 0.0],
#  [1.0, 0.0, 2.2],
#  [0.0, 2.1, 1.2]]

expect(node, inputs=[data, indices, updates], outputs=[y],
       name='test_scatter_without_axis', opset_imports=[helper.make_opsetid("", 10)])
```

</details>


### ScatterElements
There are 4 test cases, listed as following:
<details>
<summary>scatter_elements_with_axis</summary>

```python
axis = 1
node = onnx.helper.make_node(
    'ScatterElements',
    inputs=['data', 'indices', 'updates'],
    outputs=['y'],
    axis=axis,
)
data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
indices = np.array([[1, 3]], dtype=np.int64)
updates = np.array([[1.1, 2.1]], dtype=np.float32)

y = scatter_elements(data, indices, updates, axis)
# print(y) produces
# [[1.0, 1.1, 3.0, 2.1, 5.0]]

expect(node, inputs=[data, indices, updates], outputs=[y],
       name='test_scatter_elements_with_axis')
```

</details>
<details>
<summary>scatter_elements_with_duplicate_indices</summary>

```python
axis = 1
node = onnx.helper.make_node(
    'ScatterElements',
    inputs=['data', 'indices', 'updates'],
    outputs=['y'],
    axis=axis,
    reduction='add',
)
data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
indices = np.array([[1, 1]], dtype=np.int64)
updates = np.array([[1.1, 2.1]], dtype=np.float32)

y = scatter_elements(data, indices, updates, axis, reduction='add')
# print(y) produces
# [[1.0, 5.2, 3.0, 4.0, 5.0]]

expect(node, inputs=[data, indices, updates], outputs=[y],
        name='test_scatter_elements_with_duplicate_indices')
```

</details>
<details>
<summary>scatter_elements_with_negative_indices</summary>

```python
axis = 1
node = onnx.helper.make_node(
    'ScatterElements',
    inputs=['data', 'indices', 'updates'],
    outputs=['y'],
    axis=axis,
)
data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32)
indices = np.array([[1, -3]], dtype=np.int64)
updates = np.array([[1.1, 2.1]], dtype=np.float32)

y = scatter_elements(data, indices, updates, axis)
# print(y) produces
# [[1.0, 1.1, 2.1, 4.0, 5.0]]

expect(node, inputs=[data, indices, updates], outputs=[y],
       name='test_scatter_elements_with_negative_indices')
```

</details>
<details>
<summary>scatter_elements_without_axis</summary>

```python
node = onnx.helper.make_node(
    'ScatterElements',
    inputs=['data', 'indices', 'updates'],
    outputs=['y'],
)
data = np.zeros((3, 3), dtype=np.float32)
indices = np.array([[1, 0, 2], [0, 2, 1]], dtype=np.int64)
updates = np.array([[1.0, 1.1, 1.2], [2.0, 2.1, 2.2]], dtype=np.float32)

y = scatter_elements(data, indices, updates)
# print(y) produces
# [[2.0, 1.1, 0.0],
#  [1.0, 0.0, 2.2],
#  [0.0, 2.1, 1.2]]

expect(node, inputs=[data, indices, updates], outputs=[y],
       name='test_scatter_elements_without_axis')
```

</details>


### ScatterND
There are 3 test cases, listed as following:
<details>
<summary>scatternd</summary>

```python
node = onnx.helper.make_node(
    'ScatterND',
    inputs=['data', 'indices', 'updates'],
    outputs=['y'],
)
data = np.array(
    [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
     [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
     [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
     [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
indices = np.array([[0], [2]], dtype=np.int64)
updates = np.array(
    [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
     [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]], dtype=np.float32)
# Expecting output as np.array(
#    [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
#     [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
#     [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
#     [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
output = scatter_nd_impl(data, indices, updates)
expect(node, inputs=[data, indices, updates], outputs=[output],
       name='test_scatternd')
```

</details>
<details>
<summary>scatternd_add</summary>

```python
node = onnx.helper.make_node(
    'ScatterND',
    inputs=['data', 'indices', 'updates'],
    outputs=['y'],
    reduction='add',
)
data = np.array(
    [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
        [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
        [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
        [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
indices = np.array([[0], [0]], dtype=np.int64)
updates = np.array(
    [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
        [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]], dtype=np.float32)
# Expecting output as np.array(
#    [[[7, 8, 9, 10], [13, 14, 15, 16], [18, 17, 16, 15], [16, 15, 14, 13]],
#     [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
#     [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
#     [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
output = scatter_nd_impl(data, indices, updates, reduction='add')
expect(node, inputs=[data, indices, updates], outputs=[output],
       name='test_scatternd_add')
```

</details>
<details>
<summary>scatternd_multiply</summary>

```python
node = onnx.helper.make_node(
    'ScatterND',
    inputs=['data', 'indices', 'updates'],
    outputs=['y'],
    reduction='mul',
)
data = np.array(
    [[[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
        [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
        [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
        [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
indices = np.array([[0], [0]], dtype=np.int64)
updates = np.array(
    [[[5, 5, 5, 5], [6, 6, 6, 6], [7, 7, 7, 7], [8, 8, 8, 8]],
        [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]]], dtype=np.float32)
# Expecting output as np.array(
#    [[[5, 10, 15, 20], [60, 72, 84, 96], [168, 147, 126, 105], [128, 96, 64, 32]],
#     [[1, 2, 3, 4], [5, 6, 7, 8], [8, 7, 6, 5], [4, 3, 2, 1]],
#     [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]],
#     [[8, 7, 6, 5], [4, 3, 2, 1], [1, 2, 3, 4], [5, 6, 7, 8]]], dtype=np.float32)
output = scatter_nd_impl(data, indices, updates, reduction='mul')
expect(node, inputs=[data, indices, updates], outputs=[output],
       name='test_scatternd_multiply')
```

</details>


### Selu
There are 2 test cases, listed as following:
<details>
<summary>selu</summary>

```python
node = onnx.helper.make_node(
    'Selu',
    inputs=['x'],
    outputs=['y'],
    alpha=2.0,
    gamma=3.0
)

x = np.array([-1, 0, 1]).astype(np.float32)
# expected output [-3.79272318, 0., 3.]
y = np.clip(x, 0, np.inf) * 3.0 + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0 * 3.0
expect(node, inputs=[x], outputs=[y],
       name='test_selu_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, 0, np.inf) * 3.0 + (np.exp(np.clip(x, -np.inf, 0)) - 1) * 2.0 * 3.0
expect(node, inputs=[x], outputs=[y],
       name='test_selu')
```

</details>
<details>
<summary>selu_default</summary>

```python
default_alpha = 1.67326319217681884765625
default_gamma = 1.05070102214813232421875
node = onnx.helper.make_node(
    'Selu',
    inputs=['x'],
    outputs=['y'],
)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, 0, np.inf) * default_gamma + \
    (np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha * default_gamma
expect(node, inputs=[x], outputs=[y],
       name='test_selu_default')
```

</details>


### SequenceInsert
There are 1 test cases, listed as following:
<details>
<summary>sequenceinsert</summary>

```python
test_cases = {
    'at_back': [np.array([10, 11, 12]).astype(np.int64)],
    'at_front': [np.array([-2, -1, 0]), np.array([0]).astype(np.int64)]
}
sequence = [np.array([1, 2, 3, 4]).astype(np.int64), np.array([5, 6, 7]).astype(np.int64), np.array([8, 9]).astype(np.int64)]

for test_name, test_inputs in test_cases.items():
    tensor = test_inputs[0].astype(np.int64)

    if len(test_inputs) > 1:
        node = onnx.helper.make_node(
            'SequenceInsert',
            inputs=['sequence', 'tensor', 'position'],
            outputs=['output_sequence']
        )
        position = test_inputs[1]
        inserted = sequence_insert_reference_implementation(sequence, tensor, position)
        expect(node, inputs=[sequence, tensor, position], outputs=[inserted],
               name='test_sequence_insert_' + test_name)
    else:
        node = onnx.helper.make_node(
            'SequenceInsert',
            inputs=['sequence', 'tensor'],
            outputs=['output_sequence']
        )
        inserted = sequence_insert_reference_implementation(sequence, tensor)
        expect(node, inputs=[sequence, tensor], outputs=[inserted],
               name='test_sequence_insert_' + test_name)
```

</details>


### SequenceMap
There are 6 test cases, listed as following:
<details>
<summary>sequence_map_add_1_sequence_1_tensor</summary>

```python
body = onnx.helper.make_graph(
    [onnx.helper.make_node('Add', ['in0', 'in1'], ['out0'])],
    'seq_map_body',
    [onnx.helper.make_tensor_value_info('in0', onnx.TensorProto.FLOAT, ['N']),
     onnx.helper.make_tensor_value_info('in1', onnx.TensorProto.FLOAT, ['N'])],
    [onnx.helper.make_tensor_value_info(
        'out0', onnx.TensorProto.FLOAT, ['N'])]
)

node = onnx.helper.make_node(
    'SequenceMap',
    inputs=['x0', 'x1'],
    outputs=['y0'],
    body=body
)

x0 = [np.random.uniform(0.0, 1.0, 10).astype(np.float32) for k in range(3)]
x1 = np.random.uniform(0.0, 1.0, 10).astype(np.float32)
y0 = [x0[i] + x1 for i in range(3)]
input_type_protos = [
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
    onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N']),
]
output_type_protos = [
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
]
expect(node, inputs=[x0, x1], outputs=[y0],
       input_type_protos=input_type_protos,
       output_type_protos=output_type_protos,
       name='test_sequence_map_add_1_sequence_1_tensor')
```

</details>
<details>
<summary>sequence_map_add_2_sequences</summary>

```python
body = onnx.helper.make_graph(
    [onnx.helper.make_node('Add', ['in0', 'in1'], ['out0'])],
    'seq_map_body',
    [onnx.helper.make_tensor_value_info('in0', onnx.TensorProto.FLOAT, ['N']),
     onnx.helper.make_tensor_value_info('in1', onnx.TensorProto.FLOAT, ['N'])],
    [onnx.helper.make_tensor_value_info(
        'out0', onnx.TensorProto.FLOAT, ['N'])]
)

node = onnx.helper.make_node(
    'SequenceMap',
    inputs=['x0', 'x1'],
    outputs=['y0'],
    body=body
)

N = [np.random.randint(1, 10) for _ in range(3)]
x0 = [np.random.uniform(0.0, 1.0, N[k]).astype(np.float32)
      for k in range(3)]
x1 = [np.random.uniform(0.0, 1.0, N[k]).astype(np.float32)
      for k in range(3)]
y0 = [x0[k] + x1[k] for k in range(3)]
input_type_protos = [
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
]
output_type_protos = [
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
]
expect(node, inputs=[x0, x1], outputs=[y0],
       input_type_protos=input_type_protos,
       output_type_protos=output_type_protos,
       name='test_sequence_map_add_2_sequences')
```

</details>
<details>
<summary>sequence_map_extract_shapes</summary>

```python
body = onnx.helper.make_graph(
    [onnx.helper.make_node('Shape', ['x'], ['shape'])],
    'seq_map_body',
    [onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, ['H', 'W', 'C'])],
    [onnx.helper.make_tensor_value_info('shape', onnx.TensorProto.INT64, [3])]
)

node = onnx.helper.make_node(
    'SequenceMap',
    inputs=['in_seq'],
    outputs=['shapes'],
    body=body
)

shapes = [
    np.array([40, 30, 3], dtype=np.int64),
    np.array([20, 10, 3], dtype=np.int64),
    np.array([10, 5, 3], dtype=np.int64),
]
x0 = [np.zeros(shape, dtype=np.float32) for shape in shapes]
input_type_protos = [
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['H', 'W', 'C'])),
]
output_type_protos = [
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT64, [3])),
]
expect(node, inputs=[x0], outputs=[shapes],
       input_type_protos=input_type_protos,
       output_type_protos=output_type_protos,
       name='test_sequence_map_extract_shapes')
```

</details>
<details>
<summary>sequence_map_identity_1_sequence</summary>

```python
body = onnx.helper.make_graph(
    [onnx.helper.make_node('Identity', ['in0'], ['out0'])],
    'seq_map_body',
    [onnx.helper.make_tensor_value_info(
        'in0', onnx.TensorProto.FLOAT, ['N'])],
    [onnx.helper.make_tensor_value_info(
        'out0', onnx.TensorProto.FLOAT, ['M'])]
)

node = onnx.helper.make_node(
    'SequenceMap',
    inputs=['x'],
    outputs=['y'],
    body=body
)

x = [np.random.uniform(0.0, 1.0, 10).astype(np.float32)
     for _ in range(3)]
y = x
input_type_protos = [
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
]
output_type_protos = [
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
]
expect(node, inputs=[x], outputs=[y],
       input_type_protos=input_type_protos,
       output_type_protos=output_type_protos,
       name='test_sequence_map_identity_1_sequence')
```

</details>
<details>
<summary>sequence_map_identity_1_sequence_1_tensor</summary>

```python
body = onnx.helper.make_graph(
    [onnx.helper.make_node('Identity', ['in0'], ['out0']),
     onnx.helper.make_node('Identity', ['in1'], ['out1'])],
    'seq_map_body',
    [onnx.helper.make_tensor_value_info('in0', onnx.TensorProto.FLOAT, ['N']),
     onnx.helper.make_tensor_value_info('in1', onnx.TensorProto.FLOAT, ['M'])],
    [onnx.helper.make_tensor_value_info(
        'out0', onnx.TensorProto.FLOAT, ['N']),
     onnx.helper.make_tensor_value_info(
        'out1', onnx.TensorProto.FLOAT, ['M'])]
)

node = onnx.helper.make_node(
    'SequenceMap',
    inputs=['x0', 'x1'],
    outputs=['y0', 'y1'],
    body=body
)

x0 = [np.random.uniform(0.0, 1.0, np.random.randint(
    1, 10)).astype(np.float32) for _ in range(3)]
x1 = np.random.uniform(0.0, 1.0, np.random.randint(
    1, 10)).astype(np.float32)
y0 = x0
y1 = [x1 for _ in range(3)]
input_type_protos = [
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
    onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['M']),
]
output_type_protos = [
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['M'])),
]
expect(node, inputs=[x0, x1], outputs=[y0, y1],
       input_type_protos=input_type_protos,
       output_type_protos=output_type_protos,
       name='test_sequence_map_identity_1_sequence_1_tensor')
```

</details>
<details>
<summary>sequence_map_identity_2_sequences</summary>

```python
body = onnx.helper.make_graph(
    [onnx.helper.make_node('Identity', ['in0'], ['out0']),
     onnx.helper.make_node('Identity', ['in1'], ['out1'])],
    'seq_map_body',
    [onnx.helper.make_tensor_value_info('in0', onnx.TensorProto.FLOAT, ['N']),
     onnx.helper.make_tensor_value_info('in1', onnx.TensorProto.FLOAT, ['M'])],
    [onnx.helper.make_tensor_value_info('out0', onnx.TensorProto.FLOAT, ['N']),
     onnx.helper.make_tensor_value_info('out1', onnx.TensorProto.FLOAT, ['M'])]
)

node = onnx.helper.make_node(
    'SequenceMap',
    inputs=['x0', 'x1'],
    outputs=['y0', 'y1'],
    body=body
)

x0 = [np.random.uniform(0.0, 1.0, np.random.randint(
    1, 10)).astype(np.float32) for _ in range(3)]
x1 = [np.random.uniform(0.0, 1.0, np.random.randint(
    1, 10)).astype(np.float32) for _ in range(3)]
y0 = x0
y1 = x1
input_type_protos = [
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['M'])),
]
output_type_protos = [
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])),
    onnx.helper.make_sequence_type_proto(
        onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['M'])),
]
expect(node, inputs=[x0, x1], outputs=[y0, y1],
       input_type_protos=input_type_protos,
       output_type_protos=output_type_protos,
       name='test_sequence_map_identity_2_sequences')
```

</details>


### Shape
There are 1 test cases, listed as following:
<details>
<summary>shape</summary>

```python
x = np.array([
    [1, 2, 3],
    [4, 5, 6],
]).astype(np.float32)
test_shape('_example', x)  # preserve names of original test cases

x = np.random.randn(3, 4, 5).astype(np.float32)

test_shape('', x)  # preserve names of original test cases

test_shape('_start_1', x, start=1)

test_shape('_end_1', x, end=1)

test_shape('_start_negative_1', x, start=-1)

test_shape('_end_negative_1', x, end=-1)

test_shape('_start_1_end_negative_1', x, start=1, end=-1)

test_shape('_start_1_end_2', x, start=1, end=2)

test_shape('_clip_start', x, start=-10)

test_shape('_clip_end', x, end=10)
```

</details>


### Shrink
There are 2 test cases, listed as following:
<details>
<summary>hard_shrink</summary>

```python
node = onnx.helper.make_node(
    'Shrink',
    inputs=['x'],
    outputs=['y'],
    lambd=1.5,
)
X = np.arange(-2.0, 2.1, dtype=np.float32)
Y = np.array([-2, 0, 0, 0, 2], dtype=np.float32)
expect(node, inputs=[X], outputs=[Y],
       name='test_shrink_hard')
```

</details>
<details>
<summary>soft_shrink</summary>

```python
node = onnx.helper.make_node(
    'Shrink',
    inputs=['x'],
    outputs=['y'],
    lambd=1.5,
    bias=1.5,
)
X = np.arange(-2.0, 2.1, dtype=np.float32)
Y = np.array([-0.5, 0, 0, 0, 0.5], dtype=np.float32)
expect(node, inputs=[X], outputs=[Y],
       name='test_shrink_soft')
```

</details>


### Sigmoid
There are 1 test cases, listed as following:
<details>
<summary>sigmoid</summary>

```python
node = onnx.helper.make_node(
    'Sigmoid',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = 1.0 / (1.0 + np.exp(np.negative(x)))  # expected output [0.26894143, 0.5, 0.7310586]
expect(node, inputs=[x], outputs=[y],
       name='test_sigmoid_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = 1.0 / (1.0 + np.exp(np.negative(x)))
expect(node, inputs=[x], outputs=[y],
       name='test_sigmoid')
```

</details>


### Sign
There are 1 test cases, listed as following:
<details>
<summary>sign</summary>

```python
node = onnx.helper.make_node(
    'Sign',
    inputs=['x'],
    outputs=['y'],
)

x = np.array(range(-5, 6)).astype(np.float32)
y = np.sign(x)
expect(node, inputs=[x], outputs=[y],
       name='test_sign')
```

</details>


### Sin
There are 1 test cases, listed as following:
<details>
<summary>sin</summary>

```python
node = onnx.helper.make_node(
    'Sin',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.sin(x)
expect(node, inputs=[x], outputs=[y],
       name='test_sin_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.sin(x)
expect(node, inputs=[x], outputs=[y],
       name='test_sin')
```

</details>


### Sinh
There are 1 test cases, listed as following:
<details>
<summary>sinh</summary>

```python
node = onnx.helper.make_node(
    'Sinh',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.sinh(x)  # expected output [-1.17520118,  0.,  1.17520118]
expect(node, inputs=[x], outputs=[y],
       name='test_sinh_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.sinh(x)
expect(node, inputs=[x], outputs=[y],
       name='test_sinh')
```

</details>


### Size
There are 1 test cases, listed as following:
<details>
<summary>size</summary>

```python
node = onnx.helper.make_node(
    'Size',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([
    [1, 2, 3],
    [4, 5, 6],
]).astype(np.float32)
y = np.array(6).astype(np.int64)

expect(node, inputs=[x], outputs=[y],
       name='test_size_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.array(x.size).astype(np.int64)

expect(node, inputs=[x], outputs=[y],
       name='test_size')
```

</details>


### Slice
There are 8 test cases, listed as following:
<details>
<summary>slice</summary>

```python
node = onnx.helper.make_node(
    'Slice',
    inputs=['x', 'starts', 'ends', 'axes', 'steps'],
    outputs=['y'],
)

x = np.random.randn(20, 10, 5).astype(np.float32)
y = x[0:3, 0:10]
starts = np.array([0, 0], dtype=np.int64)
ends = np.array([3, 10], dtype=np.int64)
axes = np.array([0, 1], dtype=np.int64)
steps = np.array([1, 1], dtype=np.int64)

expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
       name='test_slice')
```

</details>
<details>
<summary>slice_default_axes</summary>

```python
node = onnx.helper.make_node(
    'Slice',
    inputs=['x', 'starts', 'ends'],
    outputs=['y'],
)

x = np.random.randn(20, 10, 5).astype(np.float32)
starts = np.array([0, 0, 3], dtype=np.int64)
ends = np.array([20, 10, 4], dtype=np.int64)
y = x[:, :, 3:4]

expect(node, inputs=[x, starts, ends], outputs=[y],
       name='test_slice_default_axes')
```

</details>
<details>
<summary>slice_default_steps</summary>

```python
node = onnx.helper.make_node(
    'Slice',
    inputs=['x', 'starts', 'ends', 'axes'],
    outputs=['y'],
)

x = np.random.randn(20, 10, 5).astype(np.float32)
starts = np.array([0, 0, 3], dtype=np.int64)
ends = np.array([20, 10, 4], dtype=np.int64)
axes = np.array([0, 1, 2], dtype=np.int64)
y = x[:, :, 3:4]

expect(node, inputs=[x, starts, ends, axes], outputs=[y],
       name='test_slice_default_steps')
```

</details>
<details>
<summary>slice_end_out_of_bounds</summary>

```python
node = onnx.helper.make_node(
    'Slice',
    inputs=['x', 'starts', 'ends', 'axes', 'steps'],
    outputs=['y'],
)

x = np.random.randn(20, 10, 5).astype(np.float32)
starts = np.array([1], dtype=np.int64)
ends = np.array([1000], dtype=np.int64)
axes = np.array([1], dtype=np.int64)
steps = np.array([1], dtype=np.int64)
y = x[:, 1:1000]

expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
       name='test_slice_end_out_of_bounds')
```

</details>
<details>
<summary>slice_neg</summary>

```python
node = onnx.helper.make_node(
    'Slice',
    inputs=['x', 'starts', 'ends', 'axes', 'steps'],
    outputs=['y'],
)

x = np.random.randn(20, 10, 5).astype(np.float32)
starts = np.array([0], dtype=np.int64)
ends = np.array([-1], dtype=np.int64)
axes = np.array([1], dtype=np.int64)
steps = np.array([1], dtype=np.int64)
y = x[:, 0:-1]

expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
       name='test_slice_neg')
```

</details>
<details>
<summary>slice_neg_steps</summary>

```python
node = onnx.helper.make_node(
    'Slice',
    inputs=['x', 'starts', 'ends', 'axes', 'steps'],
    outputs=['y'],
)

x = np.random.randn(20, 10, 5).astype(np.float32)
starts = np.array([20, 10, 4], dtype=np.int64)
ends = np.array([0, 0, 1], dtype=np.int64)
axes = np.array([0, 1, 2], dtype=np.int64)
steps = np.array([-1, -3, -2]).astype(np.int64)
y = x[20:0:-1, 10:0:-3, 4:1:-2]

expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
       name='test_slice_neg_steps')
```

</details>
<details>
<summary>slice_negative_axes</summary>

```python
node = onnx.helper.make_node(
    'Slice',
    inputs=['x', 'starts', 'ends', 'axes'],
    outputs=['y'],
)

x = np.random.randn(20, 10, 5).astype(np.float32)
starts = np.array([0, 0, 3], dtype=np.int64)
ends = np.array([20, 10, 4], dtype=np.int64)
axes = np.array([0, -2, -1], dtype=np.int64)
y = x[:, :, 3:4]

expect(node, inputs=[x, starts, ends, axes], outputs=[y],
       name='test_slice_negative_axes')
```

</details>
<details>
<summary>slice_start_out_of_bounds</summary>

```python
node = onnx.helper.make_node(
    'Slice',
    inputs=['x', 'starts', 'ends', 'axes', 'steps'],
    outputs=['y'],
)

x = np.random.randn(20, 10, 5).astype(np.float32)
starts = np.array([1000], dtype=np.int64)
ends = np.array([1000], dtype=np.int64)
axes = np.array([1], dtype=np.int64)
steps = np.array([1], dtype=np.int64)
y = x[:, 1000:1000]

expect(node, inputs=[x, starts, ends, axes, steps], outputs=[y],
       name='test_slice_start_out_of_bounds')
```

</details>


### Softmax
There are 2 test cases, listed as following:
<details>
<summary>softmax</summary>

```python
node = onnx.helper.make_node(
    'Softmax',
    inputs=['x'],
    outputs=['y'],
)
x = np.array([[-1, 0, 1]]).astype(np.float32)
# expected output [[0.09003058, 0.24472848, 0.66524094]]
y = softmax(x, axis=1)
expect(node, inputs=[x], outputs=[y],
       name='test_softmax_example')
```

</details>
<details>
<summary>softmax_axis</summary>

```python
x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]
             ).astype(np.float32)
# expected output
# [[0.032058604 0.08714432  0.23688284  0.6439143  ]
# [0.032058604 0.08714432  0.23688284  0.6439143  ]]
y = softmax(x)

node = onnx.helper.make_node(
    'Softmax',
    inputs=['x'],
    outputs=['y'],
)
expect(node, inputs=[x], outputs=[y],
       name='test_softmax_large_number')

x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
node = onnx.helper.make_node(
    'Softmax',
    inputs=['x'],
    outputs=['y'],
    axis=0,
)
y = softmax(x, axis=0)
expect(node, inputs=[x], outputs=[y],
       name='test_softmax_axis_0')

node = onnx.helper.make_node(
    'Softmax',
    inputs=['x'],
    outputs=['y'],
    axis=1,
)
y = softmax(x, axis=1)
expect(node, inputs=[x], outputs=[y],
       name='test_softmax_axis_1')

node = onnx.helper.make_node(
    'Softmax',
    inputs=['x'],
    outputs=['y'],
    axis=2,
)
y = softmax(x, axis=2)
expect(node, inputs=[x], outputs=[y],
       name='test_softmax_axis_2')

node = onnx.helper.make_node(
    'Softmax',
    inputs=['x'],
    outputs=['y'],
    axis=-1,
)
y = softmax(x, axis=-1)
expect(node, inputs=[x], outputs=[y],
       name='test_softmax_negative_axis')

# default axis is -1
node = onnx.helper.make_node(
    'Softmax',
    inputs=['x'],
    outputs=['y'],
)
expect(node, inputs=[x], outputs=[y],
       name='test_softmax_default_axis')
```

</details>


### SoftmaxCrossEntropyLoss
There are 34 test cases, listed as following:
<details>
<summary>input_shape_is_NCd1_mean_weight_negative_ii</summary>

```python
reduction = 'mean'
ignore_index = np.int64(-1)

node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y', 'w'],
                             outputs=['z'],
                             reduction=reduction,
                             ignore_index=ignore_index)

N, C, dim1 = 3, 5, 6
np.random.seed(0)
x = np.random.rand(N, C, dim1).astype(np.float32)
labels = np.random.randint(0, high=C, size=(N, dim1)).astype(np.int64)
labels[0][0] = -1
weight = np.random.rand(C).astype(np.float32)

sce = softmaxcrossentropy(x,
                          labels,
                          weight=weight,
                          reduction=reduction,
                          ignore_index=ignore_index)

expect(node, inputs=[x, labels, weight], outputs=[sce], name='test_sce_NCd1_mean_weight_negative_ii')
```

</details>
<details>
<summary>input_shape_is_NCd1_mean_weight_negative_ii_log_prob</summary>

```python
reduction = 'mean'
ignore_index = np.int64(-1)

node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y', 'w'],
                             outputs=['z', 'log_prob'],
                             reduction=reduction,
                             ignore_index=ignore_index)

N, C, dim1 = 3, 5, 6
np.random.seed(0)
x = np.random.rand(N, C, dim1).astype(np.float32)
labels = np.random.randint(0, high=C, size=(N, dim1)).astype(np.int64)
labels[0][0] = -1
weight = np.random.rand(C).astype(np.float32)

loss, log_prob = softmaxcrossentropy(x,
                          labels,
                          weight=weight,
                          reduction=reduction,
                          ignore_index=ignore_index,
                          get_log_prob=True)

expect(node, inputs=[x, labels, weight], outputs=[loss, log_prob], name='test_sce_NCd1_mean_weight_negative_ii_log_prob')
```

</details>
<details>
<summary>input_shape_is_NCd1d2d3_none_no_weight_negative_ii</summary>

```python
reduction = 'none'
ignore_index = np.int64(-5)

node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y'],
                             outputs=['z'],
                             reduction=reduction,
                             ignore_index=ignore_index)

N, C, dim1, dim2, dim3 = 3, 5, 6, 6, 5
np.random.seed(0)
x = np.random.rand(N, C, dim1, dim2, dim3).astype(np.float32)
labels = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3)).astype(np.int64)
labels[0][0][0][0] = -5

sce = softmaxcrossentropy(x,
                          labels,
                          reduction=reduction,
                          ignore_index=ignore_index)

expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_NCd1d2d3_none_no_weight_negative_ii')
```

</details>
<details>
<summary>input_shape_is_NCd1d2d3_none_no_weight_negative_ii_log_prob</summary>

```python
reduction = 'none'
ignore_index = np.int64(-5)

node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y'],
                             outputs=['z', 'log_prob'],
                             reduction=reduction,
                             ignore_index=ignore_index)

N, C, dim1, dim2, dim3 = 3, 5, 6, 6, 5
np.random.seed(0)
x = np.random.rand(N, C, dim1, dim2, dim3).astype(np.float32)
labels = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3)).astype(np.int64)
labels[0][0][0][0] = -5

loss, log_prob = softmaxcrossentropy(x,
                          labels,
                          reduction=reduction,
                          ignore_index=ignore_index,
                          get_log_prob=True)

expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_NCd1d2d3_none_no_weight_negative_ii_log_prob')
```

</details>
<details>
<summary>input_shape_is_NCd1d2d3_sum_weight_high_ii</summary>

```python
reduction = 'sum'
ignore_index = np.int64(10)

node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y', 'w'],
                             outputs=['z'],
                             reduction=reduction,
                             ignore_index=ignore_index)

N, C = 3, 5
np.random.seed(0)
x = np.random.rand(N, C).astype(np.float32)
labels = np.random.randint(0, high=C, size=(N)).astype(np.int64)
labels[0] = 10
weight = np.random.rand(C).astype(np.float32)

sce = softmaxcrossentropy(x,
                          labels,
                          weight=weight,
                          reduction=reduction,
                          ignore_index=ignore_index)

expect(node, inputs=[x, labels, weight], outputs=[sce], name='test_sce_NCd1d2d3_sum_weight_high_ii')
```

</details>
<details>
<summary>input_shape_is_NCd1d2d3_sum_weight_high_ii_log_prob</summary>

```python
reduction = 'sum'
ignore_index = np.int64(10)

node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y', 'w'],
                             outputs=['z', 'log_prob'],
                             reduction=reduction,
                             ignore_index=ignore_index)

N, C = 3, 5
np.random.seed(0)
x = np.random.rand(N, C).astype(np.float32)
labels = np.random.randint(0, high=C, size=(N)).astype(np.int64)
labels[0] = 10
weight = np.random.rand(C).astype(np.float32)

loss, log_prob = softmaxcrossentropy(x,
                          labels,
                          weight=weight,
                          reduction=reduction,
                          ignore_index=ignore_index,
                          get_log_prob=True)

expect(node, inputs=[x, labels, weight], outputs=[loss, log_prob], name='test_sce_NCd1d2d3_sum_weight_high_ii_log_prob')
```

</details>
<details>
<summary>input_shape_is_NCd1d2d3d4d5_mean_weight</summary>

```python
reduction = 'mean'

node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y', 'w'],
                             outputs=['z'],
                             reduction=reduction)

N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
np.random.seed(0)
x = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
labels = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)).astype(np.int64)
weight = np.random.rand(C).astype(np.float32)

sce = softmaxcrossentropy(x,
                        labels,
                        weight=weight,
                        reduction=reduction)

expect(node, inputs=[x, labels, weight], outputs=[sce], name='test_sce_NCd1d2d3d4d5_mean_weight')
```

</details>
<details>
<summary>input_shape_is_NCd1d2d3d4d5_mean_weight_log_prob</summary>

```python
reduction = 'mean'

node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y', 'w'],
                             outputs=['z', 'log_prob'],
                             reduction=reduction)

N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
np.random.seed(0)
x = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
labels = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)).astype(np.int64)
weight = np.random.rand(C).astype(np.float32)

loss, log_prob = softmaxcrossentropy(x,
                        labels,
                        weight=weight,
                        reduction=reduction,
                        get_log_prob=True)

expect(node, inputs=[x, labels, weight], outputs=[loss, log_prob], name='test_sce_NCd1d2d3d4d5_mean_weight_log_prob')
```

</details>
<details>
<summary>input_shape_is_NCd1d2d3d4d5_none_no_weight</summary>

```python
reduction = 'none'

node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y'],
                             outputs=['z'],
                             reduction=reduction)

N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
np.random.seed(0)
x = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
labels = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)).astype(np.int64)

sce = softmaxcrossentropy(x,
                        labels,
                        reduction=reduction)

expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_NCd1d2d3d4d5_none_no_weight')
```

</details>
<details>
<summary>input_shape_is_NCd1d2d3d4d5_none_no_weight_log_prob</summary>

```python
reduction = 'none'

node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y'],
                             outputs=['z', 'log_prob'],
                             reduction=reduction)

N, C, dim1, dim2, dim3, dim4, dim5 = 3, 5, 6, 6, 5, 3, 4
np.random.seed(0)
x = np.random.rand(N, C, dim1, dim2, dim3, dim4, dim5).astype(np.float32)
labels = np.random.randint(0, high=C, size=(N, dim1, dim2, dim3, dim4, dim5)).astype(np.int64)

loss, log_prob = softmaxcrossentropy(x,
                        labels,
                        reduction=reduction,
                        get_log_prob=True)

expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_NCd1d2d3d4d5_none_no_weight_log_prob')
```

</details>
<details>
<summary>softmaxcrossentropy_mean</summary>

```python
# Define operator attributes.
reduction = 'mean'

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y'],
                             outputs=['z'],
                             reduction=reduction)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)

# Compute SoftmaxCrossEntropyLoss
sce = softmaxcrossentropy(x, labels)

# Check results
expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_mean')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_3d</summary>

```python
# Define operator attributes.
reduction = 'mean'

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y'],
                             outputs=['z'],
                             reduction=reduction)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5, 2).astype(np.float32)
y = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)

# Compute SoftmaxCrossEntropyLoss
sce = softmaxcrossentropy(x, y)

# Check results
expect(node, inputs=[x, y], outputs=[sce], name='test_sce_mean_3d')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_3d_log_prob</summary>

```python
# Define operator attributes.
reduction = 'mean'

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y'],
                             outputs=['z', 'log_prob'],
                             reduction=reduction)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5, 2).astype(np.float32)
y = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)

# Compute SoftmaxCrossEntropyLoss
loss, log_prob = softmaxcrossentropy(x, y, get_log_prob=True)

# Check results
expect(node, inputs=[x, y], outputs=[loss, log_prob], name='test_sce_mean_3d_log_prob')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_log_prob</summary>

```python
# Define operator attributes.
reduction = 'mean'

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y'],
                             outputs=['z', 'log_prob'],
                             reduction=reduction)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)

# Compute SoftmaxCrossEntropyLoss
loss, log_prob = softmaxcrossentropy(x, labels, get_log_prob=True)

# Check results
expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_mean_log_prob')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_no_weights_ii</summary>

```python
# Define operator attributes.
reduction = 'mean'
ignore_index = np.int64(2)

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                            inputs=['x', 'y'],
                            outputs=['z'],
                            reduction=reduction,
                            ignore_index=ignore_index)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
labels[0] = np.int64(2)

# Compute SoftmaxCrossEntropyLoss
sce = softmaxcrossentropy(x, labels, ignore_index=ignore_index)

# Check results
expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_mean_no_weight_ii')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_no_weights_ii_3d</summary>

```python
# Define operator attributes.
reduction = 'mean'
ignore_index = np.int64(2)

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                            inputs=['x', 'y'],
                            outputs=['z'],
                            reduction=reduction,
                            ignore_index=ignore_index)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5, 2).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)
labels[0][0] = np.int64(2)

# Compute SoftmaxCrossEntropyLoss
sce = softmaxcrossentropy(x, labels, ignore_index=ignore_index)

# Check results
expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_mean_no_weight_ii_3d')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_no_weights_ii_3d_log_prob</summary>

```python
# Define operator attributes.
reduction = 'mean'
ignore_index = np.int64(2)

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                            inputs=['x', 'y'],
                            outputs=['z', 'log_prob'],
                            reduction=reduction,
                            ignore_index=ignore_index)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5, 2).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)
labels[0][0] = np.int64(2)

# Compute SoftmaxCrossEntropyLoss
loss, log_prob = softmaxcrossentropy(x, labels, ignore_index=ignore_index, get_log_prob=True)

# Check results
expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_mean_no_weight_ii_3d_log_prob')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_no_weights_ii_4d</summary>

```python
# Define operator attributes.
reduction = 'mean'
ignore_index = np.int64(2)

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                            inputs=['x', 'y'],
                            outputs=['z'],
                            reduction=reduction,
                            ignore_index=ignore_index)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5, 2, 7).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, 2, 7)).astype(np.int64)
labels[0][0][0] = np.int64(2)

# Compute SoftmaxCrossEntropyLoss
sce = softmaxcrossentropy(x, labels, reduction=reduction, ignore_index=ignore_index)

# Check results
expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_mean_no_weight_ii_4d')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_no_weights_ii_4d_log_prob</summary>

```python
# Define operator attributes.
reduction = 'mean'
ignore_index = np.int64(2)

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                            inputs=['x', 'y'],
                            outputs=['z', 'log_prob'],
                            reduction=reduction,
                            ignore_index=ignore_index)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5, 2, 7).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, 2, 7)).astype(np.int64)
labels[0][0][0] = np.int64(2)

# Compute SoftmaxCrossEntropyLoss
loss, log_prob = softmaxcrossentropy(x, labels, reduction=reduction, ignore_index=ignore_index, get_log_prob=True)

# Check results
expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_mean_no_weight_ii_4d_log_prob')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_no_weights_ii_log_prob</summary>

```python
# Define operator attributes.
reduction = 'mean'
ignore_index = np.int64(2)

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                            inputs=['x', 'y'],
                            outputs=['z', 'log_prob'],
                            reduction=reduction,
                            ignore_index=ignore_index)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
labels[0] = np.int64(2)

# Compute SoftmaxCrossEntropyLoss
loss, log_prob = softmaxcrossentropy(x, labels, ignore_index=ignore_index, get_log_prob=True)

# Check results
expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_mean_no_weight_ii_log_prob')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_weights</summary>

```python
# Define operator attributes.
reduction = 'mean'

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y', 'w'],
                             outputs=['z'],
                             reduction=reduction)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

# Compute SoftmaxCrossEntropyLoss
sce = softmaxcrossentropy(x, labels, weight=weights)

# Check results
expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_sce_mean_weight')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_weights_ii</summary>

```python
# Define operator attributes.
reduction = 'mean'
ignore_index = np.int64(0)

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y', 'w'],
                             outputs=['z'],
                             reduction=reduction,
                             ignore_index=ignore_index)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
labels[0] = np.int64(0)
weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

# Compute SoftmaxCrossEntropyLoss
sce = softmaxcrossentropy(x, labels, weight=weights, ignore_index=ignore_index)

# Check results
expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_sce_mean_weight_ii')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_weights_ii_3d</summary>

```python
# Define operator attributes.
reduction = 'mean'
ignore_index = np.int64(1)

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                            inputs=['x', 'y', 'w'],
                            outputs=['z'],
                            reduction=reduction,
                            ignore_index=ignore_index)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5, 2).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)
labels[0][0] = np.int64(1)
weights = np.array([0.2, 0.3, 0.6, 0.1, 0.5], dtype=np.float32)

# Compute SoftmaxCrossEntropyLoss
sce = softmaxcrossentropy(x, labels, weight=weights, ignore_index=ignore_index)

# Check results
expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_sce_mean_weight_ii_3d')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_weights_ii_3d_log_prob</summary>

```python
# Define operator attributes.
reduction = 'mean'
ignore_index = np.int64(1)

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                            inputs=['x', 'y', 'w'],
                            outputs=['z', 'log_prob'],
                            reduction=reduction,
                            ignore_index=ignore_index)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5, 2).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, 2)).astype(np.int64)
labels[0][0] = np.int64(1)
weights = np.array([0.2, 0.3, 0.6, 0.1, 0.5], dtype=np.float32)

# Compute SoftmaxCrossEntropyLoss
loss, log_prob = softmaxcrossentropy(x, labels, weight=weights, ignore_index=ignore_index, get_log_prob=True)

# Check results
expect(node, inputs=[x, labels, weights], outputs=[loss, log_prob], name='test_sce_mean_weight_ii_3d_log_prob')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_weights_ii_4d</summary>

```python
# Define operator attributes.
reduction = 'mean'
ignore_index = np.int64(2)

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                            inputs=['x', 'y', 'w'],
                            outputs=['z'],
                            reduction=reduction,
                            ignore_index=ignore_index)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5, 2, 7).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, 2, 7)).astype(np.int64)
labels[0][0][0] = np.int64(2)
weights = np.array([0.2, 0.3, 0.6, 0.1, 0.5], dtype=np.float32)

# Compute SoftmaxCrossEntropyLoss
sce = softmaxcrossentropy(x, labels, reduction=reduction, weight=weights, ignore_index=ignore_index)

# Check results
expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_sce_mean_weight_ii_4d')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_weights_ii_4d_log_prob</summary>

```python
# Define operator attributes.
reduction = 'mean'
ignore_index = np.int64(2)

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                            inputs=['x', 'y', 'w'],
                            outputs=['z', 'log_prob'],
                            reduction=reduction,
                            ignore_index=ignore_index)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5, 2, 7).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, 2, 7)).astype(np.int64)
labels[0][0][0] = np.int64(2)
weights = np.array([0.2, 0.3, 0.6, 0.1, 0.5], dtype=np.float32)

# Compute SoftmaxCrossEntropyLoss
loss, log_prob = softmaxcrossentropy(x, labels, reduction=reduction, weight=weights, ignore_index=ignore_index, get_log_prob=True)

# Check results
expect(node, inputs=[x, labels, weights], outputs=[loss, log_prob], name='test_sce_mean_weight_ii_4d_log_prob')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_weights_ii_log_prob</summary>

```python
# Define operator attributes.
reduction = 'mean'
ignore_index = np.int64(0)

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y', 'w'],
                             outputs=['z', 'log_prob'],
                             reduction=reduction,
                             ignore_index=ignore_index)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
labels[0] = np.int64(0)
weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

# Compute SoftmaxCrossEntropyLoss
loss, log_prob = softmaxcrossentropy(x, labels, weight=weights, ignore_index=ignore_index, get_log_prob=True)

# Check results
expect(node, inputs=[x, labels, weights], outputs=[loss, log_prob], name='test_sce_mean_weight_ii_log_prob')
```

</details>
<details>
<summary>softmaxcrossentropy_mean_weights_log_prob</summary>

```python
# Define operator attributes.
reduction = 'mean'

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y', 'w'],
                             outputs=['z', 'log_prob'],
                             reduction=reduction)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

# Compute SoftmaxCrossEntropyLoss
loss, log_prob = softmaxcrossentropy(x, labels, weight=weights, get_log_prob=True)

# Check results
expect(node, inputs=[x, labels, weights], outputs=[loss, log_prob], name='test_sce_mean_weight_log_prob')
```

</details>
<details>
<summary>softmaxcrossentropy_none</summary>

```python
# Define operator attributes.
reduction = 'none'

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y'],
                             outputs=['z'],
                             reduction=reduction)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)

# Compute SoftmaxCrossEntropyLoss
sce = softmaxcrossentropy(x, labels, reduction='none')

# Check results
expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_none')
```

</details>
<details>
<summary>softmaxcrossentropy_none_log_prob</summary>

```python
# Define operator attributes.
reduction = 'none'

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y'],
                             outputs=['z', 'log_prob'],
                             reduction=reduction)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)

# Compute SoftmaxCrossEntropyLoss
loss, log_prob = softmaxcrossentropy(x, labels, reduction='none', get_log_prob=True)

# Check results
expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_none_log_prob')
```

</details>
<details>
<summary>softmaxcrossentropy_none_weights</summary>

```python
# Define operator attributes.
reduction = 'none'

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y', 'w'],
                             outputs=['z'],
                             reduction=reduction)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

# Compute SoftmaxCrossEntropyLoss
sce = softmaxcrossentropy(x, labels, weight=weights, reduction='none')

# Check results
expect(node, inputs=[x, labels, weights], outputs=[sce], name='test_sce_none_weights')
```

</details>
<details>
<summary>softmaxcrossentropy_none_weights_log_prob</summary>

```python
# Define operator attributes.
reduction = 'none'

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y', 'w'],
                             outputs=['z', 'log_prob'],
                             reduction=reduction)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)
weights = np.array([0.9, 0.7, 0.8, 0.9, 0.9], dtype=np.float32)

# Compute SoftmaxCrossEntropyLoss
loss, log_prob = softmaxcrossentropy(x, labels, weight=weights, reduction='none', get_log_prob=True)

# Check results
expect(node, inputs=[x, labels, weights], outputs=[loss, log_prob], name='test_sce_none_weights_log_prob')
```

</details>
<details>
<summary>softmaxcrossentropy_sum</summary>

```python
# Define operator attributes.
reduction = 'sum'

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y'],
                             outputs=['z'],
                             reduction=reduction)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)

# Compute SoftmaxCrossEntropyLoss
sce = softmaxcrossentropy(x, labels, reduction='sum')

# Check results
expect(node, inputs=[x, labels], outputs=[sce], name='test_sce_sum')
```

</details>
<details>
<summary>softmaxcrossentropy_sum_log_prob</summary>

```python
# Define operator attributes.
reduction = 'sum'

# Create operator.
node = onnx.helper.make_node('SoftmaxCrossEntropyLoss',
                             inputs=['x', 'y'],
                             outputs=['z', 'log_prob'],
                             reduction=reduction)

# Define operator inputs.
np.random.seed(0)
x = np.random.rand(3, 5).astype(np.float32)
labels = np.random.randint(0, high=5, size=(3, )).astype(np.int64)

# Compute SoftmaxCrossEntropyLoss
loss, log_prob = softmaxcrossentropy(x, labels, reduction='sum', get_log_prob=True)

# Check results
expect(node, inputs=[x, labels], outputs=[loss, log_prob], name='test_sce_sum_log_prob')
```

</details>


### Softplus
There are 1 test cases, listed as following:
<details>
<summary>softplus</summary>

```python
node = onnx.helper.make_node(
    'Softplus',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.log(np.exp(x) + 1)  # expected output [0.31326166, 0.69314718, 1.31326163]
expect(node, inputs=[x], outputs=[y],
       name='test_softplus_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.log(np.exp(x) + 1)
expect(node, inputs=[x], outputs=[y],
       name='test_softplus')
```

</details>


### Softsign
There are 1 test cases, listed as following:
<details>
<summary>softsign</summary>

```python
node = onnx.helper.make_node(
    'Softsign',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.array([-0.5, 0, 0.5]).astype(np.float32)
expect(node, inputs=[x], outputs=[y],
       name='test_softsign_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = x / (1 + np.abs(x))
expect(node, inputs=[x], outputs=[y],
       name='test_softsign')
```

</details>


### SpaceToDepth
There are 2 test cases, listed as following:
<details>
<summary>example</summary>

```python
node = onnx.helper.make_node(
    'SpaceToDepth',
    inputs=['x'],
    outputs=['y'],
    blocksize=2,
)

# (1, 1, 4, 6) input tensor
x = np.array([[[[0, 6, 1, 7, 2, 8],
                [12, 18, 13, 19, 14, 20],
                [3, 9, 4, 10, 5, 11],
                [15, 21, 16, 22, 17, 23]]]]).astype(np.float32)

# (1, 4, 2, 3) output tensor
y = np.array([[[[0, 1, 2],
                [3, 4, 5]],
               [[6, 7, 8],
                [9, 10, 11]],
               [[12, 13, 14],
                [15, 16, 17]],
               [[18, 19, 20],
                [21, 22, 23]]]]).astype(np.float32)
expect(node, inputs=[x], outputs=[y],
       name='test_spacetodepth_example')
```

</details>
<details>
<summary>spacetodepth</summary>

```python
b, c, h, w = shape = (2, 2, 6, 6)
blocksize = 2
node = onnx.helper.make_node(
    'SpaceToDepth',
    inputs=['x'],
    outputs=['y'],
    blocksize=blocksize,
)
x = np.random.random_sample(shape).astype(np.float32)
tmp = np.reshape(x, [b, c,
                     h // blocksize, blocksize,
                     w // blocksize, blocksize])
tmp = np.transpose(tmp, [0, 3, 5, 1, 2, 4])
y = np.reshape(tmp, [b, c * (blocksize**2),
                     h // blocksize,
                     w // blocksize])
expect(node, inputs=[x], outputs=[y],
       name='test_spacetodepth')
```

</details>


### Split
There are 4 test cases, listed as following:
<details>
<summary>1d</summary>

```python
input = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float32)

node = onnx.helper.make_node(
    'Split',
    inputs=['input'],
    outputs=['output_1', 'output_2', 'output_3'],
    axis=0
)

expected_outputs = [np.array([1., 2.]).astype(np.float32), np.array([3., 4.]).astype(np.float32), np.array([5., 6.]).astype(np.float32)]
expect(node, inputs=[input], outputs=[y for y in expected_outputs], name='test_split_equal_parts_1d')

split = np.array([2, 4]).astype(np.int64)
node = onnx.helper.make_node(
    'Split',
    inputs=['input', 'split'],
    outputs=['output_1', 'output_2'],
    axis=0,
)

expected_outputs = [np.array([1., 2.]).astype(np.float32), np.array([3., 4., 5., 6.]).astype(np.float32)]
expect(node, inputs=[input, split], outputs=[y for y in expected_outputs], name='test_split_variable_parts_1d')
```

</details>
<details>
<summary>2d</summary>

```python
input = np.array([[1., 2., 3., 4., 5., 6.],
                  [7., 8., 9., 10., 11., 12.]]).astype(np.float32)

node = onnx.helper.make_node(
    'Split',
    inputs=['input'],
    outputs=['output_1', 'output_2'],
    axis=1
)

expected_outputs = [np.array([[1., 2., 3.], [7., 8., 9.]]).astype(np.float32),
                    np.array([[4., 5., 6.], [10., 11., 12.]]).astype(np.float32)]

expect(node, inputs=[input], outputs=[y for y in expected_outputs], name='test_split_equal_parts_2d')

split = np.array([2, 4]).astype(np.int64)
node = onnx.helper.make_node(
    'Split',
    inputs=['input', 'split'],
    outputs=['output_1', 'output_2'],
    axis=1,
)

expected_outputs = [np.array([[1., 2.], [7., 8.]]).astype(np.float32),
                    np.array([[3., 4., 5., 6.], [9., 10., 11., 12.]]).astype(np.float32)]

expect(node, inputs=[input, split], outputs=[y for y in expected_outputs], name='test_split_variable_parts_2d')
```

</details>
<details>
<summary>default_values</summary>

```python
input = np.array([1., 2., 3., 4., 5., 6.]).astype(np.float32)

# If axis is not specified, split is applied on default axis 0
node = onnx.helper.make_node(
    'Split',
    inputs=['input'],
    outputs=['output_1', 'output_2', 'output_3']
)

expected_outputs = [np.array([1., 2.]).astype(np.float32), np.array([3., 4.]).astype(np.float32), np.array([5., 6.]).astype(np.float32)]
expect(node, inputs=[input], outputs=[y for y in expected_outputs], name='test_split_equal_parts_default_axis')

split = np.array([2, 4]).astype(np.int64)
node = onnx.helper.make_node(
    'Split',
    inputs=['input', 'split'],
    outputs=['output_1', 'output_2']
)

expected_outputs = [np.array([1., 2.]).astype(np.float32), np.array([3., 4., 5., 6.]).astype(np.float32)]
expect(node, inputs=[input, split], outputs=[y for y in expected_outputs], name='test_split_variable_parts_default_axis')
```

</details>
<details>
<summary>zero_size_splits</summary>

```python
input = np.array([]).astype(np.float32)

# Split emtpy tensor to tensors of size zero
split = np.array([0, 0, 0]).astype(np.int64)
node = onnx.helper.make_node(
    'Split',
    inputs=['input', 'split'],
    outputs=['output_1', 'output_2', 'output_3']
)

expected_outputs = [np.array([]).astype(np.float32), np.array([]).astype(np.float32), np.array([]).astype(np.float32)]
expect(node, inputs=[input, split], outputs=[y for y in expected_outputs], name='test_split_zero_size_splits')
```

</details>


### Sqrt
There are 1 test cases, listed as following:
<details>
<summary>sqrt</summary>

```python
node = onnx.helper.make_node(
    'Sqrt',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([1, 4, 9]).astype(np.float32)
y = np.sqrt(x)  # expected output [1., 2., 3.]
expect(node, inputs=[x], outputs=[y],
       name='test_sqrt_example')

x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
y = np.sqrt(x)
expect(node, inputs=[x], outputs=[y],
       name='test_sqrt')
```

</details>


### Squeeze
There are 2 test cases, listed as following:
<details>
<summary>squeeze</summary>

```python
node = onnx.helper.make_node(
    'Squeeze',
    inputs=['x', 'axes'],
    outputs=['y'],
)
x = np.random.randn(1, 3, 4, 5).astype(np.float32)
axes = np.array([0], dtype=np.int64)
y = np.squeeze(x, axis=0)

expect(node, inputs=[x, axes], outputs=[y],
       name='test_squeeze')
```

</details>
<details>
<summary>squeeze_negative_axes</summary>

```python
node = onnx.helper.make_node(
    'Squeeze',
    inputs=['x', 'axes'],
    outputs=['y'],
)
x = np.random.randn(1, 3, 1, 5).astype(np.float32)
axes = np.array([-2], dtype=np.int64)
y = np.squeeze(x, axis=-2)
expect(node, inputs=[x, axes], outputs=[y],
       name='test_squeeze_negative_axes')
```

</details>


### StringNormalizer
There are 6 test cases, listed as following:
<details>
<summary>monday_casesensintive_lower</summary>

```python
input = np.array(['monday', 'tuesday', 'wednesday', 'thursday']).astype(object)
output = np.array(['tuesday', 'wednesday', 'thursday']).astype(object)
stopwords = ['monday']

node = onnx.helper.make_node(
    'StringNormalizer',
    inputs=['x'],
    outputs=['y'],
    case_change_action='LOWER',
    is_case_sensitive=1,
    stopwords=stopwords
)
expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_casesensintive_lower')
```

</details>
<details>
<summary>monday_casesensintive_nochangecase</summary>

```python
input = np.array(['monday', 'tuesday', 'wednesday', 'thursday']).astype(object)
output = np.array(['tuesday', 'wednesday', 'thursday']).astype(object)
stopwords = ['monday']

node = onnx.helper.make_node(
    'StringNormalizer',
    inputs=['x'],
    outputs=['y'],
    is_case_sensitive=1,
    stopwords=stopwords
)
expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_casesensintive_nochangecase')
```

</details>
<details>
<summary>monday_casesensintive_upper</summary>

```python
input = np.array(['monday', 'tuesday', 'wednesday', 'thursday']).astype(object)
output = np.array(['TUESDAY', 'WEDNESDAY', 'THURSDAY']).astype(object)
stopwords = ['monday']

node = onnx.helper.make_node(
    'StringNormalizer',
    inputs=['x'],
    outputs=['y'],
    case_change_action='UPPER',
    is_case_sensitive=1,
    stopwords=stopwords
)
expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_casesensintive_upper')
```

</details>
<details>
<summary>monday_empty_output</summary>

```python
input = np.array(['monday', 'monday']).astype(object)
output = np.array(['']).astype(object)
stopwords = ['monday']

node = onnx.helper.make_node(
    'StringNormalizer',
    inputs=['x'],
    outputs=['y'],
    case_change_action='UPPER',
    is_case_sensitive=1,
    stopwords=stopwords
)
expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_empty_output')
```

</details>
<details>
<summary>monday_insensintive_upper_twodim</summary>

```python
input = np.array(['Monday', 'tuesday', 'wednesday', 'Monday', 'tuesday', 'wednesday']).astype(object).reshape([1, 6])

# It does upper case cecedille, accented E
# and german umlaut but fails
# with german eszett
output = np.array(['TUESDAY', 'WEDNESDAY', 'TUESDAY', 'WEDNESDAY']).astype(object).reshape([1, 4])
stopwords = ['monday']

node = onnx.helper.make_node(
    'StringNormalizer',
    inputs=['x'],
    outputs=['y'],
    case_change_action='UPPER',
    stopwords=stopwords
)
expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_insensintive_upper_twodim')
```

</details>
<details>
<summary>nostopwords_nochangecase</summary>

```python
input = np.array(['monday', 'tuesday']).astype(object)
output = input

# No stopwords. This is a NOOP
node = onnx.helper.make_node(
    'StringNormalizer',
    inputs=['x'],
    outputs=['y'],
    is_case_sensitive=1,
)
expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_nostopwords_nochangecase')
```

</details>


### Sub
There are 2 test cases, listed as following:
<details>
<summary>sub</summary>

```python
node = onnx.helper.make_node(
    'Sub',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.array([1, 2, 3]).astype(np.float32)
y = np.array([3, 2, 1]).astype(np.float32)
z = x - y  # expected output [-2., 0., 2.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_sub_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(3, 4, 5).astype(np.float32)
z = x - y
expect(node, inputs=[x, y], outputs=[z],
       name='test_sub')

x = np.random.randint(12, 24, size=(3, 4, 5), dtype=np.uint8)
y = np.random.randint(12, size=(3, 4, 5), dtype=np.uint8)
z = x - y
expect(node, inputs=[x, y], outputs=[z],
       name='test_sub_uint8')
```

</details>
<details>
<summary>sub_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Sub',
    inputs=['x', 'y'],
    outputs=['z'],
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(5).astype(np.float32)
z = x - y
expect(node, inputs=[x, y], outputs=[z],
       name='test_sub_bcast')
```

</details>


### Sum
There are 1 test cases, listed as following:
<details>
<summary>sum</summary>

```python
data_0 = np.array([3, 0, 2]).astype(np.float32)
data_1 = np.array([1, 3, 4]).astype(np.float32)
data_2 = np.array([2, 6, 6]).astype(np.float32)
result = np.array([6, 9, 12]).astype(np.float32)
node = onnx.helper.make_node(
    'Sum',
    inputs=['data_0', 'data_1', 'data_2'],
    outputs=['result'],
)
expect(node, inputs=[data_0, data_1, data_2], outputs=[result],
       name='test_sum_example')

node = onnx.helper.make_node(
    'Sum',
    inputs=['data_0'],
    outputs=['result'],
)
expect(node, inputs=[data_0], outputs=[data_0],
       name='test_sum_one_input')

result = np.add(data_0, data_1)
node = onnx.helper.make_node(
    'Sum',
    inputs=['data_0', 'data_1'],
    outputs=['result'],
)
expect(node, inputs=[data_0, data_1], outputs=[result],
       name='test_sum_two_inputs')
```

</details>


### Tan
There are 1 test cases, listed as following:
<details>
<summary>tan</summary>

```python
node = onnx.helper.make_node(
    'Tan',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.tan(x)
expect(node, inputs=[x], outputs=[y],
       name='test_tan_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.tan(x)
expect(node, inputs=[x], outputs=[y],
       name='test_tan')
```

</details>


### Tanh
There are 1 test cases, listed as following:
<details>
<summary>tanh</summary>

```python
node = onnx.helper.make_node(
    'Tanh',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([-1, 0, 1]).astype(np.float32)
y = np.tanh(x)  # expected output [-0.76159418, 0., 0.76159418]
expect(node, inputs=[x], outputs=[y],
       name='test_tanh_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.tanh(x)
expect(node, inputs=[x], outputs=[y],
       name='test_tanh')
```

</details>


### TfIdfVectorizer
There are 7 test cases, listed as following:
<details>
<summary>tf_batch_onlybigrams_skip0</summary>

```python
input = np.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(np.int32)
output = np.array([[0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 0., 1.]]).astype(np.float32)

ngram_counts = np.array([0, 4]).astype(np.int64)
ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
pool_int64s = np.array([2, 3, 5, 4,    # unigrams
                        5, 6, 7, 8, 6, 7]).astype(np.int64)   # bigrams

helper = TfIdfVectorizerHelper(
    mode='TF',
    min_gram_length=2,
    max_gram_length=2,
    max_skip_count=0,
    ngram_counts=ngram_counts,
    ngram_indexes=ngram_indexes,
    pool_int64s=pool_int64s
)
node = helper.make_node_noweights()
expect(node, inputs=[input], outputs=[output], name='test_tfidfvectorizer_tf_batch_onlybigrams_skip0')
```

</details>
<details>
<summary>tf_batch_onlybigrams_skip5</summary>

```python
input = np.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(np.int32)
output = np.array([[0., 0., 0., 0., 0., 0., 0.], [0., 0., 0., 0., 1., 1., 1.]]).astype(np.float32)

ngram_counts = np.array([0, 4]).astype(np.int64)
ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
pool_int64s = np.array([2, 3, 5, 4,    # unigrams
                        5, 6, 7, 8, 6, 7]).astype(np.int64)   # bigrams

helper = TfIdfVectorizerHelper(
    mode='TF',
    min_gram_length=2,
    max_gram_length=2,
    max_skip_count=5,
    ngram_counts=ngram_counts,
    ngram_indexes=ngram_indexes,
    pool_int64s=pool_int64s
)
node = helper.make_node_noweights()
expect(node, inputs=[input], outputs=[output], name='test_tfidfvectorizer_tf_batch_onlybigrams_skip5')
```

</details>
<details>
<summary>tf_batch_uniandbigrams_skip5</summary>

```python
input = np.array([[1, 1, 3, 3, 3, 7], [8, 6, 7, 5, 6, 8]]).astype(np.int32)
output = np.array([[0., 3., 0., 0., 0., 0., 0.], [0., 0., 1., 0., 1., 1., 1.]]).astype(np.float32)

ngram_counts = np.array([0, 4]).astype(np.int64)
ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
pool_int64s = np.array([2, 3, 5, 4,    # unigrams
                        5, 6, 7, 8, 6, 7]).astype(np.int64)   # bigrams

helper = TfIdfVectorizerHelper(
    mode='TF',
    min_gram_length=1,
    max_gram_length=2,
    max_skip_count=5,
    ngram_counts=ngram_counts,
    ngram_indexes=ngram_indexes,
    pool_int64s=pool_int64s
)
node = helper.make_node_noweights()
expect(node, inputs=[input], outputs=[output], name='test_tfidfvectorizer_tf_batch_uniandbigrams_skip5')
```

</details>
<details>
<summary>tf_only_bigrams_skip0</summary>

```python
input = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]).astype(np.int32)
output = np.array([0., 0., 0., 0., 1., 1., 1.]).astype(np.float32)

ngram_counts = np.array([0, 4]).astype(np.int64)
ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
pool_int64s = np.array([2, 3, 5, 4,    # unigrams
                        5, 6, 7, 8, 6, 7]).astype(np.int64)    # bigrams

helper = TfIdfVectorizerHelper(
    mode='TF',
    min_gram_length=2,
    max_gram_length=2,
    max_skip_count=0,
    ngram_counts=ngram_counts,
    ngram_indexes=ngram_indexes,
    pool_int64s=pool_int64s
)
node = helper.make_node_noweights()
expect(node, inputs=[input], outputs=[output], name='test_tfidfvectorizer_tf_only_bigrams_skip0')
```

</details>
<details>
<summary>tf_onlybigrams_levelempty</summary>

```python
input = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]).astype(np.int32)
output = np.array([1., 1., 1.]).astype(np.float32)

ngram_counts = np.array([0, 0]).astype(np.int64)
ngram_indexes = np.array([0, 1, 2]).astype(np.int64)
pool_int64s = np.array([    # unigrams none
                       5, 6, 7, 8, 6, 7]).astype(np.int64)    # bigrams

helper = TfIdfVectorizerHelper(
    mode='TF',
    min_gram_length=2,
    max_gram_length=2,
    max_skip_count=0,
    ngram_counts=ngram_counts,
    ngram_indexes=ngram_indexes,
    pool_int64s=pool_int64s
)
node = helper.make_node_noweights()
expect(node, inputs=[input], outputs=[output], name='test_tfidfvectorizer_tf_onlybigrams_levelempty')
```

</details>
<details>
<summary>tf_onlybigrams_skip5</summary>

```python
input = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]).astype(np.int32)
output = np.array([0., 0., 0., 0., 1., 3., 1.]).astype(np.float32)

ngram_counts = np.array([0, 4]).astype(np.int64)
ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
pool_int64s = np.array([2, 3, 5, 4,    # unigrams
                        5, 6, 7, 8, 6, 7]).astype(np.int64)    # bigrams

helper = TfIdfVectorizerHelper(
    mode='TF',
    min_gram_length=2,
    max_gram_length=2,
    max_skip_count=5,
    ngram_counts=ngram_counts,
    ngram_indexes=ngram_indexes,
    pool_int64s=pool_int64s
)
node = helper.make_node_noweights()
expect(node, inputs=[input], outputs=[output], name='test_tfidfvectorizer_tf_onlybigrams_skip5')
```

</details>
<details>
<summary>tf_uniandbigrams_skip5</summary>

```python
input = np.array([1, 1, 3, 3, 3, 7, 8, 6, 7, 5, 6, 8]).astype(np.int32)
output = np.array([0., 3., 1., 0., 1., 3., 1.]).astype(np.float32)

ngram_counts = np.array([0, 4]).astype(np.int64)
ngram_indexes = np.array([0, 1, 2, 3, 4, 5, 6]).astype(np.int64)
pool_int64s = np.array([2, 3, 5, 4,    # unigrams
                        5, 6, 7, 8, 6, 7]).astype(np.int64)    # bigrams

helper = TfIdfVectorizerHelper(
    mode='TF',
    min_gram_length=1,
    max_gram_length=2,
    max_skip_count=5,
    ngram_counts=ngram_counts,
    ngram_indexes=ngram_indexes,
    pool_int64s=pool_int64s
)
node = helper.make_node_noweights()
expect(node, inputs=[input], outputs=[output], name='test_tfidfvectorizer_tf_uniandbigrams_skip5')
```

</details>


### ThresholdedRelu
There are 2 test cases, listed as following:
<details>
<summary>default</summary>

```python
default_alpha = 1.0
node = onnx.helper.make_node(
    'ThresholdedRelu',
    inputs=['x'],
    outputs=['y']
)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, default_alpha, np.inf)
y[y == default_alpha] = 0

expect(node, inputs=[x], outputs=[y],
       name='test_thresholdedrelu_default')
```

</details>
<details>
<summary>thresholdedrelu</summary>

```python
alpha = 2.0
node = onnx.helper.make_node(
    'ThresholdedRelu',
    inputs=['x'],
    outputs=['y'],
    alpha=alpha
)

x = np.array([-1.5, 0., 1.2, 2.0, 2.2]).astype(np.float32)
y = np.clip(x, alpha, np.inf)  # expected output [0., 0., 0., 0., 2.2]
y[y == alpha] = 0

expect(node, inputs=[x], outputs=[y],
       name='test_thresholdedrelu_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, alpha, np.inf)
y[y == alpha] = 0

expect(node, inputs=[x], outputs=[y],
       name='test_thresholdedrelu')
```

</details>


### Tile
There are 2 test cases, listed as following:
<details>
<summary>tile</summary>

```python
node = onnx.helper.make_node(
    'Tile',
    inputs=['x', 'y'],
    outputs=['z']
)

x = np.random.rand(2, 3, 4, 5).astype(np.float32)

repeats = np.random.randint(low=1, high=10, size=(np.ndim(x),)).astype(np.int64)

z = np.tile(x, repeats)

expect(node,
       inputs=[x, repeats],
       outputs=[z],
       name='test_tile')
```

</details>
<details>
<summary>tile_precomputed</summary>

```python
node = onnx.helper.make_node(
    'Tile',
    inputs=['x', 'y'],
    outputs=['z']
)

x = np.array([
    [0, 1],
    [2, 3]
], dtype=np.float32)

repeats = np.array([2, 2], dtype=np.int64)

z = np.array([
    [0, 1, 0, 1],
    [2, 3, 2, 3],
    [0, 1, 0, 1],
    [2, 3, 2, 3]
], dtype=np.float32)

expect(node,
       inputs=[x, repeats],
       outputs=[z],
       name='test_tile_precomputed')
```

</details>


### TopK
There are 3 test cases, listed as following:
<details>
<summary>top_k</summary>

```python
axis = 1
largest = 1

k = 3
node = onnx.helper.make_node(
    'TopK',
    inputs=['x', 'k'],
    outputs=['values', 'indices'],
    axis=axis
)
X = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
], dtype=np.float32)
K = np.array([k], dtype=np.int64)
values_ref, indices_ref = topk_sorted_implementation(X, k, axis, largest)

#print(values_ref)
#[[ 3.  2.  1.]
# [ 7.  6.  5.]
# [11. 10.  9.]]
#print(indices_ref)
#[[3 2 1]
# [3 2 1]
# [3 2 1]]

expect(node, inputs=[X, K], outputs=[values_ref, indices_ref],
       name='test_top_k')
```

</details>
<details>
<summary>top_k_negative_axis</summary>

```python
axis = -1
largest = 1

k = 3
node = onnx.helper.make_node(
    'TopK',
    inputs=['x', 'k'],
    outputs=['values', 'indices'],
    axis=axis
)
X = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
], dtype=np.float32)
K = np.array([k], dtype=np.int64)
values_ref, indices_ref = topk_sorted_implementation(X, k, axis, largest)

# print(values_ref)
#[[ 3.  2.  1.]
# [ 7.  6.  5.]
# [11. 10.  9.]]
# print(indices_ref)
#[[3 2 1]
# [3 2 1]
# [3 2 1]]

expect(node, inputs=[X, K], outputs=[values_ref, indices_ref],
       name='test_top_k_negative_axis')
```

</details>
<details>
<summary>top_k_smallest</summary>

```python
axis = 1
largest = 0
sorted = 1
k = 3

node = onnx.helper.make_node(
    'TopK',
    inputs=['x', 'k'],
    outputs=['values', 'indices'],
    axis=axis,
    largest=largest,
    sorted=sorted
)

X = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [11, 10, 9, 8],
], dtype=np.float32)
K = np.array([k], dtype=np.int64)
values_ref, indices_ref = topk_sorted_implementation(X, k, axis, largest)

#print(values_ref)
#[[ 0.  1.  2.]
# [ 4.  5.  6.]
# [ 8.  9. 10.]]
#print(indices_ref)
#[[0 1 2]
# [0 1 2]
# [3 2 1]]

expect(node, inputs=[X, K], outputs=[values_ref, indices_ref],
       name='test_top_k_smallest')
```

</details>


### Transpose
There are 2 test cases, listed as following:
<details>
<summary>all_permutations</summary>

```python
shape = (2, 3, 4)
data = np.random.random_sample(shape).astype(np.float32)
permutations = list(itertools.permutations(np.arange(len(shape))))

for i in range(len(permutations)):
    node = onnx.helper.make_node(
        'Transpose',
        inputs=['data'],
        outputs=['transposed'],
        perm=permutations[i]
    )
    transposed = np.transpose(data, permutations[i])
    expect(node, inputs=[data], outputs=[transposed],
           name='test_transpose_all_permutations_' + str(i))
```

</details>
<details>
<summary>default</summary>

```python
shape = (2, 3, 4)
data = np.random.random_sample(shape).astype(np.float32)

node = onnx.helper.make_node(
    'Transpose',
    inputs=['data'],
    outputs=['transposed']
)

transposed = np.transpose(data)
expect(node, inputs=[data], outputs=[transposed],
       name='test_transpose_default')
```

</details>


### Trilu
There are 18 test cases, listed as following:
<details>
<summary>tril</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x'],
    outputs=['y'],
    upper=0,
)

x = np.random.randint(10, size=(4, 5)).astype(np.int64)
# X:
#  [[4, 7, 3, 7, 9],
#   [1, 2, 8, 6, 9],
#   [9, 4, 1, 8, 7],
#   [4, 3, 4, 2, 4]]
# expect result:
#  [[4, 0, 0, 0, 0],
#   [1, 2, 0, 0, 0],
#   [9, 4, 1, 0, 0],
#   [4, 3, 4, 2, 0]]
y = tril_reference_implementation(x)
expect(node, inputs=[x], outputs=[y], name='test_tril')
```

</details>
<details>
<summary>tril_neg</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x', 'k'],
    outputs=['y'],
    upper=0,
)

x = np.random.randint(10, size=(4, 5)).astype(np.int64)
k = np.array(-1).astype(np.int64)
# X:
#  [[4, 7, 3, 7, 9],
#   [1, 2, 8, 6, 9],
#   [9, 4, 1, 8, 7],
#   [4, 3, 4, 2, 4]]
# expect result:
#  [[0, 0, 0, 0, 0],
#   [1, 0, 0, 0, 0],
#   [9, 4, 0, 0, 0],
#   [4, 3, 4, 0, 0]]
y = tril_reference_implementation(x, int(k))
expect(node, inputs=[x, k], outputs=[y], name='test_tril_neg')
```

</details>
<details>
<summary>tril_one_row</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x'],
    outputs=['y'],
    upper=0,
)

x = np.random.randint(10, size=(3, 1, 5)).astype(np.int64)
# X:
# [[[6, 2, 4, 1, 6]],
#
#  [[8, 3, 8, 7, 0]],
#
#  [[2, 2, 9, 5, 9]]]
# expect result:
# [[[6, 0, 0, 0, 0]],
#
#  [[8, 0, 0, 0, 0]],
#
#  [[2, 0, 0, 0, 0]]]
y = tril_reference_implementation(x)
expect(node, inputs=[x], outputs=[y], name='test_tril_one_row_neg')
```

</details>
<details>
<summary>tril_out_neg</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x', 'k'],
    outputs=['y'],
    upper=0,
)

x = np.random.randint(10, size=(4, 5)).astype(np.int64)
k = np.array(-7).astype(np.int64)
# X:
#  [[4, 7, 3, 7, 9],
#   [1, 2, 8, 6, 9],
#   [9, 4, 1, 8, 7],
#   [4, 3, 4, 2, 4]]
# expect result:
#  [[0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0]]
y = tril_reference_implementation(x, int(k))
expect(node, inputs=[x, k], outputs=[y], name='test_tril_out_neg')
```

</details>
<details>
<summary>tril_out_pos</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x', 'k'],
    outputs=['y'],
    upper=0,
)
x = np.random.randint(10, size=(4, 5)).astype(np.int64)
k = np.array(6).astype(np.int64)
# X:
#  [[4, 7, 3, 7, 9],
#   [1, 2, 8, 6, 9],
#   [9, 4, 1, 8, 7],
#   [4, 3, 4, 2, 4]]
# expect result:
#  [[4, 7, 3, 7, 9],
#   [1, 2, 8, 6, 9],
#   [9, 4, 1, 8, 7],
#   [4, 3, 4, 2, 4]]
y = tril_reference_implementation(x, int(k))
expect(node, inputs=[x, k], outputs=[y], name='test_tril_out_pos')
```

</details>
<details>
<summary>tril_pos</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x', 'k'],
    outputs=['y'],
    upper=0,
)

x = np.random.randint(10, size=(4, 5)).astype(np.int64)
k = np.array(2).astype(np.int64)
# X:
#  [[4, 7, 3, 7, 9],
#   [1, 2, 8, 6, 9],
#   [9, 4, 1, 8, 7],
#   [4, 3, 4, 2, 4]]
# expect result:
#  [[4, 7, 3, 0, 0],
#   [1, 2, 8, 6, 0],
#   [9, 4, 1, 8, 7],
#   [4, 3, 4, 2, 4]]
y = tril_reference_implementation(x, int(k))
expect(node, inputs=[x, k], outputs=[y], name='test_tril_pos')
```

</details>
<details>
<summary>tril_square</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x'],
    outputs=['y'],
    upper=0,
)

x = np.random.randint(10, size=(2, 3, 3)).astype(np.int64)
# X:
# [[[0, 4, 3],
#   [2, 0, 9],
#   [8, 2, 5]],
#
#  [[2, 7, 2],
#   [2, 6, 0],
#   [2, 6, 5]]]
# expect result:
# [[[0, 0, 0],
#   [2, 0, 0],
#   [8, 2, 5]],
#
#  [[2, 0, 0],
#   [2, 6, 0],
#   [2, 6, 5]]]
y = tril_reference_implementation(x)
expect(node, inputs=[x], outputs=[y], name='test_tril_square')
```

</details>
<details>
<summary>tril_square_neg</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x', 'k'],
    outputs=['y'],
    upper=0,
)

x = np.random.randint(10, size=(2, 3, 3)).astype(np.int64)
k = np.array(-1).astype(np.int64)
# X:
# [[[0, 4, 3],
#   [2, 0, 9],
#   [8, 2, 5]],
#
#  [[2, 7, 2],
#   [2, 6, 0],
#   [2, 6, 5]]]
# expect result:
# [[[0, 0, 0],
#   [2, 0, 0],
#   [8, 2, 0]],
#
#  [[0, 0, 0],
#   [2, 0, 0],
#   [2, 6, 0]]]
y = tril_reference_implementation(x, int(k))
expect(node, inputs=[x, k], outputs=[y], name='test_tril_square_neg')
```

</details>
<details>
<summary>tril_zero</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x', 'k'],
    outputs=['y'],
    upper=0,
)

x = np.random.randint(10, size=(3, 0, 5)).astype(np.int64)
k = np.array(6).astype(np.int64)
# X:
# []
# expect result:
# []
y = tril_reference_implementation(x, int(k))
expect(node, inputs=[x, k], outputs=[y], name='test_tril_zero')
```

</details>
<details>
<summary>triu</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x'],
    outputs=['y'],
)

x = np.random.randint(10, size=(4, 5)).astype(np.int64)
# X:
#  [[4, 7, 3, 7, 9],
#   [1, 2, 8, 6, 9],
#   [9, 4, 0, 8, 7],
#   [4, 3, 4, 2, 4]]
# expect result:
#  [[4, 7, 3, 7, 9],
#   [0, 2, 8, 6, 9],
#   [0, 0, 0, 8, 7],
#   [0, 0, 0, 2, 4]]
y = triu_reference_implementation(x)
expect(node, inputs=[x], outputs=[y], name='test_triu')
```

</details>
<details>
<summary>triu_neg</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x', 'k'],
    outputs=['y'],
)

x = np.random.randint(10, size=(4, 5)).astype(np.int64)
k = np.array(-1).astype(np.int64)
# X:
#  [[4, 7, 3, 7, 9],
#   [1, 2, 8, 6, 9],
#   [9, 4, 0, 8, 7],
#   [4, 3, 4, 2, 4]]
# expect result:
#  [[4, 7, 3, 7, 9],
#   [1, 2, 8, 6, 9],
#   [0, 4, 0, 8, 7],
#   [0, 0, 4, 2, 4]]
y = triu_reference_implementation(x, int(k))
expect(node, inputs=[x, k], outputs=[y], name='test_triu_neg')
```

</details>
<details>
<summary>triu_one_row</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x', 'k'],
    outputs=['y'],
)

x = np.random.randint(10, size=(3, 1, 5)).astype(np.int64)
k = np.array(1).astype(np.int64)
# X:
# [[[1, 4, 9, 7, 1]],
#
#  [[9, 2, 8, 8, 4]],
#
#  [[3, 9, 7, 4, 2]]]
# expect result:
# [[[0, 4, 9, 7, 1]],
#
#  [[0, 2, 8, 8, 4]],
#
#  [[0, 9, 7, 4, 2]]]
y = triu_reference_implementation(x, int(k))
expect(node, inputs=[x, k], outputs=[y], name='test_triu_one_row')
```

</details>
<details>
<summary>triu_out_neg_out</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x', 'k'],
    outputs=['y'],
)

x = np.random.randint(10, size=(4, 5)).astype(np.int64)
k = np.array(-7).astype(np.int64)
# X:
#  [[4, 7, 3, 7, 9],
#   [1, 2, 8, 6, 9],
#   [9, 4, 0, 8, 7],
#   [4, 3, 4, 2, 4]]
# expect result:
#  [[4, 7, 3, 7, 9],
#   [1, 2, 8, 6, 9],
#   [9, 4, 0, 8, 7],
#   [4, 3, 4, 2, 4]]
y = triu_reference_implementation(x, int(k))
expect(node, inputs=[x, k], outputs=[y], name='test_triu_out_neg_out')
```

</details>
<details>
<summary>triu_out_pos</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x', 'k'],
    outputs=['y'],
)

x = np.random.randint(10, size=(4, 5)).astype(np.int64)
k = np.array(6).astype(np.int64)
# X:
#  [[4, 7, 3, 7, 9],
#   [1, 2, 8, 6, 9],
#   [9, 4, 0, 8, 7],
#   [4, 3, 4, 2, 4]]
# expect result:
#  [[0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0],
#   [0, 0, 0, 0, 0]]
y = triu_reference_implementation(x, int(k))
expect(node, inputs=[x, k], outputs=[y], name='test_triu_out_pos')
```

</details>
<details>
<summary>triu_pos</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x', 'k'],
    outputs=['y'],
)

x = np.random.randint(10, size=(4, 5)).astype(np.int64)
k = np.array(2).astype(np.int64)
# X:
#  [[4, 7, 3, 7, 9],
#   [1, 2, 8, 6, 9],
#   [9, 4, 0, 8, 7],
#   [4, 3, 4, 2, 4]]
# expect result:
#  [[0, 0, 3, 7, 9],
#   [0, 0, 0, 6, 9],
#   [0, 0, 0, 0, 7],
#   [0, 0, 0, 0, 0]]
y = triu_reference_implementation(x, int(k))
expect(node, inputs=[x, k], outputs=[y], name='test_triu_pos')
```

</details>
<details>
<summary>triu_square</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x'],
    outputs=['y'],
)

x = np.random.randint(10, size=(2, 3, 3)).astype(np.int64)
y = triu_reference_implementation(x)
# X:
# [[[4, 6, 9],
#   [7, 5, 4],
#   [8, 1, 2]],
#
#  [[1, 4, 9],
#   [9, 6, 3],
#   [8, 9, 8]]]
# expect result:
# [[[4, 6, 9],
#   [0, 5, 4],
#   [0, 0, 2]],
#
#  [[1, 4, 9],
#   [0, 6, 3],
#   [0, 0, 8]]]
expect(node, inputs=[x], outputs=[y], name='test_triu_square')
```

</details>
<details>
<summary>triu_square_neg</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x', 'k'],
    outputs=['y'],
)

x = np.random.randint(10, size=(2, 3, 3)).astype(np.int64)
k = np.array(-1).astype(np.int64)
# X:
# [[[4, 6, 9],
#   [7, 5, 4],
#   [8, 1, 2]],
#
#  [[1, 4, 9],
#   [9, 6, 3],
#   [8, 9, 8]]]
# expect result:
# [[[4, 6, 9],
#   [7, 5, 4],
#   [0, 1, 2]],
#
#  [[1, 4, 9],
#   [9, 6, 3],
#   [0, 9, 8]]]
y = triu_reference_implementation(x, int(k))
expect(node, inputs=[x, k], outputs=[y], name='test_triu_square_neg')
```

</details>
<details>
<summary>triu_zero</summary>

```python
node = onnx.helper.make_node(
    'Trilu',
    inputs=['x', 'k'],
    outputs=['y'],
)

x = np.random.randint(10, size=(0, 5)).astype(np.int64)
k = np.array(6).astype(np.int64)
# X:
# []
# expect result:
# []
y = triu_reference_implementation(x, int(k))
expect(node, inputs=[x, k], outputs=[y], name='test_triu_zero')
```

</details>


### Unique
There are 5 test cases, listed as following:
<details>
<summary>not_sorted_without_axis</summary>

```python
node_not_sorted = onnx.helper.make_node(
    'Unique',
    inputs=['X'],
    outputs=['Y', 'indices', 'inverse_indices', 'counts'],
    sorted=0
)
# numpy unique does not retain original order (it sorts the output unique values)
# https://github.com/numpy/numpy/issues/8621
# we need to recover unsorted output and indices
x = np.array([2.0, 1.0, 1.0, 3.0, 4.0, 3.0], dtype=np.float32)
y, indices, inverse_indices, counts = np.unique(x, True, True, True)

# prepare index mapping from sorted to unsorted
argsorted_indices = np.argsort(indices)
inverse_indices_map = {i: si for i, si in zip(argsorted_indices, np.arange(len(argsorted_indices)))}

indices = indices[argsorted_indices]
y = np.take(x, indices, axis=0)
inverse_indices = np.asarray([inverse_indices_map[i] for i in inverse_indices], dtype=np.int64)
counts = counts[argsorted_indices]
indices, inverse_indices, counts = specify_int64(indices, inverse_indices, counts)
# print(y)
# [2.0, 1.0, 3.0, 4.0]
# print(indices)
# [0 1 3 4]
# print(inverse_indices)
# [0, 1, 1, 2, 3, 2]
# print(counts)
# [1, 2, 2, 1]

expect(node_not_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_not_sorted_without_axis')
```

</details>
<details>
<summary>sorted_with_axis</summary>

```python
node_sorted = onnx.helper.make_node(
    'Unique',
    inputs=['X'],
    outputs=['Y', 'indices', 'inverse_indices', 'counts'],
    sorted=1,
    axis=0
)

x = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]], dtype=np.float32)
y, indices, inverse_indices, counts = np.unique(x, True, True, True, axis=0)
indices, inverse_indices, counts = specify_int64(indices, inverse_indices, counts)
# print(y)
# [[1. 0. 0.]
#  [2. 3. 4.]]
# print(indices)
# [0 2]
# print(inverse_indices)
# [0 0 1]
# print(counts)
# [2 1]

expect(node_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_sorted_with_axis')
```

</details>
<details>
<summary>sorted_with_axis_3d</summary>

```python
node_sorted = onnx.helper.make_node(
    'Unique',
    inputs=['X'],
    outputs=['Y', 'indices', 'inverse_indices', 'counts'],
    sorted=1,
    axis=1
)

x = np.array([[[1., 1.], [0., 1.], [2., 1.], [0., 1.]],
              [[1., 1.], [0., 1.], [2., 1.], [0., 1.]]], dtype=np.float32)
y, indices, inverse_indices, counts = np.unique(x, True, True, True, axis=1)
indices, inverse_indices, counts = specify_int64(indices, inverse_indices, counts)
# print(y)
# [[[0. 1.]
#  [1. 1.]
#  [2. 1.]]
# [[0. 1.]
#  [1. 1.]
#  [2. 1.]]]
# print(indices)
# [1 0 2]
# print(inverse_indices)
# [1 0 2 0]
# print(counts)
# [2 1 1]
expect(node_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_sorted_with_axis_3d')
```

</details>
<details>
<summary>sorted_with_negative_axis</summary>

```python
node_sorted = onnx.helper.make_node(
    'Unique',
    inputs=['X'],
    outputs=['Y', 'indices', 'inverse_indices', 'counts'],
    sorted=1,
    axis=-1
)

x = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 3]], dtype=np.float32)
y, indices, inverse_indices, counts = np.unique(x, True, True, True, axis=-1)
indices, inverse_indices, counts = specify_int64(indices, inverse_indices, counts)
# print(y)
# [[0. 1.]
#  [0. 1.]
#  [3. 2.]]
# print(indices)
# [1 0]
# print(inverse_indices)
# [1 0 0]
# print(counts)
# [2 1]

expect(node_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_sorted_with_negative_axis')
```

</details>
<details>
<summary>sorted_without_axis</summary>

```python
node_sorted = onnx.helper.make_node(
    'Unique',
    inputs=['X'],
    outputs=['Y', 'indices', 'inverse_indices', 'counts']
)

x = np.array([2.0, 1.0, 1.0, 3.0, 4.0, 3.0], dtype=np.float32)
y, indices, inverse_indices, counts = np.unique(x, True, True, True)
indices, inverse_indices, counts = specify_int64(indices, inverse_indices, counts)
expect(node_sorted, inputs=[x], outputs=[y, indices, inverse_indices, counts], name='test_unique_sorted_without_axis')
```

</details>


### Unsqueeze
There are 5 test cases, listed as following:
<details>
<summary>unsqueeze_negative_axes</summary>

```python
node = onnx.helper.make_node(
    'Unsqueeze',
    inputs=['x', 'axes'],
    outputs=['y'],
)
x = np.random.randn(1, 3, 1, 5).astype(np.float32)
axes = np.array([-2]).astype(np.int64)
y = np.expand_dims(x, axis=-2)
expect(node, inputs=[x, axes], outputs=[y],
       name='test_unsqueeze_negative_axes')
```

</details>
<details>
<summary>unsqueeze_one_axis</summary>

```python
x = np.random.randn(3, 4, 5).astype(np.float32)

for i in range(x.ndim):
    axes = np.array([i]).astype(np.int64)
    node = onnx.helper.make_node(
        'Unsqueeze',
        inputs=['x', 'axes'],
        outputs=['y'],
    )
    y = np.expand_dims(x, axis=i)

    expect(node, inputs=[x, axes], outputs=[y],
           name='test_unsqueeze_axis_' + str(i))
```

</details>
<details>
<summary>unsqueeze_three_axes</summary>

```python
x = np.random.randn(3, 4, 5).astype(np.float32)
axes = np.array([2, 4, 5]).astype(np.int64)

node = onnx.helper.make_node(
    'Unsqueeze',
    inputs=['x', 'axes'],
    outputs=['y'],
)
y = np.expand_dims(x, axis=2)
y = np.expand_dims(y, axis=4)
y = np.expand_dims(y, axis=5)

expect(node, inputs=[x, axes], outputs=[y],
       name='test_unsqueeze_three_axes')
```

</details>
<details>
<summary>unsqueeze_two_axes</summary>

```python
x = np.random.randn(3, 4, 5).astype(np.float32)
axes = np.array([1, 4]).astype(np.int64)

node = onnx.helper.make_node(
    'Unsqueeze',
    inputs=['x', 'axes'],
    outputs=['y'],
)
y = np.expand_dims(x, axis=1)
y = np.expand_dims(y, axis=4)

expect(node, inputs=[x, axes], outputs=[y],
       name='test_unsqueeze_two_axes')
```

</details>
<details>
<summary>unsqueeze_unsorted_axes</summary>

```python
x = np.random.randn(3, 4, 5).astype(np.float32)
axes = np.array([5, 4, 2]).astype(np.int64)

node = onnx.helper.make_node(
    'Unsqueeze',
    inputs=['x', 'axes'],
    outputs=['y'],
)
y = np.expand_dims(x, axis=2)
y = np.expand_dims(y, axis=4)
y = np.expand_dims(y, axis=5)

expect(node, inputs=[x, axes], outputs=[y],
       name='test_unsqueeze_unsorted_axes')
```

</details>


### Upsample
There are 1 test cases, listed as following:
<details>
<summary>nearest</summary>

```python
node = onnx.helper.make_node(
    'Upsample',
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
       name='test_upsample_nearest', opset_imports=[helper.make_opsetid("", 9)])
```

</details>


### Where
There are 2 test cases, listed as following:
<details>
<summary>long</summary>

```python
node = onnx.helper.make_node(
    'Where',
    inputs=['condition', 'x', 'y'],
    outputs=['z'],
)

condition = np.array([[1, 0], [1, 1]], dtype=bool)
x = np.array([[1, 2], [3, 4]], dtype=np.int64)
y = np.array([[9, 8], [7, 6]], dtype=np.int64)
z = np.where(condition, x, y)  # expected output [[1, 8], [3, 4]]
expect(node, inputs=[condition, x, y], outputs=[z],
       name='test_where_long_example')
```

</details>
<details>
<summary>where</summary>

```python
node = onnx.helper.make_node(
    'Where',
    inputs=['condition', 'x', 'y'],
    outputs=['z'],
)

condition = np.array([[1, 0], [1, 1]], dtype=bool)
x = np.array([[1, 2], [3, 4]], dtype=np.float32)
y = np.array([[9, 8], [7, 6]], dtype=np.float32)
z = np.where(condition, x, y)  # expected output [[1, 8], [3, 4]]
expect(node, inputs=[condition, x, y], outputs=[z],
       name='test_where_example')
```

</details>


### Xor
There are 2 test cases, listed as following:
<details>
<summary>xor</summary>

```python
node = onnx.helper.make_node(
    'Xor',
    inputs=['x', 'y'],
    outputs=['xor'],
)

# 2d
x = (np.random.randn(3, 4) > 0).astype(bool)
y = (np.random.randn(3, 4) > 0).astype(bool)
z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor2d')

# 3d
x = (np.random.randn(3, 4, 5) > 0).astype(bool)
y = (np.random.randn(3, 4, 5) > 0).astype(bool)
z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor3d')

# 4d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
y = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor4d')
```

</details>
<details>
<summary>xor_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Xor',
    inputs=['x', 'y'],
    outputs=['xor'],
)

# 3d vs 1d
x = (np.random.randn(3, 4, 5) > 0).astype(bool)
y = (np.random.randn(5) > 0).astype(bool)
z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor_bcast3v1d')

# 3d vs 2d
x = (np.random.randn(3, 4, 5) > 0).astype(bool)
y = (np.random.randn(4, 5) > 0).astype(bool)
z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor_bcast3v2d')

# 4d vs 2d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
y = (np.random.randn(5, 6) > 0).astype(bool)
z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor_bcast4v2d')

# 4d vs 3d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(bool)
y = (np.random.randn(4, 5, 6) > 0).astype(bool)
z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor_bcast4v3d')

# 4d vs 4d
x = (np.random.randn(1, 4, 1, 6) > 0).astype(bool)
y = (np.random.randn(3, 1, 5, 6) > 0).astype(bool)
z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor_bcast4v4d')
```

</details>


<br/>

## &#x1F494;No Cover Common Operators
### ConcatFromSequence (call for test cases)


### GlobalLpPool (call for test cases)


### GreaterOrEqual (call for test cases)


### LessOrEqual (call for test cases)


### LpNormalization (call for test cases)


### LpPool (call for test cases)


### MaxRoiPool (call for test cases)


### Multinomial (random generator operator)


### Optional (call for test cases)


### OptionalGetElement (call for test cases)


### RandomNormal (random generator operator)


### RandomNormalLike (random generator operator)


### RandomUniform (random generator operator)


### RandomUniformLike (random generator operator)


### SequenceAt (call for test cases)


### SequenceConstruct (call for test cases)


### SequenceEmpty (call for test cases)


### SequenceErase (call for test cases)


### SequenceLength (call for test cases)


### SplitToSequence (call for test cases)


<br/>

## &#x1F49A;Covered Experimental Operators
<br/>

## &#x1F494;No Cover Experimental Operators
<br/>

# Model Test Coverage
## bvlc_alexnet

bvlc_alexnet has 24 nodes. Of these, 24 are covered by node tests (100.0%)


<details>
<summary>nodes</summary>

<details>
<summary>Conv: 4 out of 6 attributes covered</summary>

auto_pad: 0
dilations: 0
group: 1
kernel_shape: 3
pads: 3
strides: 2
</details>
<details>
<summary>Dropout: 1 out of 1 attributes covered</summary>

seed: 0
</details>
<details>
<summary>Gemm: 1 out of 4 attributes covered</summary>

alpha: 0
beta: 0
transA: 0
transB: 1
</details>
<details>
<summary>LRN: 4 out of 4 attributes covered</summary>

alpha: 1
beta: 1
bias: 1
size: 1
</details>
<details>
<summary>MaxPool: 3 out of 7 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
dilations: 0
kernel_shape: 1
pads: 2
storage_order: 0
strides: 1
</details>
</details>


## densenet121

densenet121 has 910 nodes. Of these, 910 are covered by node tests (100.0%)


<details>
<summary>nodes</summary>

<details>
<summary>AveragePool: 3 out of 6 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
count_include_pad: 0
kernel_shape: 1
pads: 1
strides: 1
</details>
<details>
<summary>BatchNormalization: 1 out of 3 attributes covered</summary>

epsilon: 1
momentum: 0
training_mode: 0
</details>
<details>
<summary>Concat: 1 out of 1 attributes covered</summary>

axis: 1
</details>
<details>
<summary>Conv: 4 out of 6 attributes covered</summary>

auto_pad: 0
dilations: 0
group: 1
kernel_shape: 5
pads: 4
strides: 3
</details>
<details>
<summary>Dropout: 1 out of 1 attributes covered</summary>

seed: 0
</details>
<details>
<summary>Gemm: 1 out of 4 attributes covered</summary>

alpha: 0
beta: 0
transA: 0
transB: 1
</details>
<details>
<summary>LRN: 4 out of 4 attributes covered</summary>

alpha: 1
beta: 1
bias: 1
size: 1
</details>
<details>
<summary>MaxPool: 3 out of 7 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
dilations: 0
kernel_shape: 1
pads: 3
storage_order: 0
strides: 1
</details>
<details>
<summary>Unsqueeze: 1 out of 0 attributes covered</summary>

</details>
</details>


## inception_v1

inception_v1 has 144 nodes. Of these, 144 are covered by node tests (100.0%)


<details>
<summary>nodes</summary>

<details>
<summary>AveragePool: 3 out of 6 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
count_include_pad: 0
kernel_shape: 2
pads: 2
strides: 2
</details>
<details>
<summary>BatchNormalization: 1 out of 3 attributes covered</summary>

epsilon: 1
momentum: 0
training_mode: 0
</details>
<details>
<summary>Concat: 1 out of 1 attributes covered</summary>

axis: 1
</details>
<details>
<summary>Conv: 4 out of 6 attributes covered</summary>

auto_pad: 0
dilations: 0
group: 1
kernel_shape: 5
pads: 4
strides: 3
</details>
<details>
<summary>Dropout: 1 out of 1 attributes covered</summary>

seed: 0
</details>
<details>
<summary>Gemm: 1 out of 4 attributes covered</summary>

alpha: 0
beta: 0
transA: 0
transB: 1
</details>
<details>
<summary>LRN: 4 out of 4 attributes covered</summary>

alpha: 1
beta: 1
bias: 1
size: 1
</details>
<details>
<summary>MaxPool: 3 out of 7 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
dilations: 0
kernel_shape: 1
pads: 3
storage_order: 0
strides: 2
</details>
<details>
<summary>Unsqueeze: 1 out of 0 attributes covered</summary>

</details>
</details>


## inception_v2

inception_v2 has 509 nodes. Of these, 509 are covered by node tests (100.0%)


<details>
<summary>nodes</summary>

<details>
<summary>AveragePool: 3 out of 6 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
count_include_pad: 0
kernel_shape: 3
pads: 3
strides: 2
</details>
<details>
<summary>BatchNormalization: 1 out of 3 attributes covered</summary>

epsilon: 1
momentum: 0
training_mode: 0
</details>
<details>
<summary>Concat: 1 out of 1 attributes covered</summary>

axis: 1
</details>
<details>
<summary>Conv: 4 out of 6 attributes covered</summary>

auto_pad: 0
dilations: 0
group: 1
kernel_shape: 5
pads: 4
strides: 3
</details>
<details>
<summary>Dropout: 1 out of 1 attributes covered</summary>

seed: 0
</details>
<details>
<summary>Gemm: 1 out of 4 attributes covered</summary>

alpha: 0
beta: 0
transA: 0
transB: 1
</details>
<details>
<summary>LRN: 4 out of 4 attributes covered</summary>

alpha: 1
beta: 1
bias: 1
size: 1
</details>
<details>
<summary>MaxPool: 3 out of 7 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
dilations: 0
kernel_shape: 1
pads: 3
storage_order: 0
strides: 2
</details>
<details>
<summary>Unsqueeze: 1 out of 0 attributes covered</summary>

</details>
</details>


## resnet50

resnet50 has 176 nodes. Of these, 176 are covered by node tests (100.0%)


<details>
<summary>nodes</summary>

<details>
<summary>AveragePool: 3 out of 6 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
count_include_pad: 0
kernel_shape: 3
pads: 3
strides: 2
</details>
<details>
<summary>BatchNormalization: 1 out of 3 attributes covered</summary>

epsilon: 2
momentum: 0
training_mode: 0
</details>
<details>
<summary>Concat: 1 out of 1 attributes covered</summary>

axis: 1
</details>
<details>
<summary>Conv: 4 out of 6 attributes covered</summary>

auto_pad: 0
dilations: 0
group: 1
kernel_shape: 5
pads: 4
strides: 3
</details>
<details>
<summary>Dropout: 1 out of 1 attributes covered</summary>

seed: 0
</details>
<details>
<summary>Gemm: 1 out of 4 attributes covered</summary>

alpha: 0
beta: 0
transA: 0
transB: 1
</details>
<details>
<summary>LRN: 4 out of 4 attributes covered</summary>

alpha: 1
beta: 1
bias: 1
size: 1
</details>
<details>
<summary>MaxPool: 3 out of 7 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
dilations: 0
kernel_shape: 1
pads: 3
storage_order: 0
strides: 2
</details>
<details>
<summary>Unsqueeze: 1 out of 0 attributes covered</summary>

</details>
</details>


## shufflenet

shufflenet has 203 nodes. Of these, 203 are covered by node tests (100.0%)


<details>
<summary>nodes</summary>

<details>
<summary>AveragePool: 3 out of 6 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
count_include_pad: 0
kernel_shape: 3
pads: 3
strides: 2
</details>
<details>
<summary>BatchNormalization: 1 out of 3 attributes covered</summary>

epsilon: 2
momentum: 0
training_mode: 0
</details>
<details>
<summary>Concat: 1 out of 1 attributes covered</summary>

axis: 1
</details>
<details>
<summary>Conv: 4 out of 6 attributes covered</summary>

auto_pad: 0
dilations: 0
group: 6
kernel_shape: 5
pads: 4
strides: 3
</details>
<details>
<summary>Dropout: 1 out of 1 attributes covered</summary>

seed: 0
</details>
<details>
<summary>Gemm: 1 out of 4 attributes covered</summary>

alpha: 0
beta: 0
transA: 0
transB: 1
</details>
<details>
<summary>LRN: 4 out of 4 attributes covered</summary>

alpha: 1
beta: 1
bias: 1
size: 1
</details>
<details>
<summary>MaxPool: 3 out of 7 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
dilations: 0
kernel_shape: 1
pads: 3
storage_order: 0
strides: 2
</details>
<details>
<summary>Transpose: 1 out of 1 attributes covered</summary>

perm: 1
</details>
<details>
<summary>Unsqueeze: 1 out of 0 attributes covered</summary>

</details>
</details>


## squeezenet_old

squeezenet_old has 66 nodes. Of these, 66 are covered by node tests (100.0%)


<details>
<summary>nodes</summary>

<details>
<summary>AveragePool: 3 out of 6 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
count_include_pad: 0
kernel_shape: 3
pads: 3
strides: 2
</details>
<details>
<summary>BatchNormalization: 1 out of 3 attributes covered</summary>

epsilon: 2
momentum: 0
training_mode: 0
</details>
<details>
<summary>Concat: 1 out of 1 attributes covered</summary>

axis: 1
</details>
<details>
<summary>Conv: 4 out of 6 attributes covered</summary>

auto_pad: 0
dilations: 0
group: 6
kernel_shape: 5
pads: 4
strides: 3
</details>
<details>
<summary>Dropout: 1 out of 1 attributes covered</summary>

seed: 0
</details>
<details>
<summary>Gemm: 1 out of 4 attributes covered</summary>

alpha: 0
beta: 0
transA: 0
transB: 1
</details>
<details>
<summary>LRN: 4 out of 4 attributes covered</summary>

alpha: 1
beta: 1
bias: 1
size: 1
</details>
<details>
<summary>MaxPool: 3 out of 7 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
dilations: 0
kernel_shape: 1
pads: 3
storage_order: 0
strides: 2
</details>
<details>
<summary>Transpose: 1 out of 1 attributes covered</summary>

perm: 1
</details>
<details>
<summary>Unsqueeze: 1 out of 0 attributes covered</summary>

</details>
</details>


## vgg19

vgg19 has 46 nodes. Of these, 46 are covered by node tests (100.0%)


<details>
<summary>nodes</summary>

<details>
<summary>AveragePool: 3 out of 6 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
count_include_pad: 0
kernel_shape: 3
pads: 3
strides: 2
</details>
<details>
<summary>BatchNormalization: 1 out of 3 attributes covered</summary>

epsilon: 2
momentum: 0
training_mode: 0
</details>
<details>
<summary>Concat: 1 out of 1 attributes covered</summary>

axis: 1
</details>
<details>
<summary>Conv: 4 out of 6 attributes covered</summary>

auto_pad: 0
dilations: 0
group: 6
kernel_shape: 5
pads: 4
strides: 3
</details>
<details>
<summary>Dropout: 1 out of 1 attributes covered</summary>

seed: 0
</details>
<details>
<summary>Gemm: 1 out of 4 attributes covered</summary>

alpha: 0
beta: 0
transA: 0
transB: 1
</details>
<details>
<summary>LRN: 4 out of 4 attributes covered</summary>

alpha: 1
beta: 1
bias: 1
size: 1
</details>
<details>
<summary>MaxPool: 3 out of 7 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
dilations: 0
kernel_shape: 2
pads: 3
storage_order: 0
strides: 2
</details>
<details>
<summary>Transpose: 1 out of 1 attributes covered</summary>

perm: 1
</details>
<details>
<summary>Unsqueeze: 1 out of 0 attributes covered</summary>

</details>
</details>


## zfnet512

zfnet512 has 22 nodes. Of these, 22 are covered by node tests (100.0%)


<details>
<summary>nodes</summary>

<details>
<summary>AveragePool: 3 out of 6 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
count_include_pad: 0
kernel_shape: 3
pads: 3
strides: 2
</details>
<details>
<summary>BatchNormalization: 1 out of 3 attributes covered</summary>

epsilon: 2
momentum: 0
training_mode: 0
</details>
<details>
<summary>Concat: 1 out of 1 attributes covered</summary>

axis: 1
</details>
<details>
<summary>Conv: 4 out of 6 attributes covered</summary>

auto_pad: 0
dilations: 0
group: 6
kernel_shape: 5
pads: 4
strides: 3
</details>
<details>
<summary>Dropout: 1 out of 1 attributes covered</summary>

seed: 0
</details>
<details>
<summary>Gemm: 1 out of 4 attributes covered</summary>

alpha: 0
beta: 0
transA: 0
transB: 1
</details>
<details>
<summary>LRN: 4 out of 4 attributes covered</summary>

alpha: 2
beta: 1
bias: 2
size: 1
</details>
<details>
<summary>MaxPool: 3 out of 7 attributes covered</summary>

auto_pad: 0
ceil_mode: 0
dilations: 0
kernel_shape: 2
pads: 3
storage_order: 0
strides: 2
</details>
<details>
<summary>Transpose: 1 out of 1 attributes covered</summary>

perm: 1
</details>
<details>
<summary>Unsqueeze: 1 out of 0 attributes covered</summary>

</details>
</details>


# Overall Test Coverage
## To be filled.
