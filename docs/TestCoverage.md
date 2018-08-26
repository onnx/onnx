# Test Coverage Report (ONNX Core Operators)
## Outlines
* [Node Test Coverage](#node-test-coverage)
* [Model Test Coverage](#model-test-coverage)
* [Overall Test Coverage](#overall-test-coverage)
# Node Test Coverage
## Summary
Node tests have covered 55/87 (63.22%, 4 generators excluded) common operators.

Node tests have covered 1/18 (5.56%, 0 generators excluded) experimental operators.

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
y = np.abs(x)

expect(node, inputs=[x], outputs=[y],
       name='test_abs')
```

</details>


### Add
There are 2 test cases, listed as following:
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
    broadcast=1,
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(5).astype(np.float32)
expect(node, inputs=[x, y], outputs=[x + y],
       name='test_add_bcast')
```

</details>


### And
There are 3 test cases, listed as following:
<details>
<summary>and</summary>

```python
node = onnx.helper.make_node(
    'And',
    inputs=['x', 'y'],
    outputs=['and'],
)

# 2d
x = (np.random.randn(3, 4) > 0).astype(np.bool)
y = (np.random.randn(3, 4) > 0).astype(np.bool)
z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_and2d')

# 3d
x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
y = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_and3d')

# 4d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
y = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_and4d')
```

</details>
<details>
<summary>and_axis</summary>

```python
x = (np.random.randn(5, 5, 5, 5) > 0).astype(np.bool)
y = (np.random.randn(5) > 0).astype(np.bool)

node = onnx.helper.make_node(
    'And',
    inputs=['x', 'y'],
    outputs=['and'],
    broadcast=1,
    axis=0,
)

z = np.logical_and(x, y[:, np.newaxis, np.newaxis, np.newaxis])
expect(node, inputs=[x, y], outputs=[z],
       name='test_and_axis0')

node = onnx.helper.make_node(
    'And',
    inputs=['x', 'y'],
    outputs=['and'],
    broadcast=1,
    axis=1,
)

z = np.logical_and(x, y[:, np.newaxis, np.newaxis])
expect(node, inputs=[x, y], outputs=[z],
       name='test_and_axis1')

node = onnx.helper.make_node(
    'And',
    inputs=['x', 'y'],
    outputs=['and'],
    broadcast=1,
    axis=2,
)

z = np.logical_and(x, y[:, np.newaxis])
expect(node, inputs=[x, y], outputs=[z],
       name='test_and_axis2')

node = onnx.helper.make_node(
    'And',
    inputs=['x', 'y'],
    outputs=['and'],
    broadcast=1,
    axis=3,
)

z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_and_axis3')
```

</details>
<details>
<summary>and_broadcast</summary>

```python
node = onnx.helper.make_node(
    'And',
    inputs=['x', 'y'],
    outputs=['and'],
    broadcast=1,
)

#3d vs 1d
x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
y = (np.random.randn(5) > 0).astype(np.bool)
z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_bcast3v1d')

#3d vs 2d
x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
y = (np.random.randn(4, 5) > 0).astype(np.bool)
z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_bcast3v2d')

#4d vs 2d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
y = (np.random.randn(5, 6) > 0).astype(np.bool)
z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_bcast4v2d')

#4d vs 3d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
y = (np.random.randn(4, 5, 6) > 0).astype(np.bool)
z = np.logical_and(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_bcast4v3d')
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
]   

for case in test_cases:
    from_type = case[0]
    to_type = case[1]
    input = np.random.random_sample(shape).astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, from_type)])
    node = onnx.helper.make_node(
        'Cast',
        inputs=['input'],
        outputs=['output'],
        to=to_type
    )
    output = input.astype(TENSOR_TYPE_TO_NP_TYPE[getattr(TensorProto, to_type)])
    expect(node, inputs=[input], outputs=[output], name='test_cast_' + from_type + '_to_' + to_type)
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
y = np.ceil(x) #expected output [-1., 2.]
expect(node, inputs=[x], outputs=[y],
       name='test_ceil_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.ceil(x)
expect(node, inputs=[x], outputs=[y],
       name='test_ceil')
```

</details>


### Clip
There are 2 test cases, listed as following:
<details>
<summary>clip</summary>

```python
node = onnx.helper.make_node(
    'Clip',
    inputs=['x'],
    outputs=['y'],
    min=-1.0,
    max=1.0
)

x = np.array([-2, 0, 2]).astype(np.float32)
y = np.clip(x, -1, 1) #expected output [-1., 0., 1.]
expect(node, inputs=[x], outputs=[y],
       name='test_clip_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, -1.0, 1.0)
expect(node, inputs=[x], outputs=[y],
       name='test_clip')
```

</details>
<details>
<summary>clip_default</summary>

```python
node = onnx.helper.make_node(
    'Clip',
    inputs=['x'],
    outputs=['y'],
    min=0.0
)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, 0.0, np.inf)
expect(node, inputs=[x], outputs=[y],
       name='test_clip_default_min')

node = onnx.helper.make_node(
    'Clip',
    inputs=['x'],
    outputs=['y'],
    max=0.0
)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, -np.inf, 0.0)
expect(node, inputs=[x], outputs=[y],
       name='test_clip_default_max')
```

</details>


### Concat
There are 1 test cases, listed as following:
<details>
<summary>concat</summary>

```python
test_cases = {
    '1d': ([1, 2],
           [3, 4]),
    '2d': ([[1, 2], [3, 4]],
           [[5, 6], [7, 8]]),
    '3d':([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
           [[[9, 10], [11, 12]], [[13, 14], [15, 16]]])
    }

for test_case, values in test_cases.items():
    values = [np.asarray(v) for v in values]
    for i in range(len(values[0].shape)):
        in_args = ['value' + str(k) for k in range(len(values))]
        node = onnx.helper.make_node(
            'Concat',
            inputs=[s for s in in_args],
            outputs=['output'],
            axis=i
        )
        output = np.concatenate(values,i)
        expect(node, inputs=[v for v in values], outputs=[output],
        name='test_concat_' + test_case + '_axis_' + str(i))
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


### Conv
There are 2 test cases, listed as following:
<details>
<summary>conv</summary>

```python

x = np.array([[[[  0.,   1.,   2.,   3.,   4.], # (1, 1, 5, 5) input tensor
                [  5.,   6.,   7.,   8.,   9.],
                [ 10.,  11.,  12.,  13.,  14.],
                [ 15.,  16.,  17.,  18.,  19.],
                [ 20.,  21.,  22.,  23.,  24.]]]]).astype(np.float32)
W = np.array([[[[ 1.,  1.,  1.], # (1, 1, 3, 3) tensor for convolution weights
                [ 1.,  1.,  1.],
                [ 1.,  1.,  1.]]]]).astype(np.float32)

# Convolution with padding
node_with_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    pads=[1, 1, 1, 1], # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
)
y_with_padding = np.array([[[[  12.,   21.,   27.,   33.,   24.], # (1, 1, 5, 5) output tensor
                             [  33.,   54.,   63.,   72.,   51.],
                             [  63.,   99.,  108.,  117.,   81.],
                             [  93.,  144.,  153.,  162.,  111.],
                             [  72.,  111.,  117.,  123.,   84.]]]]).astype(np.float32)
expect(node_with_padding, inputs=[x, W], outputs=[y_with_padding],
   name='test_basic_conv_with_padding')

# Convolution without padding
node_without_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    pads=[0, 0, 0, 0], # Default values for other attributes: strides=[1, 1], dilations=[1, 1], groups=1
)
y_without_padding = np.array([[[[  54.,   63.,   72.], # (1, 1, 3, 3) output tensor
                                [  99.,  108.,  117.],
                                [ 144.,  153.,  162.]]]]).astype(np.float32)
expect(node_without_padding, inputs=[x, W], outputs=[y_without_padding],
   name='test_basic_conv_without_padding')
```

</details>
<details>
<summary>conv_with_strides</summary>

```python

x = np.array([[[[  0.,   1.,   2.,   3.,   4.],  # (1, 1, 7, 5) input tensor
                [  5.,   6.,   7.,   8.,   9.],
                [ 10.,  11.,  12.,  13.,  14.],
                [ 15.,  16.,  17.,  18.,  19.],
                [ 20.,  21.,  22.,  23.,  24.],
                [ 25.,  26.,  27.,  28.,  29.],
                [ 30.,  31.,  32.,  33.,  34.]]]]).astype(np.float32)
W = np.array([[[[ 1.,  1.,  1.],  # (1, 1, 3, 3) tensor for convolution weights
                [ 1.,  1.,  1.],
                [ 1.,  1.,  1.]]]]).astype(np.float32)

# Convolution with strides=2 and padding
node_with_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    pads=[1, 1, 1, 1],
    strides=[2, 2], # Default values for other attributes: dilations=[1, 1], groups=1
)
y_with_padding = np.array([[[[  12.,   27.,   24.], # (1, 1, 4, 3) output tensor
                             [  63.,  108.,   81.],
                             [ 123.,  198.,  141.],
                             [ 112.,  177.,  124.]]]]).astype(np.float32)
expect(node_with_padding, inputs=[x, W], outputs=[y_with_padding],
   name='test_conv_with_strides_padding')

# Convolution with strides=2 and no padding
node_without_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    pads=[0, 0, 0, 0],
    strides=[2, 2], # Default values for other attributes: dilations=[1, 1], groups=1
)
y_without_padding = np.array([[[[  54.,   72.], # (1, 1, 3, 2) output tensor
                                [ 144., 162.],
                                [ 234.,  252.]]]]).astype(np.float32)
expect(node_without_padding, inputs=[x, W], outputs=[y_without_padding],
   name='test_conv_with_strides_no_padding')

# Convolution with strides=2 and padding only along one dimension (the H dimension in NxCxHxW tensor)
node_with_asymmetric_padding = onnx.helper.make_node(
    'Conv',
    inputs=['x', 'W'],
    outputs=['y'],
    kernel_shape=[3, 3],
    pads=[1, 0, 1, 0],
    strides=[2, 2], # Default values for other attributes: dilations=[1, 1], groups=1
)
y_with_asymmetric_padding = np.array([[[[  21.,   33.], # (1, 1, 4, 2) output tensor
                                        [  99.,  117.],
                                        [ 189.,  207.],
                                        [ 171.,  183.]]]]).astype(np.float32)
expect(node_with_asymmetric_padding, inputs=[x, W], outputs=[y_with_asymmetric_padding],
   name='test_conv_with_strides_and_asymmetric_padding')
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
z = x / y #expected output [3., 2.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_div_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.rand(3, 4, 5).astype(np.float32) + 1.0
z = x / y
expect(node, inputs=[x, y], outputs=[z],
       name='test_div')
```

</details>
<details>
<summary>div_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Div',
    inputs=['x', 'y'],
    outputs=['z'],
    broadcast=1,
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.rand(5).astype(np.float32) + 1.0
z = x / y
expect(node, inputs=[x, y], outputs=[z],
       name='test_div_bcast')
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
#expected output [-1.2642411, 0., 1.]
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
    broadcast=1,
)

x = (np.random.randn(3, 4, 5) * 10).astype(np.int32)
y = (np.random.randn(5) * 10).astype(np.int32)
z = np.equal(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_equal_bcast')
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
y = np.exp(x) #expected output [0.36787945, 1., 2.71828175]
expect(node, inputs=[x], outputs=[y],
       name='test_exp_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.exp(x)
expect(node, inputs=[x], outputs=[y],
       name='test_exp')
```

</details>


### Flatten
There are 2 test cases, listed as following:
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
    b= np.reshape(a, new_shape)
    expect(node, inputs=[a], outputs=[b],
       name='test_flatten_axis' + str(i))
```

</details>
<details>
<summary>flatten_with_default_axis</summary>

```python
node = onnx.helper.make_node(
    'Flatten',
    inputs=['a'],
    outputs=['b'], # Default value for axis: axis=1
)

shape = (5, 4, 3, 2)
a = np.random.random_sample(shape).astype(np.float32)
new_shape = (5, 24)
b= np.reshape(a, new_shape)
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
y = np.floor(x) #expected output [-2., 1., 2.]
expect(node, inputs=[x], outputs=[y],
       name='test_floor_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.floor(x)
expect(node, inputs=[x], outputs=[y],
       name='test_floor')
```

</details>


### Gather
There are 2 test cases, listed as following:
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

expect(node, inputs=[data, indices], outputs=[y],
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

expect(node, inputs=[data, indices], outputs=[y],
       name='test_gather_1')
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
spatial_shape = np.ndim(x) - 2
y = np.average(x, axis=tuple(range(spatial_shape, spatial_shape + 2)))
for _ in range(spatial_shape):
    y = np.expand_dims(y, -1)
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
spatial_shape = np.ndim(x) - 2
y = np.max(x, axis=tuple(range(spatial_shape, spatial_shape + 2)))
for _ in range(spatial_shape):
    y = np.expand_dims(y, -1)
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


### Greater
There are 2 test cases, listed as following:
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
<summary>greater_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Greater',
    inputs=['x', 'y'],
    outputs=['greater'],
    broadcast=1,
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(5).astype(np.float32)
z = np.greater(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_greater_bcast')
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
y = np.clip(x * 0.5 + 0.6, 0, 1) #expected output [0.1, 0.6, 1.]
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

x = np.array([[3, 0, 1, 2], [2, 5, 1, 0], [0, 1, 3, 2], [0, 1, 2, 3]]).astype(np.float32)
y = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).astype(np.float32)
expect(node, inputs=[x], outputs=[y],
       name='test_hardmax_example')

# For multiple occurrances of the maximal values, the first occurrence is selected for one-hot output
x = np.array([[3, 3, 3, 1]]).astype(np.float32)
y = np.array([[1, 0, 0, 0]]).astype(np.float32)
expect(node, inputs=[x], outputs=[y],
       name='test_hardmax_one_hot')
```

</details>
<details>
<summary>hardmax_axis</summary>

```python
def hardmax_2d(x):
    return np.eye(x.shape[1])[np.argmax(x,axis=1)]

x = np.random.randn(3, 4, 5).astype(np.float32)
node = onnx.helper.make_node(
    'Hardmax',
    inputs=['x'],
    outputs=['y'],
    axis=0,
)
y = hardmax_2d(x.reshape(1, 60)).reshape(3, 4, 5)
expect(node, inputs=[x], outputs=[y],
       name='test_hardmax_axis_0')

node = onnx.helper.make_node(
    'Hardmax',
    inputs=['x'],
    outputs=['y'],
    axis=1,
)
y = hardmax_2d(x.reshape(3, 20)).reshape(3, 4, 5)
expect(node, inputs=[x], outputs=[y],
       name='test_hardmax_axis_1')

# default axis is 1
node = onnx.helper.make_node(
    'Hardmax',
    inputs=['x'],
    outputs=['y'],
)
expect(node, inputs=[x], outputs=[y],
       name='test_hardmax_default_axis')

node = onnx.helper.make_node(
    'Hardmax',
    inputs=['x'],
    outputs=['y'],
    axis=2,
)
y = hardmax_2d(x.reshape(12, 5)).reshape(3, 4, 5)
expect(node, inputs=[x], outputs=[y],
       name='test_hardmax_axis_2')
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
#expected output [-0.1, 0., 1.]
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
There are 2 test cases, listed as following:
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
<summary>less_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Less',
    inputs=['x', 'y'],
    outputs=['less'],
    broadcast=1,
)

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(5).astype(np.float32)
z = np.less(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_less_bcast')
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
y = np.log(x) #expected output [0., 2.30258512]
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
y = x - np.log(np.sum(np.exp(x), axis=1)) #expected output [[-2.40760589, -1.40760589, -0.40760589]]
expect(node, inputs=[x], outputs=[y],
       name='test_logsoftmax_example_1')
```

</details>
<details>
<summary>logsoftmax_axis</summary>

```python
def logsoftmax_2d(x):
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return x - max_x - np.log(np.sum(exp_x, axis=1).reshape((-1, 1)))

x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
#expected output [[-3.4401896, -2.4401896, -1.44018972, -0.44018969],
#                 [-3.4401896, -2.4401896, -1.44018972, -0.44018969]]
y = logsoftmax_2d(x)

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
y = logsoftmax_2d(x.reshape(1, 60)).reshape(3, 4, 5)
expect(node, inputs=[x], outputs=[y],
       name='test_logsoftmax_axis_0')

node = onnx.helper.make_node(
    'LogSoftmax',
    inputs=['x'],
    outputs=['y'],
    axis=1,
)
y = logsoftmax_2d(x.reshape(3, 20)).reshape(3, 4, 5)
expect(node, inputs=[x], outputs=[y],
       name='test_logsoftmax_axis_1')

# default axis is 1
node = onnx.helper.make_node(
    'LogSoftmax',
    inputs=['x'],
    outputs=['y'],
)
expect(node, inputs=[x], outputs=[y],
       name='test_logsoftmax_default_axis')

node = onnx.helper.make_node(
    'LogSoftmax',
    inputs=['x'],
    outputs=['y'],
    axis=2,
)
y = logsoftmax_2d(x.reshape(12, 5)).reshape(3, 4, 5)
expect(node, inputs=[x], outputs=[y],
       name='test_logsoftmax_axis_2')
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


### Max
There are 1 test cases, listed as following:
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


### Min
There are 1 test cases, listed as following:
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
z = x * y #expected output [4., 10., 18.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_mul_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(3, 4, 5).astype(np.float32)
z = x * y
expect(node, inputs=[x, y], outputs=[z],
       name='test_mul')
```

</details>
<details>
<summary>mul_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Mul',
    inputs=['x', 'y'],
    outputs=['z'],
    broadcast=1,
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
y = np.negative(x) #expected output [4., -2.],
expect(node, inputs=[x], outputs=[y],
       name='test_neg_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.negative(x)
expect(node, inputs=[x], outputs=[y],
       name='test_neg')
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
x = (np.random.randn(3, 4) > 0).astype(np.bool)
expect(node, inputs=[x], outputs=[np.logical_not(x)],
       name='test_not_2d')

# 3d
x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
expect(node, inputs=[x], outputs=[np.logical_not(x)],
       name='test_not_3d')

# 4d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
expect(node, inputs=[x], outputs=[np.logical_not(x)],
       name='test_not_4d')
```

</details>


### Or
There are 3 test cases, listed as following:
<details>
<summary>or</summary>

```python
node = onnx.helper.make_node(
    'Or',
    inputs=['x', 'y'],
    outputs=['or'],
)

# 2d
x = (np.random.randn(3, 4) > 0).astype(np.bool)
y = (np.random.randn(3, 4) > 0).astype(np.bool)
z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or2d')

# 3d
x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
y = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or3d')

# 4d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
y = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or4d')
```

</details>
<details>
<summary>or_axis</summary>

```python
x = (np.random.randn(5, 5, 5, 5) > 0).astype(np.bool)
y = (np.random.randn(5) > 0).astype(np.bool)

node = onnx.helper.make_node(
    'Or',
    inputs=['x', 'y'],
    outputs=['or'],
    broadcast=1,
    axis=0,
)

z = np.logical_or(x, y[:, np.newaxis, np.newaxis, np.newaxis])
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_axis0')

node = onnx.helper.make_node(
    'Or',
    inputs=['x', 'y'],
    outputs=['or'],
    broadcast=1,
    axis=1,
)

z = np.logical_or(x, y[:, np.newaxis, np.newaxis])
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_axis1')

node = onnx.helper.make_node(
    'Or',
    inputs=['x', 'y'],
    outputs=['or'],
    broadcast=1,
    axis=2,
)

z = np.logical_or(x, y[:, np.newaxis])
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_axis2')

node = onnx.helper.make_node(
    'Or',
    inputs=['x', 'y'],
    outputs=['or'],
    broadcast=1,
    axis=3,
)

z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_axis3')
```

</details>
<details>
<summary>or_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Or',
    inputs=['x', 'y'],
    outputs=['or'],
    broadcast=1,
)

#3d vs 1d
x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
y = (np.random.randn(5) > 0).astype(np.bool)
z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_bcast3v1d')

#3d vs 2d
x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
y = (np.random.randn(4, 5) > 0).astype(np.bool)
z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_bcast3v2d')

#4d vs 2d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
y = (np.random.randn(5, 6) > 0).astype(np.bool)
z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_bcast4v2d')

#4d vs 3d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
y = (np.random.randn(4, 5, 6) > 0).astype(np.bool)
z = np.logical_or(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_or_bcast4v3d')
```

</details>


### Pad
There are 2 test cases, listed as following:
<details>
<summary>constant_pad</summary>

```python
node = onnx.helper.make_node(
    'Pad',
    inputs=['x'],
    outputs=['y'],
    mode='constant',
    value=1.2,
    pads=[0, 0, 1, 3, 0, 0, 2, 4],
)
x = np.random.randn(1, 3, 4, 5).astype(np.float32)
y = np.pad(
    x,
    pad_width=((0, 0), (0, 0), (1, 2), (3, 4)),
    mode='constant',
    constant_values=1.2,
)

expect(node, inputs=[x], outputs=[y],
       name='test_constant_pad')
```

</details>
<details>
<summary>reflection_and_edge_pad</summary>

```python
for mode in ['edge', 'reflect']:
    node = onnx.helper.make_node(
        'Pad',
        inputs=['x'],
        outputs=['y'],
        mode=mode,
        pads=[0, 0, 1, 1, 0, 0, 1, 1]
    )
    x = np.random.randn(1, 3, 4, 5).astype(np.float32)
    y = np.pad(
        x,
        pad_width=((0, 0), (0, 0), (1, 1), (1, 1)),
        mode=mode,
    )

    expect(node, inputs=[x], outputs=[y],
           name='test_{}_pad'.format(mode))
```

</details>


### Pow
There are 2 test cases, listed as following:
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
z = np.power(x, y) # expected output [1., 32., 729.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_pow_example')

x = np.arange(60).reshape(3, 4, 5).astype(np.float32)
y = np.random.randn(3, 4, 5).astype(np.float32)
z = np.power(x, y)
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
    broadcast=1,
)

x = np.array([1, 2, 3]).astype(np.float32)
y = np.array([2]).astype(np.float32)
z = np.power(x, y) # expected output [1., 4., 9.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_pow_bcast')

node = onnx.helper.make_node(
    'Pow',
    inputs=['x', 'y'],
    outputs=['z'],
    broadcast=1,
    axis=0,
)
x = np.array([[1, 2, 3], [4, 5, 6]]).astype(np.float32)
y = np.array([2, 3]).astype(np.float32)
z = np.array([[1, 4, 9], [64, 125, 216]]).astype(np.float32)
expect(node, inputs=[x, y], outputs=[z],
       name='test_pow_bcast_axis0')
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
y = np.reciprocal(x) #expected output [-0.25, 0.5],
expect(node, inputs=[x], outputs=[y],
       name='test_reciprocal_example')

x = np.random.rand(3, 4, 5).astype(np.float32) + 0.5
y = np.reciprocal(x)
expect(node, inputs=[x], outputs=[y],
       name='test_reciprocal')
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
There are 1 test cases, listed as following:
<details>
<summary>reshape</summary>

```python
original_shape = [2, 3, 4]
test_cases = {
    'reordered_dims':[4, 2, 3],
    'reduced_dims':[3, 8],
    'extended_dims':[3, 2, 2, 2],
    'one_dim':[24],
    'negative_dim':[6, -1, 2]
}
data = np.random.random_sample(original_shape).astype(np.float32)

for test_name,test_shape in test_cases.items():
    node = onnx.helper.make_node(
        'Reshape',
        inputs=['data'],
        outputs=['reshaped'],
        shape=test_shape,
    )

    reshaped = np.reshape(data, test_shape)
    expect(node, inputs=[data], outputs=[reshaped],
       name='test_reshape_' + test_name)
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
#expected output [-3.79272318, 0., 3.]
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
default_alpha = 1.6732
default_gamma = 1.0507
node = onnx.helper.make_node(
    'Selu',
    inputs=['x'],
    outputs=['y'],
)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.clip(x, 0, np.inf) * default_gamma + (np.exp(np.clip(x, -np.inf, 0)) - 1) * default_alpha * default_gamma
expect(node, inputs=[x], outputs=[y],
       name='test_selu_default')
```

</details>


### Shape
There are 1 test cases, listed as following:
<details>
<summary>shape</summary>

```python
node = onnx.helper.make_node(
    'Shape',
    inputs=['x'],
    outputs=['y'],
)

x = np.array([
    [1, 2, 3],
    [4, 5, 6],
]).astype(np.float32)
y = np.array([
    2, 3,
]).astype(np.int64)

expect(node, inputs=[x], outputs=[y],
       name='test_shape_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.array(x.shape).astype(np.int64)

expect(node, inputs=[x], outputs=[y],
       name='test_shape')
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
y = 1.0 / (1.0 + np.exp(np.negative(x))) #expected output [0.26894143, 0.5, 0.7310586]
expect(node, inputs=[x], outputs=[y],
       name='test_sigmoid_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = 1.0 / (1.0 + np.exp(np.negative(x)))
expect(node, inputs=[x], outputs=[y],
       name='test_sigmoid')
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
There are 5 test cases, listed as following:
<details>
<summary>slice</summary>

```python
node = onnx.helper.make_node(
    'Slice',
    inputs=['x'],
    outputs=['y'],
    axes=[0, 1],
    starts=[0, 0],
    ends=[3, 10],
)

x = np.random.randn(20, 10, 5).astype(np.float32)
y = x[0:3, 0:10]

expect(node, inputs=[x], outputs=[y],
       name='test_slice')
```

</details>
<details>
<summary>slice_default_axes</summary>

```python
node = onnx.helper.make_node(
    'Slice',
    inputs=['x'],
    outputs=['y'],
    starts=[0, 0, 3],
    ends=[20, 10, 4],
)

x = np.random.randn(20, 10, 5).astype(np.float32)
y = x[:, :, 3:4]

expect(node, inputs=[x], outputs=[y],
       name='test_slice_default_axes')
```

</details>
<details>
<summary>slice_end_out_of_bounds</summary>

```python
node = onnx.helper.make_node(
    'Slice',
    inputs=['x'],
    outputs=['y'],
    axes=[1],
    starts=[1],
    ends=[1000],
)

x = np.random.randn(20, 10, 5).astype(np.float32)
y = x[:, 1:1000]

expect(node, inputs=[x], outputs=[y],
       name='test_slice_end_out_of_bounds')
```

</details>
<details>
<summary>slice_neg</summary>

```python
node = onnx.helper.make_node(
    'Slice',
    inputs=['x'],
    outputs=['y'],
    axes=[1],
    starts=[0],
    ends=[-1],
)

x = np.random.randn(20, 10, 5).astype(np.float32)
y = x[:, 0:-1]

expect(node, inputs=[x], outputs=[y],
       name='test_slice_neg')
```

</details>
<details>
<summary>slice_start_out_of_bounds</summary>

```python
node = onnx.helper.make_node(
    'Slice',
    inputs=['x'],
    outputs=['y'],
    axes=[1],
    starts=[1000],
    ends=[1000],
)

x = np.random.randn(20, 10, 5).astype(np.float32)
y = x[:, 1000:1000]

expect(node, inputs=[x], outputs=[y],
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
y = np.exp(x) / np.sum(np.exp(x), axis=1) #expected output [[0.09003058, 0.24472848, 0.66524094]]
expect(node, inputs=[x], outputs=[y],
       name='test_softmax_example')
```

</details>
<details>
<summary>softmax_axis</summary>

```python
def softmax_2d(x):
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))

x = np.array([[0, 1, 2, 3], [10000, 10001, 10002, 10003]]).astype(np.float32)
#expected output [[0.0320586, 0.08714432, 0.23688284, 0.64391428],
#                 [0.0320586, 0.08714432, 0.23688284, 0.64391428]]
y = softmax_2d(x)

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
y = softmax_2d(x.reshape(1, 60)).reshape(3, 4, 5)
expect(node, inputs=[x], outputs=[y],
       name='test_softmax_axis_0')

node = onnx.helper.make_node(
    'Softmax',
    inputs=['x'],
    outputs=['y'],
    axis=1,
)
y = softmax_2d(x.reshape(3, 20)).reshape(3, 4, 5)
expect(node, inputs=[x], outputs=[y],
       name='test_softmax_axis_1')

# default axis is 1
node = onnx.helper.make_node(
    'Softmax',
    inputs=['x'],
    outputs=['y'],
)
expect(node, inputs=[x], outputs=[y],
       name='test_softmax_default_axis')

node = onnx.helper.make_node(
    'Softmax',
    inputs=['x'],
    outputs=['y'],
    axis=2,
)
y = softmax_2d(x.reshape(12, 5)).reshape(3, 4, 5)
expect(node, inputs=[x], outputs=[y],
       name='test_softmax_axis_2')
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
y = np.log(np.exp(x) + 1) #expected output [0.31326166, 0.69314718, 1.31326163]
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
y = np.sqrt(x) #expected output [1., 2., 3.]
expect(node, inputs=[x], outputs=[y],
       name='test_sqrt_example')

x = np.abs(np.random.randn(3, 4, 5).astype(np.float32))
y = np.sqrt(x)
expect(node, inputs=[x], outputs=[y],
       name='test_sqrt')
```

</details>


### Squeeze
There are 1 test cases, listed as following:
<details>
<summary>squeeze</summary>

```python
node = onnx.helper.make_node(
    'Squeeze',
    inputs=['x'],
    outputs=['y'],
    axes=[0],
)
x = np.random.randn(1, 3, 4, 5).astype(np.float32)
y = np.squeeze(x, axis=0)

expect(node, inputs=[x], outputs=[y],
       name='test_squeeze')
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
z = x - y #expected output [-2., 0., 2.]
expect(node, inputs=[x, y], outputs=[z],
       name='test_sub_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.random.randn(3, 4, 5).astype(np.float32)
z = x - y
expect(node, inputs=[x, y], outputs=[z],
       name='test_sub')
```

</details>
<details>
<summary>sub_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Sub',
    inputs=['x', 'y'],
    outputs=['z'],
    broadcast=1,
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
y = np.tanh(x) #expected output [-0.76159418, 0., 0.76159418]
expect(node, inputs=[x], outputs=[y],
       name='test_tanh_example')

x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.tanh(x)
expect(node, inputs=[x], outputs=[y],
       name='test_tanh')
```

</details>


### TopK
There are 1 test cases, listed as following:
<details>
<summary>top_k</summary>

```python
node = onnx.helper.make_node(
    'TopK',
    inputs=['x'],
    outputs=['values', 'indices'],
    k=3
)
X = np.array([
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [8, 9, 10, 11],
], dtype=np.float32)
values_ref = np.array([
    [3, 2, 1],
    [7, 6, 5],
    [11, 10, 9],
])
indices_ref = np.array([
    [3, 2, 1],
    [3, 2, 1],
    [3, 2, 1],
], dtype=np.int32)

expect(node, inputs=[X], outputs=[values_ref, indices_ref],
       name='test_top_k')
```

</details>


### Transpose
There are 2 test cases, listed as following:
<details>
<summary>all_permutations</summary>

```python
shape = (2,3,4)
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


### Unsqueeze
There are 1 test cases, listed as following:
<details>
<summary>squeeze</summary>

```python
node = onnx.helper.make_node(
    'Unsqueeze',
    inputs=['x'],
    outputs=['y'],
    axes=[0],
)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = np.expand_dims(x, axis=0)

expect(node, inputs=[x], outputs=[y],
       name='test_unsqueeze')
```

</details>


### Xor
There are 3 test cases, listed as following:
<details>
<summary>xor</summary>

```python
node = onnx.helper.make_node(
    'Xor',
    inputs=['x', 'y'],
    outputs=['xor'],
)

# 2d
x = (np.random.randn(3, 4) > 0).astype(np.bool)
y = (np.random.randn(3, 4) > 0).astype(np.bool)
z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor2d')

# 3d
x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
y = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor3d')

# 4d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
y = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor4d')
```

</details>
<details>
<summary>xor_axis</summary>

```python
x = (np.random.randn(5, 5, 5, 5) > 0).astype(np.bool)
y = (np.random.randn(5) > 0).astype(np.bool)

node = onnx.helper.make_node(
    'Xor',
    inputs=['x', 'y'],
    outputs=['xor'],
    broadcast=1,
    axis=0,
)

z = np.logical_xor(x, y[:, np.newaxis, np.newaxis, np.newaxis])
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor_axis0')

node = onnx.helper.make_node(
    'Xor',
    inputs=['x', 'y'],
    outputs=['xor'],
    broadcast=1,
    axis=1,
)

z = np.logical_xor(x, y[:, np.newaxis, np.newaxis,])
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor_axis1')

node = onnx.helper.make_node(
    'Xor',
    inputs=['x', 'y'],
    outputs=['xor'],
    broadcast=1,
    axis=2,
)

z = np.logical_xor(x, y[:, np.newaxis,])
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor_axis2')

node = onnx.helper.make_node(
    'Xor',
    inputs=['x', 'y'],
    outputs=['xor'],
    broadcast=1,
    axis=3,
)

z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor_axis3')
```

</details>
<details>
<summary>xor_broadcast</summary>

```python
node = onnx.helper.make_node(
    'Xor',
    inputs=['x', 'y'],
    outputs=['xor'],
    broadcast=1,
)

#3d vs 1d
x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
y = (np.random.randn(5) > 0).astype(np.bool)
z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor_bcast3v1d')

#3d vs 2d
x = (np.random.randn(3, 4, 5) > 0).astype(np.bool)
y = (np.random.randn(4, 5) > 0).astype(np.bool)
z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor_bcast3v2d')

#4d vs 2d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
y = (np.random.randn(5, 6) > 0).astype(np.bool)
z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor_bcast4v2d')

#4d vs 3d
x = (np.random.randn(3, 4, 5, 6) > 0).astype(np.bool)
y = (np.random.randn(4, 5, 6) > 0).astype(np.bool)
z = np.logical_xor(x, y)
expect(node, inputs=[x, y], outputs=[z],
       name='test_xor_bcast4v3d')
```

</details>


<br/>

## &#x1F494;No Cover Common Operators
### ArgMax (call for test cases)


### ArgMin (call for test cases)


### AveragePool (call for test cases)


### BatchNormalization (call for test cases)


### ConvTranspose (call for test cases)


### DepthToSpace (call for test cases)


### Dropout (call for test cases)


### GRU (call for test cases)


### Gemm (call for test cases)


### GlobalLpPool (call for test cases)


### InstanceNormalization (call for test cases)


### LRN (call for test cases)


### LSTM (call for test cases)


### LpNormalization (call for test cases)


### LpPool (call for test cases)


### MaxPool (call for test cases)


### MaxRoiPool (call for test cases)


### PRelu (call for test cases)


### RNN (call for test cases)


### RandomNormal (random generator operator)


### RandomNormalLike (random generator operator)


### RandomUniform (random generator operator)


### RandomUniformLike (random generator operator)


### ReduceL1 (call for test cases)


### ReduceL2 (call for test cases)


### ReduceLogSum (call for test cases)


### ReduceLogSumExp (call for test cases)


### ReduceMax (call for test cases)


### ReduceMean (call for test cases)


### ReduceMin (call for test cases)


### ReduceProd (call for test cases)


### ReduceSum (call for test cases)


### ReduceSumSquare (call for test cases)


### SpaceToDepth (call for test cases)


### Split (call for test cases)


### Tile (call for test cases)


<br/>

## &#x1F49A;Covered Experimental Operators
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


<br/>

## &#x1F494;No Cover Experimental Operators
### ATen (call for test cases)


### Affine (call for test cases)


### ConstantFill (call for test cases)


### Crop (call for test cases)


### FC (call for test cases)


### GRUUnit (call for test cases)


### GivenTensorFill (call for test cases)


### Identity (call for test cases)


### If (call for test cases)


### ImageScaler (call for test cases)


### Loop (call for test cases)


### LoopIndexTensor (call for test cases)


### MeanVarianceNormalization (call for test cases)


### ParametricSoftplus (call for test cases)


### Scale (call for test cases)


### ScaledTanh (call for test cases)


### Upsample (call for test cases)


<br/>

# Model Test Coverage
## To be filled.
# Overall Test Coverage
## To be filled.
