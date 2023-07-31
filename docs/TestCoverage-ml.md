<!--- SPDX-License-Identifier: Apache-2.0 -->
# Test Coverage Report (ONNX-ML Operators)
## Outlines
* [Node Test Coverage](#node-test-coverage)
* [Model Test Coverage](#model-test-coverage)
* [Overall Test Coverage](#overall-test-coverage)
# Node Test Coverage
## Summary
Node tests have covered 3/18 (16.67%, 0 generators excluded) common operators.

Node tests have covered 0/0 (N/A) experimental operators.

* [Covered Common Operators](#covered-common-operators)
* [No Cover Common Operators](#no-cover-common-operators)
* [Covered Experimental Operators](#covered-experimental-operators)
* [No Cover Experimental Operators](#no-cover-experimental-operators)

## &#x1F49A;Covered Common Operators
### ArrayFeatureExtractor
There are 1 test cases, listed as following:
<details>
<summary>arrayfeatureextractor</summary>

```python
node = onnx.helper.make_node(
    "ArrayFeatureExtractor",
    inputs=["x", "y"],
    outputs=["z"],
    domain="ai.onnx.ml",
)

x = np.arange(12).reshape((3, 4)).astype(np.float32)
y = np.array([0, 1], dtype=np.int64)
z = np.array([[0, 4, 8], [1, 5, 9]], dtype=np.float32).T
expect(
    node,
    inputs=[x, y],
    outputs=[z],
    name="test_ai_onnx_ml_array_feature_extractor",
)
```

</details>


### Binarizer
There are 1 test cases, listed as following:
<details>
<summary>binarizer</summary>

```python
threshold = 1.0
node = onnx.helper.make_node(
    "Binarizer",
    inputs=["X"],
    outputs=["Y"],
    threshold=threshold,
    domain="ai.onnx.ml",
)
x = np.random.randn(3, 4, 5).astype(np.float32)
y = compute_binarizer(x, threshold)[0]

expect(node, inputs=[x], outputs=[y], name="test_ai_onnx_ml_binarizer")
```

</details>


### LabelEncoder
There are 2 test cases, listed as following:
<details>
<summary>string_int_label_encoder</summary>

```python
node = onnx.helper.make_node(
    "LabelEncoder",
    inputs=["X"],
    outputs=["Y"],
    domain="ai.onnx.ml",
    keys_strings=["a", "b", "c"],
    values_int64s=[0, 1, 2],
    default_int64=42,
)
x = np.array(["a", "b", "d", "c", "g"]).astype(object)
y = np.array([0, 1, 42, 2, 42]).astype(np.int64)
expect(node, inputs=[x], outputs=[y], name="test_ai_onnx_ml_label_encoder_string_int")

node = onnx.helper.make_node(
    "LabelEncoder",
    inputs=["X"],
    outputs=["Y"],
    domain="ai.onnx.ml",
    keys_strings=["a", "b", "c"],
    values_int64s=[0, 1, 2],
)
x = np.array(["a", "b", "d", "c", "g"]).astype(object)
y = np.array([0, 1, -1, 2, -1]).astype(np.int64)
expect(node, inputs=[x], outputs=[y], name="test_ai_onnx_ml_label_encoder_string_int_no_default")
```

</details>
<details>
<summary>tensor_based_label_encoder</summary>

```python
node = onnx.helper.make_node(
    "LabelEncoder",
    inputs=["X"],
    outputs=["Y"],
    domain="ai.onnx.ml",
    keys_as_tensor=make_tensor("keys_as_tensor", onnx.TensorProto.STRING, (3,), ["a", "b", "c"]),
    values_as_tensor=make_tensor("values_as_tensor", onnx.TensorProto.INT16, (3,), [0, 1, 2]),
    default_as_tensor=make_tensor("default_as_tensor", onnx.TensorProto.INT16, (1,), [42]),
)
x = np.array(["a", "b", "d", "c", "g"]).astype(object)
y = np.array([0, 1, 42, 2, 42]).astype(np.int16)
expect(node, inputs=[x], outputs=[y], name="test_ai_onnx_ml_label_encoder_tensor_mapping")
```

</details>


<br/>

## &#x1F494;No Cover Common Operators
### CastMap (call for test cases)


### CategoryMapper (call for test cases)


### DictVectorizer (call for test cases)


### FeatureVectorizer (call for test cases)


### Imputer (call for test cases)


### LinearClassifier (call for test cases)


### LinearRegressor (call for test cases)


### Normalizer (call for test cases)


### OneHotEncoder (call for test cases)


### SVMClassifier (call for test cases)


### SVMRegressor (call for test cases)


### Scaler (call for test cases)


### TreeEnsembleClassifier (call for test cases)


### TreeEnsembleRegressor (call for test cases)


### ZipMap (call for test cases)


<br/>

## &#x1F49A;Covered Experimental Operators
<br/>

## &#x1F494;No Cover Experimental Operators
<br/>

# Model Test Coverage
No model tests present for selected domain
# Overall Test Coverage
## To be filled.
