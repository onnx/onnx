<!--- SPDX-License-Identifier: Apache-2.0 -->
# Test Coverage Report (ONNX-ML Operators)
## Outlines
* [Node Test Coverage](#node-test-coverage)
* [Model Test Coverage](#model-test-coverage)
* [Overall Test Coverage](#overall-test-coverage)
# Node Test Coverage
## Summary
Node tests have covered 4/19 (21.05%, 0 generators excluded) common operators.

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
expect(
    node,
    inputs=[x],
    outputs=[y],
    name="test_ai_onnx_ml_label_encoder_string_int",
)

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
expect(
    node,
    inputs=[x],
    outputs=[y],
    name="test_ai_onnx_ml_label_encoder_string_int_no_default",
)
```

</details>
<details>
<summary>tensor_based_label_encoder</summary>

```python
tensor_keys = make_tensor(
    "keys_tensor", onnx.TensorProto.STRING, (3,), ["a", "b", "c"]
)
repeated_string_keys = ["a", "b", "c"]
x = np.array(["a", "b", "d", "c", "g"]).astype(object)
y = np.array([0, 1, 42, 2, 42]).astype(np.int16)

node = onnx.helper.make_node(
    "LabelEncoder",
    inputs=["X"],
    outputs=["Y"],
    domain="ai.onnx.ml",
    keys_tensor=tensor_keys,
    values_tensor=make_tensor(
        "values_tensor", onnx.TensorProto.INT16, (3,), [0, 1, 2]
    ),
    default_tensor=make_tensor(
        "default_tensor", onnx.TensorProto.INT16, (1,), [42]
    ),
)

expect(
    node,
    inputs=[x],
    outputs=[y],
    name="test_ai_onnx_ml_label_encoder_tensor_mapping",
)

node = onnx.helper.make_node(
    "LabelEncoder",
    inputs=["X"],
    outputs=["Y"],
    domain="ai.onnx.ml",
    keys_strings=repeated_string_keys,
    values_tensor=make_tensor(
        "values_tensor", onnx.TensorProto.INT16, (3,), [0, 1, 2]
    ),
    default_tensor=make_tensor(
        "default_tensor", onnx.TensorProto.INT16, (1,), [42]
    ),
)

expect(
    node,
    inputs=[x],
    outputs=[y],
    name="test_ai_onnx_ml_label_encoder_tensor_value_only_mapping",
)
```

</details>


### TreeEnsemble
There are 2 test cases, listed as following:
<details>
<summary>tree_ensemble_set_membership</summary>

```python
node = onnx.helper.make_node(
    "TreeEnsemble",
    ["X"],
    ["Y"],
    domain="ai.onnx.ml",
    n_targets=4,
    aggregate_function=1,
    membership_values=make_tensor(
        "membership_values",
        onnx.TensorProto.FLOAT,
        (8,),
        [1.2, 3.7, 8, 9, np.nan, 12, 7, np.nan],
    ),
    nodes_missing_value_tracks_true=None,
    nodes_hitrates=None,
    post_transform=0,
    tree_roots=[0],
    nodes_modes=make_tensor(
        "nodes_modes",
        onnx.TensorProto.UINT8,
        (3,),
        np.array([0, 6, 6], dtype=np.uint8),
    ),
    nodes_featureids=[0, 0, 0],
    nodes_splits=make_tensor(
        "nodes_splits",
        onnx.TensorProto.FLOAT,
        (3,),
        np.array([11, 232344.0, np.nan], dtype=np.float32),
    ),
    nodes_trueleafs=[0, 1, 1],
    nodes_truenodeids=[1, 0, 1],
    nodes_falseleafs=[1, 0, 1],
    nodes_falsenodeids=[2, 2, 3],
    leaf_targetids=[0, 1, 2, 3],
    leaf_weights=make_tensor(
        "leaf_weights", onnx.TensorProto.FLOAT, (4,), [1, 10, 1000, 100]
    ),
)

x = np.array([1.2, 3.4, -0.12, np.nan, 12, 7], np.float32).reshape(-1, 1)
expected = np.array(
    [
        [1, 0, 0, 0],
        [0, 0, 0, 100],
        [0, 0, 0, 100],
        [0, 0, 1000, 0],
        [0, 0, 1000, 0],
        [0, 10, 0, 0],
    ],
    dtype=np.float32,
)
expect(
    node,
    inputs=[x],
    outputs=[expected],
    name="test_ai_onnx_ml_tree_ensemble_set_membership",
)
```

</details>
<details>
<summary>tree_ensemble_single_tree</summary>

```python
node = onnx.helper.make_node(
    "TreeEnsemble",
    ["X"],
    ["Y"],
    domain="ai.onnx.ml",
    n_targets=2,
    membership_values=None,
    nodes_missing_value_tracks_true=None,
    nodes_hitrates=None,
    aggregate_function=1,
    post_transform=0,
    tree_roots=[0],
    nodes_modes=make_tensor(
        "nodes_modes",
        onnx.TensorProto.UINT8,
        (3,),
        np.array([0, 0, 0], dtype=np.uint8),
    ),
    nodes_featureids=[0, 0, 0],
    nodes_splits=make_tensor(
        "nodes_splits",
        onnx.TensorProto.DOUBLE,
        (3,),
        np.array([3.14, 1.2, 4.2], dtype=np.float64),
    ),
    nodes_truenodeids=[1, 0, 1],
    nodes_trueleafs=[0, 1, 1],
    nodes_falsenodeids=[2, 2, 3],
    nodes_falseleafs=[0, 1, 1],
    leaf_targetids=[0, 1, 0, 1],
    leaf_weights=make_tensor(
        "leaf_weights",
        onnx.TensorProto.DOUBLE,
        (4,),
        np.array([5.23, 12.12, -12.23, 7.21], dtype=np.float64),
    ),
)

x = np.array([1.2, 3.4, -0.12, 1.66, 4.14, 1.77], np.float64).reshape(3, 2)
y = np.array([[5.23, 0], [5.23, 0], [0, 12.12]], dtype=np.float64)
expect(
    node,
    inputs=[x],
    outputs=[y],
    name="test_ai_onnx_ml_tree_ensemble_single_tree",
)
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
