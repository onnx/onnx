<!--- SPDX-License-Identifier: Apache-2.0 -->
# Test Coverage Report (ONNX-ML Operators)
## Outlines
* [Node Test Coverage](#node-test-coverage)
* [Model Test Coverage](#model-test-coverage)
* [Overall Test Coverage](#overall-test-coverage)
# Node Test Coverage
## Summary
Node tests have covered 1/18 (5.56%, 0 generators excluded) common operators.

Node tests have covered 0/0 (N/A) experimental operators.

* [Covered Common Operators](#covered-common-operators)
* [No Cover Common Operators](#no-cover-common-operators)
* [Covered Experimental Operators](#covered-experimental-operators)
* [No Cover Experimental Operators](#no-cover-experimental-operators)

## &#x1F49A;Covered Common Operators
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


<br/>

## &#x1F494;No Cover Common Operators
### ArrayFeatureExtractor (call for test cases)


### CastMap (call for test cases)


### CategoryMapper (call for test cases)


### DictVectorizer (call for test cases)


### FeatureVectorizer (call for test cases)


### Imputer (call for test cases)


### LabelEncoder (call for test cases)


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
