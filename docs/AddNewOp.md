# Proposing and submitting a new operator or function to ONNX

Operators are the basic building blocks that define ONNX model. With a rich set of operators, ONNX can describe most DNN and ML models from various frameworks. Improving the process of accepting and adding a new operator to ONNX is crucial for ONNX standard future and its viability.

In this document, we describe the process of accepting a new proposed operator and how to properly submit a new operator as part of ONNX standard. The goal is to improve on what we currently have based on our experience, learning and feedbacks we gathered from the community.

The ONNX specification includes a core set of operators that enable many models. It is a non-goal to add all possible operators, however more operators are added as needed to cover evolving needs.

# 4 steps to add an operator
1. Decide whether to add an operator or function
2. Submit PR to propose an operator/function
3. Review of PR by Operators SIG
4. Merging of PR and inclusion in next ONNX release

# Proposing a new operator
In order to propose a new operator, the following is needed:
1. If the operator can be composed by other ONNX operators, then it should be a function and not an operator (we have a function in ONNX : MeanVarianceNormalization).
2. If the operators can be split to new primitives, propose those primitives instead and make the operator a function.
3. Based on a model. This will help us understand the usage and that it solves an actual problem. For the case of the model being private or IP and can't be shared, the operator doesn't belong to the standard and should be implemented as custom OP.
4. The operator needs to be implemented by at-least one (well-known) framework. This help us to understand the actual behavior of the operator and its usage.
5. Operator signature and behavior:
    1. If the operator is available in numpy, prefer numpy semantics.
    2. If the operator is available in more than one frameworks, make sure that your design is general and cover those frameworks.
6. Prefer attributes over inputs.

# Submitting new operator
Once the criteria of proposing new operator has been satisfied, you will need to submit a PR for the new operator. Here the expectation of what the PR should include. The reviewer is expected to verify the completeness of the PR before signoff.
1. Description:
    1. Write a detailed description about the operator, and its expected behavior. Pretty much, the description should be clear enough to avoid confusion between implementors.
    2. Add an example in the description to illustrate the usage.
    3. Add reference to the source of the operator in the corresponding framework in the description (if possible).
    4. Write the mathematic formula or a pseudocode in the description. The core algorithm needs to be very clear.
2. Write a reference implementation in Python, this reference implementation should cover all the expected behavior of the operator. Only in extremely rare case, we will waive this requirement.
3. Operator version: check out our
[versioning doc](https://github.com/fdwr/onnx/blob/master/docs/Versioning.md#operator-versioning)
4. Write unit test, that cover main usage and corner cases. 
    1. The testing examples will be extracted to the doc. 
    2. We also generate binary data for it. 
    3. Example: https://github.com/onnx/onnx/blob/master/onnx/backend/test/case/node/abs.py
5. Update the documentation and generate the test data.
    1. Running [the script](https://github.com/onnx/onnx/blob/master/tools/update_doc.sh)
to update the doc and generate the test data.
6. Shape Inference function 
    1. Please provide a shape inference function in cases where it is meaningful and applicable.
    2. In cases where shape inference is not possible, it must have logic to perform 
rank inference at the very least (adding right amount of dimensions to the output shape)
    3. Shape inference functions must be accompanied by unit tests (https://github.com/onnx/onnx/blob/master/onnx/test/shape_inference_test.py).
    4. You can refer to the shape inference function for the `TopK` operator while implementing your own function (https://github.com/onnx/onnx/blob/master/onnx/defs/math/defs.cc#L943)

## Example to Follow
[PR 1959](https://github.com/onnx/onnx/pull/1959) is a good example to follow.

## Sign-off
At least two sign-off from the operator contributor group.
