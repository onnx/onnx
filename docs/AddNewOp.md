<!--- SPDX-License-Identifier: Apache-2.0 -->

# Table of Contents
1. [New Operator or Function](#new_operator_or_function)
    1. [Step 1: Proposing a new operator/function](#step1_new_operator_or_function)
    2. [Step 2: Submit PR](#step2_new_operator_or_function)
    3. [Step 3: PR Review by Operators SIG](#step3_new_operator_or_function)
    4. [Step 4: ONNX release](#step4_new_operator_or_function)

2. [Removing Operator or Function](#removing_operator_or_function)
    1. [Removing operator](#removing_operator)
    2. [Removing function](#removing_function)
    3. [Document removing operator or function](#document_removing_operator_or_function)

# Proposing and submitting a new operator or function to ONNX <a name="new_operator_or_function"></a>

Operators are the basic building blocks used to define ONNX models. With a rich set of operators, ONNX can describe most DNN and ML models from various frameworks. Functions enable expressing complex operators in terms of more primitive operators. The ONNX specification includes a core set of operators that enable many models. It is a non-goal to add all possible operators, however more operators are added as needed to cover evolving needs.

In this document, we describe the process of accepting a new proposed operator and how to properly submit a new operator as part of ONNX standard. The goal is to improve on what we currently have based on our experience, learning and feedbacks we gathered from the community.

## 4 steps to add an operator <a name="steps_to_add_an_operator"></a>
1. Decide what to propose
2. Submit PR for new operator/function
3. Review of PR by Operators SIG
4. Merging of PR and inclusion in next ONNX release

## Step 1: Proposing a new operator/function <a name="step1_new_operator_or_function"></a>
In order to propose a new operator/function, the following is needed:
1. If the operator can be expressed in terms of other ONNX operators, then it should be a function and not an operator (we have a function in ONNX : MeanVarianceNormalization).
2. If the operators can be split to new primitives, propose those primitives instead and make the operator a function.
3. Based on a model. This will help us understand the usage and that it solves an actual problem. For the case of the model being private or IP and can't be shared, the operator doesn't belong to the standard and should be implemented as custom OP.
4. The operator needs to be implemented by at-least one (well-known) framework. This help us to understand the actual behavior of the operator and its usage.
5. Operator signature and behavior:
    1. If the operator is available in numpy, prefer numpy semantics.
    2. If the operator is available in more than one frameworks, make sure that your design is general and cover those frameworks.
6. Prefer attributes over inputs.
7. The operator should not be made more complex than is required by the use-cases. However, the operator
should be made as general as possible, as long as it does not make the implementation more complex.
This requires carefully balancing generality and complexity. For example, generalizing from 3-D tensors to
N-D tensors is straight-forward (implementation-wise) for some operators, but complex for other operators.
The choice in such cases will be made based on the complexity of such a generalization.

## Step 2: Submit PR <a name="step2_new_operator_or_function"></a>
Once the criteria of proposing new operator/function has been satisfied, you will need to submit a PR for the new operator/function. Here the expectation of what the PR should include. The reviewer is expected to verify the completeness of the PR before signoff.
1. Description:
    1. Write a detailed description about the operator, and its expected behavior. Pretty much, the description should be clear enough to avoid confusion between implementors.
    2. Add an example in the description to illustrate the usage.
    3. Add reference to the source of the operator in the corresponding framework in the description (if possible).
    4. Write the mathematic formula or a pseudocode in the description. The core algorithm needs to be very clear.
2. Write a reference implementation in Python, this reference implementation should cover all the expected behavior of the operator. Only in extremely rare case, we will waive this requirement.
3. Operator version: check out our
[versioning doc](/docs/Versioning.md#operator-versioning)
4. Write unit test, that cover main usage and corner cases.
    1. The testing examples will be extracted to the doc.
    2. We also generate binary data for it.
    3. Example: [onnx/backend/test/case/node/abs.py](/onnx/backend/test/case/node/abs.py)
5. Add at least one automatic upgrade test for your operator in [onnx/test/automatic_upgrade_test.py](/onnx/test/automatic_upgrade_test.py) using `_test_op_upgrade`. These tests create a given operator at a given opset version (usually the version the operator was introduced in) and test that the version converter is able to convert them to the highest available version. So for a new operator `_test_op_upgrade` will not test anything, but as soon as the operator gets updated in a future opset the test will autoamtically become nontrivial.
6. Update the documentation and generate the test data.
    1. Running [the script](/tools/update_doc.sh)
to update the doc and generate the test data.
7. Shape Inference function
    1. Please provide a shape inference function in cases where it is meaningful and applicable.
    2. In cases where shape inference is not possible, it must have logic to perform
rank inference at the very least (adding right amount of dimensions to the output shape)
    3. Shape inference functions must be accompanied by unit tests ([onnx/test/shape_inference_test.py](/onnx/test/shape_inference_test.py)).
    4. You can refer to the shape inference function for the `TopK` operator while implementing your own function ([onnx/defs/math/defs.cc](/onnx/defs/math/defs.cc))

### Example to Follow
[PR 1959](https://github.com/onnx/onnx/pull/1959) is a good example to follow.

## Step 3: PR Review by Operators SIG <a name="step3_new_operator_or_function"></a>
The [Operators SIG](https://github.com/onnx/sigs/tree/main/operators) is responsible for the operators/functions in the ONNX specification. The SIG regularly meets and reviews PRs.

### Sign-off
At least two sign-off from the Operators SIG [contributors](https://github.com/onnx/onnx/tree/main/community#community-roles).

## Step 4: ONNX release <a name="step4_new_operator_or_function"></a>
Once the PR is reviewed and signed off by the Operators SIG, it will be merged. Your new operator/function will be part of the main branch and available to anyone building from source. These are not official releases. ONNX periodically releases official new versions that are a snapshot of the main branch. Your new operator/function will be part of that release.

# Removing operator or function <a name="removing_operator_or_function"></a>
There are a lot of reasons for removing existing ONNX operator or function, such us being replaced with different operator or can be decomposed by a set of other operators. This document describes the criteria of removing an existing ONNX operator from the standard.

## Removing operator <a name="removing_operator"></a>
Any operator in ONNX was added because it was required by a model and/or framework. In order to deprecate such an operator we need to do the following.
* Operator can’t be deprecated unless there is a replacement.
    * Replacement can be a more general operator that supersedes the old one.
    * Or a set of primitive operators that together can implement the same functionality and behavior of the deprecated operator (Function).
* If the deprecated operator can be decomposed by existing operators then it must be converted to a function.
* If replacement isn’t in ONNX standard yet, then add the replacement operator or set of operators first.
* Add a version adapter which turns the operator into its replacement for the version converter. Example: [onnx/version_converter/adapters/upsample_9_10.h](/onnx/version_converter/adapters/upsample_9_10.h)
* No grace period is needed for deprecated operators.

## Removing function <a name="removing_function"></a>
Function, by definition, is composed of ONNX primitives; however, function could have been accelerated by framework or runtime that support ONNX. So, removing function is not recommended, with the exception of adding another single function which supersede its functionality.

## Document removing operator or function <a name="document_removing_operator_or_function"></a>
To make sure everyone is aware of the deprecation, the following need to happen:
* Any removed operator or function from ONNX need to be mentioned in the release note.
* Their old documentation needs to be updated to show the new replacement and the mapping between the old to the new.
    * Only `def.cc` need to be remove, `old.cc` will remain.
    * `old.cc` need to be updated with the mapping to the replacement.
* ONNX checker need to be updated to error with a proper message.
* All removed operators need to be appended at the end of the `operator.md` file.
