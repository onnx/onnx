# Adding a New Operator to ONNX standard

Operator is the basic computation unit in the neural network models.
Currently, ONNX covers a core set of operators. There are operators which are
needed by users, however they are still excluded from ONNX operator set. For
example, GroupNormalization was proposed this year, and many developers already
use it in their models, but in ONNX operator set, it's still missing. To make
ONNX more comprehensive, we should encourage community to contribute high
quality operator specification to expand our operator set.

We are maintaining [a list](https://github.com/onnx/onnx/issues/1646) in our
GitHub issue. Please check it and post your request following this thread.

## Step 1: Determine Whether We Should Add the Operator
Determine whether if the proposed operator is common enough, usually it
should/will be supported by at least two frameworks, such as PyTorch/Caffe2,
Tensorflow, MxNet, etc.

## Step 2: Determine the Operator Should Be Added as Operator or Function
If the requested operator can be easily expressed as several (e.g., 2 or 3)
existing operators, we can define it as a Function (i.e., composite operator).
MeanVarianceNormalization is the first example of registering an operator
as a Function.

## Step 3: Define and polish the spec of the proposed operator.
The spec should include:
1. opset version of the added operator. Check out our
[versioning doc](https://github.com/fdwr/onnx/blob/master/docs/Versioning.md#operator-versioning)
for more details
2. description about the operators, should be with enough details to avoid
ambiguity, adding the links to refs if necessary 
3. inputs,
4. outputs,
5. attributes,
6. type constraints about input and output tensors
7. shape inference function Example:
https://github.com/onnx/onnx/blob/master/onnx/defs/nn/defs.cc#L1183

Usually, if we can find similar functions in Numpy, we will try to align with
numpy.

## Step 4: Adding Test Cases for the New Operator.
The testing examples will be extracted to the doc. Later, we also generate
binary data for it. Example:
https://github.com/onnx/onnx/blob/master/onnx/backend/test/case/node/abs.py

## Step 5: Update the Doc and Generate the Test Data
Running [the script](https://github.com/onnx/onnx/blob/master/tools/update_doc.sh)
to update the doc and generate the test data.

## Example to Follow
In [PR 1428](https://github.com/onnx/onnx/pull/1428), we add EyeLike generator operator.
It's a good example to follow.
