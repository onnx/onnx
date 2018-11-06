# Dimension Denotation

Dimension Denotation is an experimental attempt to give tensor axis semantic descriptions and thus types and perform verification steps based on them subsequently.

## Motivation

The motivation of such a mechanism can be illustrated via a simple example. In the linear neural network specification below, we assume a NCHW model input:

```
input_in_NCHW -> Transpose(input, perm=[0, 2, 1, 3]) -> AveragePool(input, ...)
```

In this neural network, a user mistakenly constructed a neural network that transposes an NCHW input to a weird NHCW format and pass through spatial pooling that assumes a NCHW input format. As clearly a mistake as it is, no existing infrastructure will report an error to the user. This is should be deeply unnerving to programmers who rely heavily on type checking as an integral part of program correctness guarantee. This proposal seeks to resolve this vacuum of proper type-checking inherent in the current paradigm of neural network specification.

This proposal consists of three key components: Denotation Definition, Denotation Propagation and Denotation Verification, each of which will be discussed in detail.

## Denotation Definition

To begin with, we define a set of types for tensor types. Such types are defined based on the following principles:
1. Be fine grain enough to eliminate potential pitfalls. For instance, the above example illustrated in the motivation section mandates that we distinguish between a channel dimension and a spatial feature dimension to ensure the correctness of execution of the AveragePool op.
2. Be coarse grain enough to alleviate the mental burden of users. For instance, in the above example, there is significantly less need to distinguish between a width dimension and a height dimension because operations like pooling and convolution often do not draw a distinction between various spatial dimensions. Thus, we summarize all the spatial dimensions as feature dimensions.
3. As an important corollary of 2, be model agnostic. For instance, the semantics of feature dimensions in recurrent neural networks (RNN) and the semantics of spatial dimensions in convolutional neural network (CNN) are almost indistinguishable and therefore we permit users and developers to describe either as a feature dimension.

Specifically, in our first proposal, we define the following set of standard denotations:

1. `DATA_BATCH` describes a batch dimension of the training data. This corresponds to the `N` dimension in the more commonly used tensor format notation `NCHW`.
2. `DATA_CHANNEL` describes a channel dimension of the training data. This corresponds to the `C` dimension.
3. `DATA_TIME` describes a time dimension.
4. `DATA_FEATURE` describes a feature dimension. This corresponds to the `H`, `W` dimension or the feature dimension in RNN.
5. `FILTER_IN_CHANNEL` describes a filter in-channel dimension. This is the dimension that is identical (in size) to the channel dimension of the input image feature maps.
6. `FILTER_OUT_CHANNEL` describes a filter out-channel dimension. This is the dimension that is identical (in size) to the channel dimension of the output image feature maps.
7. `FILTER_SPATIAL` describes a filter spatial dimension.

## Denotation Propagation

Denotation Propagation happens when an operation permutes, destroys or creates dimensions with respect to its input tensor. In such scenarios, we will implement customized, operation-specific functions to infer the output tensor dimension denotation based on the input tensor dimension denotation. An example operation where denotation propagation happens is Transpose operation where the pseudocode for output dimension denotation inference can be formulated as a function of the input dimension denotation:

```
for i, j in enumerate(perm):
    out_dim_denotaion[i] = in_dim_denotation[j]
```

## Denotation Verification

Denotation Verification happens when an operation expects its input to arrive in a particular format. An example operation where denotation verification happens is AveragePool operation where the input, if annotated with dimension denotation, in the 2D case should have the denotation [`DATA_BATCH`, `DATA_CHANNEL`, `DATA_FEATURE`, `DATA_FEATURE`]. If there is a mismatch between the expected dimension denotation and the actual dimension denotation, an error should be reported.

## Type Denotation

See the [type denotation documentation](TypeDenotation.md) for more details on how to describe images and other types.