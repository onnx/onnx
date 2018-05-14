Using Denotation For Semantic Description
------------------------------------------

Denotation is an experiment to give semantic description to models.  This enables model authors to describe the parts of their model application developers need to know in order to consume them.

There are 2 types of denotation : [Type Denotation](TypeDenotation.md) and [Dimension Denotation](DimensionDenotation.md).

### Type Denotation

Type Denotation is used to describe semantic information around what the inputs and outputs are.    It is stored on the TypeProto message.

#### Motivation

The motivation of such a mechanism can be illustrated via a simple example. In the the neural network SqueezeNet, it takes in an NCHW image input float[1,2,244,244] and produces a output float[1,1000,1,1]:

```
input_in_NCHW -> data_0 -> SqueezeNet() -> output_softmaxout_1
```

In order to run this model the user needs a lot of information.    In this case the user needs to know:
* the input is an image
* the image is in the format of NCHW
* the color channels are in the order of bgr
* the pixel data is 8 bit
* the pixel data is normalized as values 0-255

This proposal consists of three key components to provide all of this information: Type Denotation, [Dimension Denotation](DimensionDenotation.md), and [model metadata](MetadataProps.md).  Each of which will be discussed in detail.

#### Type Denotation Definition

To begin with, we define a set of semantic types that define what models generally consume as inputs and produce as outputs.

Specifically, in our first proposal we define the following set of standard denotations:

1. `IMAGE` describes that a type holds an image.  You can use dimension denotation to learn more about the layout of the image, and also the optional model metadata_props.
2. `AUDIO` describes that a type holds an audio clip.   
3. `TEXT` describes that a type holds a block of text.

Model authors SHOULD add type denotation to inputs and outputs for the model.

#### Denotation Propagation

Type Denotation propagation does not automatically occur.   It is used to describe the initial input or final outputs, but as data flows through the graph no inference is made to propogate if the data still holds an image (for example).   A model builder or conversion tool MAY apply propagation manually in the model if they knows that subsequent types share the same semantic denotation.

#### Denotation Verification

Denotation Verification is not enforced.   It is simply a method for model authors to indicate to model consumers what they should be passing in and what they should be expecting out.  No error is reported if you do not actually pass in an image (for example).

#### Combined With Dimension Denotation

Type denotation can be combined with the new experimental feature of [Dimension Denotation](DimensionDenotation.md).  For example if the Type Denotation is `IMAGE`, then that type SHOULD also have [Dimension Denotation](DimensionDenotation.md) stating the channel layout.

#### Model metadata_props

A model author then uses model metadata to describe information about ALL of the inputs and outputs for the model.   For example, `Image.BitmapPixelFormat`.  See the [model metadata documentation](MetadataProps.md) for details.
