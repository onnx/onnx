<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Type Denotation

Type Denotation is used to describe semantic information around what the inputs and outputs are.    It is stored on the TypeProto message.

## Motivation

The motivation of such a mechanism can be illustrated via a simple example. In the neural network SqueezeNet, it takes in an NCHW image input float[1,3,244,244] and produces a output float[1,1000,1,1]:

```
input_in_NCHW -> data_0 -> SqueezeNet() -> output_softmaxout_1
```

In order to run this model the user needs a lot of information.    In this case the user needs to know:
* the input is an image
* the image is in the format of NCHW
* the color channels are in the order of bgr
* the pixel data is 8 bit
* the pixel data is normalized as values 0-255

This proposal consists of three key components to provide all of this information:
* Type Denotation,
* [Dimension Denotation](DimensionDenotation.md),
* [Model Metadata](MetadataProps.md).

## Type Denotation Definition

To begin with, we define a set of semantic types that define what models generally consume as inputs and produce as outputs.

Specifically, in our first proposal we define the following set of standard denotations:

0. `TENSOR` describes that a type holds a generic tensor using the standard TypeProto message.
1. `IMAGE` describes that a type holds an image.  You can use dimension denotation to learn more about the layout of the image, and also the optional model metadata_props.
2. `AUDIO` describes that a type holds an audio clip.
3. `TEXT` describes that a type holds a block of text.

Model authors SHOULD add type denotation to inputs and outputs for the model as appropriate.

## An Example with input IMAGE

Let's use the same SqueezeNet example from above and show everything to properly annotate the model:

* First set the TypeProto.denotation =`IMAGE` for the ValueInfoProto `data_0`
* Because it's an image, the model consumer now knows to go look for image metadata on the model
* Then include 3 metadata strings on ModelProto.metadata_props
	* `Image.BitmapPixelFormat` = `Bgr8`
	* `Image.ColorSpaceGamma` = `SRGB`
	* `Image.NominalPixelRange` = `NominalRange_0_255`
* For that same ValueInfoProto, make sure to also use Dimension Denotations to denote NCHW
	* TensorShapeProto.Dimension[0].denotation = `DATA_BATCH`
	* TensorShapeProto.Dimension[1].denotation = `DATA_CHANNEL`
	* TensorShapeProto.Dimension[2].denotation = `DATA_FEATURE`
	* TensorShapeProto.Dimension[3].denotation = `DATA_FEATURE`

Now there is enough information in the model to know everything about how to pass a correct image into the model.
