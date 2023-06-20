<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Metadata

In addition to the core metadata recommendations listed in the [extensibility documentation](IR.md#optional-metadata) there is additional experimental metadata to help provide information for model inputs and outputs.

This metadata applies to all input and output tensors of a given category.  The first such category we define is: `Image`.

## Motivation

The motivation of such a mechanism is to allow model authors to convey to model consumers enough information for them to consume the model.

In the case of images there are many option for providing valid image data.  However a model which consumes images was trained with a particular set of these options which must
be used during inferencing.

The goal is this proposal is to provide enough metadata that the model consumer can perform their own featurization prior to running the model and provide a compatible input or retrieve an output and know what its format is.

## Image Category Definition

For every tensor in this model that uses [Type Denotation](TypeDenotation.md) to declare itself an `IMAGE`, you SHOULD provide metadata to assist the model consumer.  Note that any metadata provided using this mechanism is global to ALL types
with the accompanying denotation.

Keys and values are case insenstive.

Specifically, we define here the following set image metadata:

|Key|Value|Description|
|-----|----|-----------|
|`Image.BitmapPixelFormat`|__string__|Specifies the format of pixel data. Each enumeration value defines a channel ordering and bit depth. Possible values: <ul><li>`Gray8`: 1 channel image, the pixel data is 8 bpp grayscale.</li><li>`Rgb8`: 3 channel image, channel order is RGB, pixel data is 8bpp (No alpha)</li><li>`Bgr8`: 3 channel image, channel order is BGR, pixel data is 8bpp (No alpha)</li><li>`Rgba8`: 4 channel image, channel order is RGBA, pixel data is 8bpp (Straight alpha)</li><li>`Bgra8`: 4 channel image, channel order is BGRA, pixel data is 8bpp (Straight alpha)</li></ul>|
|`Image.ColorSpaceGamma`|__string__|Specifies the gamma color space used. Possible values:<ul><li>`Linear`: Linear color space, gamma == 1.0</li><li>`SRGB`: sRGB color space, gamma == 2.2</li></ul>|
|`Image.NominalPixelRange`|__string__|Specifies the range that pixel values are stored. Possible values: <ul><li>`NominalRange_0_255`:  [0...255] for 8bpp samples</li><li>`Normalized_0_1`: [0...1] pixel data is stored normalized</li><li>`Normalized_1_1`: [-1...1] pixel data is stored normalized</li><li>`NominalRange_16_235`: [16...235] for 8bpp samples</li></ul>|



