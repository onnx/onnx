# SPDX-License-Identifier: Apache-2.0

# (1) TODO: Fix https://github.com/onnx/onnx/issues/4101
# to solve version conversion failure from Softmax-12 to Softmax-13
# version_converter/adapters/softmax_12_13.h:56: adapt_softmax_12_13:
# Assertion `target_shape.size() != 0` failed:
# Version conversion for Softmax failed because input shape is unknown.


SKIP_VERSION_CONVERTER_MODELS = {
    "vision/classification/vgg/model/vgg19-bn-7.onnx",  # version_converter/adapters/transformers.h:30: operator(): Assertion `node->i(attr) == value` failed: Attribute spatial must have value 1
    "vision/classification/vgg/model/vgg16-bn-7.onnx",  # version_converter/adapters/transformers.h:30: operator(): Assertion `node->i(attr) == value` failed: Attribute spatial must have value 1
    "vision/object_detection_segmentation/ssd/model/ssd-12.onnx",  # (1) Softmax 12 to 13 failure
    "vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12.onnx",  # (1) Softmax 12 to 13 failure
    "vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12-int8.onnx",  # unordered_map::at: key not found
    "text/machine_comprehension/t5/model/t5-encoder-12.onnx",  # (1) Softmax 12 to 13 failure
    "text/machine_comprehension/t5/model/t5-decoder-with-lm-head-12.onnx",  # (1) Softmax 12 to 13 failure
    "text/machine_comprehension/gpt2-bs/model/gpt2-lm-head-bs-12.onnx",  # (1) Softmax 12 to 13 failure
    "text/machine_comprehension/bert-squad/model/bertsquad-12.onnx",  # (1) Softmax 12 to 13 failure
    "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-6.onnx",  # the converted opset 7 model cannot pass shape inference:
    # [ShapeInferenceError] (op_type:Mul, node name: ): [ShapeInferenceError] Inferred shape and existing shape differ in dimension 0: (64) vs (1)
}
