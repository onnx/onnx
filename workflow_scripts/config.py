# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0


SKIP_VERSION_CONVERTER_MODELS = {
    "vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-6.onnx",  # the converted opset 7 model cannot pass shape inference:
    # [ShapeInferenceError] (op_type:Mul, node name: ): [ShapeInferenceError] Inferred shape and existing shape differ in dimension 0: (64) vs (1)
    "vision/classification/resnet/preproc/resnet-preproc-v1-18.onnx",  # preprocessing model contains unknown domain "local"
    "vision/classification/vgg/model/vgg16-bn-7.onnx",  # version_converter/adapters/transformers.h:30: operator(): Assertion `node->i(attr) == value` failed: Attribute spatial must have value 1
    "vision/classification/vgg/model/vgg19-bn-7.onnx",  # version_converter/adapters/transformers.h:30: operator(): Assertion `node->i(attr) == value` failed: Attribute spatial must have value 1
    "vision/object_detection_segmentation/mask-rcnn/model/MaskRCNN-12-int8.onnx",  # unordered_map::at: key not found
    "vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_10.onnx",  # starts failing after ONNX 1.15.0 due to subgraph shape inference enhancement:
    # [ShapeInferenceError] Inference error(s): (op_type:Loop, node name: generic_loop_Loop__48):
    # [ShapeInferenceError] Inference error(s): (op_type:If, node name: Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond_If__115):
    # [ShapeInferenceError] Inference error(s): (op_type:Concat, node name: Postprocessor/BatchMultiClassNonMaxSuppression/map/while/PadOrClipBoxList/cond/concat):
    # [ShapeInferenceError] All inputs to Concat must have same rank. Input 1 has rank 1 != 2
    # TODO: remove this skip. Tracking by https://github.com/onnx/models/issues/628
}
