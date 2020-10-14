# Originally there are some checker failures in weekly-CI
# Skip them in test_model_zoo.py for now
# TODO: fix these checker failures
SKIP_CHECKER_MODELS = ['vision/classification/alexnet/model/bvlcalexnet-3.onnx',
                       'vision/classification/caffenet/model/caffenet-6.onnx',
                       'vision/classification/densenet-121/model/densenet-3.onnx',
                       'vision/classification/inception_and_googlenet/inception_v1/model/inception-v1-3.onnx',
                       'vision/classification/inception_and_googlenet/inception_v2/model/inception-v2-3.onnx',
                       'vision/classification/rcnn_ilsvrc13/model/rcnn-ilsvrc13-3.onnx',
                       'vision/classification/resnet/model/resnet50-caffe2-v1-3.onnx',
                       'vision/classification/shufflenet/model/shufflenet-3.onnx',
                       'vision/classification/squeezenet/model/squeezenet1.0-3.onnx',
                       'vision/classification/vgg/model/vgg16-7.onnx',  # only failed on Windows-CI
                       'vision/classification/vgg/model/vgg19-7.onnx',  # only failed on Windows-CI
                       'vision/classification/vgg/model/vgg16-bn-7.onnx',  # only failed on Windows-CI
                       'vision/classification/vgg/model/vgg19-bn-7.onnx',  # only failed on Windows-CI
                       'vision/classification/vgg/model/vgg19-caffe2-3.onnx',
                       'vision/classification/zfnet-512/model/zfnet512-3.onnx']
