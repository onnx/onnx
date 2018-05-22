Overview
========

Deep learning with neural networks is accomplished through computation over dataflow graphs. Some frameworks (such as CNTK, Caffe2, Theano, and TensorFlow) make use of static graphs, while others (such as PyTorch and Chainer) use dynamic graphs. However, they all provide interfaces that make it simple for developers to construct computation graphs and runtimes that process the graphs in an optimized way. The graph serves as an Intermediate Representation (IR) that captures the specific intent of the developer's source code, and is conducive for optimization and translation to run on specific devices (CPU, GPU, FPGA, etc.).

Why a common IR?
----------------

Today, each framework has its own proprietary representation of the graph, though they all provide similar capabilities – meaning each framework is a siloed stack of API, graph, and runtime. Furthermore, frameworks are typically optimized for some characteristic, such as fast training, supporting complicated network architectures, inference on mobile devices, etc. It's up to the developer to select a framework that is optimized for one of these characteristics. Additionally, these optimizations may be better suited for particular stages of development. This leads to significant delays between research and production due to the necessity of conversion.

With the goal of democratizing AI, we envision empowering developers to select the framework that works best for their project, at any stage of development or deployment. The Open Neural Network Exchange (ONNX) format is a common IR to help establish this powerful ecosystem.

By providing a common representation of the computation graph, ONNX helps developers choose the right framework for their task, allows authors to focus on innovative enhancements, and enables hardware vendors to streamline optimizations for their platforms.

ONNX is designed to be an open format. We welcome contributions from the community and encourage everyone to adopt ONNX in their ecosystem.

Why two variants?
-----------------

The base definition of ONNX includes the necessary support for machine learning algorithms based on neural network technologies. ONNX-ML includes additional types and standard operators commonly used in classical machine learning algorithms. The two variants were created in order to explicitly recognize the desire for some frameworks to go beyond neural network algorithms in a standardized fashion, while allowing other frameworks to support only neural networks.