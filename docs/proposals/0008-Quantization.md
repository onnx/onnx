# Quantization Support In ONNX

## Requirements:
1.	Interoperability MUST be ensured.
ONLY widely accepted quantization schema can be standardized in ONNX. In this design, 8 bits linear (scale/zero_point) quantization will be standardized.
2.	Customized quantization schema should be allowed.
ONNX should be able to represent customized quantization schemas (the schema hasn’t been standardized in ONNX yet) with a subgraph consisting of primitive operators.
3.	All ONNX operators must define a mathematical function of the following form: 
outputs = OP(inputs, attrs)
It means the data needed for mathematical calculation defined by an op must be either an input or an attribute.
4.	Enable both static and dynamic quantization.
Quantization parameters used in defining an op will be defined as inputs/outputs. Static quantization will be a special case of dynamic one, where the quantization parameter inputs are from either initializers or constant nodes.
NOTE: as a best practice, weights in an inference model should be statically quantized.
5.	Support model verification for static quantization models. The verification includes,
a.	Same tensor should have same real-value representation.
If they use same static quantization parameters, then this can be ensured.
b.	Any other kind of quantization parameters’ value check before sending a model to a hardware vendor.

## Goals of this design:
1.	Add a small set of operators to standardize 8 bits linear (scale/zero_point) quantization.
2.	Add a small set of operators to further enable ONNX to represent other quantization schemas. 
3.	Add quantization information as model level annotation for easy model verification.

## Status:
1. To support 8 bit linear (scale/zero_point) quantization, QuantizeLinear/DequantizeLinear/QLinearConv/QLinearMatmul were added.
2. To enable other quantization schemas, ConvInteger/MatmulInteger were added.
3. More operators/quantized data types will be added as needed.
