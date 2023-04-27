<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# Proposal - Symbolic Shape Inference And Partial Data Propagation  

*Note: This proposal was accepted and implemented in ONNX 1.10. Following PRs implemented this proposal: 3518, 3551, 3593, 3580*

## Introduction 
ONNX provides an implementation of shape inference on ONNX graphs. Shape inference is computed using the operator level shape inference functions. The inferred shape of an operator is used to get the shape information without having to launch the model in a session. Such static shape inference can be used to catch obvious errors before runtime, eliminate run-time checks which are otherwise guaranteed to pass, improve static memory planning and improve model visualization experience. For pytorch exporter and compiler-based execution providers like Nuphar, shape inference is required (rank inference is minimum requirement), and they cannot work with unknown shapes. 

This document explains the limitations of shape inference and lays out a proposal for addressing these limitations.  

## Current onnx shape inference limitations (Pre ONNX 1.10) 
Today, ONNX shape inference is not guaranteed to be complete. Wherever possible we fall back to rank inference however, there are scenarios when rank inference is not possible either. Here are the various limitations which block the completion of shape inference: 

1. Some dynamic behaviors block the flow of shape inference, and the shape inference stops. For example, reshape to a dynamically computed shape.

2. Shape inference works only with constants and simple variables. It does not support arithmetic expressions containing variables nor does it support symbol generation. For example, concatenation on tensors of shapes (5, 2) and (7, 2) can be inferred to produce a result of shape (12, 2), but concatenation on tensors of shapes (5, 2) and (N, 2) will simply produce (?, 2), where “?” represents a dimension with neither dim value nor dim param, rather than containing a representation of N+5 or generating a new symbol (M, 2). In such scenarios shape propagation stops. 

3. All operators are not required to have a shape inference implementation. When such an op is encountered the shape inference stops. There are also cases when rank inference is not done as a fallback mechanism. (Note: We are working on an ongoing basis to identify and fix such issues. The current document does not focus on this limitation)  
  

## Goals and Non-Goals 
Our **goal** is to fix the shape inference gap in scenarios where: 

* Shape computations are done in branches (refer to limitation 1) 

* Symbolic dimensions are present (refer to limitation 2) 

By fixing these gaps we aim to:   

* Unblock pytorch exporter from exporting models when exporting stops because of absence of shape information.  

* Improve static memory planning in the runtimes. 

* Enable pre-allocating output buffers outside of the runtimes so that its lifetime can be managed by the caller itself.  
  

### Non-goals  
* Add symbolic expressions to ONNX standard: This is not necessary for accomplishing our goals. There are advantages to having this capability, for example this can significantly reduce the number of symbols introduced and it can also provide more deterministic shape calculations in certain special cases. However, the tradeoff is the added complexity. So, at this point we are not considering it. This can be considered in future iterations. 

* Enable data computation and propagation for older operator sets. (details in the proposal section) 

Note: This work will benefit Nuphar as well but right now there is no plan to move Nuphar to use this solution.  
  

## Terminology 
Shape inference can be broken into 2 parts:  

* Node level shape inference: This refers to operator specific shape inference functions. They are defined with the operator schema itself.  

* Graph-level shape inference: This refers to the higher-level logic which walks through the entire graph, gets the inferred shape from node level shape inference functions and then makes decisions on merging these inferred shapes with existing shapes so that they are available for downstream nodes.  
 

## Proposal 
Extend current shape inference to allow:  
* Symbol generation and propagation 

* Partial data computation and propagation 

* Extend shape op to generate slice of the shape to facilitate simplifying shape computations.  


## Extend shape inference  

### Symbol generation and propagation 
Extend graph level shape inference to maintain a graph level view of symbols and generate new symbols where necessary. This will enable us to continue the shape inference of the downstream nodes. 

Example: 

For an op like “Concat” if its inputs have shapes “[M]” and “[N]” current shape-inference returns “[?]” where “?” is to indicate a dimension with neither dim-value nor dim-param set. Now, suppose the output X of “Concat” is input to a unary-op Op1() whose output Y is then input to another unary-op Op2() whose output is Z, etc. The shape “[?]” is propagated further. We infer that Y and Z have shape “[?]”. However, we do not infer that X, Y, and Z have the same shape because two “?” cannot be considered equal.  

Per the current proposal, “[?]” in inferred shapes will be replaced by a new unique symbol by the graph level shape inference so the downstream nodes can use the symbolic shapes to carry out shape inference. In the current example, “Concat” will produce “[?]” as the shape which will then be replaced by “[K]”, then subsequent shape inference will infer that X, Y, and Z all have the same shape “[K]”. Runtimes can use this information to reuse memory for these tensors.  


### Partial data computation and propagation 
When shape inputs are computed dynamically, shape inference post a reshape node stops. This can be prevented by making this data available to the reshape node during shape inference. We propose computation and propagation of data for operators which are used in shape computation. 

It is called “partial” data computation and propagation because this will only be done for shape computations. It is not meant to be a full-fledged kernel for the operator. For the same reasons data computations will be implemented for a limited set of operators. While we will increase the coverage in the future iterations it is important to note that for some operators like LSTM, convolution ops, pooling ops etc. data propagation function will never be added because such ops are not used in shape computations. 

The following operators will be picked in the first phase. (These operators are generally used for shape computations.) 

| Ops     | 
| --------| 
| Add     | 
| Sub     |
| Mul     |
| Cast    | 
| Concat  |
| Gather  |
| Reshape |
| Shape   |
| Slice   |
| Size    |
| Squeeze |
| UnSqueeze |

The OpSchema class will be extended to include an optional “PartialDataPropagationFunction” like the existing TypeAndShapeInferenceFunction. This function will provide data computation for the operators which will then be propagated to the downstream operators by the graph level shape inference. PartialDataPropagationFunction will be called by the graph level shape inference after TypeAndShapeInference runs for the node because the output shape is required for partial data computation.  

A new interface "DataPropagationContext” will be added to allow  PartialDataPropagationFunction to access all the information required to propagate shape data for the given node and allow writing of the computed data. 

Example: 

```
using DataPropagationFunction = std::function<void(DataPropagationContext&)>  

class OpSchema final {  

 public:  
  .  
  .  
  .  

  OpSchema& PartialDataPropagationFunction(DataPropagationFunction dataPropagationFunction)  {  
    partial_data_propagation_function_ = std::move(dataPropagationFunction);  
    return *this;  
  }  

  DataPropagationFunction GetDataPropagationFunction() const {
    return partial_data_propagation_function_ ? partial_data_propagation_function_ : dummyDataPropagator;  
  }  
} 

// Operator schema example 
ONNX_OPERATOR_SET_SCHEMA(  
    Shape,  
    13,  
    OpSchema()  
        .SetDoc(“”)  
        .Input(0, "data", "An input tensor.", "T", . . .)  
        .Output(0, "shape", "Shape of the input tensor", "T1", . . .)  
        .TypeConstraint("T", OpSchema::all_tensor_types())  
        .TypeConstraint("T1", {"tensor(int64)"})  
        .TypeAndShapeInferenceFunction([](InferenceContext& ctx) {  
        . . .  
        })  

        .PartialDataPropagationFunction([](DataPropagationContext& ctx) {  
          TensorShapeProto tp; 
          // compute output data for shape operator 
          // add computed data to DataPropagationContext for propagating it downstream  
          ctx.addOutputData(0, std::move(tp));  
        })); 
```

The symbol generation will happen at the graph level shape inference, therefore all the models (older opsets as well as the latest opset versions) can benefit from this enhancement. However, the data computation and propagation are tied to the OpScehma and will happen at node level. To begin with these functions will only be added to the latest op schemas. Older schemas can be extended to support data computation later, on a case by case basis to support some high priority scenarios. What this means is that older opset models will not benefit from shape inference improvements because of this enhancement.  
  

## Special Cases 
This section considers some edge cases and proposes a solution to handle them.  


### Broadcasting with symbolic dims 
If we have a broadcast between two unknown dimensions “M” and “N” we cannot infer that both M and N should have the same value. The runtime semantics allows for one of the two symbols to have the value 1 and the other to have a value different from 1. So, merging M and N and treating them as the same value is potentially unsound. In this case, a new symbol will be generated for the output shape and the shape inference will continue.  


### Inferred shape does not match output shape 
Inferred and existing shapes can be mismatched. Although failing shape inference in such cases seems like the correct approach it may not always be practical. By default, shape inference will fail when such a case is encountered however callers will have an option to override existing types with inferred types. When this option is enabled, shape inference will continue with the inferred type.  


### Handling symbolic dimensions with data propagation 
When the shape contains symbolic dimensions, we try and propagate them downstream, however in cases where some arithmetic operations are performed on these symbolic dims we create new symbols and propagate them instead.   


### Output shape is dependent on input data 
There are certain nodes like NonZero where the output shape depends on the input data. In this case it is not possible to infer the shape completely hence a new symbolic shape will be created using the inferred rank and shape inference will continue.  
