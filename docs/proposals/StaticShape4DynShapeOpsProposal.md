# Support static shapes for operators with dynamically shaped output  

## Summary 

We propose to implement a flag/option that replaces a dynamically shaped output with a statically defined shape (thus padded) output. 

An example: Consider `NonMaxSuppression` operator 

One of the NMS’s inputs is  `max_output_boxes_per_class` which is an input scalar that defines the max number of 
entries in the output: `selected_indices`. 
Hence the size of the output is dynamic, the length of this tensor can be 

`[0: max_output_boxes_per_class]`. 

Hardware accelerators might have an issue with this definition as it is much more convenient and efficient 
to know the size of the output tensor statically.  In this case, our suggestion is to allow `max_output_boxes_per_class`
to have a static shape, thus its actual meaning would be `number_of_output_boxes_per_class`. 

## Implementation details: 

The suggested change requires adding a flag to all relevant operators. Currently we identified the following: 

* `NonMaxSuppression` 
* `NonZero` 
* `Unique` 
* `Resize` 
* `Shape` 
* `...` 

Since we aim to select a fixed shape for an inherently dynamically-shaped output tensor, we must pad the 
remaining unused space with some value. This value would be a key to other operators which uses this padded output. 

## Discussion 

Some operators such as shape can be easily solved if the input is fixed and we use shape inference. Should we add them to the list?  

## Implementation Considerations 

 

 

 

 
