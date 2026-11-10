```toml
name = "Scan"
domain = "ai.onnx"
shape_inference = "defs::controlflow::utils::ScanInferenceFunction"

[attributes.body]
type = "GRAPH"
desc = "The graph run each iteration. It has N+M inputs: (loop state variables..., scan_input_elts...). It has N+K outputs: (loop state variables..., scan_output_elts...). Each scan_output is created by concatenating the value of the specified scan_output_elt value at the end of each iteration of the loop. It is an error if the dimensions of these values change across loop iterations."
required = true

[attributes.num_scan_inputs]
type = "INT"
desc = "An attribute specifying the number of scan_inputs M. "
required = true

[attributes.scan_input_axes]
type = "INTS"
desc = "An optional list of M flags. The i-th element of the list specifies the axis to be scanned (the sequence axis) for the i-th scan_input. If omitted, 0 will be used as the scan axis for every scan_input. Negative value for an axis means counting dimensions from the back. Accepted range is [-r, r-1] where r = rank(input)."

[attributes.scan_input_directions]
type = "INTS"
desc = "An optional list of M flags. The i-th element of the list specifies the direction to be scanned for the i-th scan_input tensor: 0 indicates forward direction and 1 indicates reverse direction. If omitted, all scan_input tensors will be scanned in the forward direction."

[attributes.scan_output_axes]
type = "INTS"
desc = "An optional list of K flags. The i-th element of the list specifies the axis for the i-th scan_output. The scan outputs are accumulated along the specified axis. If omitted, 0 will be used as the scan axis for every scan_output. Negative value for an axis means counting dimensions from the back. Accepted range is [-r, r-1]."

[attributes.scan_output_directions]
type = "INTS"
desc = "An optional list of K flags, one for each scan_output. The i-th element of the list specifies whether the i-th scan_output should be constructed by appending or prepending a new value in each iteration: 0 indicates appending and 1 indicates prepending. If omitted, all scan_output tensors will be produced by appending a value in each iteration."
```
