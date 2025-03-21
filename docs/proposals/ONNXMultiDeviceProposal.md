<!--
Copyright (c) ONNX Project Contributors

SPDX-License-Identifier: Apache-2.0
-->

# ONNX Multi-Device Proposal

## Background

The recent trend in increasingly larger models has spurred an interest in distributed inference. A key performance bottleneck for inference for these large models has been the memory limits of GPUs and other accelerators as well as communication bandwidth. Thus, efficient distributed inference typically requires parallelization of the computation across multiple devices taking memory and bandwidth into account.

Our goal is to extend ONNX so that it can serve as a representation of a parallelized model. This is driven by the current state-of-the-art techniques used for distributed inference (eg., see [GSPMD: General and Scalable Parallelization for ML Computation Graphs](https://arxiv.org/pdf/2105.04663.pdf)). In particular, two techniques of interest are tensor parallelism and pipelining. In tensor parallelism (also known as horizontal parallelism or operator parallelism), the computation of a single operator (node) in the graph is parallelized across multiple devices by sharding its inputs, In pipeline parallelism, different subgraphs are assigned to different devices.


## Design

See [this commit](https://github.com/kevinch-nv/onnx/commit/07e97452096b28ba7c46fec6927d195907431e07) for the proposed additions to the ONNX spec.

The key point of this design is that all multi-device specific annotations are at the node level, and do not affect the main computational graph. This means:
 - All communication operations required for multi-device execution are implicit
 - A backend may choose to ignore the annotations if the provided configurations are either not supported or not available

### Sharding Specification

Sharding refers to modifying a tensor into multiple parts to be sent across multiple devices. A tensor may be sharded across any of its axis.

Modification of a tensor generally falls into two categories: splitting and duplication. A formal description of the sharding rules can be found [here](ShardingFormalism.md).

#### Sharding as a Split

For example, consider the following 2x2 tensor:

`[[1, 2], [3, 4]]`

If a sharding across axis 0 is specified over two devices, then:
- Device 0 will receive a tensor of shape 1x2 with data `[[1, 2]]`
- Device 1 will receive a tensor of shape 1x2 with data `[[3, 4]]`

The corresponding ShardingSpecProto for the above will look like:
```
{
    device = [0, 1]
    sharded_dim =[
        {
            axis = 0
            simple_sharding =
            [
                {
                    num_shards = 2
                }
            ]
        }
    ]
}
```

If a sharding across axis 1 is specified over two devices, then:
- Device 0 will receive a tensor of shape 2x1 with data `[[1], [3]]`
- Device 1 will receive a tensor of shape 2x1 with data `[[2], [4]]`

The corresponding ShardingSpecProto for the above will look like:
```
{
    device = [0, 1]
    sharded_dim =[
        {
            axis = 1
            simple_sharding =
            [
                {
                    num_shards = 2
                }
            ]
        }
    ]
}
```

If a sharding across axis 0 and axis 1 is specified over four devices, then:
- Device 0 will receive a tensor of shape 1x1 with data `[[1]]`
- Device 1 will receive a tensor of shape 1x1 with data `[[2]]`
- Device 2 will receive a tensor of shape 1x1 with data `[[3]]`
- Device 3 will receive a tensor of shape 1x1 with data `[[4]]`

The corresponding ShardingSpecProto for the above will look like:
```
{
    device = [0, 1, 2, 3]
    sharded_dim =[
        {
            axis = 0
            simple_sharding =
            [
                {
                    num_shards = 2
                }
            ]
        }
        {
            axis = 1
            simple_sharding =
            [
                {
                    num_shards = 2
                }
            ]
        }
    ]
}
```

A key observation in the above example shows how indexing is performed when multiple sharding axes are provided. In general, the splitting is done as:

```
split_tensors = []
for a in range(num_shards_a):
    a_width = input.shape[axis0] / num_shards_a
    a_index = a * a_width
    for b in range(num_shards_b):
        b_width = input.shape[axis1] / num_shards_b
        b_index =  b * b_width
        split = input[a_index : a_index + a_width, b_index : b_index + b_width]
        split_tensors.append(split)
```

Note that the above examples assume that the num_shards are evenly divisible into the axis that's being sharded. While this is not a hard restriction, it is up to the backend on how to handle non-evenly divisble cases.


#### Sharding as a Broadcast

There may be cases where data in a tensor must be duplicated across multiple devices to ensure that operations stay functionally correct.

For example consider replicating the same 2x2 tensor across two devices. We can do so by providing the following ShardingSpecProto:

```
{
    device = [-1] // keys into device_map
    device_map = {-1: [0, 1]}
    sharded_dim =[]
}
```

It is also possible to mix splitting and broadcasting, consider the following ShardingSpecProto:

```
{
    device = [-1, -2] // keys into device_map
    device_map = {-1: [0, 1], -2: [2, 3]}
    sharded_dim =[
        {
            axis = 0
            simple_sharding =
            [
                {
                    num_shards = 2
                }
            ]
        }
    ]
}
```

On device 0 and 1, the following 1x2 tensor is produced: `[[1,2]]`
On device 2 and 3, the following 1x2 tensor is produced: `[[2,3]]`

#### Pipeline Parallelism

Pipeline stages are represented as an optional integer value in a node's NodeConfigurationProto. It is a hint to the backend on how to run a model in a pipelined fashion across multiple devices. For example, consider the following diagram:

```
Nodes below have a pipeline id of 1:

A -> B -> C -> D -> E
                    | Nodes below have a pipeline id of 2:
                    F -> G -> H -> I -> J -> K

```

It is possible to have both pipeline and tensor parallel annotations in the same ONNX graph.

