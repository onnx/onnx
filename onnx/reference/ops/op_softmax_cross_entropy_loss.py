# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from onnx.reference.op_run import OpRun


def softmaxcrossentropy(
    x, target, weight=None, reduction="mean", ignore_index=None, get_log_prob=None, xp=None
):
    input_shape = x.shape
    if len(input_shape) == 1:
        raise RuntimeError(f"Unsupported shape {input_shape!r}.")

    # Get array namespace
    from onnx.reference.array_api_namespace import get_array_api_namespace
    if xp is None:
        xp = get_array_api_namespace(x)

    target_shape = target.shape
    N = input_shape[0]
    C = input_shape[1]

    # compute log_softmax
    max_x = xp.max(x, axis=1, keepdims=True)
    exp_x = xp.exp(x - max_x)
    p = exp_x / xp.sum(exp_x, axis=1, keepdims=True)
    inp = xp.log(p)
    log_prob = None
    if get_log_prob is True:
        # Create a copy using array API
        log_prob = inp + xp.zeros_like(inp)

    # initialize the positional weights when required
    gather_weight = None
    if weight is not None:
        # Gather weight values based on target indices using array indexing
        # Create gather_weight by indexing weight with target
        # Note: take is not in array API, so we need to use indexing
        # Flatten target to 1D, gather from weight, then reshape
        target_flat = xp.reshape(target, (-1,))
        weight_flat = weight[target_flat.astype(xp.int32)]
        gather_weight = xp.reshape(weight_flat, target.shape)
        
        if ignore_index is not None:
            gather_weight = xp.where(target == ignore_index, 0, gather_weight).astype(
                dtype=x.dtype
            )
    elif ignore_index is not None:
        gather_weight = xp.where(target == ignore_index, 0, 1).astype(dtype=x.dtype)

    # if input is 4-d and above, make it 3-d
    if len(input_shape) != 3:
        inp = xp.reshape(inp, (N, C, -1))
        target = xp.reshape(target, (N, -1))

    # Get a dimension from the reshaped input.
    # If the original input shape is [N, C, H, W],
    # the D here should be H * W because we reshape
    # [N, C, H, W] to [N, C, H * W].
    D = inp.shape[2]
    neg_gather_element_input = xp.zeros((N, D), dtype=x.dtype)
    for i in range(N):
        for d in range(D):
            if target[i, d] != ignore_index:
                neg_gather_element_input[i, d] = -inp[i, target[i, d], d]

    loss = neg_gather_element_input

    # if the input was 4-d or above reshape to the right shape
    if len(input_shape) != 3:
        loss = xp.reshape(loss, target_shape)

    # apply the weights when required
    if gather_weight is not None:
        loss = gather_weight * loss
        if reduction == "mean":
            loss = xp.sum(loss) / xp.sum(gather_weight)
            if get_log_prob is True:
                return loss, log_prob
            return (loss,)

    if reduction == "mean":
        loss = xp.mean(loss)
    elif reduction == "sum":
        loss = xp.sum(loss)

    loss = loss.astype(x.dtype)
    if get_log_prob is True:
        return loss, log_prob.astype(x.dtype)  # type: ignore[union-attr]
    return (loss,)


class SoftmaxCrossEntropyLoss(OpRun):
    def _run(self, x, target, weight=None, ignore_index=None, reduction=None):
        xp = self._get_array_api_namespace(x)
        n_outputs = len(self.onnx_node.output)
        return softmaxcrossentropy(
            x,
            target,
            weight=weight,
            reduction=reduction,
            ignore_index=ignore_index,
            get_log_prob=n_outputs == 2,
            xp=xp,
        )
