# Copyright (c) ONNX Project Contributors
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np

from onnx.reference.op_run import OpRun


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Compute softmax along the specified axis."""
    x_max = np.max(x, axis=axis, keepdims=True)
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    return tmp / s


def _compute_flex_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    scale: float | None = None,
    score_mod: callable | None = None,
    mask_mod: callable | None = None,
    prob_mod: callable | None = None,
) -> np.ndarray:
    """
    Compute flexible attention with custom modification functions.
    
    Args:
        Q: Query tensor with shape (batch_size, num_heads, seq_length_q, head_size)
        K: Key tensor with shape (batch_size, num_heads, seq_length_k, head_size)
        V: Value tensor with shape (batch_size, num_heads, seq_length_k, head_size_v)
        scale: Scaling factor for attention scores. If None, defaults to 1/sqrt(head_size)
        score_mod: Optional function to modify attention scores after Q@K^T
        mask_mod: Optional function to apply masking to attention scores
        prob_mod: Optional function to modify attention probabilities after softmax
        
    Returns:
        Output tensor with shape (batch_size, num_heads, seq_length_q, head_size_v)
    """
    assert len(Q.shape) == len(K.shape) == len(V.shape) == 4, "Q, K, V must be 4D tensors"
    
    # Calculate scaling factor if not provided
    if scale is None:
        head_size = Q.shape[3]
        scale = 1 / np.sqrt(head_size)
    
    # Step 1: Compute Q @ K^T
    # Q: (batch_size, num_heads, seq_length_q, head_size)
    # K^T: (batch_size, num_heads, head_size, seq_length_k)
    # Result: (batch_size, num_heads, seq_length_q, seq_length_k)
    K_transposed = np.transpose(K, (0, 1, 3, 2))
    attention_scores = np.matmul(Q, K_transposed)
    
    # Step 2: Apply scaling
    attention_scores = attention_scores * scale
    
    # Step 3: Apply score_mod if provided
    if score_mod is not None:
        attention_scores = score_mod(attention_scores)
    
    # Step 4: Apply mask_mod if provided
    if mask_mod is not None:
        attention_scores = mask_mod(attention_scores)
    
    # Step 5: Apply Softmax
    attention_probs = _softmax(attention_scores, axis=-1)
    
    # Step 6: Apply prob_mod if provided
    if prob_mod is not None:
        attention_probs = prob_mod(attention_probs)
    
    # Step 7: Compute output = probs @ V
    # probs: (batch_size, num_heads, seq_length_q, seq_length_k)
    # V: (batch_size, num_heads, seq_length_k, head_size_v)
    # Result: (batch_size, num_heads, seq_length_q, head_size_v)
    output = np.matmul(attention_probs, V)
    
    return output


class FlexAttention(OpRun):
    """
    Reference implementation for FlexAttention operator.
    
    This is a simplified reference implementation that demonstrates the basic
    computation pattern. Note that graph attributes (score_mod, mask_mod, prob_mod)
    require special handling that is not fully implemented in this reference.
    
    For a complete implementation with graph attribute support, backend-specific
    implementations should inline the provided subgraphs at the appropriate stages.
    """
    
    op_domain = "ai.onnx.preview"
    
    def _run(self, Q, K, V, scale=None, score_mod=None, mask_mod=None, prob_mod=None):
        """
        Run FlexAttention operation.
        
        Args:
            Q: Query tensor
            K: Key tensor  
            V: Value tensor
            scale: Optional scaling factor
            score_mod: Optional graph for modifying attention scores (not fully supported in reference)
            mask_mod: Optional graph for masking attention scores (not fully supported in reference)
            prob_mod: Optional graph for modifying attention probabilities (not fully supported in reference)
            
        Returns:
            Output tensor
        """
        # Note: This reference implementation doesn't execute the graph attributes
        # (score_mod, mask_mod, prob_mod) as that would require a full graph executor.
        # Backend implementations should inline these subgraphs properly.
        
        # For the reference implementation, we compute basic attention
        output = _compute_flex_attention(
            Q, K, V, scale=scale,
            # Graph attributes would be executed here in a full implementation
            score_mod=None,
            mask_mod=None,
            prob_mod=None
        )
        
        return (output,)
