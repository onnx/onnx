import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect
from torch.nn import functional as F
import torch


def dropout(X, drop_probability=0.5, training_mode=False):
    if training_mode is False:
            return X
    mask = np.random.uniform(0, 1.0, X.shape) >= drop_probability
    scale = (1 / (1 - drop_probability))
    return mask * X * scale


class MultiHeadAttention(Base):

    @staticmethod
    def export() -> None:
        node = onnx.helper.make_node(
            'MultiHeadAttention',
            inputs=[
                'query', 'key', 'value', \
                'q_weight', 'k_weight', 'v_weight', \
                'q_bias', 'k_bias', 'v_bias', \
                'out_weight', 'out_bias', \
                'padding_mask', 'attn_mask'
                ],
            outputs=['attn_out'],
        )
        query = np.array([4, 16, 16]).astype(np.float32)
        key = np.array([4, 16, 20]).astype(np.float32)
        value = np.array([4, 16, 20]).astype(np.float32)

        q_weight = np.array([16, 16]).astype(np.float32)
        k_weight = np.array([20, 16]).astype(np.float32)
        v_weight = np.array([20, 16]).astype(np.float32)

        q_bias = np.array([1, 1, 16]).astype(np.float32)
        k_bias = np.array([1, 1, 16]).astype(np.float32)
        v_bias = np.array([1, 1, 16]).astype(np.float32)

        out_weight = np.array([16, 16]).astype(np.float32)
        out_bias = np.array([1, 1, 16]).astype(np.float32)

        padding_mask = np.array([4, 20]).astype(np.float32)
        attn_mask = np.array([4 * 5, 20, 16]).astype(np.float32)

        q = query.dot(q_weight) + q_bias
        k = key.dot(k_weight) + k_bias
        v = value.dot(v_weight) + v_bias

        bsz, embed_dim, tgt_len = query.size()
        bsz, embed_dim, src_len = key.size()

        num_heads = 5
        head_dim = embed_dim // num_heads 
        scaling = float(head_dim) ** -0.5
        q = q * scaling
        q = q.reshape(tgt_len, bsz*num_heads, head_dim).transpose(0, 1)
        k = k.reshape(src_len, bsz*num_heads, head_dim).transpose(0, 1)
        v = v.reshape(src_len, bsz*num_heads, head_dim).transpose(0, 1)
        attn_output_weights = q.dot(k.transpose(1, 2)) / scaling

        t = np.exp(attn_output_weights)
        attention = t / np.sum(t, axis=-1)
        attention = dropout(attention)
        attn_output = attention.dot(v).reshape(bsz, embed_dim, tgt_len)

        expect(node, inputs=[query, key, value, q_weight, k_weight, v_weight, \
            q_bias, k_bias, v_bias, out_weight, out_bias, padding_mask, attn_mask], 
            outputs=[attn_output],
            name='test_multiheadattention')

    @staticmethod
    def export_with_attn_mask() -> None:
        node = onnx.helper.make_node(
            'MultiHeadAttention',
            inputs=[
                'query', 'key', 'value', \
                'q_weight', 'k_weight', 'v_weight', \
                'q_bias', 'k_bias', 'v_bias', \
                'out_weight', 'out_bias', \
                'padding_mask', 'attn_mask'
                ],
            outputs=['attn_out'],
        )
        query = np.array([4, 16, 16]).astype(np.float32)
        key = np.array([4, 16, 20]).astype(np.float32)
        value = np.array([4, 16, 20]).astype(np.float32)

        q_weight = np.array([16, 16]).astype(np.float32)
        k_weight = np.array([20, 16]).astype(np.float32)
        v_weight = np.array([20, 16]).astype(np.float32)

        q_bias = np.array([1, 1, 16]).astype(np.float32)
        k_bias = np.array([1, 1, 16]).astype(np.float32)
        v_bias = np.array([1, 1, 16]).astype(np.float32)

        out_weight = np.array([16, 16]).astype(np.float32)
        out_bias = np.array([1, 1, 16]).astype(np.float32)

        padding_mask = np.array([4, 20]).astype(np.float32)
        attn_mask = np.array([4 * 5, 20, 16]).astype(np.float32)

        q = query.dot(q_weight) + q_bias
        k = key.dot(k_weight) + k_bias
        v = value.dot(v_weight) + v_bias

        bsz, embed_dim, tgt_len = query.size()
        bsz, embed_dim, src_len = key.size()

        num_heads = 5
        head_dim = embed_dim // num_heads 
        scaling = float(head_dim) ** -0.5
        q = q * scaling
        q = q.reshape(tgt_len, bsz*num_heads, head_dim).transpose(0, 1)
        k = k.reshape(src_len, bsz*num_heads, head_dim).transpose(0, 1)
        v = v.reshape(src_len, bsz*num_heads, head_dim).transpose(0, 1)
        attn_output_weights = q.dot(k.transpose(1, 2)) / scaling

        if attn_mask is not None:
            attn_mask = np.array([4 * 5, tgt_len, src_len]).astype(np.float32)
            attn_output_weights += attn_mask

        t = np.exp(attn_output_weights)
        attention = t / np.sum(t, axis=-1)
        attention = dropout(attention)
        attn_output = attention.dot(v).reshape(bsz, embed_dim, tgt_len)

        expect(node, inputs=[query, key, value, q_weight, k_weight, v_weight, \
            q_bias, k_bias, v_bias, out_weight, out_bias, padding_mask, attn_mask], 
            outputs=[attn_output],
            name='test_multiheadattention')

    @staticmethod
    def export_with_padding_mask() -> None:
        node = onnx.helper.make_node(
            'MultiHeadAttention',
            inputs=[
                'query', 'key', 'value', \
                'q_weight', 'k_weight', 'v_weight', \
                'q_bias', 'k_bias', 'v_bias', \
                'out_weight', 'out_bias', \
                'padding_mask', 'attn_mask'
                ],
            outputs=['attn_out'],
        )
        query = np.array([4, 16, 16]).astype(np.float32)
        key = np.array([4, 16, 20]).astype(np.float32)
        value = np.array([4, 16, 20]).astype(np.float32)

        q_weight = np.array([16, 16]).astype(np.float32)
        k_weight = np.array([20, 16]).astype(np.float32)
        v_weight = np.array([20, 16]).astype(np.float32)

        q_bias = np.array([1, 1, 16]).astype(np.float32)
        k_bias = np.array([1, 1, 16]).astype(np.float32)
        v_bias = np.array([1, 1, 16]).astype(np.float32)

        out_weight = np.array([16, 16]).astype(np.float32)
        out_bias = np.array([1, 1, 16]).astype(np.float32)

        padding_mask = np.array([4, 20]).astype(np.float32)
        attn_mask = np.array([4 * 5, 20, 16]).astype(np.float32)

        q = query.dot(q_weight) + q_bias
        k = key.dot(k_weight) + k_bias
        v = value.dot(v_weight) + v_bias

        bsz, embed_dim, tgt_len = query.size()
        bsz, embed_dim, src_len = key.size()

        num_heads = 5
        head_dim = embed_dim // num_heads 
        scaling = float(head_dim) ** -0.5
        q = q * scaling
        q = q.reshape(tgt_len, bsz*num_heads, head_dim).transpose(0, 1)
        k = k.reshape(src_len, bsz*num_heads, head_dim).transpose(0, 1)
        v = v.reshape(src_len, bsz*num_heads, head_dim).transpose(0, 1)
        attn_output_weights = q.dot(k.transpose(1, 2)) / scaling

        if padding_mask is not None:
            np.expand_dims
            attn_output_weights = attn_output_weights.reshape(bsz, num_heads, tgt_len, src_len)
            mask = np.ma.masked_where(np.expand_dims(np.expand_dims(padding_mask, axis=0), axis=0), attn_output_weights)
            mask = np.ma.filled(mask, -np.inf)
            attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)  

        t = np.exp(attn_output_weights)
        attention = t / np.sum(t, axis=-1)
        attention = dropout(attention)
        attn_output = attention.dot(v).reshape(bsz, embed_dim, tgt_len)

        expect(node, inputs=[query, key, value, q_weight, k_weight, v_weight, \
            q_bias, k_bias, v_bias, out_weight, out_bias, padding_mask, attn_mask], 
            outputs=[attn_output],
            name='test_multiheadattention')
