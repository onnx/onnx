import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def dropout(X, drop_probability=0.2, training_mode=False):  # type: ignore
    if training_mode is False:
        return X
    mask = np.random.uniform(0, 1.0, X.shape) >= drop_probability
    scale = (1 / (1 - drop_probability))
    return mask * X * scale


class MultiHeadAttention(Base):

    @staticmethod
    def export() -> None:
        query = np.random.randn(4, 16, 16)
        key = np.random.randn(4, 20, 16)
        value = np.random.randn(4, 20, 16)

        q_weight = np.random.randn(16, 16)
        k_weight = np.random.randn(16, 16)
        v_weight = np.random.randn(16, 16)

        q_bias = np.random.randn(1, 1, 16)
        k_bias = np.random.randn(1, 1, 16)
        v_bias = np.random.randn(1, 1, 16)

        out_weight = np.random.randn(16, 16)
        out_bias = np.random.randn(1, 1, 16)

        q = query.dot(q_weight) + q_bias
        k = key.dot(k_weight) + k_bias
        v = value.dot(v_weight) + v_bias

        bsz, tgt_len, embed_dim = query.shape
        bsz, src_len, embed_dim = key.shape

        num_heads = 4
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5
        q = q * scaling
        q = q.reshape(bsz * num_heads, tgt_len, head_dim)
        k = k.reshape(bsz * num_heads, head_dim, src_len)
        v = v.reshape(bsz * num_heads, src_len, head_dim)
        attn_output_weights = np.matmul(q, k) / scaling

        t = np.exp(attn_output_weights)
        attention = t / np.expand_dims(np.sum(t, axis=-1), -1)
        attention = dropout(attention)
        attn_output = np.matmul(attention, v).reshape(bsz, tgt_len, embed_dim)

        node = onnx.helper.make_node(
            'MultiHeadAttention',
            inputs=['query', 'key', 'value', 'q_weight', 'k_weight', 'v_weight',
                    'q_bias', 'k_bias', 'v_bias', 'out_weight', 'out_bias'],
            outputs=['attn_out'],
            embedding_dim=embed_dim,
            num_heads=num_heads,
        )

        expect(node, inputs=[query, key, value, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias, out_weight, out_bias],
               outputs=[attn_output],
               name='test_multiheadattention')

    @staticmethod
    def export_with_attn_mask() -> None:
        query = np.random.randn(4, 16, 16)
        key = np.random.randn(4, 20, 16)
        value = np.random.randn(4, 20, 16)

        q_weight = np.random.randn(16, 16)
        k_weight = np.random.randn(16, 16)
        v_weight = np.random.randn(16, 16)

        q_bias = np.random.randn(1, 1, 16)
        k_bias = np.random.randn(1, 1, 16)
        v_bias = np.random.randn(1, 1, 16)

        out_weight = np.random.randn(16, 16)
        out_bias = np.random.randn(1, 1, 16)

        q = query.dot(q_weight) + q_bias
        k = key.dot(k_weight) + k_bias
        v = value.dot(v_weight) + v_bias

        bsz, tgt_len, embed_dim = query.shape
        bsz, src_len, embed_dim = key.shape

        num_heads = 4
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5
        q = q * scaling
        q = q.reshape(bsz * num_heads, tgt_len, head_dim)
        k = k.reshape(bsz * num_heads, head_dim, src_len)
        v = v.reshape(bsz * num_heads, src_len, head_dim)
        attn_output_weights = np.matmul(q, k) / scaling

        attn_mask = np.random.randn(bsz * num_heads, tgt_len, src_len)
        attn_output_weights += attn_mask

        t = np.exp(attn_output_weights)
        attention = t / np.expand_dims(np.sum(t, axis=-1), -1)
        attention = dropout(attention)
        attn_output = np.matmul(attention, v).reshape(bsz, tgt_len, embed_dim)

        node = onnx.helper.make_node(
            'MultiHeadAttention',
            inputs=['query', 'key', 'value', 'q_weight', 'k_weight', 'v_weight',
                    'q_bias', 'k_bias', 'v_bias', 'out_weight', 'out_bias', 'attn_mask'],
            outputs=['attn_out'],
            embedding_dim=embed_dim,
            num_heads=num_heads,
        )

        expect(node, inputs=[query, key, value, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias, out_weight, out_bias, attn_mask],
               outputs=[attn_output],
               name='test_multiheadattention')

    @staticmethod
    def export_with_padding_mask() -> None:
        query = np.random.randn(4, 16, 16)
        key = np.random.randn(4, 20, 16)
        value = np.random.randn(4, 20, 16)

        q_weight = np.random.randn(16, 16)
        k_weight = np.random.randn(16, 16)
        v_weight = np.random.randn(16, 16)

        q_bias = np.random.randn(1, 1, 16)
        k_bias = np.random.randn(1, 1, 16)
        v_bias = np.random.randn(1, 1, 16)

        out_weight = np.random.randn(16, 16)
        out_bias = np.random.randn(1, 1, 16)

        padding_mask = np.random.randn(4, 20) > 0

        q = query.dot(q_weight) + q_bias
        k = key.dot(k_weight) + k_bias
        v = value.dot(v_weight) + v_bias

        bsz, tgt_len, embed_dim = query.shape
        bsz, src_len, embed_dim = key.shape

        num_heads = 4
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5
        q = q * scaling
        q = q.reshape(bsz * num_heads, tgt_len, head_dim)
        k = k.reshape(bsz * num_heads, head_dim, src_len)
        v = v.reshape(bsz * num_heads, src_len, head_dim)
        attn_output_weights = np.matmul(q, k) / scaling
        attn_output_weights = attn_output_weights.reshape(
            bsz, num_heads, tgt_len, src_len)
        mask = np.concatenate([padding_mask] * num_heads
                              * tgt_len).reshape(bsz, num_heads, tgt_len, src_len)
        mask = np.ma.masked_where(mask, attn_output_weights)
        mask = np.ma.filled(np.array(mask), -np.inf)
        attn_output_weights = attn_output_weights.reshape(
            bsz * num_heads, tgt_len, src_len)
        t = np.exp(attn_output_weights)
        attention = t / np.expand_dims(np.sum(t, axis=-1), -1)
        attention = dropout(attention)
        attn_output = np.matmul(attention, v).reshape(bsz, tgt_len, embed_dim)

        node = onnx.helper.make_node(
            'MultiHeadAttention',
            inputs=['query', 'key', 'value', 'q_weight', 'k_weight', 'v_weight',
                    'q_bias', 'k_bias', 'v_bias', 'out_weight', 'out_bias', 'padding_mask'],
            outputs=['attn_out'],
            embedding_dim=embed_dim,
            num_heads=num_heads,
        )

        expect(node, inputs=[query, key, value, q_weight, k_weight, v_weight, q_bias, k_bias, v_bias, out_weight, out_bias, padding_mask],
               outputs=[attn_output],
               name='test_multiheadattention')
