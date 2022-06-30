import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def dropout(X, drop_probability=0.0, training_mode=False):  # type: ignore
    if training_mode is False:
        return X
    mask = np.random.uniform(0, 1.0, X.shape) >= drop_probability
    scale = (1 / (1 - drop_probability))
    return mask * X * scale


class ScaledDotProductAttention(Base):
    @staticmethod
    def export() -> None:
        bsz, tgt_len, src_len, head_dim = 4, 16, 20, 4
        drop_probability, training_mode = 0.2, 0
        query = np.random.randn(bsz, tgt_len, head_dim)
        key = np.random.randn(bsz, src_len, head_dim)
        value = np.random.randn(bsz, src_len, head_dim)

        scaling = float(head_dim) ** -0.5
        attn_output_weights = np.matmul(query, key.transpose(0, 2, 1)) / scaling

        t = np.exp(attn_output_weights)
        attention = t / np.expand_dims(np.sum(t, axis=-1), -1)
        attention = dropout(attention, drop_probability, training_mode > 0)
        output = np.matmul(attention, value).reshape(bsz, tgt_len, head_dim)

        node = onnx.helper.make_node(
            'ScaledDotProductAttention',
            inputs=['query', 'key', 'value'],
            outputs=['output'],
            training_mode=training_mode,
            dropout=drop_probability
        )

        expect(node, inputs=[query, key, value],
               outputs=[output],
               name='test_ScaledDotProductAttention')

    @staticmethod
    def export_with_attn_mask() -> None:
        bsz, tgt_len, src_len, head_dim = 4, 16, 20, 4
        drop_probability, training_mode = 0.2, 0
        query = np.random.randn(bsz, tgt_len, head_dim)
        key = np.random.randn(bsz, src_len, head_dim)
        value = np.random.randn(bsz, src_len, head_dim)
        attn_mask = np.random.randn(bsz, tgt_len, src_len)

        scaling = float(head_dim) ** -0.5
        attn_output_weights = np.matmul(query, key.transpose(0, 2, 1)) / scaling
        mask = np.ma.masked_where(attn_mask > 0, attn_output_weights)
        attn_output_weights = np.ma.filled(np.array(mask), -np.inf)

        t = np.exp(attn_output_weights)
        attention = t / np.expand_dims(np.sum(t, axis=-1), -1)
        attention = dropout(attention, drop_probability, training_mode > 0)
        output = np.matmul(attention, value).reshape(bsz, tgt_len, head_dim)

        node = onnx.helper.make_node(
            'ScaledDotProductAttention',
            inputs=['query', 'key', 'value', 'attn_mask'],
            outputs=['output'],
            training_mode=training_mode,
            dropout=drop_probability
        )

        expect(node, inputs=[query, key, value, attn_mask],
               outputs=[output],
               name='test_ScaledDotProductAttention')

    @staticmethod
    def export_out_with_attention() -> None:
        bsz, tgt_len, src_len, head_dim = 4, 16, 20, 4
        drop_probability, training_mode = 0.2, 0
        query = np.random.randn(bsz, tgt_len, head_dim)
        key = np.random.randn(bsz, src_len, head_dim)
        value = np.random.randn(bsz, src_len, head_dim)

        scaling = float(head_dim) ** -0.5
        attn_output_weights = np.matmul(query, key.transpose(0, 2, 1)) / scaling

        t = np.exp(attn_output_weights)
        attention = t / np.expand_dims(np.sum(t, axis=-1), -1)
        attention = dropout(attention, drop_probability, training_mode > 0)
        output = np.matmul(attention, value).reshape(bsz, tgt_len, head_dim)

        node = onnx.helper.make_node(
            'ScaledDotProductAttention',
            inputs=['query', 'key', 'value'],
            outputs=['output', 'attn'],
            training_mode=training_mode,
            dropout=drop_probability
        )

        expect(node, inputs=[query, key, value],
               outputs=[output, attention],
               name='test_ScaledDotProductAttention')

    @staticmethod
    def export_with_attn_mask_and_out_with_attention() -> None:
        bsz, tgt_len, src_len, head_dim = 4, 16, 20, 4
        drop_probability, training_mode = 0.2, 0
        query = np.random.randn(bsz, tgt_len, head_dim)
        key = np.random.randn(bsz, src_len, head_dim)
        value = np.random.randn(bsz, src_len, head_dim)
        attn_mask = np.random.randn(bsz, tgt_len, src_len)

        scaling = float(head_dim) ** -0.5
        attn_output_weights = np.matmul(query, key.transpose(0, 2, 1)) / scaling
        mask = np.ma.masked_where(attn_mask > 0, attn_output_weights)
        attn_output_weights = np.ma.filled(np.array(mask), -np.inf)

        t = np.exp(attn_output_weights)
        attention = t / np.expand_dims(np.sum(t, axis=-1), -1)
        attention = dropout(attention, drop_probability, training_mode > 0)
        output = np.matmul(attention, value).reshape(bsz, tgt_len, head_dim)

        node = onnx.helper.make_node(
            'ScaledDotProductAttention',
            inputs=['query', 'key', 'value', 'attn_mask'],
            outputs=['output', 'attn'],
            training_mode=training_mode,
            dropout=drop_probability
        )

        expect(node, inputs=[query, key, value, attn_mask],
               outputs=[output, attention],
               name='test_ScaledDotProductAttention')
