# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect

padval = u'0xdeadbeaf'
start_mark = b'\x02'.decode('ascii')
end_mark = b'\x03'.decode('ascii')


class Tokenizer(Base):

    @staticmethod
    def export_chartokenization_single_dim():    # type: () -> None
        input = np.array([u'abcdef', u'abcd']).astype(np.object).reshape([2])
        output = np.array([
            u'a',
            u'b',
            u'c',
            u'd',
            u'e',
            u'f',
            u'a',
            u'b',
            u'c',
            u'd',
            padval,
            padval]
        ).astype(np.object).reshape([2, 6])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=0,
            separators=[u''],
            pad_value=padval,
            mincharnum=1
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_chartokenization_single_dim')

    @staticmethod
    def export_chartokenization_single_dim_mark():    # type: () -> None
        input = np.array([u'abcdef', u'abcd']).astype(np.object).reshape([2])
        output = np.array([
            start_mark,
            u'a',
            u'b',
            u'c',
            u'd',
            u'e',
            u'f',
            end_mark,
            start_mark,
            u'a',
            u'b',
            u'c',
            u'd',
            end_mark,
            padval,
            padval]
        ).astype(np.object).reshape([2, 8])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=1,
            separators=[u''],
            pad_value=padval,
            mincharnum=1
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_chartokenization_single_dim_mark')

    @staticmethod
    def export_monday_chartokenization_single_2dim():    # type: () -> None
        input = np.array([u'abcd', u'abcd', u'abcd', u'abcdef']).astype(np.object).reshape([2, 2])
        output = np.array([
            "a",
            "b",
            "c",
            "d",
            padval,
            padval,
            "a",
            "b",
            "c",
            "d",
            padval,
            padval,
            "a",
            "b",
            "c",
            "d",
            padval,
            padval,
            "a",
            "b",
            "c",
            "d",
            "e",
            "f"
        ]).astype(np.object).reshape([2, 2, 6])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=0,
            separators=[u''],
            pad_value=padval,
            mincharnum=1
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_chartokenization_single_2dim')

    @staticmethod
    def export_chartokenization_single_2dim_mark():    # type: () -> None
        input = np.array([u'abcd', u'abcd', u'abcd', u'abcdef']).astype(np.object).reshape([2, 2])
        output = np.array([
            start_mark,
            "a",
            "b",
            "c",
            "d",
            end_mark,
            padval,
            padval,
            start_mark,
            "a",
            "b",
            "c",
            "d",
            end_mark,
            padval,
            padval,
            start_mark,
            "a",
            "b",
            "c",
            "d",
            end_mark,
            padval,
            padval,
            start_mark,
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            end_mark
        ]).astype(np.object).reshape([2, 2, 8])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=1,
            separators=[u''],
            pad_value=padval,
            mincharnum=1
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_chartokenization_single_2dim_mark')

    @staticmethod
    def export_chartokenization_empty_output():    # type: () -> None
        # Special case where empty output is produced
        # For [C] we expect [C][0] output
        input = np.array([u'', u'']).astype(np.object).reshape([2])
        output = np.array([]).astype(np.object).reshape([2, 0])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=1,
            separators=[u''],
            pad_value=padval,
            mincharnum=1
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_chartokenization_empty_output')

    @staticmethod
    def export_chartokenization_empty_output_2dim():    # type: () -> None
        # Special case where empty output is produced
        # For [N][C] we expect [N][C][0] output
        input = np.array([u'', u'', u'', u'']).astype(np.object).reshape([2, 2])
        output = np.array([]).astype(np.object).reshape([2, 2, 0])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=1,
            separators=[u''],
            pad_value=padval,
            mincharnum=1
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_chartokenization_empty_output_2dim')

    @staticmethod
    def export_separators_mark():    # type: () -> None
        input = np.array([u'Monday', u'Tuesday', u'Wednesday']).astype(np.object).reshape([3])
        output = np.array([
            start_mark,
            u'Mo',
            u'da',
            end_mark,
            start_mark,
            u'Tuesda',
            end_mark,
            padval,
            start_mark,
            u'Wed',
            u'esda',
            end_mark
        ]).astype(np.object).reshape([3, 4])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=1,
            separators=[u'y', u'n'],
            pad_value=padval,
            mincharnum=1
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_separators_mark')

    @staticmethod
    def export_separators_completematch_empty_output():    # type: () -> None
        input = np.array([u'Monday', u'Tuesday']).astype(np.object).reshape([2])
        output = np.array([]).astype(np.object).reshape([2, 0])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=1,
            separators=[u'Monday', u'Tuesday'],
            pad_value=padval,
            mincharnum=1
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_separators_completematch_empty_output')

    @staticmethod
    def export_separators_startmatch_mark():    # type: () -> None
        input = np.array([u'Monday', u'Tuesday']).astype(np.object).reshape([2])
        output = np.array([
            start_mark,
            u'onday',
            end_mark,
            start_mark,
            u'uesday',
            end_mark]
        ).astype(np.object).reshape([2, 3])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=1,
            separators=[u'M', u'T'],
            pad_value=padval,
            mincharnum=1
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_separators_startmatch_mark')

    @staticmethod
    def export_separators_endmatch_mark():    # type: () -> None
        input = np.array([u'Monday', u'Tuesday']).astype(np.object).reshape([2])
        output = np.array([
            start_mark,
            u'Monda',
            end_mark,
            start_mark,
            u'Tuesda',
            end_mark]
        ).astype(np.object).reshape([2, 3])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=1,
            separators=[u'y'],
            pad_value=padval,
            mincharnum=1
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_separators_endmatch_mark')

    @staticmethod
    def export_separators_endmatch_mark_michar():    # type: () -> None
        input = np.array([u'Monday', u'Kono']).astype(np.object).reshape([2])
        output = np.array([
            start_mark,
            u'Monda',
            end_mark,
            start_mark,
            padval,
            end_mark]
        ).astype(np.object).reshape([2, 3])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=1,
            separators=[u'y', u'o'],
            pad_value=padval,
            mincharnum=4
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_separators_endmatch_mark_michar')

    @staticmethod
    def export_separators_emptyinput_emptyoutput():    # type: () -> None
        input = np.array([u'', u'']).astype(np.object).reshape([2])
        output = np.array([]).astype(np.object).reshape([2, 0])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=1,
            separators=[u'y', u'o'],
            pad_value=padval,
            mincharnum=4
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_separators_emptyinput_emptyoutput')

    @staticmethod
    def export_separators_overlapshortfirst():    # type: () -> None
        input = np.array([u'Absurd']).astype(np.object).reshape([1])
        output = np.array([u'Ab', u'rd']).astype(np.object).reshape([1, 2])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=0,
            separators=[u'su', u'Absu'],
            pad_value=padval,
            mincharnum=1
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_separators_overlapshortfirst')

    @staticmethod
    def export_separators_overlaplongfirst():    # type: () -> None
        input = np.array([u'Absurd']).astype(np.object).reshape([1])
        output = np.array([u'rd']).astype(np.object).reshape([1, 1])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=0,
            separators=[u'Absu', u'su'],
            pad_value=padval,
            mincharnum=1
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_separators_overlaplongfirst')

    @staticmethod
    def export_separators_overlaplongfirst_repeatedshort():    # type: () -> None
        input = np.array([u'Absususurd']).astype(np.object).reshape([1])
        output = np.array([u'rd']).astype(np.object).reshape([1, 1])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=0,
            separators=[u'Absu', u'su'],
            pad_value=padval,
            mincharnum=1
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_separators_overlaplongfirst_repeatedshort')

    @staticmethod
    def export_separators_overlapping_match():    # type: () -> None
        input = np.array([u'Absususurd']).astype(np.object).reshape([1])
        output = np.array([u'Abs', u'rd']).astype(np.object).reshape([1, 2])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=0,
            separators=[u'usu', u'Absu'],
            pad_value=padval,
            mincharnum=1
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_separators_overlapping_match')

    @staticmethod
    def export_separators_common_prefix_mark():    # type: () -> None
        input = np.array([u'a;b', u'a;;;b', u'b;c;;;d;e', u'a;;b;;;c']).astype(np.object).reshape([4])
        output = np.array([
            start_mark,
            u'a',
            u'b',
            end_mark,
            padval,
            padval,
            start_mark,
            u'a',
            u'b',
            end_mark,
            padval,
            padval,
            start_mark,
            u'b',
            u'c',
            u'd',
            u'e',
            end_mark,
            start_mark,
            u'a',
            u'b',
            u'c',
            end_mark,
            padval]
        ).astype(np.object).reshape([4, 6])

        node = onnx.helper.make_node(
            'Tokenizer',
            inputs=['x'],
            outputs=['y'],
            name=None,
            doc_string=None,
            domain='ai.onnx.ml',
            mark=1,
            separators=[u';', u';;;'],
            pad_value=padval,
            mincharnum=1
        )
        expect(node, inputs=[input], outputs=[output], name='test_tokenizer_separators_common_prefix_mark')
