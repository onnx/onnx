from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class StringConcat(Base):

    @staticmethod
    def export_stringconcat_default_separator():    # type: () -> None
        input = np.array([u'a', u'b', u'c']).astype(np.object)
        output = np.array([u'a b c']).astype(np.object)

        # No stopwords. This is a NOOP
        node = onnx.helper.make_node(
            'StringConcat',
            inputs=['input'],
            outputs=['output'],
        )
        expect(node, inputs=[input], outputs=[output], name='test_stringconcat_default_separator')

    @staticmethod
    def export_stringconcat_with_separator():    # type: () -> None
        input = np.array([u'a', u'b', u'c']).astype(np.object)
        output = np.array([u'a:b:c']).astype(np.object)

        node = onnx.helper.make_node(
            'StringConcat',
            inputs=['input'],
            outputs=['output'],
            separator=':'
        )
        expect(node, inputs=[input], outputs=[output], name='test_stringconcat_with_separator')
