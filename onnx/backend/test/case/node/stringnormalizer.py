from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class StringNormalizer(Base):

    @staticmethod
    def export_nostopwords_nochangecase():    # type: () -> None
        input = np.array([u'monday', u'tuesday']).astype(np.object)
        output = input

        # No stopwords. This is a NOOP
        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            casechangeaction='NONE',
            is_case_sensitive=1,
        )
        expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_nostopwords_nochangecase')

    @staticmethod
    def export_monday_casesensintive_nochangecase():    # type: () -> None
        input = np.array([u'monday', u'tuesday', u'wednesday', u'thursday']).astype(np.object)
        output = np.array([u'tuesday', u'wednesday', u'thursday']).astype(np.object)
        stopwords = [u'monday']

        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            casechangeaction='NONE',
            is_case_sensitive=1,
            stopwords=stopwords
        )
        expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_casesensintive_nochangecase')

    @staticmethod
    def export_monday_casesensintive_lower():    # type: () -> None
        input = np.array([u'monday', u'tuesday', u'wednesday', u'thursday']).astype(np.object)
        output = np.array([u'tuesday', u'wednesday', u'thursday']).astype(np.object)
        stopwords = [u'monday']

        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            casechangeaction='LOWER',
            is_case_sensitive=1,
            stopwords=stopwords
        )
        expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_casesensintive_lower')

    @staticmethod
    def export_monday_casesensintive_upper():    # type: () -> None
        input = np.array([u'monday', u'tuesday', u'wednesday', u'thursday']).astype(np.object)
        output = np.array([u'TUESDAY', u'WEDNESDAY', u'THURSDAY']).astype(np.object)
        stopwords = [u'monday']

        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            casechangeaction='UPPER',
            is_case_sensitive=1,
            stopwords=stopwords
        )
        expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_casesensintive_upper')

    @staticmethod
    def export_monday_empty_output():    # type: () -> None
        input = np.array([u'monday', u'monday']).astype(np.object)
        output = np.array([]).astype(np.object)
        stopwords = [u'monday']

        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            casechangeaction='UPPER',
            is_case_sensitive=1,
            stopwords=stopwords
        )
        expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_empty_output')

    # mypy 0,600 and 0.650 fails to handle unicode strings
    # https://github.com/python/mypy/issues/6198
    # The issue occurs even if we comment out the code!
    # So we have to remove French, Russian, German and Chinese strings from the below two tests
    @staticmethod
    def export_monday_casesensintive_upper_langmix():    # type: () -> None
        input = np.array([u'monday', u'tuesday', u'wednesday']).astype(np.object)

        # It does upper case cecedille, accented E
        # and german umlaut but fails
        # with german eszett
        output = np.array([u'TUESDAY', u'WEDNESDAY']).astype(np.object)
        stopwords = [u'monday']

        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            casechangeaction='UPPER',
            is_case_sensitive=1,
            stopwords=stopwords
        )
        expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_casesensintive_upper_langmix')

    @staticmethod
    def export_monday_insensintive_upper_langmix():    # type: () -> None
        input = np.array([u'Monday', u'tuesday', u'wednesday']).astype(np.object)

        # It does upper case cecedille, accented E
        # and german umlaut but fails
        # with german eszett
        output = np.array([u'TUESDAY', u'WEDNESDAY']).astype(np.object)
        stopwords = [u'monday']

        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            casechangeaction='UPPER',
            is_case_sensitive=0,
            stopwords=stopwords
        )
        expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_insensintive_upper_langmix')

    @staticmethod
    def export_monday_insensintive_upper_twodim():    # type: () -> None
        input = np.array([u'Monday', u'tuesday', u'wednesday', u'Monday', u'tuesday', u'wednesday']).astype(np.object).reshape([2, 3])

        # It does upper case cecedille, accented E
        # and german umlaut but fails
        # with german eszett
        output = np.array([u'TUESDAY', u'WEDNESDAY', u'TUESDAY', u'WEDNESDAY']).astype(np.object).reshape([2, 2])
        stopwords = [u'monday']

        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            casechangeaction='UPPER',
            is_case_sensitive=0,
            stopwords=stopwords
        )
        expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_insensintive_upper_twodim')
