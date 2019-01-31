# coding: utf-8

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
            case_change_action='LOWER',
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
            case_change_action='UPPER',
            is_case_sensitive=1,
            stopwords=stopwords
        )
        expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_casesensintive_upper')

    @staticmethod
    def export_monday_empty_output():    # type: () -> None
        input = np.array([u'monday', u'monday']).astype(np.object)
        output = np.array([u'']).astype(np.object)
        stopwords = [u'monday']

        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            case_change_action='UPPER',
            is_case_sensitive=1,
            stopwords=stopwords
        )
        expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_empty_output')

    @staticmethod
    def export_monday_insensintive_upper_twodim():    # type: () -> None
        input = np.array([u'Monday', u'tuesday', u'wednesday', u'Monday', u'tuesday', u'wednesday']).astype(np.object).reshape([1, 6])

        # It does upper case cecedille, accented E
        # and german umlaut but fails
        # with german eszett
        output = np.array([u'TUESDAY', u'WEDNESDAY', u'TUESDAY', u'WEDNESDAY']).astype(np.object).reshape([1, 4])
        stopwords = [u'monday']

        node = onnx.helper.make_node(
            'StringNormalizer',
            inputs=['x'],
            outputs=['y'],
            case_change_action='UPPER',
            stopwords=stopwords
        )
        expect(node, inputs=[input], outputs=[output], name='test_strnormalizer_export_monday_insensintive_upper_twodim')
