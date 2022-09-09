# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import locale
import unicodedata
import warnings

import numpy  # type: ignore

from ..op_run import RuntimeTypeError
from ._op import OpRunUnary


class StringNormalizer(OpRunUnary):
    """
    The operator is not really threadsafe as python cannot
    play with two locales at the same time. stop words
    should not be implemented here as the tokenization
    usually happens after this steps.
    """

    def __init__(self, onnx_node, run_params):  # type: ignore
        OpRunUnary.__init__(self, onnx_node, run_params)
        self.slocale = self.locale  # type: ignore
        if self.stopwords is None:  # type: ignore
            self.raw_stops = set()
            self.stops = set()
        else:
            self.raw_stops = set(self.stopwords)  # type: ignore
            if self.case_change_action == "LOWER":  # type: ignore
                self.stops = set(w.lower() for w in self.stopwords)  # type: ignore
            elif self.case_change_action == "UPPER":  # type: ignore
                self.stops = set(w.upper() for w in self.stopwords)  # type: ignore
            else:
                self.stops = set(self.stopwords)  # type: ignore

    def _run(self, x):  # type: ignore
        """
        Normalizes strings.
        """
        res = numpy.empty(x.shape, dtype=x.dtype)
        if len(x.shape) == 2:
            for i in range(0, x.shape[1]):
                self._run_column(x[:, i], res[:, i])
        elif len(x.shape) == 1:
            self._run_column(x, res)
        else:
            raise RuntimeTypeError("x must be a matrix or a vector.")
        if len(res.shape) == 2 and res.shape[0] == 1:
            res = numpy.array([[w for w in res.tolist()[0] if len(w) > 0]])
            if res.shape[1] == 0:
                res = numpy.array([[""]])
        elif len(res.shape) == 1:
            res = numpy.array([w for w in res.tolist() if len(w) > 0])
            if len(res) == 0:
                res = numpy.array([""])
        return (res,)

    def _run_column(self, cin, cout):  # type: ignore
        """
        Normalizes string in a columns.
        """
        if locale.getlocale() != self.slocale:
            try:
                locale.setlocale(locale.LC_ALL, self.slocale)
            except locale.Error as e:
                warnings.warn(
                    f"Unknown local setting {self.slocale!r} (current: {locale.getlocale()!r}) - {e!r}."
                )
        cout[:] = cin[:]

        for i in range(0, cin.shape[0]):
            if isinstance(cout[i], float):
                # nan
                cout[i] = ""  # pragma: no cover
            else:
                cout[i] = self.strip_accents_unicode(cout[i])

        if self.is_case_sensitive and len(self.stops) > 0:  # type: ignore
            for i in range(0, cin.shape[0]):
                cout[i] = self._remove_stopwords(cout[i], self.raw_stops)

        if self.case_change_action == "LOWER":  # type: ignore
            for i in range(0, cin.shape[0]):
                cout[i] = cout[i].lower()
        elif self.case_change_action == "UPPER":  # type: ignore
            for i in range(0, cin.shape[0]):
                cout[i] = cout[i].upper()
        elif self.case_change_action != "NONE":  # type: ignore
            raise RuntimeError(
                f"Unknown option for case_change_action: {self.case_change_action!r}."  # type: ignore
            )

        if not self.is_case_sensitive and len(self.stops) > 0:  # type: ignore
            for i in range(0, cin.shape[0]):
                cout[i] = self._remove_stopwords(cout[i], self.stops)

        return cout

    def _remove_stopwords(self, text, stops):  # type: ignore
        spl = text.split(" ")
        return " ".join(filter(lambda s: s not in stops, spl))

    def strip_accents_unicode(self, s):  # type: ignore
        """
        Transforms accentuated unicode symbols into their simple counterpart.
        Source: `sklearn/feature_extraction/text.py
        <https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/
        feature_extraction/text.py#L115>`_.

        :param s: string
            The string to strip
        :return: the cleaned string
        """
        try:
            # If `s` is ASCII-compatible, then it does not contain any accented
            # characters and we can avoid an expensive list comprehension
            s.encode("ASCII", errors="strict")
            return s
        except UnicodeEncodeError:
            normalized = unicodedata.normalize("NFKD", s)
            s = "".join([c for c in normalized if not unicodedata.combining(c)])
            return s
