# SPDX-License-Identifier: Apache-2.0
# pylint: disable=W0221

import numpy as np  # type: ignore

from ..op_run import OpRun


class TfIdfVectorizer(OpRun):
    def _run(
        self,
        X,
        max_gram_length=None,
        max_skip_count=None,
        min_gram_length=None,
        mode=None,
        ngram_counts=None,
        ngram_indexes=None,
        pool_int64s=None,
        pool_strings=None,
        weights=None,
    ):
        pass
