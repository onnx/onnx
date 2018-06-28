from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


class MVN(Base):

    @staticmethod
    def export():  # type: () -> None
        node = onnx.helper.make_node(
            'FuncMeanVarianceNormalization',
            inputs=['X'],
            outputs=['Y'],
        )

        x = np.array([[[[0.8439683], [0.5665144], [0.05836735]],
            [[0.02916367], [0.12964272], [0.5060197]],
            [[0.79538304], [0.9411346], [0.9546573]]],
            [[[0.17730942], [0.46192095], [0.26480448]],
            [[0.6746842], [0.01665257], [0.62473077]],
            [[0.9240844], [0.9722341], [0.11965699]]],
            [[[0.41356155], [0.9129373], [0.59330076]],
            [[0.81929934], [0.7862604], [0.11799799]],
            [[0.69248444], [0.54119414], [0.07513223]]]], dtype=np.float32)

        # Hard-coded pre calculated data for avoid the overhead in
        # calculating test data (with across_channels=False)
        y = np.array([[[[1.3546424], [0.330535], [-1.5450816]],
            [[-1.2106763], [-0.89259505], [0.29888144]],
            [[0.38083088], [0.81808794], [0.85865635]]],
            [[[-1.1060556], [-0.05552877], [-0.78310347]],
            [[0.8328137], [-1.250282], [0.6746787]],
            [[0.7669372], [0.9113869], [-1.6463585]]],
            [[[-0.23402767], [1.6092132], [0.429406]],
            [[1.290614], [1.1860244], [-0.92945826]],
            [[0.0721334], [-0.38174], [-1.7799333]]]], dtype=np.float32)
        expect(node, inputs=[x], outputs=[y],
               name='test_mvn')
