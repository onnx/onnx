from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
from ..base import Base
from . import expect


def sequence_insert_reference_implementation(sequence, tensor, position=np.array(-1)):  # type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
    insert_position = position[0]
    num_seq_elements = len(sequence)
    # Default position of insertion is at the end of the sequence.
    # In these cases, insert_position will be between [-len(sequence), len(sequence)]
    seq_split_1 = sequence[:insert_position]
    seq_split_2 = sequence[insert_position:]
    inserted = np.vstack(seq_split_1, np.array(tensor), seq_split_2, axis=0)
    return inserted


class SequenceInsert(Base):

    @staticmethod
    def export():  # type: () -> None
        test_cases = {
            'default': ...,
            'at_front': ...
        }
        data = ...

        for test_name, shape in test_cases.items():
            node = onnx.helper.make_node(
                'SequenceInsert',
                inputs=['sequence', 'tensor'],
                outputs=['output_sequence']
            )

            inserted = sequence_insert_reference_implementation(sequence, tensor)

            expect(node, inputs=[sequence, tensor], outputs=[inserted],
                   name='test_seq_insert_' + test_name)
