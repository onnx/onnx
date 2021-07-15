# SPDX-License-Identifier: Apache-2.0

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import onnx
import torch
from ..base import Base
from . import expect


class GridSampler(Base):
    @staticmethod
    def export_gridsampler_torch():  # type: () -> None
        node = onnx.helper.make_node(
            'GridSampler',
            inputs=['X', 'Grid'],
            outputs=['Y'],
            mode='bilinear',
            padding_mode='zeros',
            align_corners=0,
        )

        # X shape, [N, C, H, W] - [1, 1, 4, 4]
        # Grid shape, [N, H_out, W_out, 2] - [1, 6, 6, 2]
        # Y shape, [N, C, H_out, W_out] - [1, 1, 6, 6]
        X = torch.arange(3 * 3).view(1, 1, 3, 3).float()
        d = torch.linspace(-1, 1, 6)
        meshx, meshy = torch.meshgrid((d, d))
        grid = torch.stack((meshy, meshx), 2)
        Grid = grid.unsqueeze(0)
        Y = torch.nn.functional.grid_sample(X, Grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        expect(node, inputs=[X.numpy(), Grid.numpy()], outputs=[Y.numpy()],
               name='test_gridsampler_torch')
