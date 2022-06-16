# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from onnx import checker, helper, numpy_helper, TensorProto, NodeProto, GraphProto, ValueInfoProto, ModelProto, ONNX_ML, SparseTensorProto, TypeProto
from onnx.defs import ONNX_DOMAIN, ONNX_ML_DOMAIN, AI_ONNX_PREVIEW_TRAINING_DOMAIN
from onnx.helper import make_node, make_tensor, make_tensor_value_info, make_empty_tensor_value_info, make_opsetid, make_tensor_sequence_value_info
from typing import Sequence, Union, Tuple, Type, List, Any, Optional
import onnx.shape_inference
import unittest
import os
import numpy as np  # type: ignore


class TestDataPropagation(unittest.TestCase):
    def _make_graph(self,
                    seed_values: Sequence[Union[str, Tuple[str, TensorProto.DataType, Any]]],
                    nodes: List[NodeProto],
                    value_info: List[ValueInfoProto],
                    initializer: Optional[Sequence[TensorProto]] = None
                    ) -> GraphProto:
        if initializer is None:
            initializer = []
        names_in_initializer = {x.name for x in initializer}
        input_value_infos = []
        # If the starting values are not also initializers,
        # introduce the starting values as the output of reshape,
        # so that the sizes are guaranteed to be unknown
        for seed_value in seed_values:
            if isinstance(seed_value, tuple):
                seed_name, proto_type = seed_value[:2]
                seed_value_info = make_tensor_value_info(*seed_value)
            else:
                seed_name, proto_type = seed_value, TensorProto.UNDEFINED
                seed_value_info = make_empty_tensor_value_info(seed_value)
            if seed_name in names_in_initializer:
                input_value_infos.append(seed_value_info)
            else:
                value_info.append(seed_value_info)
                input_value_infos.append(make_tensor_value_info('SEED_' + seed_name, proto_type, ()))
                input_value_infos.append(make_tensor_value_info('UNKNOWN_SHAPE_' + seed_name, TensorProto.INT64, ()))
                nodes[:0] = [make_node("Reshape", ['SEED_' + seed_name, 'UNKNOWN_SHAPE_' + seed_name], [seed_name])]
        return helper.make_graph(nodes, "test", input_value_infos, [], initializer=initializer, value_info=value_info)

    def _inferred(self, graph: GraphProto, **kwargs: Any) -> ModelProto:
        kwargs['producer_name'] = 'onnx-test'
        data_prop = kwargs.pop('data_prop', False)
        orig_model = helper.make_model(graph, **kwargs)
        inferred_model = onnx.shape_inference.infer_shapes(orig_model, strict_mode=True, data_prop=data_prop)
        checker.check_model(inferred_model)
        return inferred_model

    def _assert_inferred(self, graph: GraphProto, vis: List[ValueInfoProto], **kwargs: Any) -> None:
        names_in_vis = {x.name for x in vis}
        vis = list(x for x in graph.value_info if x.name not in names_in_vis) + vis
        inferred_model = self._inferred(graph, **kwargs)
        inferred_vis = list(inferred_model.graph.value_info)
        vis = list(sorted(vis, key=lambda x: x.name))
        inferred_vis = list(sorted(inferred_vis, key=lambda x: x.name))
        assert len(vis) == len(inferred_vis)
        for i in range(len(vis)):
            self._compare_value_infos(vis[i].type, inferred_vis[i].type)

    def _compare_value_infos(self, vi_type: TypeProto, inferred_vi_type: TypeProto) -> None:
        if vi_type.HasField('tensor_type'):
            assert inferred_vi_type.HasField('tensor_type')
            assert vi_type.tensor_type.HasField('elem_type')
            assert inferred_vi_type.tensor_type.HasField('elem_type')
            assert vi_type.tensor_type.elem_type == inferred_vi_type.tensor_type.elem_type
            assert vi_type.tensor_type.HasField('shape') == inferred_vi_type.tensor_type.HasField('shape')
            if vi_type.tensor_type.HasField('shape'):
                assert len(vi_type.tensor_type.shape.dim) == len(inferred_vi_type.tensor_type.shape.dim)
                for dim_i in range(len(vi_type.tensor_type.shape.dim)):
                    dim = vi_type.tensor_type.shape.dim[dim_i]
                    inferred_dim = inferred_vi_type.tensor_type.shape.dim[dim_i]
                    # if it is a symbolic shape, make sure the inferred symbol has generated (dim_param)
                    if dim.dim_param:
                        assert dim.dim_param == inferred_dim.dim_param, f'\n{vi_type}\n{inferred_vi_type}\n'
                    else:
                        assert dim.dim_value == inferred_dim.dim_value, f'\n{vi_type}\n{inferred_vi_type}\n'
        elif vi_type.HasField('sequence_type'):
            assert inferred_vi_type.HasField('sequence_type')
            vi = vi_type.sequence_type.elem_type
            inferred_vi = inferred_vi_type.sequence_type.elem_type
            self._compare_value_infos(vi, inferred_vi)
        elif vi_type.HasField('optional_type'):
            assert inferred_vi_type.HasField('optional_type')
            vi = vi_type.optional_type.elem_type
            inferred_vi = inferred_vi_type.optional_type.elem_type
            self._compare_value_infos(vi, inferred_vi)
        else:
            raise NotImplementedError(
                "Unrecognized value info type in _compare_value_infos: ", str(vi_type))

    def test_expand_symbolic_input(self) -> None:
        graph = self._make_graph(
            [('x', TensorProto.INT32, (3, 1, 2)),
             ('y', TensorProto.INT32, (1, 4, 2))],
            [make_node("Shape", ['y'], ['shape']),
             make_node("Expand", ['x', 'shape'], ['z'])],
            [])
        self._assert_inferred(graph, [
            make_tensor_value_info('shape', TensorProto.INT64, (3,)),
            make_tensor_value_info('z', TensorProto.INT32, (3, 4, 2))],
            data_prop=True)

    def test_constantofshape_with_symbolic_shape(self) -> None:
        graph = self._make_graph(
            [('x', TensorProto.FLOAT, (3, 4, 5))],
            [make_node("Shape", ['x'], ['shape']),
             make_node("ConstantOfShape", ['shape'], ['y'], value=make_tensor('value', TensorProto.INT32, (1, ), (2, )))],
            [])
        self._assert_inferred(graph,
            [make_tensor_value_info('shape', TensorProto.INT64, (3,)),
             make_tensor_value_info('y', TensorProto.INT32, (3, 4, 5))], data_prop=True)  # type: ignore


if __name__ == '__main__':
    unittest.main()
