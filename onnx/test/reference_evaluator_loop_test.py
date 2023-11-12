# Copyright (c) ONNX Project Contributors

# SPDX-License-Identifier: Apache-2.0
# type: ignore
import unittest

import numpy as np

from onnx import TensorProto
from onnx.helper import (
    make_graph,
    make_model,
    make_node,
    make_opsetid,
    make_tensor_value_info,
)
from onnx.numpy_helper import from_array
from onnx.reference import ReferenceEvaluator


def create_model():
    opset_imports = [
        make_opsetid("pkg.onnxscript.torch_lib", 1),
        make_opsetid("", 18),
        make_opsetid("pkg.onnxscript.torch_lib.common", 1),
    ]
    inputs = []
    outputs = []
    nodes = []
    initializers = []
    sparse_initializers = []

    initializers.append(
        from_array(
            np.array(
                [
                    0.00000000e00,
                    9.63807106e-05,
                    3.85493040e-04,
                    8.67187977e-04,
                    1.54134631e-03,
                    2.40764022e-03,
                    3.46577168e-03,
                    4.71532345e-03,
                    6.15581870e-03,
                    7.78672099e-03,
                    9.60734487e-03,
                    1.16170645e-02,
                    1.38150454e-02,
                    1.62004530e-02,
                    1.87723637e-02,
                    2.15298235e-02,
                    2.44717300e-02,
                    2.75969803e-02,
                    3.09043229e-02,
                    3.43925357e-02,
                    3.80602479e-02,
                    4.19060290e-02,
                    4.59284186e-02,
                    5.01258671e-02,
                    5.44967353e-02,
                    5.90393841e-02,
                    6.37519956e-02,
                    6.86328113e-02,
                    7.36799240e-02,
                    7.88913965e-02,
                    8.42652023e-02,
                    8.97992849e-02,
                    9.54914987e-02,
                    1.01339698e-01,
                    1.07341558e-01,
                    1.13494784e-01,
                    1.19797021e-01,
                    1.26245826e-01,
                    1.32838756e-01,
                    1.39573216e-01,
                    1.46446615e-01,
                    1.53456330e-01,
                    1.60599649e-01,
                    1.67873800e-01,
                    1.75275981e-01,
                    1.82803363e-01,
                    1.90453023e-01,
                    1.98222041e-01,
                    2.06107378e-01,
                    2.14106023e-01,
                    2.22214907e-01,
                    2.30430871e-01,
                    2.38750756e-01,
                    2.47171342e-01,
                    2.55689412e-01,
                    2.64301658e-01,
                    2.73004740e-01,
                    2.81795382e-01,
                    2.90670127e-01,
                    2.99625576e-01,
                    3.08658302e-01,
                    3.17764759e-01,
                    3.26941490e-01,
                    3.36184919e-01,
                    3.45491529e-01,
                    3.54857683e-01,
                    3.64279807e-01,
                    3.73754233e-01,
                    3.83277357e-01,
                    3.92845452e-01,
                    4.02454883e-01,
                    4.12101895e-01,
                    4.21782821e-01,
                    4.31493819e-01,
                    4.41231310e-01,
                    4.50991422e-01,
                    4.60770458e-01,
                    4.70564604e-01,
                    4.80370104e-01,
                    4.90183175e-01,
                    5.00000000e-01,
                    5.09816885e-01,
                    5.19629955e-01,
                    5.29435456e-01,
                    5.39229572e-01,
                    5.49008608e-01,
                    5.58768749e-01,
                    5.68506241e-01,
                    5.78217208e-01,
                    5.87898135e-01,
                    5.97545147e-01,
                    6.07154608e-01,
                    6.16722703e-01,
                    6.26245797e-01,
                    6.35720253e-01,
                    6.45142376e-01,
                    6.54508531e-01,
                    6.63815141e-01,
                    6.73058569e-01,
                    6.82235301e-01,
                    6.91341758e-01,
                    7.00374484e-01,
                    7.09329903e-01,
                    7.18204677e-01,
                    7.26995289e-01,
                    7.35698462e-01,
                    7.44310677e-01,
                    7.52828717e-01,
                    7.61249304e-01,
                    7.69569218e-01,
                    7.77785182e-01,
                    7.85894036e-01,
                    7.93892622e-01,
                    8.01777959e-01,
                    8.09546947e-01,
                    8.17196608e-01,
                    8.24724019e-01,
                    8.32126200e-01,
                    8.39400411e-01,
                    8.46543670e-01,
                    8.53553414e-01,
                    8.60426784e-01,
                    8.67161274e-01,
                    8.73754144e-01,
                    8.80203009e-01,
                    8.86505246e-01,
                    8.92658472e-01,
                    8.98660302e-01,
                    9.04508471e-01,
                    9.10200715e-01,
                    9.15734828e-01,
                    9.21108603e-01,
                    9.26320076e-01,
                    9.31367218e-01,
                    9.36248064e-01,
                    9.40960646e-01,
                    9.45503294e-01,
                    9.49874163e-01,
                    9.54071641e-01,
                    9.58094001e-01,
                    9.61939812e-01,
                    9.65607524e-01,
                    9.69095707e-01,
                    9.72403049e-01,
                    9.75528300e-01,
                    9.78470206e-01,
                    9.81227636e-01,
                    9.83799577e-01,
                    9.86184955e-01,
                    9.88382936e-01,
                    9.90392685e-01,
                    9.92213249e-01,
                    9.93844151e-01,
                    9.95284677e-01,
                    9.96534228e-01,
                    9.97592330e-01,
                    9.98458624e-01,
                    9.99132812e-01,
                    9.99614477e-01,
                    9.99903619e-01,
                    1.00000000e00,
                    9.99903619e-01,
                    9.99614477e-01,
                    9.99132812e-01,
                    9.98458624e-01,
                    9.97592330e-01,
                    9.96534228e-01,
                    9.95284677e-01,
                    9.93844151e-01,
                    9.92213249e-01,
                    9.90392625e-01,
                    9.88382936e-01,
                    9.86184955e-01,
                    9.83799517e-01,
                    9.81227636e-01,
                    9.78470147e-01,
                    9.75528240e-01,
                    9.72403049e-01,
                    9.69095707e-01,
                    9.65607464e-01,
                    9.61939752e-01,
                    9.58094001e-01,
                    9.54071581e-01,
                    9.49874163e-01,
                    9.45503235e-01,
                    9.40960646e-01,
                    9.36247945e-01,
                    9.31367159e-01,
                    9.26320076e-01,
                    9.21108603e-01,
                    9.15734768e-01,
                    9.10200715e-01,
                    9.04508471e-01,
                    8.98660302e-01,
                    8.92658412e-01,
                    8.86505187e-01,
                    8.80202949e-01,
                    8.73754144e-01,
                    8.67161214e-01,
                    8.60426784e-01,
                    8.53553295e-01,
                    8.46543610e-01,
                    8.39400291e-01,
                    8.32126141e-01,
                    8.24723959e-01,
                    8.17196667e-01,
                    8.09546888e-01,
                    8.01777959e-01,
                    7.93892503e-01,
                    7.85893977e-01,
                    7.77785003e-01,
                    7.69569159e-01,
                    7.61249185e-01,
                    7.52828717e-01,
                    7.44310498e-01,
                    7.35698342e-01,
                    7.26995111e-01,
                    7.18204618e-01,
                    7.09329724e-01,
                    7.00374365e-01,
                    6.91341579e-01,
                    6.82235241e-01,
                    6.73058391e-01,
                    6.63815022e-01,
                    6.54508531e-01,
                    6.45142257e-01,
                    6.35720253e-01,
                    6.26245737e-01,
                    6.16722703e-01,
                    6.07154489e-01,
                    5.97545207e-01,
                    5.87898076e-01,
                    5.78217268e-01,
                    5.68506062e-01,
                    5.58768690e-01,
                    5.49008489e-01,
                    5.39229572e-01,
                    5.29435277e-01,
                    5.19629896e-01,
                    5.09816706e-01,
                    5.00000000e-01,
                    4.90183026e-01,
                    4.80370075e-01,
                    4.70564455e-01,
                    4.60770428e-01,
                    4.50991273e-01,
                    4.41231281e-01,
                    4.31493670e-01,
                    4.21782732e-01,
                    4.12101686e-01,
                    4.02454793e-01,
                    3.92845273e-01,
                    3.83277267e-01,
                    3.73754025e-01,
                    3.64279717e-01,
                    3.54857504e-01,
                    3.45491439e-01,
                    3.36184949e-01,
                    3.26941401e-01,
                    3.17764789e-01,
                    3.08658183e-01,
                    2.99625605e-01,
                    2.90670037e-01,
                    2.81795382e-01,
                    2.73004651e-01,
                    2.64301658e-01,
                    2.55689263e-01,
                    2.47171313e-01,
                    2.38750607e-01,
                    2.30430841e-01,
                    2.22214788e-01,
                    2.14106023e-01,
                    2.06107259e-01,
                    1.98222011e-01,
                    1.90452904e-01,
                    1.82803333e-01,
                    1.75275862e-01,
                    1.67873770e-01,
                    1.60599500e-01,
                    1.53456300e-01,
                    1.46446496e-01,
                    1.39573157e-01,
                    1.32838637e-01,
                    1.26245797e-01,
                    1.19796902e-01,
                    1.13494724e-01,
                    1.07341409e-01,
                    1.01339638e-01,
                    9.54913795e-02,
                    8.97992253e-02,
                    8.42652023e-02,
                    7.88913369e-02,
                    7.36799240e-02,
                    6.86327517e-02,
                    6.37519956e-02,
                    5.90393245e-02,
                    5.44967353e-02,
                    5.01258075e-02,
                    4.59284186e-02,
                    4.19059694e-02,
                    3.80602181e-02,
                    3.43924761e-02,
                    3.09043229e-02,
                    2.75969207e-02,
                    2.44717300e-02,
                    2.15297937e-02,
                    1.87723637e-02,
                    1.62004232e-02,
                    1.38150156e-02,
                    1.16170347e-02,
                    9.60734487e-03,
                    7.78669119e-03,
                    6.15581870e-03,
                    4.71529365e-03,
                    3.46577168e-03,
                    2.40761042e-03,
                    1.54131651e-03,
                    8.67187977e-04,
                    3.85493040e-04,
                    9.63807106e-05,
                ],
                dtype=np.float32,
            ),
            name="_tensor_constant0",
        )
    )
    inputs.append(make_tensor_value_info("arg0", TensorProto.FLOAT, shape=(1, 16000)))
    nodes.append(make_node("Constant", [], ["_val_1"], value_ints=[1, 1, 16000]))
    nodes.append(make_node("Cast", ["_val_1"], ["size_0__1"], to=7))
    nodes.append(make_node("Reshape", ["arg0", "size_0__1"], ["view"]))
    nodes.append(make_node("Constant", [], ["_val_3"], value_ints=[256, 256]))
    nodes.append(make_node("Constant", [], ["neg_1__2"], value_ints=[-1]))
    nodes.append(make_node("Shape", ["view"], ["tmp__2"]))
    nodes.append(make_node("Size", ["tmp__2"], ["rank__2"]))
    nodes.append(make_node("Constant", [], ["int64_2__2"], value_int=2))
    nodes.append(make_node("CastLike", ["int64_2__2", "rank__2"], ["int64_2_cast__2"]))
    nodes.append(make_node("Mul", ["rank__2", "int64_2_cast__2"], ["tmp_0__2"]))
    nodes.append(make_node("Size", ["_val_3"], ["tmp_1__2"]))
    nodes.append(make_node("Sub", ["tmp_0__2", "tmp_1__2"], ["zero_count__2"]))
    nodes.append(
        make_node("Reshape", ["zero_count__2", "neg_1__2"], ["zero_count_2__2"])
    )
    nodes.append(make_node("Constant", [], ["zero__2"], value_ints=[0]))
    nodes.append(make_node("Expand", ["zero__2", "zero_count_2__2"], ["zeros__2"]))
    nodes.append(
        make_node("Concat", ["_val_3", "zeros__2"], ["torch_paddings__2"], axis=0)
    )
    nodes.append(make_node("Size", ["torch_paddings__2"], ["size_d__2"]))
    nodes.append(make_node("Constant", [], ["steps__2"], value_ints=[-2]))
    nodes.append(make_node("Sub", ["steps__2", "size_d__2"], ["ends__2"]))
    nodes.append(
        make_node(
            "Slice",
            ["torch_paddings__2", "steps__2", "ends__2", "zero__2", "steps__2"],
            ["odd_elements__2"],
        )
    )
    nodes.append(make_node("Sub", ["neg_1__2", "size_d__2"], ["ends_3__2"]))
    nodes.append(
        make_node(
            "Slice",
            ["torch_paddings__2", "neg_1__2", "ends_3__2", "zero__2", "steps__2"],
            ["even_elements__2"],
        )
    )
    nodes.append(
        make_node(
            "Concat",
            ["odd_elements__2", "even_elements__2"],
            ["onnx_padding__2"],
            axis=0,
        )
    )
    nodes.append(make_node("Constant", [], ["value__2"], value_float=0.0))
    nodes.append(make_node("CastLike", ["value__2", "view"], ["value_cast__2"]))
    nodes.append(
        make_node(
            "Pad", ["view", "onnx_padding__2", "value_cast__2"], ["constant_pad_nd"]
        )
    )
    nodes.append(make_node("Constant", [], ["_val_5"], value_ints=[1, 16512]))
    nodes.append(make_node("Cast", ["_val_5"], ["size_0__3"], to=7))
    nodes.append(make_node("Reshape", ["constant_pad_nd", "size_0__3"], ["view_1"]))
    nodes.append(make_node("Constant", [], ["_val_8"], value_ints=[96, 96]))
    nodes.append(make_node("Constant", [], ["neg_1__4"], value_ints=[-1]))
    nodes.append(make_node("Shape", ["_tensor_constant0"], ["tmp__4"]))
    nodes.append(make_node("Size", ["tmp__4"], ["rank__4"]))
    nodes.append(make_node("Constant", [], ["int64_2__4"], value_int=2))
    nodes.append(make_node("CastLike", ["int64_2__4", "rank__4"], ["int64_2_cast__4"]))
    nodes.append(make_node("Mul", ["rank__4", "int64_2_cast__4"], ["tmp_0__4"]))
    nodes.append(make_node("Size", ["_val_8"], ["tmp_1__4"]))
    nodes.append(make_node("Sub", ["tmp_0__4", "tmp_1__4"], ["zero_count__4"]))
    nodes.append(
        make_node("Reshape", ["zero_count__4", "neg_1__4"], ["zero_count_2__4"])
    )
    nodes.append(make_node("Constant", [], ["zero__4"], value_ints=[0]))
    nodes.append(make_node("Expand", ["zero__4", "zero_count_2__4"], ["zeros__4"]))
    nodes.append(
        make_node("Concat", ["_val_8", "zeros__4"], ["torch_paddings__4"], axis=0)
    )
    nodes.append(make_node("Size", ["torch_paddings__4"], ["size_d__4"]))
    nodes.append(make_node("Constant", [], ["steps__4"], value_ints=[-2]))
    nodes.append(make_node("Sub", ["steps__4", "size_d__4"], ["ends__4"]))
    nodes.append(
        make_node(
            "Slice",
            ["torch_paddings__4", "steps__4", "ends__4", "zero__4", "steps__4"],
            ["odd_elements__4"],
        )
    )
    nodes.append(make_node("Sub", ["neg_1__4", "size_d__4"], ["ends_3__4"]))
    nodes.append(
        make_node(
            "Slice",
            ["torch_paddings__4", "neg_1__4", "ends_3__4", "zero__4", "steps__4"],
            ["even_elements__4"],
        )
    )
    nodes.append(
        make_node(
            "Concat",
            ["odd_elements__4", "even_elements__4"],
            ["onnx_padding__4"],
            axis=0,
        )
    )
    nodes.append(make_node("Constant", [], ["value__4"], value_float=0.0))
    nodes.append(
        make_node("CastLike", ["value__4", "_tensor_constant0"], ["value_cast__4"])
    )
    nodes.append(
        make_node(
            "Pad",
            ["_tensor_constant0", "onnx_padding__4", "value_cast__4"],
            ["constant_pad_nd_1"],
        )
    )
    nodes.append(make_node("Constant", [], ["tmp__5"], value_int=-1))
    nodes.append(make_node("Constant", [], ["tmp_0__5"], value_ints=[-1]))
    nodes.append(make_node("Reshape", ["tmp__5", "tmp_0__5"], ["dims__5"]))
    nodes.append(make_node("SequenceEmpty", [], ["seq_result__5"]))
    nodes.append(make_node("Constant", [], ["i__5"], value_ints=[0]))
    nodes.append(make_node("Constant", [], ["target_end__5"], value_int=101))
    nodes.append(
        make_node("CastLike", ["target_end__5", "i__5"], ["target_end_cast__5"])
    )
    nodes.append(make_node("Less", ["i__5", "target_end_cast__5"], ["cond__5"]))
    nodes.append(make_node("Constant", [], ["true__5_int64"], value_int=1))
    nodes.append(make_node("Cast", ["true__5_int64"], ["true__5"], to=TensorProto.BOOL))

    def _make_local_graph_body():
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []

        inputs.append(
            make_tensor_value_info("infinite_loop__5", TensorProto.INT64, shape=[])
        )
        inputs.append(make_tensor_value_info("cond__5_0", TensorProto.BOOL, shape=[]))
        inputs.append(make_tensor_value_info("i_1__5", TensorProto.INT64, None))
        inputs.append(
            make_tensor_value_info("seq_result_2__5", TensorProto.FLOAT, None)
        )
        nodes.append(make_node("Constant", [], ["step__5"], value_int=160))
        nodes.append(make_node("CastLike", ["step__5", "i_1__5"], ["step_cast__5"]))
        nodes.append(make_node("Mul", ["i_1__5", "step_cast__5"], ["starts__5"]))
        nodes.append(make_node("Constant", [], ["size__5"], value_int=512))
        nodes.append(make_node("CastLike", ["size__5", "starts__5"], ["size_cast__5"]))
        nodes.append(make_node("Add", ["starts__5", "size_cast__5"], ["ends__5"]))
        nodes.append(
            make_node(
                "Slice",
                ["view_1", "starts__5", "ends__5", "dims__5"],
                ["slice_result__5"],
            )
        )
        nodes.append(
            make_node("Cast", ["slice_result__5"], ["slice_result_float32__5"], to=1)
        )
        nodes.append(
            make_node(
                "SequenceInsert",
                ["seq_result_2__5", "slice_result_float32__5"],
                ["seq_result_3__5"],
            )
        )
        nodes.append(make_node("Constant", [], ["int64_1__5"], value_int=1))
        nodes.append(
            make_node("CastLike", ["int64_1__5", "i_1__5"], ["int64_1_cast__5"])
        )
        nodes.append(make_node("Add", ["i_1__5", "int64_1_cast__5"], ["i_4__5"]))
        nodes.append(make_node("Constant", [], ["target_end_5__5"], value_int=101))
        nodes.append(
            make_node(
                "CastLike", ["target_end_5__5", "i_4__5"], ["target_end_5_cast__5"]
            )
        )
        nodes.append(
            make_node("Less", ["i_4__5", "target_end_5_cast__5"], ["cond_6__5"])
        )
        nodes.append(make_node("Identity", ["cond_6__5"], ["cond_out__5"]))
        outputs.append(
            make_tensor_value_info("cond_out__5", TensorProto.BOOL, shape=[])
        )
        outputs.append(make_tensor_value_info("i_4__5", TensorProto.INT64, None))
        outputs.append(
            make_tensor_value_info("seq_result_3__5", TensorProto.FLOAT, None)
        )
        graph = make_graph(
            nodes,
            "noname",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        return graph

    body = _make_local_graph_body()
    nodes.append(
        make_node(
            "Loop",
            ["", "true__5", "i__5", "seq_result__5"],
            ["i_7__5", "seq_result_8__5"],
            body=body,
        )
    )
    nodes.append(
        make_node(
            "ConcatFromSequence",
            ["seq_result_8__5"],
            ["concat_result__5"],
            axis=-1,
            new_axis=1,
        )
    )
    nodes.append(
        make_node("Transpose", ["concat_result__5"], ["result__5"], perm=[1, 2, 0])
    )
    nodes.append(make_node("CastLike", ["result__5", "view_1"], ["unfold"]))
    nodes.append(make_node("CastLike", ["constant_pad_nd_1", "unfold"], ["other_0__6"]))
    nodes.append(make_node("Mul", ["unfold", "other_0__6"], ["mul"]))
    nodes.append(make_node("Constant", [], ["_val_12"], value_ints=[-1]))
    nodes.append(make_node("Unsqueeze", ["mul", "_val_12"], ["_val_13"]))
    nodes.append(make_node("Constant", [], ["_val_14"], value_ints=[0]))
    nodes.append(make_node("Unsqueeze", ["_val_13", "_val_14"], ["_val_15"]))
    nodes.append(
        make_node("DFT", ["_val_15"], ["_val_16"], axis=3, inverse=0, onesided=1)
    )
    nodes.append(make_node("Constant", [], ["_val_17"], value_ints=[0]))
    nodes.append(make_node("Squeeze", ["_val_16", "_val_17"], ["_val_18"]))
    nodes.append(make_node("Shape", ["_val_13"], ["self_shape__7"]))
    nodes.append(make_node("Constant", [], ["dims__7"], value_ints=[2]))
    nodes.append(
        make_node(
            "Gather",
            ["self_shape__7", "dims__7"],
            ["self_shape_subscripted__7"],
            axis=0,
        )
    )
    nodes.append(
        make_node(
            "ReduceProd",
            ["self_shape_subscripted__7"],
            ["total_sample_count__7"],
            keepdims=0,
        )
    )
    nodes.append(
        make_node(
            "CastLike",
            ["total_sample_count__7", "_val_18"],
            ["total_sample_count_0__7"],
        )
    )
    nodes.append(make_node("Constant", [], ["normalization__7"], value_int=0))
    nodes.append(make_node("Constant", [], ["int64_1__7"], value_int=1))
    nodes.append(make_node("Equal", ["normalization__7", "int64_1__7"], ["cond__7"]))

    def _make_local_graph_then_branch():
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []

        nodes.append(make_node("Constant", [], ["forward__7"], value_int=1))
        nodes.append(make_node("Cast", ["forward__7"], ["forward_as_bool__7"], to=9))

        def _make_local_graph_then_branch():
            inputs = []
            outputs = []
            nodes = []
            initializers = []
            sparse_initializers = []

            nodes.append(make_node("Sqrt", ["total_sample_count_0__7"], ["tmp__7"]))
            nodes.append(make_node("Div", ["_val_18", "tmp__7"], ["result__7"]))
            outputs.append(make_tensor_value_info("result__7", TensorProto.FLOAT, None))
            graph = make_graph(
                nodes,
                "noname",
                inputs,
                outputs,
                initializers,
                sparse_initializer=sparse_initializers,
            )
            return graph

        then_branch = _make_local_graph_then_branch()

        def _make_local_graph_else_branch():
            inputs = []
            outputs = []
            nodes = []
            initializers = []
            sparse_initializers = []

            nodes.append(make_node("Sqrt", ["total_sample_count_0__7"], ["tmp_1__7"]))
            nodes.append(make_node("Mul", ["_val_18", "tmp_1__7"], ["result_2__7"]))
            outputs.append(
                make_tensor_value_info("result_2__7", TensorProto.FLOAT, None)
            )
            graph = make_graph(
                nodes,
                "noname",
                inputs,
                outputs,
                initializers,
                sparse_initializer=sparse_initializers,
            )
            return graph

        else_branch = _make_local_graph_else_branch()
        nodes.append(
            make_node(
                "If",
                ["forward_as_bool__7"],
                ["result_3__7"],
                then_branch=then_branch,
                else_branch=else_branch,
            )
        )
        outputs.append(make_tensor_value_info("result_3__7", TensorProto.FLOAT, None))
        graph = make_graph(
            nodes,
            "noname",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        return graph

    then_branch = _make_local_graph_then_branch()

    def _make_local_graph_else_branch():
        inputs = []
        outputs = []
        nodes = []
        initializers = []
        sparse_initializers = []

        nodes.append(make_node("Constant", [], ["normalization_4__7"], value_int=0))
        nodes.append(make_node("Constant", [], ["int64_2__7"], value_int=2))
        nodes.append(
            make_node("Equal", ["normalization_4__7", "int64_2__7"], ["cond_5__7"])
        )

        def _make_local_graph_then_branch():
            inputs = []
            outputs = []
            nodes = []
            initializers = []
            sparse_initializers = []

            nodes.append(make_node("Constant", [], ["forward_6__7"], value_int=1))
            nodes.append(
                make_node("Cast", ["forward_6__7"], ["forward_6_as_bool__7"], to=9)
            )

            def _make_local_graph_then_branch():
                inputs = []
                outputs = []
                nodes = []
                initializers = []
                sparse_initializers = []

                nodes.append(
                    make_node(
                        "Div", ["_val_18", "total_sample_count_0__7"], ["result_7__7"]
                    )
                )
                outputs.append(
                    make_tensor_value_info("result_7__7", TensorProto.FLOAT, None)
                )
                graph = make_graph(
                    nodes,
                    "noname",
                    inputs,
                    outputs,
                    initializers,
                    sparse_initializer=sparse_initializers,
                )
                return graph

            then_branch = _make_local_graph_then_branch()

            def _make_local_graph_else_branch():
                inputs = []
                outputs = []
                nodes = []
                initializers = []
                sparse_initializers = []
                nodes.append(make_node("Identity", ["_val_18"], ["result_8__7"]))
                outputs.append(
                    make_tensor_value_info("result_8__7", TensorProto.FLOAT, None)
                )
                graph = make_graph(
                    nodes,
                    "noname",
                    inputs,
                    outputs,
                    initializers,
                    sparse_initializer=sparse_initializers,
                )
                return graph

            else_branch = _make_local_graph_else_branch()
            nodes.append(
                make_node(
                    "If",
                    ["forward_6_as_bool__7"],
                    ["result_9__7"],
                    then_branch=then_branch,
                    else_branch=else_branch,
                )
            )
            outputs.append(
                make_tensor_value_info("result_9__7", TensorProto.FLOAT, None)
            )
            graph = make_graph(
                nodes,
                "noname",
                inputs,
                outputs,
                initializers,
                sparse_initializer=sparse_initializers,
            )
            return graph

        then_branch = _make_local_graph_then_branch()

        def _make_local_graph_else_branch():
            inputs = []
            outputs = []
            nodes = []
            initializers = []
            sparse_initializers = []

            nodes.append(make_node("Constant", [], ["forward_10__7"], value_int=1))
            nodes.append(
                make_node("Cast", ["forward_10__7"], ["forward_10_as_bool__7"], to=9)
            )

            def _make_local_graph_then_branch():
                inputs = []
                outputs = []
                nodes = []
                initializers = []
                sparse_initializers = []
                nodes.append(make_node("Identity", ["_val_18"], ["result_11__7"]))
                outputs.append(
                    make_tensor_value_info("result_11__7", TensorProto.FLOAT, None)
                )
                graph = make_graph(
                    nodes,
                    "noname",
                    inputs,
                    outputs,
                    initializers,
                    sparse_initializer=sparse_initializers,
                )
                return graph

            then_branch = _make_local_graph_then_branch()

            def _make_local_graph_else_branch():
                inputs = []
                outputs = []
                nodes = []
                initializers = []
                sparse_initializers = []

                nodes.append(
                    make_node(
                        "Mul", ["_val_18", "total_sample_count_0__7"], ["result_12__7"]
                    )
                )
                outputs.append(
                    make_tensor_value_info("result_12__7", TensorProto.FLOAT, None)
                )
                graph = make_graph(
                    nodes,
                    "noname",
                    inputs,
                    outputs,
                    initializers,
                    sparse_initializer=sparse_initializers,
                )
                return graph

            else_branch = _make_local_graph_else_branch()
            nodes.append(
                make_node(
                    "If",
                    ["forward_10_as_bool__7"],
                    ["result_13__7"],
                    then_branch=then_branch,
                    else_branch=else_branch,
                )
            )
            outputs.append(
                make_tensor_value_info("result_13__7", TensorProto.FLOAT, None)
            )
            graph = make_graph(
                nodes,
                "noname",
                inputs,
                outputs,
                initializers,
                sparse_initializer=sparse_initializers,
            )
            return graph

        else_branch = _make_local_graph_else_branch()
        nodes.append(
            make_node(
                "If",
                ["cond_5__7"],
                ["result_14__7"],
                then_branch=then_branch,
                else_branch=else_branch,
            )
        )
        outputs.append(make_tensor_value_info("result_14__7", TensorProto.FLOAT, None))
        graph = make_graph(
            nodes,
            "noname",
            inputs,
            outputs,
            initializers,
            sparse_initializer=sparse_initializers,
        )
        return graph

    else_branch = _make_local_graph_else_branch()
    nodes.append(
        make_node(
            "If",
            ["cond__7"],
            ["_fft_r2c"],
            then_branch=then_branch,
            else_branch=else_branch,
        )
    )
    nodes.append(make_node("Transpose", ["_fft_r2c"], ["transpose"], perm=[0, 2, 1, 3]))
    outputs.append(
        make_tensor_value_info("transpose", TensorProto.FLOAT, shape=(1, 257, 101, 2))
    )
    graph = make_graph(
        nodes,
        "noname",
        inputs,
        outputs,
        initializers,
        sparse_initializer=sparse_initializers,
    )
    model = make_model(graph, opset_imports=opset_imports)
    return model


class TestReferenceEvaluatorLoop(unittest.TestCase):
    def test_loop_fft(self):
        model = create_model()
        batch_size = 1
        signal_length = 16000
        np_signals = np.random.random(size=[batch_size, signal_length]).astype(
            np.float32
        )
        print(f"Run ONNX graph with signals of shape {np_signals.shape}")

        # try:
        #     from onnxruntime import InferenceSession
        #     ort_session = InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        # except ImportError:
        #     # onnxruntime is not available
        #     pass
        session = ReferenceEvaluator(model, verbose=10)
        session.run(None, {"arg0": np_signals})


if __name__ == "__main__":
    unittest.main(verbosity=2)
