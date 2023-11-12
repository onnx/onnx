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
                    0.0,
                    9.638071060180664e-05,
                    0.00038549304008483887,
                    0.0008671879768371582,
                    0.0015413463115692139,
                    0.002407640218734741,
                    0.0034657716751098633,
                    0.004715323448181152,
                    0.006155818700790405,
                    0.0077867209911346436,
                    0.00960734486579895,
                    0.011617064476013184,
                    0.013815045356750488,
                    0.01620045304298401,
                    0.018772363662719727,
                    0.021529823541641235,
                    0.02447172999382019,
                    0.027596980333328247,
                    0.030904322862625122,
                    0.03439253568649292,
                    0.03806024789810181,
                    0.04190602898597717,
                    0.04592841863632202,
                    0.05012586712837219,
                    0.05449673533439636,
                    0.05903938412666321,
                    0.06375199556350708,
                    0.0686328113079071,
                    0.07367992401123047,
                    0.07889139652252197,
                    0.08426520228385925,
                    0.08979928493499756,
                    0.09549149870872498,
                    0.10133969783782959,
                    0.10734155774116516,
                    0.11349478363990784,
                    0.11979702115058899,
                    0.12624582648277283,
                    0.13283875584602356,
                    0.13957321643829346,
                    0.1464466154575348,
                    0.15345633029937744,
                    0.16059964895248413,
                    0.1678737998008728,
                    0.175275981426239,
                    0.18280336260795593,
                    0.19045302271842957,
                    0.19822204113006592,
                    0.20610737800598145,
                    0.214106023311615,
                    0.22221490740776062,
                    0.23043087124824524,
                    0.23875075578689575,
                    0.2471713423728943,
                    0.255689412355423,
                    0.2643016576766968,
                    0.2730047404766083,
                    0.2817953824996948,
                    0.29067012667655945,
                    0.29962557554244995,
                    0.30865830183029175,
                    0.3177647590637207,
                    0.32694149017333984,
                    0.33618491888046265,
                    0.34549152851104736,
                    0.3548576831817627,
                    0.3642798066139221,
                    0.37375423312187195,
                    0.38327735662460327,
                    0.3928454518318176,
                    0.4024548828601837,
                    0.4121018946170807,
                    0.42178282141685486,
                    0.4314938187599182,
                    0.44123131036758423,
                    0.4509914219379425,
                    0.46077045798301697,
                    0.470564603805542,
                    0.48037010431289673,
                    0.49018317461013794,
                    0.5,
                    0.5098168849945068,
                    0.519629955291748,
                    0.5294354557991028,
                    0.5392295718193054,
                    0.5490086078643799,
                    0.5587687492370605,
                    0.5685062408447266,
                    0.5782172083854675,
                    0.5878981351852417,
                    0.5975451469421387,
                    0.6071546077728271,
                    0.6167227029800415,
                    0.6262457966804504,
                    0.6357202529907227,
                    0.6451423764228821,
                    0.6545085310935974,
                    0.6638151407241821,
                    0.6730585694313049,
                    0.6822353005409241,
                    0.691341757774353,
                    0.7003744840621948,
                    0.7093299031257629,
                    0.71820467710495,
                    0.7269952893257141,
                    0.7356984615325928,
                    0.7443106770515442,
                    0.7528287172317505,
                    0.761249303817749,
                    0.7695692181587219,
                    0.7777851819992065,
                    0.7858940362930298,
                    0.7938926219940186,
                    0.8017779588699341,
                    0.809546947479248,
                    0.8171966075897217,
                    0.824724018573761,
                    0.8321262001991272,
                    0.8394004106521606,
                    0.8465436697006226,
                    0.8535534143447876,
                    0.8604267835617065,
                    0.8671612739562988,
                    0.8737541437149048,
                    0.8802030086517334,
                    0.8865052461624146,
                    0.8926584720611572,
                    0.8986603021621704,
                    0.9045084714889526,
                    0.9102007150650024,
                    0.9157348275184631,
                    0.921108603477478,
                    0.9263200759887695,
                    0.9313672184944153,
                    0.9362480640411377,
                    0.9409606456756592,
                    0.945503294467926,
                    0.9498741626739502,
                    0.9540716409683228,
                    0.9580940008163452,
                    0.961939811706543,
                    0.9656075239181519,
                    0.9690957069396973,
                    0.9724030494689941,
                    0.9755282998085022,
                    0.9784702062606812,
                    0.9812276363372803,
                    0.9837995767593384,
                    0.9861849546432495,
                    0.9883829355239868,
                    0.9903926849365234,
                    0.992213249206543,
                    0.9938441514968872,
                    0.9952846765518188,
                    0.9965342283248901,
                    0.9975923299789429,
                    0.9984586238861084,
                    0.9991328120231628,
                    0.9996144771575928,
                    0.9999036192893982,
                    1.0,
                    0.9999036192893982,
                    0.9996144771575928,
                    0.9991328120231628,
                    0.9984586238861084,
                    0.9975923299789429,
                    0.9965342283248901,
                    0.9952846765518188,
                    0.9938441514968872,
                    0.992213249206543,
                    0.9903926253318787,
                    0.9883829355239868,
                    0.9861849546432495,
                    0.9837995171546936,
                    0.9812276363372803,
                    0.9784701466560364,
                    0.9755282402038574,
                    0.9724030494689941,
                    0.9690957069396973,
                    0.9656074643135071,
                    0.9619397521018982,
                    0.9580940008163452,
                    0.954071581363678,
                    0.9498741626739502,
                    0.9455032348632812,
                    0.9409606456756592,
                    0.9362479448318481,
                    0.9313671588897705,
                    0.9263200759887695,
                    0.921108603477478,
                    0.9157347679138184,
                    0.9102007150650024,
                    0.9045084714889526,
                    0.8986603021621704,
                    0.8926584124565125,
                    0.8865051865577698,
                    0.8802029490470886,
                    0.8737541437149048,
                    0.867161214351654,
                    0.8604267835617065,
                    0.853553295135498,
                    0.8465436100959778,
                    0.8394002914428711,
                    0.8321261405944824,
                    0.8247239589691162,
                    0.8171966671943665,
                    0.8095468878746033,
                    0.8017779588699341,
                    0.793892502784729,
                    0.785893976688385,
                    0.7777850031852722,
                    0.7695691585540771,
                    0.7612491846084595,
                    0.7528287172317505,
                    0.7443104982376099,
                    0.7356983423233032,
                    0.7269951105117798,
                    0.7182046175003052,
                    0.7093297243118286,
                    0.7003743648529053,
                    0.6913415789604187,
                    0.6822352409362793,
                    0.6730583906173706,
                    0.6638150215148926,
                    0.6545085310935974,
                    0.6451422572135925,
                    0.6357202529907227,
                    0.6262457370758057,
                    0.6167227029800415,
                    0.6071544885635376,
                    0.5975452065467834,
                    0.5878980755805969,
                    0.5782172679901123,
                    0.5685060620307922,
                    0.5587686896324158,
                    0.5490084886550903,
                    0.5392295718193054,
                    0.5294352769851685,
                    0.5196298956871033,
                    0.5098167061805725,
                    0.5,
                    0.490183025598526,
                    0.48037007451057434,
                    0.47056445479393005,
                    0.4607704281806946,
                    0.45099127292633057,
                    0.44123128056526184,
                    0.4314936697483063,
                    0.4217827320098877,
                    0.412101686000824,
                    0.40245479345321655,
                    0.3928452730178833,
                    0.3832772672176361,
                    0.37375402450561523,
                    0.36427971720695496,
                    0.35485750436782837,
                    0.3454914391040802,
                    0.33618494868278503,
                    0.3269414007663727,
                    0.3177647888660431,
                    0.3086581826210022,
                    0.29962560534477234,
                    0.2906700372695923,
                    0.2817953824996948,
                    0.2730046510696411,
                    0.2643016576766968,
                    0.25568926334381104,
                    0.2471713125705719,
                    0.2387506067752838,
                    0.23043084144592285,
                    0.22221478819847107,
                    0.214106023311615,
                    0.2061072587966919,
                    0.19822201132774353,
                    0.19045290350914001,
                    0.18280333280563354,
                    0.17527586221694946,
                    0.16787376999855042,
                    0.1605994999408722,
                    0.15345630049705505,
                    0.14644649624824524,
                    0.13957315683364868,
                    0.132838636636734,
                    0.12624579668045044,
                    0.11979690194129944,
                    0.11349472403526306,
                    0.10734140872955322,
                    0.10133963823318481,
                    0.09549137949943542,
                    0.08979922533035278,
                    0.08426520228385925,
                    0.0788913369178772,
                    0.07367992401123047,
                    0.06863275170326233,
                    0.06375199556350708,
                    0.05903932452201843,
                    0.05449673533439636,
                    0.05012580752372742,
                    0.04592841863632202,
                    0.0419059693813324,
                    0.03806021809577942,
                    0.034392476081848145,
                    0.030904322862625122,
                    0.02759692072868347,
                    0.02447172999382019,
                    0.021529793739318848,
                    0.018772363662719727,
                    0.01620042324066162,
                    0.0138150155544281,
                    0.011617034673690796,
                    0.00960734486579895,
                    0.007786691188812256,
                    0.006155818700790405,
                    0.004715293645858765,
                    0.0034657716751098633,
                    0.0024076104164123535,
                    0.0015413165092468262,
                    0.0008671879768371582,
                    0.00038549304008483887,
                    9.638071060180664e-05,
                ],
                dtype=np.float32,
            ),
            name="_tensor_constant0",
        )
    )
    inputs.append(make_tensor_value_info("arg0", TensorProto.FLOAT, shape=(1, 16000)))
    nodes.append(
        make_node(
            "Constant",
            [],
            ["_val_1"],
            value=from_array(np.array([1, 1, 16000], dtype=np.int64), name="value"),
        )
    )
    nodes.append(make_node("Cast", ["_val_1"], ["size_0__1"], to=7))
    nodes.append(make_node("Reshape", ["arg0", "size_0__1"], ["view"]))
    nodes.append(
        make_node(
            "Constant",
            [],
            ["_val_3"],
            value=from_array(np.array([256, 256], dtype=np.int64), name="value"),
        )
    )
    nodes.append(make_node("Constant", [], ["neg_1__2"], value_ints=[-1]))
    nodes.append(make_node("Shape", ["view"], ["tmp__2"]))
    nodes.append(make_node("Size", ["tmp__2"], ["rank__2"]))
    nodes.append(
        make_node(
            "Constant",
            [],
            ["int64_2__2"],
            value=from_array(np.array(2, dtype=np.int64), name="value"),
        )
    )
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
    nodes.append(
        make_node(
            "Constant",
            [],
            ["_val_5"],
            value=from_array(np.array([1, 16512], dtype=np.int64), name="value"),
        )
    )
    nodes.append(make_node("Cast", ["_val_5"], ["size_0__3"], to=7))
    nodes.append(make_node("Reshape", ["constant_pad_nd", "size_0__3"], ["view_1"]))
    nodes.append(
        make_node(
            "Constant",
            [],
            ["_val_8"],
            value=from_array(np.array([96, 96], dtype=np.int64), name="value"),
        )
    )
    nodes.append(make_node("Constant", [], ["neg_1__4"], value_ints=[-1]))
    nodes.append(make_node("Shape", ["_tensor_constant0"], ["tmp__4"]))
    nodes.append(make_node("Size", ["tmp__4"], ["rank__4"]))
    nodes.append(
        make_node(
            "Constant",
            [],
            ["int64_2__4"],
            value=from_array(np.array(2, dtype=np.int64), name="value"),
        )
    )
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
    nodes.append(
        make_node(
            "Constant",
            [],
            ["true__5"],
            value=from_array(np.array(True, dtype=np.bool_), name="value"),
        )
    )

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
        inputs.append(make_tensor_value_info("i_1__5", TensorProto.UNDEFINED, []))
        inputs.append(
            make_tensor_value_info("seq_result_2__5", TensorProto.UNDEFINED, [])
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
        nodes.append(
            make_node(
                "Constant",
                [],
                ["int64_1__5"],
                value=from_array(np.array(1, dtype=np.int64), name="value"),
            )
        )
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
        outputs.append(make_tensor_value_info("i_4__5", TensorProto.UNDEFINED, []))
        outputs.append(
            make_tensor_value_info("seq_result_3__5", TensorProto.UNDEFINED, [])
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
    nodes.append(
        make_node(
            "Constant",
            [],
            ["_val_12"],
            value=from_array(np.array([-1], dtype=np.int64), name="value"),
        )
    )
    nodes.append(make_node("Unsqueeze", ["mul", "_val_12"], ["_val_13"]))
    nodes.append(
        make_node(
            "Constant",
            [],
            ["_val_14"],
            value=from_array(np.array([0], dtype=np.int64), name="value"),
        )
    )
    nodes.append(make_node("Unsqueeze", ["_val_13", "_val_14"], ["_val_15"]))
    nodes.append(
        make_node("DFT", ["_val_15"], ["_val_16"], axis=3, inverse=0, onesided=1)
    )
    nodes.append(
        make_node(
            "Constant",
            [],
            ["_val_17"],
            value=from_array(np.array([0], dtype=np.int64), name="value"),
        )
    )
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
    nodes.append(
        make_node(
            "Constant",
            [],
            ["int64_1__7"],
            value=from_array(np.array(1, dtype=np.int64), name="value"),
        )
    )
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
            outputs.append(
                make_tensor_value_info("result__7", TensorProto.UNDEFINED, [])
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

            nodes.append(make_node("Sqrt", ["total_sample_count_0__7"], ["tmp_1__7"]))
            nodes.append(make_node("Mul", ["_val_18", "tmp_1__7"], ["result_2__7"]))
            outputs.append(
                make_tensor_value_info("result_2__7", TensorProto.UNDEFINED, [])
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
        outputs.append(make_tensor_value_info("result_3__7", TensorProto.UNDEFINED, []))
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
        nodes.append(
            make_node(
                "Constant",
                [],
                ["int64_2__7"],
                value=from_array(np.array(2, dtype=np.int64), name="value"),
            )
        )
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
                    make_tensor_value_info("result_7__7", TensorProto.UNDEFINED, [])
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
                    make_tensor_value_info("result_8__7", TensorProto.UNDEFINED, [])
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
                make_tensor_value_info("result_9__7", TensorProto.UNDEFINED, [])
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
                    make_tensor_value_info("result_11__7", TensorProto.UNDEFINED, [])
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
                    make_tensor_value_info("result_12__7", TensorProto.UNDEFINED, [])
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
                make_tensor_value_info("result_13__7", TensorProto.UNDEFINED, [])
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
        outputs.append(
            make_tensor_value_info("result_14__7", TensorProto.UNDEFINED, [])
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
