# -*- coding: utf-8 -*-
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from showcase.ai import trim_model as tm
from showcase.data import make_labels as ml

UNITS = [{"start": 0.0, "end": 2.0, "text": "今日はね"},
         {"start": 2.2, "end": 2.6, "text": "えーと"},
         {"start": 2.8, "end": 5.0, "text": "マジで全力でやる"},
         {"start": 5.1, "end": 7.0, "text": "マジで全力でやるって"},
         {"start": 7.5, "end": 9.0, "text": "うん"}]


def test_numeric_features_shape_and_flags():
    f = tm.numeric_features(UNITS, 1)
    assert len(f) == len(tm.NUM_NAMES) == 14
    assert f[8] == 1.0                                 # フィラーだけ
    assert tm.numeric_features(UNITS, 3)[12] >= 0.7    # 直前とほぼ同じ＝言い直し


def test_row_order_numeric_text_voice():
    r = tm.row(UNITS, 2, voice=[1.0] * 9, text_prob=0.3)
    assert len(r) == 14 + 1 + 9 and r[14] == 0.3 and r[15:] == [1.0] * 9
    assert np.isnan(tm.row(UNITS, 2)[14])              # 文章モデルが無ければ欠損のまま渡す


def test_used_intervals_groups_same_offset():
    chunks = [{"clip_t": t, "src_t": 100 + t, "score": 0.9} for t in (0.0, 0.5, 1.0)] + \
             [{"clip_t": t, "src_t": 120 + t, "score": 0.9} for t in (1.5, 2.0)]
    used = ml.used_intervals(chunks)
    assert len(used) == 2                              # ずれが変わった所＝編集でつないだ所
    assert used[0]["src_start"] == 100.0 and used[1]["src_start"] == 121.5


def test_label_units_half_rule():
    lab = ml.label_units([{"start": 0, "end": 2, "text": "a"}, {"start": 2, "end": 4, "text": "b"}],
                         [{"src_start": 0.0, "src_end": 2.9}])
    assert [u["keep"] for u in lab] == [True, False]   # 2つ目は 0.9/2 秒しか使われていない


def test_loop_detection():
    assert ml.has_loop("はっはっはっはっはっはっはっ")
    assert not ml.has_loop("はいはいはい")
