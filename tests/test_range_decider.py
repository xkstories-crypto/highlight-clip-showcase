# -*- coding: utf-8 -*-
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from showcase.ai import range_decider as rd

SEGS = [{"start": float(i * 5), "end": float(i * 5 + 4), "text": f"発話{i}"} for i in range(10)]


def test_peak_is_marked():
    body = rd.build_body(SEGS, peak=26.0)              # 5番(25〜29秒)の中
    assert body.splitlines()[5].startswith(">>> [5](-1s)")
    assert sum(l.startswith(">>>") for l in body.splitlines()) == 1


def test_takes_the_earliest_start():
    fake = lambda system, user: '{"start_min": 4, "start_topic": 3, "start_prev": 2, "end": 7}'
    r = rd.decide_range(SEGS, 26.0, fake)
    assert r["start"] == 10.0 and r["end"] == 39.0


@pytest.mark.parametrize("raw", [
    '{"start_min": 4, "start_topic": 3, "start_prev": 2, "end": 99}',   # 範囲外
    '{"start_min": 2, "start_topic": 3, "start_prev": 4, "end": 7}',    # 順序違反
    "わかりません",                                                      # JSONなし
])
def test_invalid_answers_raise_instead_of_guessing(raw):
    with pytest.raises(RuntimeError):
        rd.decide_range(SEGS, 26.0, lambda s, u: raw)
