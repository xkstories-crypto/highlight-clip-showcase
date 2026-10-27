# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from showcase.telop import layout


def test_wrap_never_breaks_inside_a_word():
    lines = layout.wrap_words("さっきまで普通に喋ってたのに", 7)
    assert "".join(lines) == "さっきまで普通に喋ってたのに"
    assert all(layout.char_units(l) <= 7 for l in lines)
    assert "喋" not in [l[-1] for l in lines]          # 「喋|って」で割らない


def test_kinsoku_no_forbidden_char_at_line_start():
    hard = []
    lines = layout.wrap_words("すごーい、ほんとにすごーい", 5, hard)
    assert not hard                                    # 語を割らずに収まるケース
    for l in lines:
        assert l[0] not in "、。ぁぃぅぇぉっゃゅょー"


def test_lead_filler_becomes_its_own_card():
    phrases = layout.split_phrases("なんか、さっきまで普通に喋ってた")
    assert phrases[0] == "なんか"
    assert layout.is_weak(phrases[0])


def test_punctuation_dropped_but_emotion_kept():
    phrases = layout.split_phrases("いや、待って!?")
    assert all("、" not in p for p in phrases)
    assert phrases[-1].endswith("!?")


def test_distribute_time_respects_minimum():
    out = layout.distribute_time(["あいう", "えおか", "きくけ"], 0.0, 0.9)
    assert len(out) == 2                               # 0.45秒×2 までしか置けない
    assert out[0][0] == 0.0 and out[-1][1] == 0.9


def test_overlaps_resolved_to_one_card_at_a_time():
    items = [[0.0, 3.0, "前の発話です"], [1.5, 4.0, "次の発話です"], [1.6, 2.0, "かぶり"]]
    dropped = []
    out = layout.resolve_overlaps(items, dropped)
    for a, b in zip(out, out[1:]):
        assert a[1] <= b[0] + 1e-6
    assert len(out) + len(dropped) == 3                # 捨てた分は必ず記録される


def test_repeat_hallucination_collapsed():
    text, changed = layout.collapse_repeats("はいはいはいはいはいはい")
    assert changed and text == "はいはい"
    assert layout.collapse_repeats("はいはいはい") == ("はいはいはい", False)
