# -*- coding: utf-8 -*-
"""
切り抜きの開始・終了を LLM に決めさせる。

山場（盛り上がった瞬間）の位置は、前段の信号処理＋ランキングモデルが見つける。
ここでは「その山場を1本の動画にするなら、どこから始めてどこで終わるか」だけを任せる。

設計で外せない4点:
  ① 秒を作らせない。セグメント番号を選ばせる（LLM は秒数をもっともらしく捏造する）
  ② 選べる範囲は、渡した文字起こしの窓の中だけ。窓の外は物理的に選べない
  ③ 山場の位置を >>> 印と「山場からの秒」で明示する
  ④ 結果をキャッシュする（LLM は同じ入力でも答えが揺れる＝再現性のため。本番のみ・ここでは省略）

開始は3段階で答えさせ、一番手前を採る:
  「はみ出しは後段で削れるが、欠けは取り返せない」。
  余分な前置きは後段の削り工程で落とす前提で、話の入口を欠かさないことを優先する。
  15本の検証で、正解範囲を欠いた本は 0。

失敗は例外で返す。黙って既定値にしない（呼び出し側が従来の範囲に戻して通知する）。
"""
from __future__ import annotations

import json
import re
from typing import Callable

SYSTEM = (
    "あなたは配信の切り抜き編集者です。文字起こしを読んで、切り抜きの開始と終了の位置だけを決めます。"
    "秒数は一切書かないでください。必ず与えられたセグメント番号の中から選びます。"
)

PROMPT = u"""配信の文字起こしの一部です。各行は [番号](山場からの秒) 発話 の形式です。
マイナスは山場より前、プラスは後ろを表します。
`>>>` が付いている行が、切り抜きたい山場（盛り上がっている場所）です。

{body}

この山場を1本の切り抜きにします。次の4つの行番号を答えてください。

開始の候補（手前ほど番号が小さい。同じ行になっても構わない）:
S1. start_min: 山場を理解するのに **最低限** 必要な前置きが始まる行。
S2. start_topic: 山場の話題が **会話に持ち出された** 行（質問・振り・話題転換の一言）。
S3. start_prev: S2 の話題が **その前の話題から流れてきている** なら、その流れが始まった行。
    地続きでなければ S2 と同じ行にする。山場と無関係な雑談まで遡らないこと。

終了:
E. end: この山場について、視聴者がもう何も待たなくてよくなる行。
   オチとその直後の反応（ツッコミ・笑い）が済んだら終わり。次の話題に入ってからは含めない。

制約:
- 番号は {lo} 〜 {hi} の範囲から選ぶ
- start_prev <= start_topic <= start_min < end
- 秒数は書かない。行番号で答える

次のJSONだけを出力してください。
{{"start_min": <番号>, "start_topic": <番号>, "start_prev": <番号>, "end": <番号>, "chain": "<話の繋がりを30字以内>", "why_end": "<20字以内>"}}
"""


def build_body(segments: list[dict], peak: float, tag: Callable[[dict], str] | None = None) -> str:
    """各行に山場からの経過秒を付ける（読むための目盛り。答えは行番号のまま）。
    tag を渡すと、行頭に声のトーン（笑い・大声など）を添えられる。"""
    lines = []
    for i, s in enumerate(segments):
        mark = ">>>" if s["start"] <= peak <= s["end"] else "   "
        rel = int(round(s["start"] - peak))
        extra = tag(s) if tag else ""
        lines.append(u"%s [%d](%+ds) %s%s" % (mark, i, rel, extra, (s.get("text") or "").strip()))
    if not any(l.startswith(">>>") for l in lines):
        # 山場が無音に落ちた場合は最も近いセグメントに印を付ける
        j = min(range(len(segments)), key=lambda i: abs(segments[i]["start"] - peak))
        lines[j] = ">>>" + lines[j][3:]
    return u"\n".join(lines)


def parse_answer(raw: str, lo: int, hi: int):
    """開始3段階と終了を取り出す。制約違反は黙って直さず例外にする。"""
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        raise ValueError("JSONが無い: %s" % (raw or "")[:200])
    obj = json.loads(m.group(0))
    starts = {k: int(obj[k]) for k in ("start_min", "start_topic", "start_prev")}
    end = int(obj["end"])
    for n in list(starts.values()) + [end]:
        if not (lo <= n <= hi):
            raise ValueError("範囲外の番号: %s (%s〜%s)" % (n, lo, hi))
    if not (starts["start_prev"] <= starts["start_topic"] <= starts["start_min"] < end):
        raise ValueError("順序違反: %s end=%s" % (starts, end))
    return starts, end, obj.get("chain", ""), obj.get("why_end", "")


def decide_range(segments: list[dict], peak: float,
                 llm: Callable[[str, str], str], retries: int = 1,
                 tag: Callable[[dict], str] | None = None) -> dict:
    """山場の前後の文字起こしから、切り抜きの開始・終了（秒）を決める。

    segments: 山場±数分の文字起こし [{"start", "end", "text"}, ...]（絶対秒）
    llm:      (system, user) -> 応答テキスト。使うモデルは呼び出し側が決める
    """
    # ──（省略）本体ではここで、手元の音声から山場の前後180秒を文字起こしし、同じ入力の答えが保存されていればそれを返す ──
    if len(segments) < 3:
        raise RuntimeError(f"文字起こしが少なすぎる: {len(segments)}セグメント")
    hi = len(segments) - 1
    user = PROMPT.format(body=build_body(segments, peak, tag), lo=0, hi=hi)

    last_err: Exception | None = None
    for _ in range(retries + 1):
        try:
            starts, end, chain, why_end = parse_answer(llm(SYSTEM, user), 0, hi)
            break
        except Exception as e:
            last_err = e
    else:
        raise RuntimeError(f"{retries + 1}回とも失敗: {last_err}")

    a = starts["start_prev"]          # 一番手前を採る
    # ──（省略）本体では結果を「元配信＋山場の秒」をキーに保存し、次回から使い回す ──
    return dict(
        start=round(float(segments[a]["start"]), 2),
        end=round(float(segments[end]["end"]), 2),
        starts={k: round(float(segments[v]["start"]), 2) for k, v in starts.items()},
        peak=round(peak, 2), chain=chain, why_end=why_end,
    )
