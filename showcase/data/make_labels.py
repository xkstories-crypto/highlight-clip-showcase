# -*- coding: utf-8 -*-
"""
学習データの作り方：プロの切り抜き動画が、元配信のどこを使って、どこを削ったかを「音」で求める。

  切り抜き動画 ──┐
                 ├─ ① 音で照合して「使われた区間」を出す
  元配信の音声 ──┘
  ② 使われた最初〜最後を「範囲」とし、範囲を書き起こして発話に分ける
  ③ 発話ごとに、長さの半分以上が使われた区間に入っていれば「残した」、それ以外は「削った」

最初は字幕の文字（3-gram）で照合していたが、書き起こしの誤りでずれるため音に切り替えた。
音の照合はテロップや効果音が乗っていても、話し声の部分の形が同じなら一致する。

書き起こしには注意が要る: 温度0だけだと「はっはっ…」のような繰り返し崩れが数十秒の発言を
飲み込む（18%の組で発生）。崩れを見つけた組だけ、温度を段階的に上げて書き直す。
照合の質が低い組（無音以外の小片のうち一致したものが少ない）は学習に使わない。
"""
from __future__ import annotations

import re

import numpy as np

SR, HOP, NMEL = 16000, 160, 40           # 10ms 刻みのログメル
CHUNK, STEP = 100, 50                    # 1秒の小片を 0.5秒ずつずらす
SAME = 3                                 # ずれの差が 30ms 以内なら同じ区間
MIN_SCORE = 0.8                          # 一致とみなす相関


def logmel(y: np.ndarray) -> np.ndarray:
    import librosa
    m = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=400, hop_length=HOP, n_mels=NMEL)
    return np.log(m + 1e-6).astype(np.float32)


def ncc(src: np.ndarray, chunk: np.ndarray) -> np.ndarray:
    """src（40×N）の上で chunk（40×L）をずらしながら相関係数を計算（GPU があれば畳み込みで一括）"""
    import torch
    import torch.nn.functional as F
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = torch.from_numpy(src).to(dev)[None]
    C = torch.from_numpy(chunk).to(dev)
    C = C - C.mean()
    L, n = C.shape[1], C.numel()
    num = F.conv1d(S, C[None])[0, 0]
    ones = torch.ones(1, S.shape[1], L, device=dev)
    s1, s2 = F.conv1d(S, ones)[0, 0], F.conv1d(S * S, ones)[0, 0]
    var = (s2 - s1 * s1 / n).clamp_min(1e-6)
    return (num / (var.sqrt() * (C.norm() + 1e-6))).cpu().numpy()


def match_chunks(clip_audio: np.ndarray, src_audio: np.ndarray, src_base: float) -> list[dict]:
    """① 切り抜きの音を1秒ずつ、元配信（の該当付近）の中で一番似ている位置に当てる"""
    fc, fs = logmel(clip_audio), logmel(src_audio)
    energy = fc.mean(0)
    quiet = np.percentile(energy, 20)
    out = []
    for c0 in range(0, fc.shape[1] - CHUNK + 1, STEP):
        if energy[c0:c0 + CHUNK].mean() < quiet:          # ほぼ無音の小片は当てにならない
            continue
        r = ncc(fs, fc[:, c0:c0 + CHUNK])
        k = int(r.argmax())
        out.append({"clip_t": c0 * HOP / SR, "src_t": src_base + k * HOP / SR, "score": float(r[k])})
    return out


def used_intervals(chunks: list[dict]) -> list[dict]:
    """ずれ（元配信の時刻 − 切り抜きの時刻）が同じ小片の並び ＝ 編集でつながずに使われた区間"""
    segs, cur = [], None
    for ch in chunks:
        if ch["score"] < MIN_SCORE:
            continue
        off = ch["src_t"] - ch["clip_t"]
        if cur and abs(off - cur["off"]) <= SAME * HOP / SR and ch["clip_t"] - cur["clip_end"] <= 1.0:
            cur["clip_end"] = ch["clip_t"]
        else:
            if cur:
                segs.append(cur)
            cur = {"off": off, "clip_start": ch["clip_t"], "clip_end": ch["clip_t"]}
    if cur:
        segs.append(cur)
    return [{"src_start": s["clip_start"] + s["off"],
             "src_end": s["clip_end"] + s["off"] + CHUNK * HOP / SR} for s in segs]


# ──（省略）② 範囲の書き起こし（faster-whisper large-v3・温度0・前文条件なし）は本体のみ ──


def label_units(units: list[dict], used: list[dict]) -> list[dict]:
    """③ 発話の長さの半分以上が使われた区間に入っていれば「残した」"""
    out = []
    for u in units:
        ln = u["end"] - u["start"]
        cov = sum(max(0.0, min(u["end"], s["src_end"]) - max(u["start"], s["src_start"])) for s in used)
        out.append(dict(u, keep=ln > 0 and cov / ln >= 0.5))
    return out


GOOD_FRAC = 0.6                          # 一致した小片がこれ未満の組は学習に使わない


def good_enough(chunks: list[dict]) -> bool:
    return bool(chunks) and sum(c["score"] >= MIN_SCORE for c in chunks) / len(chunks) >= GOOD_FRAC


_LOOP = re.compile(r"(.{1,4}?)\1{5,}")


def has_loop(text: str) -> bool:
    """書き起こしの繰り返し崩れ（1〜4文字が6回以上続く）。見つけたら設定を変えて書き直す:
    temperature=[0, 0.2, …, 1.0], compression_ratio_threshold=2.4, hallucination_silence_threshold=2.0"""
    return bool(_LOOP.search(text or ""))


# ──（省略）崩れた組だけ書き直してラベルを付け直す処理、配信の音声の読み込み・全件の一括処理は本体のみ ──
