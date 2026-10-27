# -*- coding: utf-8 -*-
"""
字幕セグメントに「声のトーン」を付ける。

文字起こしだけでは、同じ「やばい」が叫びなのか呟きなのか分からない。
そこで信号処理で取れる情報を数値にして、AI（LLM）の判断材料に添える。

  - 笑い : PANNs（音響イベント検出）の笑い系ラベルの最大値 0〜1
  - 音量 : 前後5秒の平均との差（dB）＝その瞬間だけ大きいか
  - 話速 : モーラ（拍）/ 秒
  - 間   : 次のセグメントまでの無音秒

方針: 特徴抽出はAIに頼らない。音量・話速・間は信号処理とルールで決める。
      AIを使うのは「笑い声かどうか」の検出だけで、その出力も人が付けた正解秒数で較正する。
依存（PANNs 等）が無い・失敗した場合はセグメントを変更せずに返す（安全側）。
"""
import math
import os
import struct
import subprocess
import tempfile

SR = 4000          # 音量抽出用サンプルレート
ROLL = 5           # 相対音量の前後窓（秒）
LAUGH_KEYS = ["Laughter", "Baby laughter", "Giggle", "Snicker",
              "Belly laugh", "Chuckle, chortle"]
_DEFAULT_CKPT = os.path.expanduser("~/panns_data/Cnn14_DecisionLevelMax_mAP=0.385.pth")
FFMPEG = os.getenv("FFMPEG_PATH", "ffmpeg")


def _extract_wav(audio_path, clip_start, clip_end):
    """[clip_start, clip_end] を 0 始まりの一時 wav に切り出す。"""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        [FFMPEG, "-y", "-ss", f"{clip_start:.3f}", "-t", f"{max(0.0, clip_end - clip_start):.3f}",
         "-i", audio_path, "-vn", "-ac", "1", "-ar", "32000", tmp.name],
        check=True, capture_output=True)
    return tmp.name


def _volumes(wav_path, total_sec):
    """1秒ごとの音量（dBFS）。"""
    raw = subprocess.run(
        [FFMPEG, "-i", wav_path, "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"],
        capture_output=True).stdout
    bps = SR * 2
    vols = []
    for i in range(total_sec):
        chunk = raw[i * bps:(i + 1) * bps]
        if len(chunk) < 2:
            vols.append(-60.0)
            continue
        n = len(chunk) // 2
        samples = struct.unpack(f"{n}h", chunk[:n * 2])
        rms = (sum(s * s for s in samples) / n) ** 0.5
        vols.append(20 * math.log10(max(rms / 32768, 1e-10)))
    return vols


def rel_volumes(vols):
    """前後 ROLL 秒の平均との差。配信ごとのマイク音量の違いを打ち消す。"""
    n = len(vols)
    rel = [0.0] * n
    for i in range(n):
        w = vols[max(0, i - ROLL):min(n, i + ROLL + 1)]
        rel[i] = vols[i] - sum(w) / len(w)
    return rel


def _laughter_persec(wav_path, total_sec, ckpt):
    import librosa
    import torch
    from panns_inference import SoundEventDetection, labels
    audio, _ = librosa.load(wav_path, sr=32000)
    dur = len(audio) / 32000
    sed = SoundEventDetection(checkpoint_path=ckpt,
                              device="cuda" if torch.cuda.is_available() else "cpu")
    fw = sed.inference(audio[None, :])[0]
    idxs = [i for i, l in enumerate(labels) if l in LAUGH_KEYS]
    laugh = fw[:, idxs].max(axis=1)
    fps = fw.shape[0] / dur if dur > 0 else 1.0
    ps = []
    for s in range(total_sec):
        seg = laugh[int(s * fps):int((s + 1) * fps)]
        ps.append(float(seg.max()) if len(seg) else 0.0)
    return ps


def count_mora(text):
    """ひらがなに直して拍を数える（拗音の小書きは前の字と1拍）。"""
    import pykakasi
    hira = "".join(it["hira"] for it in pykakasi.kakasi().convert(text))
    return sum(1 for c in hira if c not in "ゃゅょ" and (("ぁ" <= c <= "ん") or c == "ー"))


def annotate_tone(segments, audio_path, clip_start=0.0, clip_end=None, ckpt=None):
    """segments の各 dict に laugh / vol_rel / mora_per_sec / gap を付けて返す。
    start/end は clip_start 起点の相対秒。失敗時は無変更で返す。"""
    if not segments:
        return segments
    ckpt = ckpt or os.getenv("PANNS_CKPT", _DEFAULT_CKPT)
    if clip_end is None:
        clip_end = clip_start + max(s["end"] for s in segments)

    wav = None
    try:
        wav = _extract_wav(audio_path, clip_start, clip_end)
        total_sec = int(math.ceil(max(s["end"] for s in segments))) + 1
        rel = rel_volumes(_volumes(wav, total_sec))
        try:
            laugh = _laughter_persec(wav, total_sec, ckpt)
        except Exception as e:
            print(f"[tone] 笑い検出スキップ ({e})")
            laugh = [0.0] * total_sec

        for i, s in enumerate(segments):
            lo, hi = int(s["start"]), min(int(s["end"]) + 1, total_sec)
            rng = range(lo, hi) if hi > lo else [min(lo, total_sec - 1)]
            s["vol_rel"] = round(max((rel[t] for t in rng), default=0.0), 1)
            s["laugh"] = round(max((laugh[t] for t in rng), default=0.0), 2)
            dur = s["end"] - s["start"]
            s["mora_per_sec"] = round(count_mora(s.get("text", "")) / dur, 1) if dur > 0 else 0.0
            nxt = segments[i + 1]["start"] if i + 1 < len(segments) else s["end"]
            s["gap"] = round(nxt - s["end"], 1)
        return segments
    except Exception as e:
        print(f"[tone] トーン付与スキップ ({e})。テキストのみで判断")
        return segments
    finally:
        if wav and os.path.exists(wav):
            os.unlink(wav)


def tone_tag(seg, laugh_thr=0.15, vol_thr=3.0, fast_mora=9.0, gap_thr=0.6):
    """トーン情報を LLM プロンプト用のタグにする。例: "<笑い0.4 大声+5dB> " """
    if "laugh" not in seg and "vol_rel" not in seg:
        return ""
    tags = []
    if seg.get("laugh", 0.0) >= laugh_thr:
        tags.append(f"笑い{seg['laugh']}")
    if seg.get("vol_rel", 0.0) >= vol_thr:
        tags.append(f"大声+{seg['vol_rel']:.0f}dB")
    if seg.get("mora_per_sec", 0.0) >= fast_mora:
        tags.append("早口")
    if seg.get("gap", 0.0) >= gap_thr:
        tags.append(f"間{seg['gap']}s")
    return f"<{' '.join(tags)}> " if tags else ""
