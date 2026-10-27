# -*- coding: utf-8 -*-
"""
削り段の学習モデル：切り抜く範囲の中の発話を、1つずつ「削る／残す」と判定する。

プロの切り抜き師が実際に何を削ったかを学習データにして、3種類の手がかりを重ねる。
  - 数値の特徴14個  … 位置・長さ・話す速さ・前後の間・フィラー/相づち・直前との重複 など
  - 声の特徴9個     … 音量・声の高さ・有声の割合・音の速さ（Praat）
  - 文章モデルの確率 … 配信の話し言葉に慣れさせた ModernBERT（trim_text.py）
  これらを XGBoost に並べて入れる（スタッキング）。LLM は使わない。

封印したテスト（人が付けた正解・63組）での AUC:
  数値だけ 0.593 / ＋声 0.627 / ＋声＋文章モデル 0.645
声の特徴は「切り抜く範囲ごと」の中央値との差にそろえる。配信全体で正規化すると、
マイクの音量や地声の違いを打ち消す意味が変わってしまうため。
"""
from __future__ import annotations

import re

import numpy as np

FILLER = re.compile(r"^(え+|あ+|う+|お+|ん+|えっと|えーと|あの+|まあ|まぁ)+$")
BACKCH = re.compile(r"^(うん+|はい|そう|そうそう|ね+|へ+|ふーん|ほー|なるほど|ああ|おお|確かに|たしかに)+$")
NUM_NAMES = ["相対位置", "端までの距離", "発話数", "長さ秒", "文字数", "話す速さ", "前の間", "後の間",
             "フィラーだけ", "相づちだけ", "疑問", "笑い", "直前との重なり", "直後との重なり"]
VOICE_NAMES = ["音量平均", "音量最大", "音量ばらつき", "高さ中央", "高さ幅", "有声割合", "音の速さ",
               "前との音量差", "前との高さ差"]

XGB_PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss")


# ─────────────────────────────────────────────────────────────
# 数値の特徴
# ─────────────────────────────────────────────────────────────
def norm(s: str) -> str:
    return re.sub(r"[\s、。！？!?…・「」『』（）()]+", "", s or "")


def grams(s: str) -> set:
    s = norm(s)
    return {s[i:i + 3] for i in range(len(s) - 2)} or ({s} if s else set())


def overlap(a: str, b: str) -> float:
    """a の文字3-gramのうち b にも現れる割合（言い直し・繰り返しの手がかり）"""
    ga = grams(a)
    return len(ga & grams(b)) / len(ga) if ga else 0.0


def numeric_features(units: list[dict], i: int) -> list[float]:
    """i 番目の発話（範囲の最初と最後は除く）の14個"""
    n, u = len(units), units[i]
    t = norm(u["text"])
    d = max(u["end"] - u["start"], 0.1)
    return [i / (n - 1), min(i, n - 1 - i) / n, n, d, len(t), len(t) / d,
            max(0.0, u["start"] - units[i - 1]["end"]), max(0.0, units[i + 1]["start"] - u["end"]),
            float(bool(FILLER.match(t))), float(bool(BACKCH.match(t))),
            float("?" in u["text"] or "？" in u["text"]),
            float(bool(re.search(r"(あは|はは|ふふ|笑|w{2,})", u["text"]))),
            overlap(u["text"], units[i - 1]["text"]), overlap(u["text"], units[i + 1]["text"])]


# ─────────────────────────────────────────────────────────────
# 声の特徴
# ─────────────────────────────────────────────────────────────
SR, STEP, F0_MIN, F0_MAX, PEAK_PROM = 16000, 0.01, 75, 600, 2.0


def voice_features(units: list[dict], y: np.ndarray, base: float) -> list[list]:
    """範囲の音 y（16kHz モノラル, 時刻 base から）から発話ごとに9個。取れない所は None"""
    import parselmouth
    from scipy.signal import find_peaks
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)
    it = snd.to_intensity(minimum_pitch=F0_MIN, time_step=STEP)
    db, tdb = it.values[0], it.xs() + base
    pt = snd.to_pitch(time_step=STEP, pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
    f0, tf0 = pt.selected_array["frequency"], pt.xs() + base
    st = np.where(f0 > 0, 12 * np.log2(np.maximum(f0, 1e-6) / 100.0), np.nan)   # 半音
    db_med = np.median(db)
    st_med = np.nanmedian(st) if np.isfinite(st).any() else np.nan
    # 音の速さ＝有声の音量の山の数 ÷ 秒（de Jong & Wempe 2009 の簡易版）
    pk, _ = find_peaks(db, prominence=PEAK_PROM, height=db_med)
    pk_t = tdb[pk][np.interp(tdb[pk], tf0, np.isfinite(st).astype(float)) > 0.5]
    rows, prev = [], None
    for u in units:
        d = max(u["end"] - u["start"], 0.1)
        m, mf = (tdb >= u["start"]) & (tdb < u["end"]), (tf0 >= u["start"]) & (tf0 < u["end"])
        v = db[m] if m.any() else np.array([db_med])
        s = st[mf]
        s = s[np.isfinite(s)]
        vol = float(np.mean(v) - db_med)
        hi = float(np.median(s) - st_med) if len(s) >= 5 else np.nan
        r = [vol, float(np.max(v) - db_med), float(np.std(v)), hi,
             float(np.percentile(s, 90) - np.percentile(s, 10)) if len(s) >= 5 else np.nan,
             float(mf.sum() and np.isfinite(st[mf]).mean()),
             float(((pk_t >= u["start"]) & (pk_t < u["end"])).sum() / d),
             vol - prev[0] if prev else np.nan,
             hi - prev[3] if prev and np.isfinite(hi) and np.isfinite(prev[3]) else np.nan]
        rows.append([None if not np.isfinite(x) else round(x, 4) for x in r])
        prev = r
    return rows


# ─────────────────────────────────────────────────────────────
# 重ねる（スタッキング）
# ─────────────────────────────────────────────────────────────
def row(units, i, voice=None, text_prob=None) -> list[float]:
    """1発話ぶんの入力。列の順は 数値14 → 文章モデル → 声9（学習と推論で必ず同じ関数を通す）"""
    nan = float("nan")
    v = [nan if a is None else float(a) for a in (voice or [None] * 9)]
    return numeric_features(units, i) + [nan if text_prob is None else float(text_prob)] + v


def train(clips: list[dict], voice: dict, text_oof: dict, seed: int = 0):
    """clips: [{"clip_id", "units": [{"start","end","text","keep"}]}]
    text_oof は文章モデルの予測。学習に使った組を自分で予測した値を入れると過学習するので、
    配信単位で分割し「その組を見ていないモデル」の予測（out-of-fold）を使う。"""
    import xgboost as xgb
    # ──（省略）本体では、配信単位の5分割で予測を作って削る割合が正解と同じになるしきい値を決め、モデルと一緒に保存する ──
    X, y = [], []
    for c in clips:
        U = c["units"]
        for i in range(1, len(U) - 1):               # 範囲の最初と最後は判定しない
            X.append(row(U, i, voice[c["clip_id"]][i], text_oof[c["clip_id"]][i]))
            y.append(0 if U[i]["keep"] else 1)
    return xgb.XGBClassifier(**XGB_PARAMS, random_state=seed).fit(np.array(X), np.array(y))


def keep_mask(model, units, voice, text_probs, threshold: float) -> list[bool]:
    """発話ごとに残すか。最初と最後は必ず残す。
    しきい値は「削る割合が学習データの正解と同じになる値」を out-of-fold の予測から決めておく。"""
    # ──（省略）本体では、音声や文章モデルが使えないときに「数値だけ」「数値＋声」のモデルへ自動で切り替え、切り替えたことをログに残す ──
    X = np.array([row(units, i, voice[i], text_probs[i]) for i in range(1, len(units) - 1)])
    p = model.predict_proba(X)[:, 1]
    return [True] + [bool(x < threshold) for x in p] + [True]
