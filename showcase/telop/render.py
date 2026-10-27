# -*- coding: utf-8 -*-
"""
テロップの描画：距離変換で縁取りしたPNGを作り、ffmpeg で動画に重ねる。

なぜ ASS（字幕スタイル）の縁取りをやめたか:
  ASS の \\bord は「1文字ずつ描いた縁の和集合」なので、隣の字と縁が融合した所で
  外側の黒が内側の色フチに食われ、幅が一定にならない。字間を空けて逃げても限界がある。
  → 文字の形を一度マスクに描き、そこからの「距離」で帯を塗り分ける。
    帯は輪郭からの距離で定義されるので、字がくっついても幅は必ず一定になる。

縁は4層（プロの切り抜き11本・編集者10人の実測）:
  白の塗り → 黒(内) → 話者色 → 黒(外)
  幅は文字サイズに比例させる（固定幅だと小さい札でフチが塗りを食い潰す）。

出現アニメは「ところどころ・タイミングはバラバラ」:
  プロは基本「入りの演出なし（即時）」で、高速テンポで一律にポップさせるともたつく。
  一部の札だけをポップ／スライドさせ、強さ・速さも札ごとに散らす。
  乱数の種は札の開始秒なので、焼き直しても同じ結果になる。
"""
from __future__ import annotations

import random
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

SUPERSAMPLE = 2            # 2倍で描いて縮小＝輪郭のアンチエイリアス
CORE_RATIO = 0.072         # 黒(内)フチ幅／文字サイズ
INNER_RATIO = 0.061        # 話者色フチ幅／文字サイズ
OUTER_RATIO = 0.067        # 黒(外)フチ幅／文字サイズ
LINE_RATIO = 1.15          # 行送り／文字サイズ
LETTER_SPACING = 0.03      # 字間／文字サイズ

# ポップ: (出だし%, 行き過ぎ%, 定位置%, 行き過ぎまでms, 定位置までms, 加速の指数)
POP = (62, 106, 100, 110, 190, 1.6)
POP_EMPH = (50, 116, 100, 120, 210, 1.7)   # 強調札は大きく跳ねる
POP_RATE = 0.28            # アニメを入れる札の割合
SLIDE_RATE = 0.35          # そのうちスライドにする割合


def _rgb(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def render_card(lines: list[str], font_path: str, font_px: float,
                fill: str = "#FFFFFF", color: str = "#FF4FA0",
                core: str = "#000000", outer: str = "#000000") -> Image.Image:
    """テロップ1枚を RGBA 画像で返す。"""
    ss = SUPERSAMPLE
    f = ImageFont.truetype(font_path, int(round(font_px * ss)))
    ls = font_px * LETTER_SPACING * ss
    core_px, inner_px, outer_px = (font_px * r * ss for r in (CORE_RATIO, INNER_RATIO, OUTER_RATIO))

    def line_w(s: str) -> float:
        return max(0.0, sum(f.getlength(c) + ls for c in s) - ls) if s else 0.0

    asc, desc = f.getmetrics()
    step = font_px * LINE_RATIO * ss
    W = int(max((line_w(s) for s in lines), default=1))
    H = int(step * (len(lines) - 1) + asc + desc)
    pad = int(core_px + inner_px + outer_px + 6 * ss)
    cw, ch = W + pad * 2, H + pad * 2

    # 1) 文字の形だけをマスクに描く
    mask = Image.new("L", (cw, ch), 0)
    d = ImageDraw.Draw(mask)
    for i, s in enumerate(lines):
        x = pad + (W - line_w(s)) / 2.0
        for c in s:
            d.text((x, pad + i * step), c, font=f, fill=255)
            x += f.getlength(c) + ls

    # 2) 文字の外側の各画素について「輪郭までの距離」を求める
    m = np.array(mask)
    dist = cv2.distanceTransform((m < 128).astype(np.uint8), cv2.DIST_L2, 5)

    # 3) 距離で帯を塗り分ける（外側から順に上書き）
    w0 = core_px
    w1 = w0 + inner_px
    w2 = w1 + outer_px
    rgb = np.zeros((ch, cw, 3), dtype=np.uint8)
    rgb[dist <= w2] = _rgb(outer)
    rgb[dist <= w1] = _rgb(color)
    rgb[dist <= w0] = _rgb(core)
    rgb[m >= 128] = _rgb(fill)
    alpha = np.where(dist <= w2, 255, 0).astype(np.uint8)

    im = Image.fromarray(np.dstack([rgb, alpha]))
    return im.resize((cw // ss, ch // ss), Image.LANCZOS)


# ─────────────────────────────────────────────────────────────
# 出現アニメ
# ─────────────────────────────────────────────────────────────
def anim_for(card: dict, idx: int):
    """この札に付ける出現アニメ。None なら即時表示。
    戻り値: ("pop", プロファイル) / ("slide", 持ち上げ量px, 秒) / None"""
    emph = bool(card.get("emph"))
    rnd = random.Random(f"{idx}:{card.get('start', 0):.3f}")
    if not emph and rnd.random() > POP_RATE:
        return None
    if not emph and rnd.random() < SLIDE_RATE:
        return ("slide", rnd.uniform(26, 70), rnd.uniform(0.10, 0.20))
    s0, s1, s2, t1, t2, _ = POP_EMPH if emph else POP
    s0 = max(30.0, s0 + rnd.uniform(-12, 10))          # 出だしの小ささ
    s1 = max(s2 + 1.0, s1 + rnd.uniform(-3, 5))        # 行き過ぎ量
    t1 = max(50.0, t1 * rnd.uniform(0.70, 1.30))       # 立ち上がりの速さ
    t2 = t1 + max(40.0, (t2 - t1) * rnd.uniform(0.70, 1.35))
    return ("pop", (s0, s1, s2, t1, t2, rnd.uniform(1.3, 1.9)))


def pop_scale_expr(prof: tuple, start: float) -> str:
    """出現時の拡大率を、時刻 t の ffmpeg 式として組み立てる。
    s0 →（加速して）s1 → s2 と動き、以降は s2 で固定。"""
    s0, s1, s2, t1_ms, t2_ms, acc = prof
    t1, t2 = t1_ms / 1000.0, t2_ms / 1000.0
    u = f"max(0,t-{start:.3f})"
    rise = f"{s0:g}+{s1 - s0:g}*pow({u}/{t1:g},{acc:g})"
    fall = f"{s1:g}+{s2 - s1:g}*({u}-{t1:g})/{max(t2 - t1, 1e-3):g}"
    return f"(if(lt({u},{t1:g}),{rise},if(lt({u},{t2:g}),{fall},{s2:g}))/100)"


def slide_y_expr(bottom: float, start: float, dy: float, dur: float) -> str:
    """下から定位置へ持ち上がる y の式。終わりに向かって減速する。"""
    u = f"max(0,t-{start:.3f})"
    return f"{bottom:.0f}-h+{dy:.0f}*pow(max(0,1-{u}/{dur:g}),2)"


def _probe_fps(video_path: str, default: float = 30.0) -> float:
    try:
        r = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path],
                           capture_output=True, text=True)
        num, _, den = r.stdout.strip().partition("/")
        fps = float(num) / float(den or 1)
        return fps if 1.0 <= fps <= 240.0 else default
    except Exception:
        return default


def burn(video_path: str, cards: list[dict], out_path: str) -> None:
    """描画済みPNGを時間指定つきで動画に重ねる。
    cards: [{"png", "start", "end", "cx": 中心X, "bottom": 下端Y, "w", "h", "emph"}]

    文字を描き直すのではなく、縁取りごと描き上がったPNGレイヤーを拡大縮小する
    （編集ソフトと同じ考え方）。描画結果に触れないので縁の均一さは保たれる。
    overlay の x,y を「中心X・下端Y固定」の式で書くので、拡大の軸は下中央になる。"""
    fps = _probe_fps(video_path)
    inputs, chain, cur = [], [], "[0:v]"
    for i, c in enumerate(cards):
        anim = anim_for(c, i)
        tag = f"[v{i}]" if i < len(cards) - 1 else "[vout]"
        enable = f"enable='between(t,{c['start']:.3f},{c['end']:.3f})'"
        if not anim:
            inputs += ["-i", c["png"]]
            chain.append(f"{cur}[{i + 1}:v]overlay=x={c['cx']}-w/2:y={c['bottom']}-h:{enable}{tag}")
            cur = tag
            continue
        # 表示区間ぶんだけ静止画をループさせ、-itsoffset で本編の時刻に合わせる
        # → 式の t がそのまま本編の絶対時刻になる
        dur = max(0.05, c["end"] - c["start"]) + 0.1
        inputs += ["-loop", "1", "-framerate", f"{fps:.5f}", "-t", f"{dur:.3f}",
                   "-itsoffset", f"{c['start']:.3f}", "-i", c["png"]]
        if anim[0] == "pop":
            s = pop_scale_expr(anim[1], c["start"])
            chain.append(f"[{i + 1}:v]scale=w='max(2,{c['w']}*{s})':h='max(2,{c['h']}*{s})':"
                         f"eval=frame:flags=bicubic[p{i}]")
            src, y_expr = f"[p{i}]", f"{c['bottom']}-h"
        else:
            src, y_expr = f"[{i + 1}:v]", slide_y_expr(c["bottom"], c["start"], anim[1], anim[2])
        # x/y は必ず引用符で囲む。pow(...,2) のカンマをフィルタ区切りと誤読されるため
        chain.append(f"{cur}{src}overlay=x='{c['cx']}-w/2':y='{y_expr}':eval=frame:{enable}{tag}")
        cur = tag
    cmd = ([FFMPEG, "-y", "-v", "error", "-i", video_path] + inputs +
           ["-filter_complex", ";".join(chain), "-map", "[vout]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-c:a", "copy", out_path])
    # ──（省略）本体では GPU（NVENC）で書き出し、失敗したら CPU でやり直す ──
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"テロップ合成に失敗: {r.stderr[-400:]}")
    # ──（省略）本体ではここで札ごとの上端の位置を記録し、話者アイコンがテロップに重ならないよう追いかけさせる ──
