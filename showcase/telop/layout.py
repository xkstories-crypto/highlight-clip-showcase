# -*- coding: utf-8 -*-
"""
テロップの組版：字幕1セグメントを「1枚ずつ見せるテロップ」に割り、折り返し、時間を配る。

プロの切り抜き動画（12本を目視実測）に寄せるための規則:
  - 1テロップ＝1フレーズ。1行7〜9字 × 最大3行、1文まるごと1枚が基本
  - 句読点は打たない（改行とカットで間を作る）。ただし ?!…〜 は感情なので残す
  - 単語の途中では絶対に割らない（janome の文節境界 → 形態素境界 → 文字 の順に落とす）
  - 文字サイズは音量と無相関で、行の字数の従属変数（r=-0.52）
    ＝「文字を大きくする」とは「1テロップを短く割る」こと

すべて決定論（AIを使わない）。同じ入力からは必ず同じテロップが出る。
"""
from __future__ import annotations

import re

try:
    from janome.tokenizer import Tokenizer as _JTokenizer
    _JT = _JTokenizer()
except Exception:          # janome が無ければ1文字ずつに落ちる
    _JT = None

# ── 寸法（1080x1920 の縦型ショート基準）───────────────────────────
CANVAS_W = 1080
WIDTH_RATIO = 0.8          # 1行が使える画面幅の割合（SNSのセーフエリア目安）
FONT_PX = 176              # 基準の文字高
MIN_PX = 100               # 語を割らないための縮小はここまで
MAX_LINES = 3
LETTER_SPACING = 0.03      # 字間／文字サイズ

# ── フレーズ分割 ───────────────────────────────────────────────
TARGET_CHARS = 14          # 1テロップの目標字数
MAX_CHARS = 18             # これを超えたら必ず割る
MIN_CHARS = 3              # これ未満の欠片は隣とくっつける（チカチカ防止）
MAX_CARDS_PER_SENTENCE = 2 # 1文あたりの枚数上限（＝切り替え1回）
MIN_PHRASE_SEC = 0.45      # 1テロップの最低表示秒

# 重なり解消
MIN_VISIBLE_SEC = 0.8      # 前のテロップに残す最低表示秒
MERGE_GAP_SEC = 0.8        # 同じ文の断片をまとめる最大ギャップ

# 直前の自立語にくっつける品詞（付属語）
_ATTACH_POS = ("助詞", "助動詞", "記号")
_ATTACH_SUBPOS = ("接尾", "非自立")

# 禁則: 行頭に来てはいけない文字 / 行末に来てはいけない文字
_NO_LINE_START = set("、。，．,.・：；:;?？!！)）]｝}」』】〕》〉〞”ー～…‥ゝゞ々ぁぃぅぇぉっゃゅょゎ゛゜")
_NO_LINE_END = set("(（[｛{「『【〔《〈“〝")

# 区切ってよい助詞・接続（この直後で切ると意味が壊れにくい）
_BREAK_TAIL = ("けど", "けれど", "から", "ので", "のに", "たら", "れば", "ながら",
               "って", "では", "には", "とか", "みたいな",
               "は", "が", "を", "に", "で", "と", "も", "ね", "よ", "さ")
# 行頭に立てる接続詞（この直前で切る）
_BREAK_HEAD = ("でも", "だから", "だけど", "それで", "そんで", "つまり", "ていうか",
               "というか", "ってか", "あと", "まあ")
# 先頭の言い淀み。落とさずに独立した1枚にして「タメ」に使う（プロ版の実測）
_LEAD_FILLER = ("なんか", "えーと", "えっと", "えー", "あのー", "あの", "その",
                "まあ", "うーん", "んー")
# ASR が空白を置くのは間・文の切れ目。ここも1文の境界として扱う
_PUNCT_BREAK = "、。，．,.！!？?・… 　"
_EMPH_MARKS = ("!!", "!?", "?!", "！！", "!", "！")
# 4回以上の連続反復だけ畳む（「はいはいはい」等の実発話は壊さない）
_REPEAT_RE = re.compile(r"(.{1,20}?)\1{3,}")


# ─────────────────────────────────────────────────────────────
# 単語境界
# ─────────────────────────────────────────────────────────────
def char_units(s: str) -> float:
    """全角=1.0, 半角ASCII=0.5 で幅を数える。"""
    return sum(0.5 if 0x20 <= ord(ch) < 0x7f else 1.0 for ch in s)


def word_chunks(text: str) -> list[str]:
    """文節らしいチャンク列。自立語で区切り、付属語は前にくっつける。"""
    if _JT is None:
        # ──（省略）本体では janome が無いとき pykakasi で区切り、それも無ければ1文字ずつにする ──
        return list(text)
    chunks, cur, prev_prefix = [], "", False
    for tok in _JT.tokenize(text):
        f = tok.part_of_speech.split(",")
        pos0, pos1 = f[0], (f[1] if len(f) > 1 else "")
        attach = pos0 in _ATTACH_POS or pos1 in _ATTACH_SUBPOS
        if cur and not attach and not prev_prefix:
            chunks.append(cur)
            cur = ""
        cur += tok.surface
        prev_prefix = (pos0 == "接頭詞")
    if cur:
        chunks.append(cur)
    return [c for c in chunks if c] or list(text)


def _morphemes(text: str) -> list[str]:
    """形態素（表層形）列。文節より細かい、意味を壊しにくい第2の境界。"""
    if _JT is None:
        return list(text)
    return [t.surface for t in _JT.tokenize(text) if t.surface] or list(text)


# ─────────────────────────────────────────────────────────────
# 折り返し
# ─────────────────────────────────────────────────────────────
def wrap_words(text: str, max_units: float, hard_out: list | None = None) -> list[str]:
    """文節境界でのみ折り返す。禁則処理つき。
    文節が max_units を超えたら 形態素境界 → 文字 の順に落とし、
    語を断ち切ったら hard_out に True を積む（呼び出し側はフォントを縮めて救う）。"""
    lines, cur = [], ""
    for ch in word_chunks(text):
        cu = char_units(ch)
        if cu > max_units:                         # 文節が長すぎる → 形態素で割る
            if cur:
                lines.append(cur); cur = ""
            buf = ""
            for mo in _morphemes(ch):
                if char_units(mo) > max_units:     # 形態素単体で溢れる → 文字割り
                    if buf:
                        lines.append(buf); buf = ""
                    if hard_out is not None:
                        hard_out.append(True)
                    for c in mo:
                        if buf and char_units(buf + c) > max_units:
                            lines.append(buf); buf = c
                        else:
                            buf += c
                    continue
                if buf and char_units(buf) + char_units(mo) > max_units:
                    if hard_out is not None:
                        hard_out.append(True)
                    lines.append(buf); buf = mo
                else:
                    buf += mo
            cur = buf
            continue
        if cur and char_units(cur) + cu > max_units:
            lines.append(cur); cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)

    # 禁則: 行頭NGは前行末尾へ寄せる / 行末NGは次行へ送る。
    # 「ー」1文字だけの行は見苦しいので幅を超えてでも前行へ寄せる。
    # ただし文字割りが起きた語（「えぇぇぇ…」）ではどこで切っても行頭NGになり、
    # 全部が前行に吸い込まれて1行に潰れるので、その場合は寄せない。
    allow_orphan_pull = not (hard_out is not None and hard_out)
    fixed: list[str] = []
    for ln in lines:
        while fixed and ln and ln[0] in _NO_LINE_START and (
                char_units(fixed[-1]) + char_units(ln[0]) <= max_units
                or (len(ln) == 1 and allow_orphan_pull)):
            fixed[-1] += ln[0]; ln = ln[1:]
        moved = ""
        while ln and ln[-1] in _NO_LINE_END:
            moved = ln[-1] + moved; ln = ln[:-1]
        if ln:
            fixed.append(ln)
        if moved:
            fixed.append(moved)
    return [x for x in fixed if x] or ([text] if text else [])


def fit_card(text: str, font_px: int = FONT_PX, canvas_w: int = CANVAS_W) -> tuple[list[str], int]:
    """1枚のテロップを画面幅に収める行分けと文字サイズを決める。

    縮小の探索は3段階:
      ① 下限 MIN_PX までの範囲で「文節境界だけで折り返せる」サイズを探す
      ② 無ければ同じ範囲で文節内の折り返しを許し、大きい方から採る
         （文字が小さくなりすぎる方が、語が割れるより見た目の損失が大きい）
      ③ それでも入らない長文だけ下限を割って縮める
    """
    def chars_per_line(px: int) -> float:
        advance = px * (0.9 + LETTER_SPACING)
        return max(1.0, canvas_w * WIDTH_RATIO / advance)

    def fit(px: int, allow_hard: bool) -> tuple[list[str], bool]:
        cpl = chars_per_line(px)
        hard: list = []
        lines = wrap_words(text, cpl, hard)
        ok = len(lines) <= MAX_LINES and max(map(char_units, lines), default=0) <= cpl + 1e-6
        return lines, ok and (allow_hard or not hard)

    for allow_hard in (False, True):
        for px in range(font_px, MIN_PX - 1, -4):
            lines, ok = fit(px, allow_hard)
            if ok:
                return lines, px
    for px in range(MIN_PX - 4, 23, -4):
        lines, ok = fit(px, True)
        if ok:
            return lines, px
    return wrap_words(text, chars_per_line(24)), 24


# ─────────────────────────────────────────────────────────────
# フレーズ分割
# ─────────────────────────────────────────────────────────────
def _strip_punct(text: str) -> str:
    for ch in "、。，．,.・":
        text = text.replace(ch, " ")
    return " ".join(text.split())


def _peel_lead_filler(text: str) -> tuple[str, str]:
    """先頭の言い淀みを切り出して (言い淀み, 残り) を返す。落とさずに分ける。"""
    lead, changed = "", True
    while changed:
        changed = False
        for f in _LEAD_FILLER:
            if text.startswith(f) and len(text) > len(f) + 1:
                lead += f
                text = text[len(f):].lstrip("、 　")
                changed = True
    return lead, text


def _limit_cards(pieces: list[str], budget: int) -> list[str]:
    """1文が budget 枚を超えたら、隣同士で最も短くなる組から結合して収める。"""
    out = list(pieces)
    while budget > 0 and len(out) > budget:
        i = min(range(len(out) - 1), key=lambda k: char_units(out[k]) + char_units(out[k + 1]))
        out[i:i + 2] = [out[i] + out[i + 1]]
    return out


def _split_one(part: str) -> list[str]:
    """句読点で切った断片を、接続詞の直前・助詞の直後・単語境界でさらに割る。"""
    if char_units(part) <= TARGET_CHARS:
        return [part]
    chunks = word_chunks(part)
    out, cur = [], ""
    for i, ch in enumerate(chunks):
        nxt = "".join(chunks[i:])
        if cur and char_units(cur) >= MIN_CHARS and any(nxt.startswith(h) for h in _BREAK_HEAD):
            out.append(cur); cur = ""
        if cur and char_units(cur) + char_units(ch) > MAX_CHARS:
            out.append(cur); cur = ""
        cur += ch
        rest = "".join(chunks[i + 1:])
        if not rest:
            break
        cu = char_units(cur)
        if cu >= TARGET_CHARS * 0.6 and cur.endswith(_BREAK_TAIL) and char_units(rest) >= MIN_CHARS:
            out.append(cur); cur = ""
            continue
        if cu >= MAX_CHARS:
            out.append(cur); cur = ""
    if cur:
        out.append(cur)
    return out or [part]


def split_phrases(text: str) -> list[str]:
    """1セグメントのテキストを「1テロップ＝1フレーズ」に割る。
    優先順: ①句読点 ②接続詞の直前 ③助詞の直後 ④単語境界での強制分割。"""
    text, _ = collapse_repeats(text.strip())
    lead, text = _peel_lead_filler(text)
    # 言い淀みを独立させると本体が長くなりすぎる場合は、独立をあきらめて本体に戻す
    if lead and char_units(text) > MAX_CHARS:
        lead, text = "", lead + text
    if not text:
        return [lead] if lead else []

    rough, buf = [], ""
    for ch in text:
        if ch in _PUNCT_BREAK:
            if ch in "！!？?…":
                buf += ch
            if buf.strip():
                rough.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        rough.append(buf.strip())

    out: list[str] = []
    for i, part in enumerate(rough):
        budget = MAX_CARDS_PER_SENTENCE - (1 if (lead and i == 0) else 0)
        out.extend(_limit_cards(_split_one(_strip_punct(part)), max(1, budget)))

    merged: list[str] = []
    for p in out:
        if merged and char_units(p) < MIN_CHARS and char_units(merged[-1]) + char_units(p) <= MAX_CHARS:
            merged[-1] += p
        else:
            merged.append(p)
    if len(merged) >= 2 and char_units(merged[0]) < MIN_CHARS and \
            char_units(merged[0]) + char_units(merged[1]) <= MAX_CHARS:
        merged[1] = merged[0] + merged[1]
        merged.pop(0)
    if lead:
        if merged and char_units(lead) < MIN_CHARS:
            merged[0] = lead + merged[0]
        else:
            merged.insert(0, lead)
    return [m for m in merged if m.strip()]


def distribute_time(phrases: list[str], start: float, end: float) -> list[tuple[float, float, str]]:
    """フレーズ列に区間 [start, end] を字数比で配る。
    最低表示秒を割るときは、割れるだけの本数まで隣同士を結合する。"""
    if not phrases:
        return []
    phrases = list(phrases)
    span = max(0.0, end - start)
    max_n = max(1, int(span / MIN_PHRASE_SEC))
    while len(phrases) > max_n:
        i = min(range(len(phrases) - 1),
                key=lambda k: char_units(phrases[k]) + char_units(phrases[k + 1]))
        phrases[i:i + 2] = [phrases[i] + phrases[i + 1]]
    units = [max(char_units(p), 1.0) for p in phrases]
    total = sum(units)
    out, t = [], start
    for p, u in zip(phrases, units):
        d = span * u / total
        out.append((t, min(end, t + d), p))
        t += d
    out[-1] = (out[-1][0], end, out[-1][2])
    return out


# ─────────────────────────────────────────────────────────────
# 見せ方の判定（記号と長さだけで決める。AIに頼らない）
# ─────────────────────────────────────────────────────────────
def is_emphasis(text: str) -> bool:
    """特大で見せる札か。プロは1本に数回しか使わないので条件は絞る。"""
    t = (text or "").strip()
    if any(m in t for m in ("!!", "!?", "?!", "！！")):
        return True
    return bool(t) and char_units(t) <= 8 and t.endswith(_EMPH_MARKS)


def is_weak(text: str) -> bool:
    """札の中身が言い淀みだけなら「弱め」（小さく・細い書体）で見せる。"""
    t = (text or "").strip()
    if not t:
        return False
    while t:
        for f in _LEAD_FILLER:
            if t.startswith(f):
                t = t[len(f):].lstrip("、 　")
                break
        else:
            return False
    return True


def collapse_repeats(text: str) -> tuple[str, bool]:
    """音声認識のリピート幻聴（同じ片の4回以上の連続）を2回に畳む。"""
    new = _REPEAT_RE.sub(lambda m: m.group(1) * 2, text)
    return new, new != text


# ─────────────────────────────────────────────────────────────
# 重なり解消（話者が被ると同時刻に2枚出て重ね描きになる）
# ─────────────────────────────────────────────────────────────
def merge_same_text(items: list[list]) -> list[list]:
    """近接する同じ文の断片を1枚に畳む。items: [[start, end, text], ...]"""
    out: list[list] = []
    for e in sorted(items, key=lambda x: (x[0], x[1])):
        hit = next((p for p in reversed(out)
                    if p[2] == e[2] and e[0] <= p[1] + MERGE_GAP_SEC), None)
        if hit:
            hit[0], hit[1] = min(hit[0], e[0]), max(hit[1], e[1])
        else:
            out.append(list(e))
    return sorted(out, key=lambda x: x[0])


def _split_union(prev: list, e: list) -> bool:
    """クロストーク2件を、合計区間内で字数比に応じて前後へ並べ直す。"""
    us, ue = min(prev[0], e[0]), max(prev[1], e[1])
    if ue - us < 2 * MIN_VISIBLE_SEC:
        return False
    wa, wb = max(char_units(prev[2]), 1.0), max(char_units(e[2]), 1.0)
    cut = us + (ue - us) * wa / (wa + wb)
    cut = min(max(cut, us + MIN_VISIBLE_SEC), ue - MIN_VISIBLE_SEC)
    prev[0], prev[1] = us, cut
    e[0], e[1] = cut, ue
    return True


def resolve_overlaps(items: list[list], dropped: list | None = None) -> list[list]:
    """画面に出るテロップを常に1枚にする。
      A) 前の札が十分見えている → 前を次の開始で打ち切る（逐次発話）
      B) ほぼ同時に始まる（クロストーク） → 合計区間に並べ直し、無理なら長い方を残す
    捨てた分は dropped に積む（黙ってテキストを失わない）。"""
    items = sorted((list(x) for x in items), key=lambda x: (x[0], -(x[1] - x[0])))
    out: list[list] = []
    for e in items:
        keep = True
        while keep and out and e[0] < out[-1][1] - 1e-6:
            prev = out[-1]
            if e[0] - prev[0] >= MIN_VISIBLE_SEC:
                prev[1] = e[0]
                break
            if _split_union(prev, e):
                break
            if (e[1] - e[0]) <= (prev[1] - prev[0]):
                if dropped is not None:
                    dropped.append(list(e) + ["crosstalk_shorter"])
                keep = False
            else:
                if dropped is not None:
                    dropped.append(list(prev) + ["crosstalk_shorter"])
                out.pop()
        if keep and e[1] - e[0] >= 0.05:
            out.append(e)
        elif keep and dropped is not None:
            dropped.append(list(e) + ["too_short"])
    return out


def build_cards(segments: list[dict]) -> list[dict]:
    """字幕セグメント列 → 表示するテロップ札の列（組版の入口）。
    segments: [{"start": 秒, "end": 秒, "text": str}, ...]"""
    items = []
    for s in segments:
        for t0, t1, p in distribute_time(split_phrases(s["text"]), s["start"], s["end"]):
            items.append([t0, t1, p])
    cards = []
    for t0, t1, p in resolve_overlaps(merge_same_text(items)):
        lines, px = fit_card(p)
        cards.append({"start": round(t0, 3), "end": round(t1, 3), "text": p,
                      "lines": lines, "font_px": px,
                      "emph": is_emphasis(p), "weak": is_weak(p)})
    # ──（省略）本体ではこのあと、話者ごとの色・強調/弱めの書体の切り替え・笑い声に合わせた「www」テロップを付ける ──
    return cards
