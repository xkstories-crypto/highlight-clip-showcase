# -*- coding: utf-8 -*-
"""
削り段の文章モデル：範囲の発話を1列に並べ、各発話の「削る確率」を一度に出す。

  [CLS] <utt> 発話1 <utt> 発話2 <utt> 発話3 ... [SEP]
             ↑            ↑            ↑   この位置の出力 → 線形層 → 削る確率

発話を1つずつ判定するのではなく範囲全体を読ませるので、「前と同じことを言っている」
「この一言がないと次がつながらない」といった前後関係を見られる。

元のモデルは ModernBERT（日本語 310M）を、正解ラベルの無い配信の書き起こし約100時間で
追加学習（DAPT）して話し言葉に慣れさせたもの。慣れさせる前より、まっさらなテストで 66.7 → 69.3。
400時間に増やしても良くならなかった。
乱数を変えて3回学習し、予測を平均する（作り直すたびに予測が揺れるため）。
"""
from __future__ import annotations

import torch

MARK = "<utt>"
MAXLEN = 4096


def encode(tok, units: list[dict], labels: list[int] | None = None) -> dict:
    mark_id = tok.convert_tokens_to_ids(MARK)
    ids, pos = [tok.cls_token_id], []
    for u in units:
        pos.append(len(ids))
        ids += [mark_id] + tok(u["text"], add_special_tokens=False)["input_ids"]
    ids = (ids + [tok.sep_token_id])[:MAXLEN]
    keep = [i for i, p in enumerate(pos) if p < MAXLEN]
    out = {"ids": ids, "pos": [pos[i] for i in keep]}
    if labels is not None:
        out["y"] = [labels[i] for i in keep]
    return out


class TrimEncoder(torch.nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(enc.config.hidden_size, 1))

    def forward(self, ids, att, pos_list):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
            hs = self.enc(input_ids=ids, attention_mask=att).last_hidden_state
        # 各発話の目印の位置だけを取り出して1つの値にする
        return [self.head(hs[b, p].float()).squeeze(-1) for b, p in enumerate(pos_list)]


def loss_fn(batch, logits):
    """1組の中の全発話について、削る=1 / 残す=0 の二値交差エントロピー"""
    return sum(torch.nn.functional.binary_cross_entropy_with_logits(
        l, torch.tensor(x["y"], dtype=torch.float, device=l.device)) for x, l in zip(batch, logits)) / len(batch)

# ──（省略）学習の繰り返し処理（バッチ作成・早期終了・重みの保存・5分割の予測づくり）は本体のみ ──
# 学習設定: AdamW lr 3e-5 / weight_decay 0.01 / 線形減衰（最初の10%で立ち上げ）/
#           最大10周・検証のAUCが3周伸びなければ止める / バッチ4 / 勾配の大きさを1.0で抑える
