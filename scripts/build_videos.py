"""JRA公式YouTubeの重賞のレース映像（data/youtube/stakes_<年>.txt）を、成績のレース（race_id）に紐付けて、data/videos.csv に書く

使い方:
    python scripts/build_videos.py

映像のタイトル（例「2025年 天皇賞（秋）（GⅠ） | マスカレードボール」）のレース名と、成績PDFの見出しから取ったレース名（data/race_extra_<年>.csv の race_name）を、
同じ年の中で突き合わせる。紐付けが正しいかは、勝ち馬の名前（映像の「 | 」の後ろと、成績CSVの1着馬）で確かめ、一致しないものは書かない。
列: race_id（年付き。train.load の race_id と同じ）, video_id, title, url
"""
import re
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


# 映像のタイトルと、成績PDFの見出しで、名前が違うレース（正式名と通称、外字・表記の違い）
ALIAS = {"日本ダービー": "東京優駿", "オークス": "優駿牝馬", "ダービー卿チャレンジトロフィー": "ダービーチャレンジトロフィー"}


def norm(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"[(](?:G|J・G)(?:III|II|I)[)]|[(]L[)]", "", s)  # NFKC で、Ⅰ・Ⅱ・Ⅲ は、I・II・III になる
    s = re.sub(r"第\d+回", "", s)
    return re.sub(r"\s+", "", s)


def same_name(a, b):
    """勝ち馬の名前の照合。PDFの文字化け（同じ字が2回続く）や、長い名前の途中切れ（「…・シンティレーション」）は、同じとみなす"""
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    dedup = lambda x: re.sub(r"(.)\1", r"\1", x)
    return a == b or dedup(a) == b or a.startswith(b) or b.startswith(a) or dedup(a) == dedup(b)


if __name__ == "__main__":
    rows, unmatched, mismatch = [], [], []
    for y in (2024, 2025, 2026):
        res = pd.read_csv(DATA / f"all_{y}.csv", dtype={"race_id": str})
        winners = res[res["finish"] == 1].set_index("race_id")["horse_name"]
        ex = pd.read_csv(DATA / f"race_extra_{y}.csv", dtype={"race_id": str}).drop_duplicates("race_id")
        ex["key"] = ex["race_name"].map(norm)
        for ln in open(DATA / "youtube" / f"stakes_{y}.txt", encoding="utf-8"):
            parts = ln.rstrip("\n").split("|", 1)
            m = re.match(r"(\d{4})年\s*(.*?)\s*\|\s*(.*)$", parts[1].strip()) if len(parts) > 1 else None
            if not m:
                continue
            vid, title, winner = parts[0], m.group(2), m.group(3).strip()
            tk = norm(title)
            tk = ALIAS.get(tk, tk)
            cands = ex[ex["key"] == tk]
            if len(cands) == 0 and len(tk) >= 3:  # 「日刊スポーツ賞シンザン記念」のように、スポンサー名が付いているレース名は、後ろが一致すれば良い
                cands = ex[ex["key"].map(lambda k: isinstance(k, str) and (k.endswith(tk) or tk.endswith(k) and len(k) >= 3))]
            if len(cands) > 1:  # 同じ名前が複数あるときは、勝ち馬で選ぶ
                cands = cands[cands["race_id"].map(winners).map(lambda w: same_name(w, winner))]
            if len(cands) != 1:
                unmatched.append((y, title, winner, len(cands)))
                continue
            rid = cands["race_id"].iloc[0]
            if not same_name(winners.get(rid), winner) and not title.startswith("第"):
                mismatch.append((y, title, winner, winners.get(rid)))
                continue
            rows.append(dict(race_id=f"{y}_{rid}", video_id=vid, title=f"{y}年 {title}", url=f"https://www.youtube.com/watch?v={vid}"))
    out = pd.DataFrame(rows).drop_duplicates("race_id")
    out.to_csv(DATA / "videos.csv", index=False, encoding="utf-8-sig")
    print(f"{len(out)} 本を紐付け → data/videos.csv")
    print(f"紐付かない映像 {len(unmatched)} 本（成績のレース名に無い。海外・条件戦・名前の表記違いなど）:", *unmatched[:12], sep="\n  ")
    print(f"勝ち馬が一致しない映像 {len(mismatch)} 本（除外）:", *mismatch[:12], sep="\n  ")
