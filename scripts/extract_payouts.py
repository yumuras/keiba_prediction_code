"""成績PDFの払戻金の欄から、レースごとの 馬連・ワイド・3連複・馬単・3連単 の払戻（100円あたりの円）を data/payouts_<年>.csv に保存する。

単勝・複勝は extract_pdf.py が all_*.csv に入れているので、ここでは扱わない。
ワイドは3つ（同着で増減）の払戻が並ぶが、どの組み合わせの払戻かは、PDFの文字からは読めない（馬番が図形のため）。
そのため wide1〜3 は PDF に並んでいる順のまま保存し、組み合わせとの対応は、検証側で確かめる。

使い方: uv run python scripts/extract_payouts.py 2026
"""
import csv
import re
import sys
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import extract_pdf as X  # noqa: E402

HEAD = re.compile(r"(\d{5}) ?\d+月\d+日 .*第\d+競走")
NUM = lambda t: [int(re.sub(r"\D", "", x)) for x in re.findall(r"([\d，,]+)円", t)]
PATS = {
    "umaren": re.compile(r"馬\s*連\s+((?:[\d，,]+円\s*)+)"),
    "wide": re.compile(r"ワイ\s*ド\s+((?:[\d，,]+円\s*)+)"),
    "trio": re.compile(r"3連\s*複\s+((?:[\d，,]+円\s*)+)"),
    "exacta": re.compile(r"馬\s*単\s+((?:[\d，,]+円\s*)+)"),
    "trifecta": re.compile(r"3連\s*単\s+((?:[\d，,]+円\s*)+)"),
}
COLUMNS = ["race_id", "umaren", "wide1", "wide2", "wide3", "trio", "exacta", "trifecta", "tie"]


def parse_text(text):
    blocks, cur = {}, None
    for ln in text.split("\n"):
        m = HEAD.search(ln)
        if m:
            cur = m.group(1)
            blocks.setdefault(cur, [])
        elif cur is not None:
            blocks[cur].append(ln)
    out = {}
    for rid, lines in blocks.items():
        s = "\n".join(lines)
        v = {k: NUM(p.search(s).group(1)) if p.search(s) else [] for k, p in PATS.items()}
        if not v["umaren"] or not v["trio"]:
            continue
        w = v["wide"] + [None] * 3
        out[rid] = dict(race_id=rid, umaren=v["umaren"][0], wide1=w[0], wide2=w[1], wide3=w[2], trio=v["trio"][0],
                        exacta=(v["exacta"] or [None])[0], trifecta=(v["trifecta"] or [None])[0],
                        tie=int(any(len(v[k]) > 1 for k in ("umaren", "trio", "exacta", "trifecta")) or len(v["wide"]) != 3))
    return out


def main(year):
    pdfs = sorted(Path(ROOT.parent / year).rglob("*.pdf"))
    rows = {}
    for i, f in enumerate(pdfs, 1):
        with pdfplumber.open(f) as pdf:
            for p in pdf.pages:
                for x0, x1 in X.HALVES:
                    chars = [c for c in p.chars if x0 <= c["x0"] < min(x1, p.width)]
                    if not chars:
                        continue
                    text = X.fix_text(p.crop((x0, 0, min(x1, p.width), p.height)).extract_text() or "")
                    rows.update(parse_text(text))
        if i % 10 == 0:
            print(f"{i}/{len(pdfs)} ファイル {len(rows)}レース", flush=True)
    out = ROOT / "data" / f"payouts_{year}.csv"
    with open(out, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, COLUMNS)
        w.writeheader()
        w.writerows(rows.values())
    print(f"{len(rows)}レース → {out}")


if __name__ == "__main__":
    main(sys.argv[1])
