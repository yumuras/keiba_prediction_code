"""PDFのレースごとの情報のうち、成績CSVに無いもの（年齢条件・ハロンタイム・上り3F）を data/race_extra_<年>.csv に書く

使い方:
    python scripts/extract_race_extra.py 2026
列: race_id, race_name（レース名）, age_min（出走できる最低年齢）, age_open（1 = 「以上」＝古馬も出られる、0 = その年齢だけの限定戦）,
    laps（ハロンタイム。「12.9―11.5―…」）, last3f（上り3F、レース全体の最後の3ハロン）
"""
import sys
from pathlib import Path

import pandas as pd
import pdfplumber

sys.path.insert(0, str(Path(__file__).parent))
from extract_pdf import HALVES, fix_text, parse_headers  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]

if __name__ == "__main__":
    year = sys.argv[1]
    rows = []
    for path in sorted((ROOT / year).rglob("*.pdf")):
        with pdfplumber.open(path) as pdf:
            for p in pdf.pages:
                for x0, x1 in HALVES:
                    t = fix_text(p.crop((x0, 0, min(x1, p.width), p.height)).extract_text() or "")
                    rows += [{k: h[k] for k in ("race_id", "race_name", "age_min", "age_open", "laps", "last3f")} for h in parse_headers(t)]
    df = pd.DataFrame(rows)
    out = ROOT / "keiba_prediction" / "data" / f"race_extra_{year}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{year}: {len(df)} レース → {out}  年齢条件 {df.age_min.notna().mean():.1%}  ラップ {df.laps.notna().mean():.1%}  上り3F {df.last3f.notna().mean():.1%}")
