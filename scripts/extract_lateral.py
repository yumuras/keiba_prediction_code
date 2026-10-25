"""PDFの『通過順位』の（ ）から、コーナーごとの内外の位置を取り出し、data/lateral_<年>.csv に書く

使い方:
    python scripts/extract_lateral.py 2026          # ~/keiba_data/2026/ の下の全PDFを読む
列: race_id, umaban, horse_name, corner{1-4}_grp（並走のかたまりの頭数）, corner{1-4}_lat（かたまりの中の位置。0=内、1=外）
成績CSVとは、(年, race_id, umaban) で結ぶ。
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from extract_pdf import parse_pdf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
COLS = ["race_id", "umaban", "horse_name"] + [f"corner{k}_{n}" for k in (1, 2, 3, 4) for n in ("grp", "lat")]


def one(path):
    df = parse_pdf(str(path))
    return df[COLS] if len(df) else df


if __name__ == "__main__":
    year = sys.argv[1]
    pdfs = sorted((ROOT / year).rglob("*.pdf"))
    with Pool(8) as pool:
        parts = pool.map(one, pdfs)
    df = pd.concat(parts, ignore_index=True)
    out = ROOT / "keiba_prediction" / "data" / f"lateral_{year}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{year}: {len(pdfs)} PDF, {len(df)} 行 → {out}")
