"""気象庁「過去の気象データ検索」から、開催日の1時間ごとの気象データを取得する

使い方:
    python scripts/fetch_weather.py                 # data/all_*.csv の開催日を取得 → data/weather_<年>.csv

各競馬場の最寄りで湿度まで観測している地点（気象台、または湿度のあるアメダス）を使う。
取得したHTMLは data/.jma_cache/ に保存し、2回目以降は気象庁にアクセスしない。
アクセスは1秒に1回までに抑える。
"""
import argparse
import io
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / ".jma_cache"

# 競馬場 → (観測所の種類, 都府県番号, 地点番号, 地点名)。s = 気象台など、a = アメダス
STATIONS = {
    "札幌": ("s", 14, "47412", "札幌"),
    "函館": ("s", 23, "47430", "函館"),
    "福島": ("s", 36, "47595", "福島"),
    "新潟": ("s", 54, "47604", "新潟"),
    "東京": ("a", 44, "1133", "府中"),      # 東京競馬場は府中市
    "中山": ("a", 45, "1236", "船橋"),      # 中山競馬場は船橋市
    "中京": ("s", 51, "47636", "名古屋"),
    "京都": ("s", 61, "47759", "京都"),
    "阪神": ("s", 62, "47772", "大阪"),
    "小倉": ("a", 82, "0780", "八幡"),      # アメダス小倉は湿度などを観測していないため隣の八幡
}
URL = "https://www.data.jma.go.jp/stats/etrn/view/hourly_{kind}1.php?prec_no={prec}&block_no={block}&year={y}&month={m}&day={d}&view="


def fetch_html(venue, date):
    kind, prec, block, _ = STATIONS[venue]
    path = CACHE / f"{block}_{date:%Y%m%d}.html"
    if not path.exists():
        url = URL.format(kind=kind, prec=prec, block=block, y=date.year, m=date.month, d=date.day)
        with urlopen(Request(url, headers={"User-Agent": "keiba_prediction research script"}), timeout=30) as r:
            path.write_bytes(r.read())
        time.sleep(1)  # 気象庁のサーバーに負担をかけないよう間隔を空ける
    return path.read_text(encoding="utf-8")


def to_num(s):
    """気象庁の表の値を数値にする。「--」は現象なし（降水量なら0）、「)」付きは準正常値、「///」「×」は欠測"""
    s = s.astype(str).str.replace(r"[\)\]]", "", regex=True).str.strip()
    return pd.to_numeric(s.replace({"--": "0"}), errors="coerce")


def parse_hourly(html):
    t = pd.read_html(io.StringIO(html), attrs={"id": "tablefix1"})[0]
    t.columns = [b if a == b else f"{a}_{b}" for a, b in t.columns]
    col = lambda key: next(c for c in t.columns if key in c)
    return pd.DataFrame({
        "hour": t[col("時")].astype(int),
        "precip": to_num(t[col("降水量")]),
        "temp": to_num(t[col("気温")]),
        "humidity": to_num(t[col("湿度")]),
        "wind_speed": to_num(t[next(c for c in t.columns if "風速" in c and "風向" not in c.split("_")[-1])]),
        "sunshine": to_num(t[col("日照")]),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--races", nargs="*", default=sorted((ROOT / "data").glob("all_*.csv")),
                    help="開催日を読み取る成績CSV（既定: data/all_*.csv すべて）")
    args = ap.parse_args()

    CACHE.mkdir(parents=True, exist_ok=True)
    races = pd.concat([pd.read_csv(f, usecols=["venue", "date"], parse_dates=["date"]) for f in args.races])
    races = races[["venue", "date"]].drop_duplicates().sort_values(["date", "venue"])
    rows = []
    for i, (venue, date) in enumerate(races.itertuples(index=False), 1):
        h = parse_hourly(fetch_html(venue, date))
        rows.append(h.assign(venue=venue, date=date.date(), station=STATIONS[venue][3]))
        if i % 20 == 0:
            print(f"{i}/{len(races)}", flush=True)
    out = pd.concat(rows)[["venue", "date", "station", "hour", "temp", "humidity", "wind_speed", "precip", "sunshine"]]
    for year, g in out.groupby(pd.to_datetime(out["date"]).dt.year):  # 年ごとに data/weather_<年>.csv へ保存
        path = ROOT / "data" / f"weather_{year}.csv"
        g.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"{year}: {g[['venue', 'date']].drop_duplicates().shape[0]} 開催日 / {len(g)} 行 → {path}")
    print("欠測:", out[["temp", "humidity", "wind_speed", "precip"]].isna().sum().to_dict())


if __name__ == "__main__":
    main()
