"""JRA公式サイトの「レース結果」から、終わったレースの結果を、成績CSV（data/all_*.csv）と同じ形に取り込む

使い方:
    python src/jra_results.py 2026-10-25                    # data/all_live_2026.csv に、その日の全レースを書く（同じ日が有れば置き換える）
    python src/jra_results.py 2026-10-18 --out /tmp/x.csv   # 別のファイルに書く（PDF由来のCSVとの突き合わせ用）

成績のPDF（年度別全成績）が出るまでの、つなぎ。PDFを取り込んだら、同じ（日付・競馬場・レース番号）の行は、PDF由来のほうを使う（train.load）。
PDFにあって、結果のページに無い列（毛色・産地・ブリンカー）は空欄（ブリンカーは False）。個人の利用の範囲で使うこと（1秒間隔）。
"""
import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from lxml import html as LH

sys.path.insert(0, str(Path(__file__).parent))
import jra_card as J  # noqa: E402
import jra_odds as O  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.jra.go.jp/JRADB/accessS.html"
HEADERS = {"User-Agent": "keiba_prediction personal research (https://github.com/yumuras/keiba_prediction)"}
VENUE_CODE = {v: f"{i:02d}" for i, v in enumerate(J.VENUES, 1)}
COLUMNS = ["race_id", "date", "venue", "kai", "nichi", "race_no", "weather", "going", "surface", "course", "distance", "race_class", "handicap",
           "start_time", "finish", "waku", "umaban", "horse_name", "sex", "age", "color", "carried_weight", "base_weight", "jockey_mark", "jockey",
           "trainer", "region", "blinker", "body_weight", "body_weight_diff", "time", "time_sec", "odds", "corner1", "corner2", "corner3", "corner4",
           "win_payout", "place_payout", "popularity", "num_runners", "win"]
MARKS = {"☆": 1.0, "▲": 3.0, "△": 2.0, "◇": 2.0, "★": 4.0}  # 騎手の減量（kg）。成績PDFの斤量との突き合わせで確認

_s = requests.Session()
_s.headers.update(HEADERS)


def post(cname):
    r = _s.post(URL, data={"cname": cname}, timeout=30)
    r.raise_for_status()
    r.encoding = "shift_jis"
    time.sleep(1.0)
    return r.text


def txt(el):
    return re.sub(r"\s+", " ", el.text_content()).strip() if el is not None else ""


def to_sec(t):
    m = re.fullmatch(r"(?:(\d+):)?(\d+\.\d)", t or "")
    return int(m.group(1) or 0) * 60 + float(m.group(2)) if m else None


def payouts(text):
    """払戻金の欄から、単勝・複勝の（馬番 → 100円あたりの払戻）"""
    num = lambda s: int(s.replace(",", ""))
    win = {int(a): num(b) for a, b in re.findall(r"(\d+)\s+([\d,]+)円\s+\d+番人気", re.search(r"単勝(.*?)複勝", text, re.S).group(1))}
    place = {int(a): num(b) for a, b in re.findall(r"(\d+)\s+([\d,]+)円\s+\d+番人気", re.search(r"複勝(.*?)枠連", text, re.S).group(1))}
    return win, place


def parse_race(page, date, odds_page=None):
    d = LH.fromstring(page)
    head = " ".join(txt(x) for x in d.xpath('//div[@class="header_line no-mb"] | //div[@class="race_header"]'))
    m = re.search(r"(\d+)回(\S+?)(\d+)日", head)
    kai, venue, nichi = int(m.group(1)), next(v for v in J.VENUES if v in m.group(2)), int(m.group(3))
    race_no = int(re.search(r"回\S+?\d+日\s*(\d+)レース", txt(d)).group(1))
    hd = d.xpath('//div[@class="race_header"]')[0]
    st = re.search(r"発走時刻：\s*(\d+)時(\d+)分", head)
    wt = re.search(r"天候\s*(晴|曇|小雨|雨|小雪|雪)\s*(.*?)\s*(第|\d歳|サラ|障害|コース)", head)
    weather = wt.group(1)
    jump = "障害" in head.split("コース")[0]
    gs = re.findall(r"(芝|ダート)\s*(良|稍重|重|不良)", wt.group(2))
    going = gs[0][1] if gs else "良"
    dist = int(re.search(r"([\d,]+)メートル", head).group(1).replace(",", ""))
    detail = re.search(r"メートル（([^）]*)）", head).group(1)
    surface = "障害" if jump else ("芝" if detail.startswith("芝") else "ダート")
    course = re.sub(r"\s+", "・", detail.strip())
    course = re.sub(r"・直$", "・直線", course)
    name = txt(hd)
    cls_tok = next((k for k in ("新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン") if k in head), "オープン")
    grade = " ".join(("L" if a == "リステッド" else a) for a in d.xpath('//div[@class="race_header"]//img/@alt') if re.search(r"[ⅠⅡⅢ]|^[LＬ]$|リステッド", a))
    race_class = J._race_class(cls_tok, grade, head)
    handicap = "ハンデ" in head
    seg = txt(d)
    cm = re.search(r"上り\s*\d?F.*?コーナー通過順位\s+(?=[1-4]コーナー)", seg)  # レース全体の通過順位（行の前に、「上り」の欄がある）
    seqs = []
    if cm:
        body = seg[cm.end():seg.find("払戻金", cm.end())]
        # 見出しは「1コーナー」「3コーナー(2周目)」など。2周するレースは、4つより多く並ぶことがあるので、成績PDFと同じく、最後の4つを、最終コーナーから詰める
        pat = r"[1-4]コーナー(?:\(\d周目\))?\s*([\d,\(\)\*\-=\s]+?)(?=\s*[1-4]コーナー|$)"
        seqs = [[int(n) for n in re.findall(r"\d+", v)] for v in re.findall(pat, body)][-4:]
    cpos = {4 - len(seqs) + 1 + i: {u: j + 1 for j, u in enumerate(sq)} for i, sq in enumerate(seqs)}  # 書かれている順（成績PDFと同じ数え方）
    win_pay, place_pay = payouts(txt(d.xpath('//div[@class="refund_area mt30"]')[0]))
    odds = {}
    if odds_page:
        odds, _ = O.parse_win_place(LH.fromstring(odds_page))
    rows = []
    for tr in d.xpath('//table[contains(@class,"basic")]//tbody/tr[td[@class="place"]]'):
        g = lambda c: txt(tr.find(f'./td[@class="{c}"]'))
        place = g("place")
        if place in ("取", "除", "取消", "除外"):
            continue
        age = g("age")
        jk = g("jockey")
        mark = next((c for c in jk if c in MARKS), "")
        jockey = re.sub(r"[\s☆▲△◇★]", "", jk)
        w = float(g("weight"))
        hw = re.match(r"(\d+)\s*(?:\(([+\-±]?\d+)\))?", g("h_weight"))
        u = int(g("num"))
        t = g("time")
        rows.append(dict(
            date=date, venue=venue, kai=kai, nichi=nichi, race_no=race_no, weather=weather, going=going, surface=surface, course=course, distance=dist,
            race_class=race_class, handicap=handicap, start_time=f"{int(st.group(1)):02d}:{st.group(2)}", finish=int(place) if place.isdigit() else None,
            waku=int(re.search(r"枠(\d)", "".join(tr.xpath('./td[@class="waku"]//img/@alt'))).group(1)), umaban=u, horse_name=g("horse"),
            sex="セ" if age[0] in "セせ" else age[0], age=int(re.search(r"\d+", age).group()), color=None, carried_weight=w, base_weight=w + MARKS.get(mark, 0.0),
            jockey_mark=mark or None, jockey=jockey, trainer=re.sub(r"\s", "", g("trainer")), region=None, blinker=False,
            body_weight=int(hw.group(1)) if hw and int(hw.group(1)) > 200 else None,
            body_weight_diff=int(hw.group(2).replace("±", "0")) if hw and hw.group(2) else None, time=t or None, time_sec=to_sec(t),
            odds=odds.get(u), **{f"corner{k}": cpos.get(k, {}).get(u) for k in (1, 2, 3, 4)},
            win_payout=win_pay.get(u, 0), place_payout=place_pay.get(u, 0), popularity=int(g("pop")) if g("pop").isdigit() else None))
    df = pd.DataFrame(rows)
    df["num_runners"] = len(df)
    if df["odds"].notna().all():
        df["popularity"] = df["odds"].rank(method="min")  # 成績PDFと同じ（単勝オッズの順位）
    df["win"] = (df["finish"] == 1).astype(int)
    df["race_id"] = f"9{date[5:7]}{date[8:]}{VENUE_CODE[venue]}{race_no:02d}"
    # 3コーナーまでのレースなど、通過順位が欄の後ろに詰められている（成績PDFと同じ並びにそろえる）
    return df, re.findall(r"pw151ou\w+/[0-9A-F]{2}", page)


def fetch_day(date):
    ymd = date.replace("-", "")
    index = post("pw01sli00/AF")
    venues = sorted(set(re.findall(rf"pw01srl\d{{12}}{ymd}/[0-9A-F]{{2}}", index)))
    if not venues:
        raise RuntimeError(f"{date} の結果が、結果ページの一覧にありません（直近の開催だけが載っています）")
    out = []
    for vc in venues:
        page = post(vc)
        races = {}
        for mm in re.finditer(r"(pw01sde\d{12}(\d{2})\d{8}/[0-9A-F]{2})", page):
            races.setdefault(int(mm.group(2)), mm.group(1))
        for rn, cn in sorted(races.items()):
            rp = post(cn)
            odds_cn = re.findall(r"pw151ou\w+/[0-9A-F]{2}", rp)
            odds_page = O._post(odds_cn[0]) if odds_cn else None
            df, _ = parse_race(rp, date, odds_page)
            print(f"  {df['venue'].iloc[0]}{rn}R  {len(df)}頭  1着 {df.loc[df.finish == 1, 'horse_name'].tolist()}", flush=True)
            out.append(df)
    return pd.concat(out, ignore_index=True)[COLUMNS]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("date")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = fetch_day(a.date)
    out = Path(a.out) if a.out else ROOT / "data" / f"all_live_{a.date[:4]}.csv"
    if out.exists() and not a.out:
        old = pd.read_csv(out, dtype={"race_id": str})
        df = pd.concat([old[old["date"] != a.date], df], ignore_index=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"{len(df)} 行 / {df['race_id'].nunique()} レース → {out}")
