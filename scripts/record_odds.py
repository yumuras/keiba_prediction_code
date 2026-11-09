"""開催日に、発走前の単勝・複勝オッズを何度も記録する（買う時点のオッズでの検証のため）

使い方:
    python scripts/record_odds.py            # 1回分。5分ごとに動かす（scripts/odds_recorder.plist を参照）
    python scripts/record_odds.py --window 60   # 発走60分前から記録する（既定は40分前から）

1回の実行で行うこと:
1. 今日が開催日かを確かめる（JRAの「出馬表」の一覧。開催がない日は、その日の残りは何もしない）
2. 今日の開催の出馬表を取り込む（その日の最初の1回だけ。data/odds_log/cards/ に保存し、以後は使い回す）
3. 発走まで window 分以内で、まだ発走していないレースの、単勝・複勝オッズを取り込み、data/odds_log/<日付>.csv に追記する
4. 17時30分を過ぎたら、その日の全レースの結果を1回だけ取り込み、data/odds_log/results_<日付>.csv に保存する（scripts/paper_bets.py の精算用）

JRAへのアクセスは、1回の実行で「記録するレース数」＋α（1秒間隔）。個人の利用の範囲で使うこと
（JRAの「ご利用に際して」: 掲載内容の二次利用は、私的使用・引用の範囲を超える場合に許諾が必要）。
記録した列: fetched_at, minutes_to_post, date, venue, race_no, start_time, umaban, horse_name, win_odds, place_lo, place_hi, popularity
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import jra_card as J  # noqa: E402
import jra_odds as O  # noqa: E402

LOG = ROOT / "data" / "odds_log"
COLS = ["fetched_at", "minutes_to_post", "date", "venue", "race_no", "start_time", "umaban", "horse_name",
        "win_odds", "place_lo", "place_hi", "popularity"]


def today_cards(today):
    """今日の開催の出馬表（キャッシュ）。開催がない日は []"""
    cdir = LOG / "cards"
    cdir.mkdir(parents=True, exist_ok=True)
    none_flag = cdir / f"{today}_none"
    if none_flag.exists():
        return []
    cached = sorted(cdir.glob(f"{today}_*.json"))
    if cached:
        return [json.loads(p.read_text(encoding="utf-8")) for p in cached]
    meetings = [m for m in J.list_meetings() if m["date"] == today]
    if not meetings:
        none_flag.write_text("")
        return []
    cards = []
    for m in meetings:
        try:
            card = J.fetch_card(m["cname"])
        except Exception as e:  # 出馬表がまだ出ていないなど。保存せず、次の実行でやり直す
            print(f"{m['venue']}: {e}", file=sys.stderr)
            continue
        (cdir / f"{today}_{m['venue']}.json").write_text(json.dumps(card, ensure_ascii=False), encoding="utf-8")
        cards.append(card)
    return cards


def save_results(today):
    """その日の全レースの結果を data/odds_log/results_<日付>.csv に保存する（17時30分以降の最初の1回だけ）"""
    out = LOG / f"results_{today}.csv"
    if out.exists():
        return
    import jra_results as R
    try:
        df = R.fetch_day(today)
    except Exception as e:
        print(f"結果の取り込み: {e}", file=sys.stderr)
        return
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"結果 {df['race_id'].nunique()}レース → {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=40, help="発走の何分前から記録するか")
    a = ap.parse_args()
    now = datetime.now()
    today = now.strftime("%Y-%m-%d")
    if not 8 <= now.hour <= 18:
        return
    cards = today_cards(today)
    if not cards:
        return
    if now.hour >= 17 and now.minute >= 30 or now.hour == 18:
        save_results(today)
        return
    out = LOG / f"{today}.csv"
    new = not out.exists()
    n_races = 0
    with out.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        for card in cards:
            for r in card["races"]:
                if not r.get("start_time") or not r.get("odds_cname"):
                    continue
                post = datetime.strptime(f"{today} {r['start_time']}", "%Y-%m-%d %H:%M")
                mins = (post - now) / timedelta(minutes=1)
                if not 0 < mins <= a.window:
                    continue
                try:
                    win, place = O.fetch_win_place(r["odds_cname"])
                except Exception as e:  # 1レースの失敗で、残りを止めない
                    print(f"{card['venue']}{r['race_no']}R: {e}", file=sys.stderr)
                    continue
                if not win:
                    continue
                order = {u: i + 1 for i, u in enumerate(sorted(win, key=lambda u: (win[u], u)))}
                names = {h["umaban"]: h["name"] for h in r["horses"]}
                stamp = datetime.now().strftime("%H:%M:%S")
                for u in sorted(win):
                    lo, hi = place.get(u, (None, None))
                    w.writerow(dict(fetched_at=stamp, minutes_to_post=round(mins, 1), date=today, venue=card["venue"],
                                    race_no=r["race_no"], start_time=r["start_time"], umaban=u, horse_name=names.get(u, ""),
                                    win_odds=win[u], place_lo=lo, place_hi=hi, popularity=order[u]))
                n_races += 1
    if n_races:
        print(f"{now:%H:%M} {n_races}レースを記録 → {out}")


if __name__ == "__main__":
    main()
