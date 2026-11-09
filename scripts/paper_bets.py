"""記録したオッズ（data/odds_log/*.csv）で、決めたルールを「記録上で」買い、結果で精算する

使い方:
    python scripts/paper_bets.py                 # 発走5分前の時点のオッズで判定（既定）
    python scripts/paper_bets.py --decide 10     # 発走10分前の時点のオッズで判定
    python scripts/paper_bets.py --list          # 買い目を1件ずつ表示する

ルール（scripts/check_roi_patterns.py で、最終オッズでは有望だったもの）:
- 本命1.3倍未満 → 2番人気の複勝を100円
- 本命1.4倍未満 → 2番人気の複勝を100円
参考として、全レースの1番人気・2番人気の複勝も、同じ方法で集計する。

判定に使うのは、発走 decide 分前より前の、最後の記録（その時点で本当に買える情報だけ）。
結果は data/all_<年>.csv、data/all_live_<年>.csv、data/odds_log/results_<日付>.csv（record_odds.py が17時30分以降に取り込む）の、日付・競馬場・レース番号・馬番で結ぶ。
結果がまだ無いレースは「未確定」として数える。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "data" / "odds_log"
RULES = {
    "本命1.3倍未満→2番人気の複勝": lambda fav: fav < 1.3,
    "本命1.4倍未満→2番人気の複勝": lambda fav: fav < 1.4,
    "参考: 全レース・2番人気の複勝": lambda fav: True,
}


def load_log():
    files = sorted(p for p in LOG.glob("*.csv") if not p.name.startswith("results_"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_results(years):
    out = []
    for y in years:
        for name in (f"all_{y}.csv", f"all_live_{y}.csv"):
            p = ROOT / "data" / name
            if p.exists():
                d = pd.read_csv(p, encoding="utf-8-sig", usecols=["date", "venue", "race_no", "umaban", "finish", "odds", "place_payout"])
                out.append(d.assign(src=name))
    for p in sorted(LOG.glob("results_*.csv")):
        d = pd.read_csv(p, encoding="utf-8-sig", usecols=["date", "venue", "race_no", "umaban", "finish", "odds", "place_payout"])
        out.append(d.assign(src="all_live_" + p.stem))
    if not out:
        return pd.DataFrame(columns=["date", "venue", "race_no", "umaban", "finish", "odds", "place_payout"])
    r = pd.concat(out, ignore_index=True)
    # PDF由来（all_<年>.csv）を優先する
    r["pri"] = r["src"].str.startswith("all_live").astype(int)
    return r.sort_values("pri").drop_duplicates(["date", "venue", "race_no", "umaban"]).drop(columns=["src", "pri"])


def decide(log, before):
    """各レースで、発走 before 分前より前の最後の記録"""
    d = log[log["minutes_to_post"] >= before]
    last = d.groupby(["date", "venue", "race_no"])["minutes_to_post"].transform("min")
    return d[d["minutes_to_post"] == last]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decide", type=float, default=5, help="発走の何分前のオッズで判定するか")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    log = load_log()
    if log.empty:
        print("記録がありません（scripts/record_odds.py が開催日に動くと、data/odds_log/ にたまります）")
        return
    snap = decide(log, a.decide)
    res = load_results(sorted({d[:4] for d in snap["date"]}))
    races = snap.groupby(["date", "venue", "race_no"])
    rows = []
    for key, g in races:
        g = g.sort_values(["win_odds", "umaban"])
        if len(g) < 2:
            continue
        fav, second = g.iloc[0], g.iloc[1]
        rows.append(dict(date=key[0], venue=key[1], race_no=key[2], minutes=fav["minutes_to_post"], fav_odds=fav["win_odds"],
                         umaban=int(second["umaban"]), horse=second["horse_name"], odds_at_bet=second["win_odds"],
                         place_lo=second["place_lo"], place_hi=second["place_hi"],
                         fav_umaban=int(fav["umaban"])))
    bets = pd.DataFrame(rows)
    bets = bets.merge(res, on=["date", "venue", "race_no", "umaban"], how="left")
    final_fav = res.groupby(["date", "venue", "race_no"])["odds"].min().rename("final_fav_odds").reset_index()
    bets = bets.merge(final_fav, on=["date", "venue", "race_no"], how="left")
    bets["settled"] = bets["finish"].notna() | bets["place_payout"].notna()
    print(f"記録 {log['date'].nunique()}日・{races.ngroups}レース。判定は発走{a.decide:g}分前の記録（平均 {bets['minutes'].mean():.1f}分前）\n")
    for name, rule in RULES.items():
        b = bets[bets["fav_odds"].apply(rule)]
        s = b[b["settled"]]
        pay = s["place_payout"].fillna(0)
        line = f"{name}: {len(b)}件（確定 {len(s)}件）"
        if len(s):
            line += f"  的中 {(pay > 0).mean():.0%}  回収率 {pay.mean():.1f}%"
            if len(s) >= 2:
                se = pay.std(ddof=1) / np.sqrt(len(s))
                line += f"（±{1.96 * se:.0f}）"
        if name.startswith("本命"):
            thr = 1.3 if "1.3" in name else 1.4
            ok = b["final_fav_odds"].notna()
            same = (b.loc[ok, "final_fav_odds"] < thr).mean() if ok.any() else np.nan
            line += f"  最終オッズでも条件を満たした割合 {same:.0%}" if ok.any() else ""
        print(line)
        if a.list and len(b):
            print(b[["date", "venue", "race_no", "minutes", "fav_odds", "final_fav_odds", "umaban", "horse", "odds_at_bet",
                     "place_lo", "place_hi", "odds", "finish", "place_payout"]].to_string(index=False))
            print()


if __name__ == "__main__":
    main()
