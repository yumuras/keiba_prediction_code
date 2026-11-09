"""回収率の「構造的な偏り」を、年ごとに分けて確かめる（モデルを使わない検証と、バックテストを使う検証）

使い方:
    uv run python scripts/check_roi_patterns.py                      # 人気・本命のオッズ・頭数による偏り（2024〜2026年）
    uv run python scripts/check_roi_patterns.py models/backtest.pkl  # ＋「能力モデルと市場の意見が割れる馬」の回収率

見ること:
1. 本命が1倍台前半のレースの「2番人気の複勝」、少頭数のレースの「2番人気の複勝」（全体の84%より高いか、毎年そうか）
2. 本命が1.5倍未満のレースで、本命を外した3連複ボックスと、本命を軸にした流し（払戻のある2025・2026年）
3. 能力モデル（オッズなし）が市場より高く評価した馬ほど、回収率は上がるか（十分位ごと）
いずれも最終オッズで分類しているので、実際に買う時点（オッズが動く前）では、ここより悪くなる方向に偏る。
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)


def ci(s):
    """平均（回収率%）と、復元抽出の95%区間"""
    s = np.asarray(s, dtype=float)
    bs = [rng.choice(s, len(s)).mean() for _ in range(2000)]
    return f"{len(s):5d}点  回収率 {s.mean():6.1f}%  [{np.percentile(bs, 2.5):.0f}〜{np.percentile(bs, 97.5):.0f}]"


def load(years=(2024, 2025, 2026)):
    out = []
    for y in years:
        a = pd.read_csv(ROOT / f"data/all_{y}.csv", encoding="utf-8-sig", dtype={"race_id": str})
        out.append(a.assign(year=y))
    a = pd.concat(out)
    a = a[a["popularity"].notna()].copy()
    a["fav_odds"] = a.groupby(["year", "race_id"])["odds"].transform("min")
    a["place_payout"] = a["place_payout"].fillna(0)
    # 人気が同じ（オッズが同じ）馬がいるレースがあるので、オッズ→馬番の順で、重ならない人気順を作り直す
    a = a.sort_values(["year", "race_id", "odds", "umaban"])
    a["popularity"] = a.groupby(["year", "race_id"]).cumcount() + 1
    return a


def place_patterns(a):
    print("■ 2番人気の複勝（100円あたりの回収率。カッコ内は年ごと 2024/2025/2026）")
    p2 = a[a["popularity"] == 2]
    for name, m in [
        ("全レース", p2["fav_odds"] > 0),
        ("本命 1.3倍未満", p2["fav_odds"] < 1.3),
        ("本命 1.4倍未満", p2["fav_odds"] < 1.4),
        ("本命 1.5倍未満", p2["fav_odds"] < 1.5),
        ("本命 2.0倍未満", p2["fav_odds"] < 2.0),
        ("8頭以下", p2["num_runners"] <= 8),
        ("本命1.4倍未満 または 8頭以下", (p2["fav_odds"] < 1.4) | (p2["num_runners"] <= 8)),
    ]:
        s = p2[m]
        by = "/".join(f"{v:.0f}" for v in s.groupby("year")["place_payout"].mean())
        print(f"  {name:<22} {ci(s['place_payout'])}  ({by})")
    p1 = a[a["popularity"] == 1]
    print(f"  参考: 1番人気の複勝・全レース {ci(p1['place_payout'])}")


def trio_without_favorite(a):
    print("\n■ 本命が1.5倍未満のレースの3連複（人気順で買い目を決める）")
    for y in (2025, 2026):
        pay = pd.read_csv(ROOT / f"data/payouts_{y}.csv", encoding="utf-8-sig", dtype={"race_id": str})
        pay = pay[pay["tie"] == 0].set_index("race_id")["trio"]
        g_all = a[(a["year"] == y) & (a["fav_odds"] < 1.5) & a["race_id"].isin(pay.index)]
        strat = {
            "本命軸・2〜6番人気へ流し（10点）": [(1,) + c for c in itertools.combinations(range(2, 7), 2)],
            "本命抜き・2〜6番人気ボックス（10点）": list(itertools.combinations(range(2, 7), 3)),
            "1〜5番人気ボックス（10点）": list(itertools.combinations(range(1, 6), 3)),
        }
        res = {k: [0, 0.0] for k in strat}
        n_races = 0
        for rid, g in g_all.groupby("race_id"):
            if len(g) < 7:
                continue
            n_races += 1
            top3 = set(g.loc[g["finish"] <= 3, "popularity"].astype(int))
            for k, combos in strat.items():
                res[k][0] += len(combos)
                if any(set(c) == top3 for c in combos):
                    res[k][1] += pay[rid] / 100
        print(f"  {y}年（{n_races}レース）: " + "、".join(f"{k} {r / n:.0%}" for k, (n, r) in res.items()))


def disagreement(path):
    import joblib

    p = joblib.load(path)
    p = p[p["finish"].notna() & p["popularity"].notna()].copy()
    p["place_payout"] = p["place_payout"].fillna(0)
    p["ratio"] = p["p_ability"] / p["p_mkt3"]
    q = pd.qcut(p["ratio"], 10, labels=False, duplicates="drop")
    print(f"\n■ 能力モデル÷市場の3着内確率の十分位（左＝市場より低く評価、右＝市場より高く評価）{path}")
    print("  複勝回収率%:", p.groupby(q)["place_payout"].mean().round(0).astype(int).tolist())
    print("  単勝回収率%:", p.groupby(q)["win_payout"].mean().round(0).astype(int).tolist())


if __name__ == "__main__":
    a = load()
    place_patterns(a)
    trio_without_favorite(a)
    for path in sys.argv[1:]:
        disagreement(path)
