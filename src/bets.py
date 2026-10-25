"""券種ごとの買い目（馬連・ワイド・馬単・3連複・3連単）の確率と、期待値の目安

使い方:
    table = combo_table(umaban, names, p_basis, p_market, "3連単", top=10, sort="prob")

馬ごとの勝率（合計1）から、1着・2着・3着の順位の確率を、Harville（ハービル）の方法で出す。
    P(i が1着) = p_i
    P(i が1着、j が2着) = p_i × p_j / (1 − p_i)
    P(i, j, k が1・2・3着) = p_i × p_j / (1 − p_i) × p_k / (1 − p_i − p_j)
馬連・3連複などは、この順位の確率を、組み合わせごとに足し合わせて作る。

期待値の目安は、「基準の確率 ÷ 市場の確率（オッズから作った確率）」に、券種ごとの払戻率をかけた値。
市場の確率と同じなら、払戻率（0.725〜0.80）になるので、1を超えるのは、市場より高く見積もった買い目だけ。
実際のオッズ（馬連・3連単など）は、取り込んでいないので、「目安」にとどまる。

検証（2026年3〜9月、2,004レース）: 「市場＋能力モデル」の確率は、馬連・馬単・3連複・3連単のどれでも、
市場だけの確率と区別がつかなかった（実際の着順に付けた確率の対数の差が、0をまたぐ）。
"""
import itertools

import numpy as np
import pandas as pd

# JRAの払戻率（売上のうち、的中者に払い戻される割合）
RATES = {"単勝": 0.80, "複勝": 0.80, "馬連": 0.775, "ワイド": 0.775, "馬単": 0.75, "3連複": 0.75, "3連単": 0.725}
# 期待値の順に並べるとき、確率が低すぎる買い目（見積もりの誤差だけで、期待値が大きく出る）を除くための下限
MIN_PROB = {"単勝": 0.02, "複勝": 0.10, "馬連": 0.01, "ワイド": 0.03, "馬単": 0.005, "3連複": 0.005, "3連単": 0.002}
BETS = ["単勝", "複勝", "馬連", "ワイド", "馬単", "3連複", "3連単"]


def all_probs(p):
    """勝率 p（合計1）から、券種ごとの {買い目（馬のインデックスの組）: 確率} を作る"""
    p = np.clip(np.asarray(p, dtype=float), 1e-9, None)
    p = p / p.sum()
    n = len(p)
    exacta = {(i, j): p[i] * p[j] / (1 - p[i]) for i, j in itertools.permutations(range(n), 2)}
    trifecta = {(i, j, k): exacta[(i, j)] * p[k] / max(1 - p[i] - p[j], 1e-9) for i, j, k in itertools.permutations(range(n), 3)}
    quinella, trio = {}, {}
    for (i, j), v in exacta.items():
        key = tuple(sorted((i, j)))
        quinella[key] = quinella.get(key, 0.0) + v
    for key3, v in trifecta.items():
        key = tuple(sorted(key3))
        trio[key] = trio.get(key, 0.0) + v
    wide = {}
    for (i, j, k), v in trio.items():  # 3頭が3着以内に入る確率を、2頭の組に足す（ワイド = 2頭がともに3着以内）
        for pair in ((i, j), (i, k), (j, k)):
            wide[pair] = wide.get(pair, 0.0) + v
    return {"単勝": {(i,): p[i] for i in range(n)}, "馬連": quinella, "ワイド": wide, "馬単": exacta, "3連複": trio, "3連単": trifecta}


def label(combo, umaban, names, ordered):
    nums = [int(umaban[i]) for i in combo]
    sep = " → " if ordered else "－"
    return sep.join(str(x) for x in nums) + "　" + "／".join(names[i] for i in combo)


def combo_table(umaban, names, p_basis, p_market, bet, top=5, sort="prob"):
    """券種 bet（馬連・ワイド・馬単・3連複・3連単）の買い目の表。

    p_basis: 基準の勝率（合計1）、p_market: 市場の勝率（オッズから作る。合計1）。
    sort="prob": 的中しやすい順、sort="ev": 期待値の目安が高い順（確率が MIN_PROB 以上のものだけ）。
    """
    basis = all_probs(p_basis)[bet]
    market = all_probs(p_market)[bet] if p_market is not None else None
    ordered = bet in ("馬単", "3連単")
    rows = []
    for combo, pb in basis.items():
        pm = market[combo] if market else np.nan
        rows.append((combo, pb, pm, pb / pm * RATES[bet] if market else np.nan))
    df = pd.DataFrame(rows, columns=["combo", "確率", "市場の確率", "期待値の目安"])
    if sort == "ev" and market:
        df = df[df["確率"] >= MIN_PROB[bet]]
    df = df.sort_values("期待値の目安" if (sort == "ev" and market) else "確率", ascending=False).head(top)
    df.insert(0, "買い目", [label(c, umaban, names, ordered) for c in df["combo"]])
    df = df.drop(columns="combo").reset_index(drop=True)
    return df if market else df[["買い目", "確率"]]


def single_table(umaban, names, p_basis, p_market, odds, bet, top=5, sort="prob", pay_hat=None, p_market_top3=None):
    """単勝・複勝の表。

    単勝: 確率 = 基準の勝率、期待値 = 勝率 × 単勝オッズ。
    複勝: 確率 = 3着以内の確率（p_basis に渡す）、期待値 = 確率 × 予想払戻（単勝オッズから決めた、3着以内に入ったときの払戻 pay_hat ÷ 100）。
    """
    p_basis = np.asarray(p_basis, dtype=float)
    has = odds is not None and not pd.isna(np.asarray(odds, dtype=float)).any()
    df = pd.DataFrame({"買い目": [f"{int(u)}　{n}" for u, n in zip(umaban, names)], "確率": p_basis})
    if has:
        df["市場の確率"] = np.asarray(p_market if bet == "単勝" else p_market_top3, dtype=float)
        df["単勝オッズ"] = np.asarray(odds, dtype=float)
        df["期待値の目安"] = df["確率"] * (df["単勝オッズ"] if bet == "単勝" else np.asarray(pay_hat, dtype=float) / 100)
    if sort == "ev" and has:
        df = df[df["確率"] >= MIN_PROB[bet]]
    return df.sort_values("期待値の目安" if (sort == "ev" and has) else "確率", ascending=False).head(top).reset_index(drop=True)


# ---- 実際のオッズ（jra_odds.fetch_odds の結果）を使った表 ----------------------------------------

ODDS_KEY = {"単勝": "win", "複勝": "place", "馬連": "quinella", "ワイド": "wide", "馬単": "exacta", "3連複": "trio", "3連単": "trifecta"}


def real_table(umaban, names, p_basis, p_top3, odds, bet, top=5, sort="prob"):
    """実際のオッズで、期待値（確率 × オッズ）を出した表。

    単勝・馬連・馬単・3連複・3連単: オッズは1つの値。複勝・ワイド: オッズは、下限〜上限の幅があるので、期待値も、下限〜上限（並べ替えは、中央の値）。
    確率: 単勝は p_basis、複勝は p_top3（3着内の確率）、ほかは、p_basis から Harville で作った組み合わせの確率。
    発売されていない組み合わせ（取消・除外の馬を含む）は、表に出ない。
    """
    table = odds[ODDS_KEY[bet]]
    if bet == "単勝":
        probs = {(i,): p_basis[i] for i in range(len(p_basis))}
    elif bet == "複勝":
        probs = {(i,): p_top3[i] for i in range(len(p_top3))}
    else:
        probs = all_probs(p_basis)[bet]
    ordered = bet in ("馬単", "3連単")
    ranged = bet in ("複勝", "ワイド")
    rows = []
    for combo, pb in probs.items():
        nums = tuple(int(umaban[i]) for i in combo)
        key = nums[0] if bet in ("単勝", "複勝") else (nums if ordered else tuple(sorted(nums)))
        v = table.get(key)
        if v is None:
            continue
        lo, hi = (v if ranged else (v, v))
        rows.append((combo, pb, lo, hi, pb * lo, pb * hi, pb * (lo + hi) / 2))
    df = pd.DataFrame(rows, columns=["combo", "確率", "オッズ下限", "オッズ上限", "期待値下限", "期待値上限", "期待値"])
    if sort == "ev":
        df = df[df["確率"] >= MIN_PROB[bet]]
    df = df.sort_values("期待値" if sort == "ev" else "確率", ascending=False).head(top)
    df.insert(0, "買い目", [label(c, umaban, names, ordered) if bet not in ("単勝", "複勝") else f"{int(umaban[c[0]])}　{names[c[0]]}" for c in df["combo"]])
    df = df.drop(columns="combo").reset_index(drop=True)
    if ranged:
        df["実際のオッズ"] = [f"{a:.1f}〜{b:.1f}" for a, b in zip(df["オッズ下限"], df["オッズ上限"])]
        df["期待値（実際のオッズ）"] = [f"{a:.2f}〜{b:.2f}" for a, b in zip(df["期待値下限"], df["期待値上限"])]
    else:
        df["実際のオッズ"] = df["オッズ下限"]
        df["期待値（実際のオッズ）"] = df["期待値"]
    return df[["買い目", "確率", "実際のオッズ", "期待値（実際のオッズ）"]]
