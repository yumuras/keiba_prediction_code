"""モデルが「オッズ（市場）より良いか」を測る評価

使い方:
    python src/evaluate.py

AUCやLogLossが高くても、それがオッズから得られる情報の範囲なら、馬券では役に立たない。
そこで次の3つで、市場と比べた改善幅を、レース単位のブートストラップによる95%区間つきで出す。

1. 勝ち馬の予測（Benter型）: レース内の多項ロジット
       P(馬iが勝つ) = softmax_i( a * log(市場の勝率_i) + b * 能力モデルのlogit_i )
   を学習し、「市場だけ（b = 0）」より、勝ち馬の対数尤度がどれだけ上がるか。
2. 3着以内の予測: オッズありモデルが、オッズだけのモデルよりLogLoss・レース内AUCで良いか。
3. 校正: モデルが市場より高く（低く）評価した馬は、実際にその分、結果が良い（悪い）か。
   期待値（モデルの勝率 × オッズ）の区分ごとの単勝回収率と、その誤差。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

import train as T

RNG_SEED = 0
N_BOOT = 2000


# ---- レース内の多項ロジット（勝ち馬の予測） ----------------------------------------------

def race_starts(race_ids):
    """レースごとの開始位置。同じレースの行が連続して並んでいる前提（load() が日付・場・R・馬番で整列済み）"""
    codes, _ = pd.factorize(race_ids)
    return np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])


def race_logsumexp(score, starts):
    count = np.diff(np.r_[starts, len(score)])
    shifted = score - np.repeat(np.maximum.reduceat(score, starts), count)
    return np.log(np.add.reduceat(np.exp(shifted), starts)) + np.maximum.reduceat(score, starts)


def win_log_likelihood(score, starts, win):
    """レースごとの「勝ち馬の対数確率」"""
    return score[win == 1] - race_logsumexp(score, starts)


def fit_conditional_logit(X, starts, win, init):
    """勝ち馬の対数尤度を最大化する係数を求める（レース内ソフトマックス）"""
    nll = lambda w: -win_log_likelihood(X @ w, starts, win).mean()
    return minimize(nll, init, method="BFGS").x


def softmax_in_race(score, starts):
    count = np.diff(np.r_[starts, len(score)])
    return np.exp(score - np.repeat(race_logsumexp(score, starts), count))


# ---- 予測の作成 -----------------------------------------------------------------------

# 勝ち馬の予測の比較対象。(表示名, 予測の列名)
WIN_VARIANTS = [
    ("市場（オッズそのまま）", "p_win_raw"),
    ("市場（補正あり）", "p_win_mkt"),
    ("市場＋3着内の能力モデル", "p_win_model"),
    ("市場＋勝ちの能力モデル", "p_win_w"),
    ("市場＋両方", "p_win_both"),
    ("勝ちの能力モデル単独（オッズなし）", "p_win_alone"),
]
MODEL_COLS = ["p_win_model", "p_win_w", "p_win_both"]  # 市場に能力モデルを足した版（期待値の計算に使う候補）


def collect_predictions(df, months=T.TEST_MONTHS):
    """ウォークフォワードで、評価に必要な予測をまとめて作る"""
    outs = []
    for m in months:
        train, test = df[df["month"] < m], df[df["month"] == m]
        model = T.Top3Model(use_odds=True).fit(train)
        p_top3_ab = model.predict_ability(test)               # 3着以内を目的にした能力モデル
        win_model, win_oof = T.fit_with_oof(train, "win")     # 勝ちを目的にした能力モデル
        p_win_ab = win_model.predict_proba(test[T.FEATURES])[:, 1]

        # 勝ち馬モデル: 学習データの 市場の勝率 と 能力モデルの out-of-fold 予測 から係数を求める
        s_tr, s_te = race_starts(train["race_id"]), race_starts(test["race_id"])
        lg = lambda p: logit(np.clip(p, 1e-4, 1 - 1e-4))
        cols = lambda d, p3, pw: np.c_[np.log(d["mkt_prob"].to_numpy()), lg(p3), lg(pw)]
        X_tr, X_te = cols(train, model.oof, win_oof), cols(test, p_top3_ab, p_win_ab)
        win_tr = train["win"].to_numpy()
        fit = lambda idx, init: fit_conditional_logit(X_tr[:, idx], s_tr, win_tr, init)
        w_mkt, w_top3, w_win = fit([0], [1.0]), fit([0, 1], [1.0, 0.0]), fit([0, 2], [1.0, 0.0])
        w_both, w_alone = fit([0, 1, 2], [1.0, 0.0, 0.0]), fit([2], [1.0])
        prob = lambda idx, w: softmax_in_race(X_te[:, idx] @ w, s_te)

        # 3着内: 市場だけの回帰（頭数込み）。T.market_baseline と同じ
        Xm = lambda d: np.c_[np.log(d["mkt_prob"]), np.log(d["num_runners"])]
        lr = LogisticRegression().fit(Xm(train), train["top3"])

        # 複勝の予想払戻: 3着以内に入ったときの払戻が、単勝オッズでどれだけ決まるか（学習データから、対数どうしの直線）
        placed = train[train["place_payout"] > 0]
        slope, icpt = np.polyfit(np.log(placed["odds"]), np.log(placed["place_payout"]), 1)
        smear = np.exp(np.log(placed["place_payout"]) - (slope * np.log(placed["odds"]) + icpt)).mean()  # 対数を戻すときの補正
        pay_hat = np.exp(slope * np.log(test["odds"]) + icpt) * smear  # 100円あたりの円

        outs.append(test.assign(
            pay_hat=pay_hat.to_numpy(),
            p_ability=T.normalize_in_race(test, p_top3_ab),
            p_top3=model.predict(test),
            p_mkt3=T.normalize_in_race(test, lr.predict_proba(Xm(test))[:, 1]),
            p_win_raw=test["mkt_prob"].to_numpy(),
            p_win_mkt=prob([0], w_mkt), p_win_model=prob([0, 1], w_top3), p_win_w=prob([0, 2], w_win),
            p_win_both=prob([0, 1, 2], w_both), p_win_alone=prob([2], w_alone),
            coef_top3_mkt=w_top3[0], coef_top3_ab=w_top3[1], coef_w_mkt=w_win[0], coef_w_ab=w_win[1],
            coef_both_mkt=w_both[0], coef_both_top3=w_both[1], coef_both_win=w_both[2],
        ))
    return pd.concat(outs)


# ---- 指標 -----------------------------------------------------------------------------

def per_race_sum(values, race_ids):
    """馬ごとの値をレースごとに合計する"""
    return pd.Series(values).groupby(np.asarray(race_ids)).sum()


def bootstrap_ci(per_race, weight=None, n_boot=N_BOOT, seed=RNG_SEED):
    """レースを復元抽出して、平均の95%区間を求める。weight があれば「合計 / 重みの合計」"""
    rng = np.random.default_rng(seed)
    x = np.asarray(per_race, dtype=float)
    w = np.ones_like(x) if weight is None else np.asarray(weight, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    stats = x[idx].sum(axis=1) / w[idx].sum(axis=1)
    return x.sum() / w.sum(), np.percentile(stats, [2.5, 97.5])


def within_race_auc(d, col):
    """レースごとに3着以内の馬と、それ以外の馬を並べたAUC（各レースで計算し、レースごとに返す）"""
    out = []
    for _, g in d.groupby("race_id", sort=False):
        y, p = g["top3"].to_numpy(), g[col].to_numpy()
        npos, nneg = y.sum(), (1 - y).sum()
        if npos == 0 or nneg == 0:
            out.append(np.nan)
            continue
        r = rankdata(p)
        out.append((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))
    return pd.Series(out, index=d["race_id"].drop_duplicates().to_numpy())


def horse_loss(d, col):
    p = d[col].clip(1e-6, 1 - 1e-6)
    return -(d["top3"] * np.log(p) + (1 - d["top3"]) * np.log(1 - p))


def win_table(pred):
    """勝ち馬の対数尤度（1レースあたり）。0に近いほど良い。市場（補正あり）との差には、95%区間をつける"""
    win = pred["win"].to_numpy() == 1
    ll = {name: np.log(pred[col].to_numpy()[win]) for name, col in WIN_VARIANTS}
    base = ll["市場（補正あり）"]
    rows = []
    for name, col in WIN_VARIANTS:
        diff = ll[name] - base
        mean, (lo, hi) = bootstrap_ci(diff)
        rows.append({"モデル": name, "列": col, "対数尤度/レース": ll[name].mean(), "市場（補正あり）との差": mean,
                     "区間の下限": lo, "区間の上限": hi})
    table = pd.DataFrame(rows)
    best = table[table["列"].isin(MODEL_COLS)].sort_values("市場（補正あり）との差").iloc[-1]["列"]
    return table, best, pred["race_id"].nunique()


def top3_table(pred):
    d = pred.reset_index(drop=True)
    models = [("市場（オッズ＋頭数）", "p_mkt3"), ("能力モデル（オッズなし）", "p_ability"), ("オッズありモデル", "p_top3")]
    from sklearn.metrics import roc_auc_score
    rows, aucs, losses = [], {}, {}
    for name, col in models:
        aucs[col] = within_race_auc(d, col)
        losses[col] = per_race_sum(horse_loss(d, col), d["race_id"]).reindex(aucs[col].index)
        rows.append({"モデル": name, "AUC（全体）": roc_auc_score(d["top3"], d[col]),
                     "AUC（レース内の平均）": aucs[col].mean(), "LogLoss": horse_loss(d, col).mean()})
    n_horses = d.groupby("race_id", sort=False).size().reindex(aucs["p_mkt3"].index)
    out = {}
    for name, col in models[1:]:
        dl = losses[col] - losses["p_mkt3"]
        da = (aucs[col] - aucs["p_mkt3"]).dropna()
        out[name] = {"LogLoss": bootstrap_ci(dl, weight=n_horses), "AUC（レース内）": bootstrap_ci(da)}
    return pd.DataFrame(rows), out


def calibration_tables(pred, win_col="p_win_model"):
    d = pred.copy()
    # 3着内: モデル ÷ 市場 の比で区分
    d["比"] = d["p_top3"] / d["p_mkt3"]
    d["区分"] = pd.cut(d["比"], [0, 0.85, 0.95, 1.05, 1.15, np.inf], labels=["〜0.85", "0.85〜0.95", "0.95〜1.05", "1.05〜1.15", "1.15〜"])
    cal = d.groupby("区分", observed=True).agg(頭数=("top3", "size"), 市場の確率=("p_mkt3", "mean"), モデルの確率=("p_top3", "mean"),
                                              実際の複勝率=("top3", "mean"))
    cal["実際−市場"] = cal["実際の複勝率"] - cal["市場の確率"]
    cal["実際−モデル"] = cal["実際の複勝率"] - cal["モデルの確率"]
    # 勝ち馬: 期待値（モデルの勝率 × オッズ）の区分ごとの単勝回収率と、その誤差
    d["期待値"] = d[win_col] * d["odds"]
    d["期待値区分"] = pd.cut(d["期待値"], [0, 0.7, 0.85, 1.0, 1.15, np.inf], labels=["〜0.7", "0.7〜0.85", "0.85〜1.0", "1.0〜1.15", "1.15〜"])
    d["払戻"] = d["win"] * d["odds"]
    roi = d.groupby("期待値区分", observed=True).agg(頭数=("払戻", "size"), 予想の回収率=("期待値", "mean"), 実際の回収率=("払戻", "mean"),
                                                  誤差=("払戻", lambda x: x.std() / np.sqrt(len(x))))
    roi["95%区間"] = [f"{m - 1.96 * se:.0%}〜{m + 1.96 * se:.0%}" for m, se in zip(roi["実際の回収率"], roi["誤差"])]
    return cal, roi


def place_tables(pred):
    """複勝（3着以内に入れば払戻）の期待値と、実際の回収率

    払戻は PDF の払戻欄から読んだ実際の払戻（100円あたり）。読めなかったレース（同着など）は除く。
    予想払戻 `pay_hat` は、単勝オッズから決めた「3着以内に入ったときの払戻」の予想。
    """
    d = pred.dropna(subset=["place_payout"]).copy()
    d["ret"] = d["place_payout"] / 100  # 100円あたりの払戻（円ではなく倍率）
    d["ev"] = d["p_top3"] * d["pay_hat"] / 100        # モデルの期待値（1円あたりの期待払戻）
    d["ev_mkt"] = d["p_mkt3"] * d["pay_hat"] / 100    # 市場だけの期待値

    def roi(sub, label):
        """レースごとの払戻合計と賭け数から、回収率と95%区間（レースを復元抽出）"""
        per = sub.groupby("race_id").agg(r=("ret", "sum"), n=("ret", "size"))
        m, (lo, hi) = bootstrap_ci(per["r"], weight=per["n"])
        return {"戦略": label, "賭け数": int(per["n"].sum()), "的中率": (sub["ret"] > 0).mean(), "回収率": m, "下限": lo, "上限": hi}

    top = lambda col, asc=False: d.loc[d.groupby("race_id")[col].idxmin() if asc else d.groupby("race_id")[col].idxmax()]
    strategies = pd.DataFrame([
        roi(d, "全馬に賭ける（基準）"),
        roi(top("odds", asc=True), "1番人気"),
        roi(top("p_mkt3"), "市場の予測1位（オッズ＋頭数）"),
        roi(top("p_ability"), "能力モデル（オッズなし）の1位"),
        roi(top("p_top3"), "オッズありモデルの1位"),
    ])
    thresholds = pd.DataFrame([roi(d[d["ev"] >= x], f"期待値 {x:.2f} 以上") for x in (0.8, 0.9, 1.0, 1.1)
                               if (d["ev"] >= x).any()])
    d["期待値区分"] = pd.cut(d["ev"], [0, 0.7, 0.8, 0.9, 1.0, np.inf], labels=["〜0.7", "0.7〜0.8", "0.8〜0.9", "0.9〜1.0", "1.0〜"])
    bins = d.groupby("期待値区分", observed=True).agg(頭数=("ret", "size"), 予想=("ev", "mean"), 市場だけの予想=("ev_mkt", "mean"),
                                                  実際=("ret", "mean"), 誤差=("ret", lambda x: x.std() / np.sqrt(len(x))))
    bins["95%区間"] = [f"{m - 1.96 * se:.0%}〜{m + 1.96 * se:.0%}" for m, se in zip(bins["実際"], bins["誤差"])]
    return strategies, thresholds, bins, d["race_id"].nunique(), pred["race_id"].nunique()


def report(pred):
    """表をまとめて返す（ノートブックからも使う）"""
    win, best, n_races = win_table(pred)
    top3, top3_diff = top3_table(pred)
    cal, roi = calibration_tables(pred, win_col=best)
    place, place_thr, place_bins, n_place, _ = place_tables(pred)
    return dict(win=win, best=best, n_races=n_races, top3=top3, top3_diff=top3_diff, cal=cal, roi=roi,
                place=place, place_thr=place_thr, place_bins=place_bins, n_place=n_place)


def print_report(r, pred):
    pct = "{:.1%}".format
    print(f"評価: {r['n_races']}レース / {len(pred)}頭（ウォークフォワード）\n")
    print("■ 1. 勝ち馬の予測（1レースあたりの対数尤度。0に近いほど良い。区間は市場（補正あり）との差の95%区間）")
    w = r["win"].copy()
    w["判定"] = ["" if row["モデル"].startswith("市場（") else ("市場より良い" if row["区間の下限"] > 0 else ("市場より悪い" if row["区間の上限"] < 0 else "差は誤差の範囲"))
               for _, row in w.iterrows()]
    w["区間"] = [f"[{lo:+.4f}, {hi:+.4f}]" if not row["モデル"].startswith("市場（") else "" for (_, row), lo, hi in zip(w.iterrows(), w["区間の下限"], w["区間の上限"])]
    print(w[["モデル", "対数尤度/レース", "市場（補正あり）との差", "区間", "判定"]].to_string(
        index=False, float_format=lambda x: f"{x:+.4f}"))
    c = pred.drop_duplicates("race_id").groupby("month").first()
    f = lambda col: " ".join(f"{x:+.2f}" for x in c[col])
    print("  係数（月ごと）市場＋3着内: 市場", f("coef_top3_mkt"), "/ 3着内能力", f("coef_top3_ab"))
    print("  係数（月ごと）市場＋勝ち  : 市場", f("coef_w_mkt"), "/ 勝ち能力", f("coef_w_ab"))
    print(f"  期待値・校正には「{dict((c_, n_) for n_, c_ in WIN_VARIANTS)[r['best']]}」を使用\n")
    print("■ 2. 3着以内の予測")
    print(r["top3"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for name, d in r["top3_diff"].items():
        for metric, (m, (lo, hi)) in d.items():
            good = (hi < 0) if metric == "LogLoss" else (lo > 0)
            print(f"  {name} − 市場: {metric} {m:+.4f}  95%区間 [{lo:+.4f}, {hi:+.4f}]  {'→ 市場より良い' if good else ''}")
    print("\n■ 3. 校正（モデル ÷ 市場 の比ごとの、実際の複勝率）")
    print(r["cal"].to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n■ 3'. 期待値（モデルの勝率 × オッズ）ごとの単勝回収率")
    print(r["roi"][["頭数", "予想の回収率", "実際の回収率", "95%区間"]].to_string(
        formatters={"予想の回収率": pct, "実際の回収率": pct}))


def print_place(r):
    pct = "{:.1%}".format
    f = dict(的中率=pct, 回収率=pct, 下限=pct, 上限=pct)
    print(f"\n■ 4. 複勝（払戻を読めた{r['n_place']}レース。100円あたりの回収率、区間は95%）")
    print(r["place"].to_string(index=False, formatters=f))
    print("\n  期待値（モデルの3着内確率 × 予想払戻）が一定以上の馬を、すべて買った場合")
    print(r["place_thr"].to_string(index=False, formatters=f))
    print("\n  期待値の区分ごとの回収率")
    print(r["place_bins"][["頭数", "予想", "市場だけの予想", "実際", "95%区間"]].to_string(
        formatters=dict(予想=pct, 市場だけの予想=pct, 実際=pct)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", help="予測の保存先（あれば読み込み、なければ作成する。表示だけを繰り返したいとき用）")
    args = ap.parse_args()
    if args.cache and Path(args.cache).exists():
        pred = pd.read_pickle(args.cache)
    else:
        pred = collect_predictions(T.add_features(T.load()))
        if args.cache:
            pred.to_pickle(args.cache)
    r = report(pred)
    print_report(r, pred)
    print_place(r)


if __name__ == "__main__":
    main()
