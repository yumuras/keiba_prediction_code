"""全データで最終モデルを学習し、Streamlitの画面で使う files を models/ に保存する

使い方:
    python scripts/train_model.py              # models/models.joblib と models/backtest.pkl を作る（数分）
    python scripts/train_model.py --skip-backtest   # モデルだけ（約1分）

- models.joblib: 3着内（オッズあり／なし）・能力勝率（オッズを使わない）・市場だけのモデル
- backtest.pkl : 2026年3〜9月を、各月より前のデータだけで予測した結果（「過去のレースで確かめる」画面用）
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate as E  # noqa: E402
import predict as P  # noqa: E402
import train as T  # noqa: E402


def fit_win_sharpen(df):
    """独立した能力勝率（市場の特徴量なし）の「尖らせ具合」a（P ∝ p^a）を求める。

    各月を、それより前のデータだけで学習して予測し（ウォークフォワード）、全部まとめて、勝ち馬の対数尤度が最大になる a を出す。
    a > 1 なら、弱い馬をより弱く見積もる（最も弱い馬の勝率を、少し高く見積もりがちな偏りを直す）。
    """
    outs = []
    for m in T.TEST_MONTHS:
        train, test = df[df["month"] < m], df[df["month"] == m].copy()
        p = T.ability_model().fit(train[T.FEATURES_INDEP], train["win"]).predict_proba(test[T.FEATURES_INDEP])[:, 1]
        test["p"] = p / pd.Series(p, index=test.index).groupby(test["race_id"]).transform("sum")
        outs.append(test)
    d = pd.concat(outs).sort_values(["month", "date", "venue", "race_no", "umaban"]).reset_index(drop=True)
    starts = E.race_starts(d["race_id"])
    return float(E.fit_conditional_logit(np.log(d[["p"]].clip(1e-6).to_numpy()), starts, d["win"].to_numpy(), [1.0])[0])


def fit_win_benter(df, top3_model):
    """市場込みの勝率（Benter型）の係数 (a, b)。

    P(馬iが勝つ) = softmax_i( a × log(市場の勝率_i) + b × 能力モデル（3着内）のlogit_i )  （レースの中で）。
    能力モデルのlogitには、学習データに対する out-of-fold の予測を使う（evaluate.py と同じ方法）。
    """
    from scipy.special import logit

    ok = df.groupby("race_id")["mkt_prob"].transform(lambda s: s.notna().all()).to_numpy()
    d = df[ok]
    X = np.c_[np.log(d["mkt_prob"].to_numpy()), logit(np.clip(top3_model.oof[ok], 1e-4, 1 - 1e-4))]
    return E.fit_conditional_logit(X, E.race_starts(d["race_id"]), d["win"].to_numpy(), [1.0, 0.0])


def fit_place_pay(df):
    """3着以内に入ったときの複勝払戻（100円あたり）の予想: log(払戻) = slope × log(単勝オッズ) + icpt。smear は、対数を戻すときの補正"""
    placed = df[df["place_payout"] > 0]
    slope, icpt = np.polyfit(np.log(placed["odds"]), np.log(placed["place_payout"]), 1)
    smear = float(np.exp(np.log(placed["place_payout"]) - (slope * np.log(placed["odds"]) + icpt)).mean())
    return dict(slope=float(slope), icpt=float(icpt), smear=smear)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-backtest", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    df = T.add_features(T.load())
    print(f"特徴量 {len(df)}行 {time.time() - t0:.0f}s", flush=True)

    P.MODEL_DIR.mkdir(exist_ok=True)
    top3_odds = T.Top3Model(use_odds=True).fit(df)
    models = {
        "top3_odds": top3_odds,
        "win_benter": [float(x) for x in fit_win_benter(df, top3_odds)],
        "place_pay": fit_place_pay(df),
        "top3_noodds": T.Top3Model(use_odds=False).fit(df),
        # 能力勝率: オッズ・他人の評価（過去のオッズ由来の特徴量）を使わない（docs/principles.md の15）
        "win_indep": T.ability_model().fit(df[T.FEATURES_INDEP], df["win"]),
        "win_sharpen": fit_win_sharpen(df),
        "market_lr": LogisticRegression().fit(np.c_[np.log(df["mkt_prob"]), np.log(df["num_runners"])], df["top3"]),
        "meta": dict(trained_at=datetime.now().strftime("%Y-%m-%d %H:%M"), n_races=int(df["race_id"].nunique()), n_rows=len(df),
                     date_min=str(df["date"].min().date()), date_max=str(df["date"].max().date()), n_features=len(T.FEATURES),
                     features=list(T.FEATURES)),
    }
    joblib.dump(models, P.MODEL_PATH, compress=3)
    print(f"モデルを保存 {P.MODEL_PATH} {time.time() - t0:.0f}s", flush=True)

    if not args.skip_backtest:
        pred = E.collect_predictions(df)
        keep = ["race_id", "date", "venue", "race_no", "race_class", "surface", "distance", "going", "num_runners", "umaban", "horse_name",
                "jockey", "finish", "top3", "win", "odds", "popularity", "win_payout", "place_payout", "mkt_prob",
                "p_ability", "p_top3", "p_mkt3", "p_win_raw", "p_win_mkt", "p_win_model", "p_win_w"]
        pred[keep].to_pickle(P.BACKTEST_PATH, compression="gzip")
        print(f"バックテストを保存 {P.BACKTEST_PATH} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
