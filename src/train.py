"""JRA全場の成績CSV（2025〜2026年）から「3着以内に入るか」を予測するモデル

使い方:
    python src/train.py                  # オッズ + 馬・騎手・調教師の情報で予測
    python src/train.py --no-odds        # オッズを使わず、馬・騎手・調教師の情報だけで予測

モデルは2段構成:
    1段目（能力モデル）: オッズを使わない特徴量から3着内確率を予測する勾配ブースティング
    2段目（オッズあり時のみ）: 市場の勝率（単勝オッズから計算）と1段目の予測を
                              ロジスティック回帰で組み合わせる

評価はウォークフォワード（月ごとに、その月より前のデータだけで学習して予測）。
過去成績系の特徴量は、そのレースより前の日付のデータだけから計算してリークを防ぐ。
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = sorted((ROOT / "data").glob("all_*.csv"))  # 年ごとの成績CSV（all_2025.csv, all_2026.csv, ...）
EXTRA = sorted((ROOT / "data").glob("race_extra_*.csv"))  # scripts/extract_race_extra.py で作る（年齢条件・ハロンタイム）
LATERAL = sorted((ROOT / "data").glob("lateral_*.csv"))  # scripts/extract_lateral.py で作る
WEATHER = sorted((ROOT / "data").glob("weather_*.csv"))  # scripts/fetch_weather.py で取得

CATEGORICAL = ["venue", "surface", "going", "race_class", "sex", "weather", "turn", "inout"]
NUMERIC = [
    "distance", "handicap", "waku", "umaban", "age", "carried_weight", "weight_reduced",
    "blinker", "body_weight", "body_weight_diff", "num_runners",
    # 過去成績（そのレースより前の日付のみ）
    "horse_runs", "horse_prev_finish_rate", "horse_mean_finish_rate", "days_since_last",
    "prev_behind", "mean_behind", "prev_mkt_prob", "prev_same_surface", "prev_dist_diff",
    "horse_top3_rate", "best_finish_rate", "mean_mkt_prob",
    "class_rank", "prev_class_diff", "jockey_change", "carried_weight_diff",
    "surface_mean_finish_rate", "venue_runs",
    "jockey_runs", "jockey_top3_rate", "trainer_runs", "trainer_top3_rate",
    "jockey_surf_runs", "jockey_surf_top3_rate",
    # 相性: 馬×騎手（コンビ）、騎手×調教師（厩舎との関係）
    "horse_jockey_runs", "horse_jockey_top3_rate", "jockey_trainer_runs", "jockey_trainer_top3_rate",
    # 脚質（過去のコーナー通過順位から）と、このレースの先行馬の多さ
    "prev_early_pos", "mean_early_pos", "front_rate", "mean_last_corner_pos", "mean_gain", "n_front_runners",
    # 道悪適性（稍重・重・不良での過去成績と、良馬場との差）
    "wet_runs", "wet_mean_finish_rate", "dry_mean_finish_rate", "wet_diff", "wet_behind_diff", "wet_fit_today",
    "cond_mean_finish_rate",
    # 気象（発走時刻の観測値。気象庁）
    "temp", "humidity", "wind_speed", "precip", "precip_day",
]
# レース内での相対値（同じレースの他の馬と比べて上か下か）にする列
RELATIVE = ["jockey_top3_rate", "trainer_top3_rate", "horse_mean_finish_rate", "mean_behind",
            "carried_weight", "body_weight", "age", "horse_top3_rate", "best_finish_rate", "mean_mkt_prob",
            "mean_early_pos", "cond_mean_finish_rate"]
# 内外の位置（通過順位の（ ）の中の、内から外の並び。0 = 内、1 = 外）と、並走のかたまりの大きさ。
# 検証（2026年3〜9月）で精度が変わらなかった（AUC 0.7861 → 0.7862）ので、モデルには入れていない。列は add_features で作る
LATERAL_FEATURES = ["prev_lat4", "mean_lat4", "mean_grp4"]
# 年齢条件・古馬混合戦、ペース・展開（ハロンタイム）、レースレベル（前走の相手のその後の成績）も、同じ検証で精度が変わらなかったので、モデルには入れていない
EXPERIMENTAL_FEATURES = {
    "年齢条件・古馬混合": ["age_limited", "age_over_cond", "age_min", "age_open", "mixed_mean_finish_rate", "mixed_runs"],
    "ペース・展開": ["prev_pace_rel", "prev_pace_load", "mean_pace_load"],
    "レースレベル": ["rival_level_prev", "rival_n_prev"],
}
FEATURES = CATEGORICAL + NUMERIC + [f"{c}_z" for c in RELATIVE]  # 能力モデルの特徴量（オッズなし）

# クラスの格（昇級・降級を数値で表すため）
CLASS_RANK = {"新馬": 0, "未勝利": 0, "1勝クラス": 1, "2勝クラス": 2, "3勝クラス": 3, "オープン": 4, "L": 4,
              "GⅢ": 5, "GⅡ": 6, "GⅠ": 7, "J・GⅢ": 5, "J・GⅡ": 6, "J・GⅠ": 7}

# ウォークフォワードで評価する月。month は 2026年1月 = 1 とした通算の月番号（2025年12月 = 0、2025年1月 = -11）。
# 2026年3〜9月を評価し、それより前の2025年〜2026年2月は学習のみに使う
# 市場（他人の評価）から作られた特徴量。能力勝率にはオッズや他人の評価を混ぜない（docs/principles.md の15）ので、
# 「過去のオッズ（市場評価）」の特徴量を外したものを、独立した能力勝率の特徴量にする
MARKET_DERIVED = ["prev_mkt_prob", "mean_mkt_prob", "mean_mkt_prob_z"]
FEATURES_INDEP = [f for f in FEATURES if f not in MARKET_DERIVED]

TEST_MONTHS = tuple(range(3, 10))


def load(paths=DATA):
    df = pd.concat([pd.read_csv(p, dtype={"race_id": str}, parse_dates=["date"]) for p in paths], ignore_index=True)
    # race_id は年ごとの通し番号で、年をまたぐと重複するので、年を付けて一意にする
    df["race_id"] = df["date"].dt.year.astype(str) + "_" + df["race_id"]
    if EXTRA:
        ex = pd.concat([pd.read_csv(p, dtype={"race_id": str}).assign(year=p.stem[-4:]) for p in EXTRA], ignore_index=True)
        ex["race_id"] = ex["year"] + "_" + ex["race_id"]
        df = df.merge(ex.drop(columns="year").drop_duplicates("race_id"), on="race_id", how="left")
    if LATERAL:
        lat = pd.concat([pd.read_csv(p, dtype={"race_id": str}).assign(year=p.stem[-4:]) for p in LATERAL], ignore_index=True)
        lat["race_id"] = lat["year"] + "_" + lat["race_id"]
        df = df.merge(lat.drop(columns=["year", "horse_name"]), on=["race_id", "umaban"], how="left")
    return prepare(df)


def prepare(df):
    """成績（または、結果のない出馬表）の行から、目的変数・市場の確率・位置取り・気象などの基本の列を作る。

    出馬表の行は finish・time_sec・corner* などが空欄のままでよい（結果の列は、そのレースの特徴量には使われない）。
    """
    df = df.sort_values(["date", "venue", "race_no", "umaban"]).reset_index(drop=True)
    df["month"] = (df["date"].dt.year - 2026) * 12 + df["date"].dt.month
    df["top3"] = (df["finish"] <= 3).astype(int)  # 競走中止（finish欠損）は0
    # 着順を頭数で割って 0〜1 に正規化（中止は最下位扱い）
    df["finish_rate"] = df["finish"].fillna(df["num_runners"]) / df["num_runners"]
    df["weight_reduced"] = df["base_weight"] - df["carried_weight"]
    for c in ["handicap", "blinker"]:
        df[c] = df[c].astype(int)

    # 市場の勝率: 1/オッズ をレース内で合計1に正規化（JRAの控除分を取り除く）
    inv = 1 / df["odds"]
    df["mkt_prob"] = inv / inv.groupby(df["race_id"]).transform("sum")
    # 勝ち馬とのタイム差（秒）。大敗の影響を抑えるため5秒で頭打ち
    win_time = df[df["finish"] == 1].groupby("race_id")["time_sec"].min()
    df["behind"] = (df["time_sec"] - df["race_id"].map(win_time)).clip(upper=5)
    df["class_rank"] = df["race_class"].map(CLASS_RANK)
    df["wet"] = df["going"].isin(["稍重", "重", "不良"]).astype(int)  # 道悪

    # コースの形: 「芝・右・外」→ 回り（右/左/直線）と内外（内/外/外内）
    parts = df["course"].str.split("・")
    df["turn"] = parts.apply(lambda p: next((x for x in p if x in ("右", "左", "直線")), "不明"))
    df["inout"] = parts.apply(lambda p: next((x for x in p if x in ("外", "外内")), "内"))

    # 脚質の元になる位置取り（このレースの結果なので、特徴量には「前走まで」の値として使う）
    corners = df[["corner1", "corner2", "corner3", "corner4"]]
    first = corners.bfill(axis=1).iloc[:, 0]  # 最初に記録されたコーナーの順位
    df["early_pos"] = (first - 1) / (df["num_runners"] - 1)  # 0 = 先頭、1 = 最後方
    df["last_corner_pos"] = (df["corner4"] - 1) / (df["num_runners"] - 1)
    df["is_front"] = (first == 1).astype(float).where(first.notna())  # 逃げたか
    df["gain"] = (df["corner4"] - df["finish"]) / df["num_runners"]  # 最終コーナーから何頭抜いたか
    for c in ("corner4_lat", "corner4_grp"):  # 内外の位置の列が無い行（出馬表など）は、空欄にそろえる
        if c not in df:
            df[c] = np.nan

    # ハロンタイム・年齢条件（PDFのレースごとの情報。無い行は空欄）
    for c in ("age_min", "age_open", "laps", "last3f"):
        if c not in df:
            df[c] = np.nan
    first3 = df["laps"].map(lambda t: sum(float(x) for x in t.split("―")[:3]) if isinstance(t, str) and len(t.split("―")) >= 3 else np.nan)
    df["pace_rel"] = (first3 - df["last3f"]) / (first3 + df["last3f"])  # マイナス = 前半が速い（ハイペース）、プラス = 前半が遅い（スロー）
    df["age_limited"] = (df["age_open"] == 0).astype(float).where(df["age_open"].notna())  # 1 = その年齢だけの限定戦
    df["age_over_cond"] = df["age"] - df["age_min"]

    # 気象: 発走時刻の直前の正時の観測値（15:30発走なら15時の値）と、その日の発走までの降水量
    if WEATHER:
        w = pd.concat([pd.read_csv(f, parse_dates=["date"]) for f in WEATHER]).sort_values(["venue", "date", "hour"])
        w["precip_day"] = w.groupby(["venue", "date"])["precip"].cumsum()
        df["hour"] = df["start_time"].str[:2].astype(int)
        df = df.merge(w[["venue", "date", "hour", "temp", "humidity", "wind_speed", "precip", "precip_day"]],
                      on=["venue", "date", "hour"], how="left")
    return df


def prior_stats(df, key, prefix, k=20, prior=None):
    """key（騎手・調教師など）ごとに、その日より前の騎乗数と3着内率を計算する"""
    daily = df.groupby([key, "date"]).agg(n=("top3", "size"), s=("top3", "sum")).reset_index()
    daily = daily.sort_values([key, "date"])
    g = daily.groupby(key)
    daily[f"{prefix}_runs"] = g["n"].cumsum() - daily["n"]
    prev_s = g["s"].cumsum() - daily["s"]
    # 騎乗数が少ないときに極端な値にならないよう、全体平均で平滑化
    prior = df["top3"].mean() if prior is None else prior  # 平滑化の基準（全体の3着内率）。一部の行だけで計算するときは、全体の値を渡す
    daily[f"{prefix}_top3_rate"] = (prev_s + prior * k) / (daily[f"{prefix}_runs"] + k)
    return df.merge(daily[[key, "date", f"{prefix}_runs", f"{prefix}_top3_rate"]], on=[key, "date"], how="left")


def rival_level(df, min_n=3):
    """前走のレースレベル: 前走で戦った相手が、その後のレース（今回の日付より前のもの）で、人気より良い着順だったか。

    相手の「次走の着順率が、人気の順位率よりどれだけ良かったか」の平均。プラス = 相手が、その後に人気以上の成績（レースレベルが高かった）。
    今回の日付より前に終わったレースの結果だけを使う（未来は見ない）。
    """
    d = df[["race_id", "horse_name", "date", "finish_rate", "popularity", "num_runners", "top3"]].sort_values(["horse_name", "date"])
    g = d.groupby("horse_name")
    rv = pd.DataFrame({"race_id": d["race_id"], "horse_name": d["horse_name"], "next_date": g["date"].shift(-1),
                       "perf": (g["popularity"].shift(-1) / g["num_runners"].shift(-1) - g["finish_rate"].shift(-1))}).dropna()
    t = df[["race_id", "horse_name", "date"]].sort_values(["horse_name", "date"])
    t["prev_race"] = t.groupby("horse_name")["race_id"].shift(1)
    t = t.dropna(subset=["prev_race"]).reset_index()
    m = t.merge(rv, left_on="prev_race", right_on="race_id", suffixes=("", "_rv"))
    m = m[(m["horse_name"] != m["horse_name_rv"]) & (m["next_date"] < m["date"])]
    agg = m.groupby("index")["perf"].agg(["mean", "size"])
    ok = agg["size"] >= min_n
    df["rival_level_prev"] = agg["mean"].where(ok).reindex(df.index)
    df["rival_n_prev"] = agg["size"].reindex(df.index)
    return df


def add_features(df, prior=None):
    df = prior_stats(df, "jockey", "jockey", prior=prior)
    df = prior_stats(df, "trainer", "trainer", prior=prior)
    # 騎手の芝/ダート/障害別の成績（芝は上手いがダートは苦手、などを捉える）
    df["jockey_surface"] = df["jockey"] + "_" + df["surface"]
    df = prior_stats(df, "jockey_surface", "jockey_surf", prior=prior)
    # 相性。コンビの回数は少ないので、平滑化は弱め（k=5）にして本人たちの成績が出やすくする
    df["horse_jockey"] = df["horse_name"] + "_" + df["jockey"]
    df = prior_stats(df, "horse_jockey", "horse_jockey", k=5, prior=prior)
    df["jockey_trainer"] = df["jockey"] + "_" + df["trainer"]
    df = prior_stats(df, "jockey_trainer", "jockey_trainer", prior=prior)

    # 馬ごとの過去成績。shift(1) で「前走まで」の値にする（同じ馬が同じ日に2回走ることはない）
    df = df.sort_values(["horse_name", "date"])
    g = df.groupby("horse_name")
    past_mean = lambda s: s.shift(1).expanding().mean()

    def pm(col, keys="horse_name"):
        """グループごとの、前走までの平均（空欄は除く）。transform(past_mean) と同じ値を、速く出す"""
        v = df[col]
        n = v.notna().astype(int)
        v0 = v.fillna(0)
        gk = df[keys] if isinstance(keys, str) else [df[k] for k in keys]
        prev_n = n.groupby(gk).cumsum() - n
        return (v0.groupby(gk).cumsum() - v0) / prev_n.replace(0, np.nan)
    df["horse_runs"] = g.cumcount()
    df["horse_prev_finish_rate"] = g["finish_rate"].shift(1)
    df["horse_mean_finish_rate"] = pm("finish_rate")
    df["days_since_last"] = g["date"].diff().dt.days
    df["prev_behind"] = g["behind"].shift(1)
    df["mean_behind"] = pm("behind")
    df["prev_mkt_prob"] = g["mkt_prob"].shift(1)  # 前走で市場からどれだけ評価されていたか
    df["prev_same_surface"] = (g["surface"].shift(1) == df["surface"]).astype(int)
    df["prev_dist_diff"] = df["distance"] - g["distance"].shift(1)
    # 過去の3着内率（出走数が少ない馬は全体平均0.2寄りに平滑化）・最高着順率・平均の市場評価
    past_top3 = g["top3"].transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
    df["horse_top3_rate"] = (past_top3 + 0.2 * 3) / (df["horse_runs"] + 3)
    df["best_finish_rate"] = g["finish_rate"].transform(lambda s: s.shift(1).expanding().min())
    df["mean_mkt_prob"] = pm("mkt_prob")
    # 前走からの変化: 昇級(+)/降級(-)、乗り替わり、斤量の増減
    df["prev_class_diff"] = df["class_rank"] - g["class_rank"].shift(1)
    prev_jockey = g["jockey"].shift(1)
    df["jockey_change"] = (prev_jockey != df["jockey"]).astype(float).where(prev_jockey.notna())
    df["carried_weight_diff"] = df["carried_weight"] - g["carried_weight"].shift(1)
    # コース適性: 同じ芝/ダートでの過去平均着順率、同じ競馬場での出走数
    df["surface_mean_finish_rate"] = pm("finish_rate", ["horse_name", "surface"])
    df["venue_runs"] = df.groupby(["horse_name", "venue"]).cumcount()
    # 脚質: 前走・過去平均の序盤の位置（0 = 先頭）、逃げた割合、最終コーナーの位置、直線で抜いた頭数
    df["prev_early_pos"] = g["early_pos"].shift(1)
    df["mean_early_pos"] = pm("early_pos")
    df["front_rate"] = pm("is_front")
    df["mean_last_corner_pos"] = pm("last_corner_pos")
    df["mean_gain"] = pm("gain")
    # 展開: 前走のペースと、その中でどこを走ったか（ハイペースを先行 = 負担が大きい）
    df["prev_pace_rel"] = g["pace_rel"].shift(1)
    df["pace_load"] = -df["pace_rel"] * (1 - df["early_pos"])  # プラス = ハイペースを前で走った、マイナス = スローを後ろで走った
    df["prev_pace_load"] = g["pace_load"].shift(1)
    df["mean_pace_load"] = pm("pace_load")
    df["prev_lat4"] = g["corner4_lat"].shift(1)
    df["mean_lat4"] = pm("corner4_lat")
    df["mean_grp4"] = pm("corner4_grp")

    # 道悪適性: 道悪（稍重・重・不良）と良馬場、それぞれでの前走までの平均
    def past_mean_when(col, cond):
        v = df[col].where(cond == 1)
        prev_sum = v.fillna(0).groupby(df["horse_name"]).cumsum() - v.fillna(0)
        prev_n = v.notna().astype(int).groupby(df["horse_name"]).cumsum() - v.notna().astype(int)
        return prev_sum / prev_n.replace(0, np.nan), prev_n

    df["wet_mean_finish_rate"], df["wet_runs"] = past_mean_when("finish_rate", df["wet"])
    df["dry_mean_finish_rate"], _ = past_mean_when("finish_rate", 1 - df["wet"])
    df["wet_diff"] = df["wet_mean_finish_rate"] - df["dry_mean_finish_rate"]  # プラス = 道悪で着順が悪くなる
    df["wet_behind_diff"] = (past_mean_when("behind", df["wet"])[0]
                             - past_mean_when("behind", 1 - df["wet"])[0])   # 道悪での勝ち馬とのタイム差の広がり
    # 古馬混合戦（「以上」の条件）での、過去の平均着順率。古馬と走った経験が無い馬は空欄
    df["mixed_mean_finish_rate"], df["mixed_runs"] = past_mean_when("finish_rate", (df["age_open"] == 1).astype(int))
    df["wet_fit_today"] = df["wet_diff"] * df["wet"]  # 今日が道悪のときだけ値が入る（良馬場なら0）
    # 今日と同じ条件（芝/ダート × 道悪/良）での過去平均着順率
    df["cond_mean_finish_rate"] = pm("finish_rate", ["horse_name", "surface", "wet"])
    df = df.sort_values(["date", "venue", "race_no", "umaban"]).reset_index(drop=True)
    df = rival_level(df)

    # このレースに先行タイプ（過去平均で前から1/4以内）が何頭いるか。多いとペースが速くなりやすい
    df["n_front_runners"] = (df["mean_early_pos"] < 0.25).groupby(df["race_id"]).transform("sum")

    # レース内の相対値（zスコア）
    for c in RELATIVE:
        gr = df.groupby("race_id")[c]
        df[f"{c}_z"] = (df[c] - gr.transform("mean")) / gr.transform("std").replace(0, np.nan)

    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df


def normalize_in_race(d, p):
    """レース内の確率の合計が「3着以内に入る頭数」（通常3）になるようにそろえる"""
    p = pd.Series(p, index=d.index)
    n_top3 = np.minimum(3, d.groupby("race_id")["race_id"].transform("size"))
    return (p / p.groupby(d["race_id"]).transform("sum") * n_top3).clip(1e-4, 1 - 1e-4).to_numpy()


def ability_model():
    """1段目: オッズを使わない能力モデル。データが少ないので浅い木・低い学習率で過学習を抑える"""
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype", max_iter=300, learning_rate=0.03,
        max_leaf_nodes=7, min_samples_leaf=80, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.2, random_state=42,
    )


def fit_with_oof(train, target="top3", features=None):
    """能力モデルを `target`（"top3" か "win"）を目的に学習し、学習データに対する out-of-fold 予測も返す。

    out-of-fold は、学習データのレースを5つに分け、4つで学習して残り1つを予測する、を繰り返した予測。
    学習済みのレースに対する予測は実際より良く当たって見えるので、2段目の学習にはこちらを使う。
    """
    features = FEATURES if features is None else features
    model = ability_model().fit(train[features], train[target])
    oof = np.zeros(len(train))
    for tr_idx, va_idx in GroupKFold(5).split(train, groups=train["race_id"]):
        m = ability_model().fit(train.iloc[tr_idx][features], train.iloc[tr_idx][target])
        oof[va_idx] = m.predict_proba(train.iloc[va_idx][features])[:, 1]
    return model, oof


class Top3Model:
    """3着以内確率を予測するモデル。use_odds=False なら1段目だけを使う"""

    def __init__(self, use_odds=True):
        self.use_odds = use_odds

    def _stack_X(self, d, p_ability):
        # 2段目の入力: 市場の勝率と能力モデルの予測（どちらも対数オッズに変換）、頭数
        return pd.DataFrame({
            "log_mkt_prob": np.log(d["mkt_prob"].to_numpy()),
            "ability_logit": logit(np.clip(p_ability, 1e-4, 1 - 1e-4)),
            "log_runners": np.log(d["num_runners"].to_numpy()),
        })

    def fit(self, train):
        self.ability = ability_model().fit(train[FEATURES], train["top3"])
        if self.use_odds:
            # 2段目の学習には、1段目が「見ていないレース」に対して出した予測（out-of-fold）を使う。
            # 学習済みのレースに対する予測を使うと、能力モデルを実力以上に信用してしまうため
            oof = np.zeros(len(train))
            for tr_idx, va_idx in GroupKFold(5).split(train, groups=train["race_id"]):
                m = ability_model().fit(train.iloc[tr_idx][FEATURES], train.iloc[tr_idx]["top3"])
                oof[va_idx] = m.predict_proba(train.iloc[va_idx][FEATURES])[:, 1]
            self.oof = oof  # 学習データに対する out-of-fold 予測（evaluate.py の勝率モデルの学習にも使う）
            self.stacker = LogisticRegression().fit(self._stack_X(train, oof), train["top3"])
        return self

    def predict_ability(self, d):
        """能力モデル（オッズなし）だけの3着内確率"""
        return self.ability.predict_proba(d[FEATURES])[:, 1]

    def predict(self, d):
        p = self.predict_ability(d)
        if self.use_odds:
            p = self.stacker.predict_proba(self._stack_X(d, p))[:, 1]
        return normalize_in_race(d, p)


def walk_forward(df, use_odds=True, months=TEST_MONTHS):
    """月ごとに「その月より前のデータで学習 → その月を予測」を繰り返し、予測をまとめて返す"""
    outs = []
    for m in months:
        train, test = df[df["month"] < m], df[df["month"] == m]
        model = Top3Model(use_odds).fit(train)
        outs.append(test.assign(prob=model.predict(test)))
    return pd.concat(outs), model  # model は最後の月の学習結果


def market_baseline(df, months=TEST_MONTHS):
    """比較用: オッズだけを使うロジスティック回帰（市場の評価そのもの）"""
    X = lambda d: np.c_[np.log(d["mkt_prob"]), np.log(d["num_runners"])]
    outs = []
    for m in months:
        train, test = df[df["month"] < m], df[df["month"] == m]
        lr = LogisticRegression().fit(X(train), train["top3"])
        outs.append(test.assign(prob=normalize_in_race(test, lr.predict_proba(X(test))[:, 1])))
    return pd.concat(outs)


def evaluate_bets(test, score):
    """各レースでスコア最上位の馬の単勝を100円ずつ買った場合の成績"""
    t = test.assign(score=score)
    pick = t.loc[t.groupby("race_id")["score"].idxmax()]
    fav = t.loc[t.groupby("race_id")["odds"].idxmin()]  # 比較用: 1番人気を買い続ける
    rows = []
    for name, p in [("モデル最上位", pick), ("1番人気", fav)]:
        rows.append(dict(
            戦略=name,
            レース数=len(p),
            勝率=p["win"].mean(),
            複勝率=p["top3"].mean(),
            単勝回収率=(p["win"] * p["odds"]).sum() / len(p),
        ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-odds", action="store_true", help="オッズを使わず能力モデルだけで予測する")
    args = ap.parse_args()

    df = add_features(load())
    pred, model = walk_forward(df, use_odds=not args.no_odds)
    print(f"評価: 2026年{TEST_MONTHS[0]}〜{TEST_MONTHS[-1]}月 の {pred.race_id.nunique()}レース {len(pred)}頭"
          f"（各月、それより前のデータだけで学習）")
    print(f"能力モデルの特徴量: {len(FEATURES)}個\n")

    rows = [("モデル", pred), ("参考: オッズだけ", market_baseline(df))]
    for name, p in rows:
        print(f"{name:14s} AUC {roc_auc_score(p['top3'], p['prob']):.3f}  LogLoss {log_loss(p['top3'], p['prob']):.3f}")
    if not args.no_odds:
        coef = dict(zip(["市場の勝率", "能力モデル", "頭数"], (round(float(c), 2) for c in model.stacker.coef_[0])))
        print(f"2段目の係数（最後の月）: {coef}")
    print()

    pct = "{:.1%}".format
    print(evaluate_bets(pred, pred["prob"]).to_string(
        index=False, formatters={"勝率": pct, "複勝率": pct, "単勝回収率": pct}))


if __name__ == "__main__":
    main()
