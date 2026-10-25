"""学習済みモデルで、これからのレース（出馬表）を予想する

使い方（Streamlitの画面から呼ばれる。コードからも使える）:
    models = joblib.load("models/models.joblib")
    history = load_history()
    result = predict_card(history, race, card, models)

出馬表の行は、過去の成績と同じ形の行（結果の列は空欄）にして、成績の履歴につなげ、
学習のときと同じ特徴量を作ってから予測する。過去成績系の特徴量は、そのレースの日付より前の履歴だけから
作られるので、出馬表の日付より後の履歴は使わない。
"""
from pathlib import Path

import joblib
from scipy.special import logit
import numpy as np
import pandas as pd

import train as T

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "models.joblib"
BACKTEST_PATH = MODEL_DIR / "backtest.pkl"
FUTURE_ID = "future_race"

VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
SURFACES = ["芝", "ダート", "障害"]
GOINGS = ["良", "稍重", "重", "不良"]
WEATHERS = ["晴", "曇", "小雨", "雨", "小雪", "雪"]
CLASSES = ["新馬", "未勝利", "1勝クラス", "2勝クラス", "3勝クラス", "オープン", "L", "GⅢ", "GⅡ", "GⅠ", "J・GⅢ", "J・GⅡ", "J・GⅠ"]

# 画面の表の見出し → 内部の列名
CARD_COLUMNS = {
    "馬名": "horse_name", "枠": "waku", "馬番": "umaban", "性別": "sex", "年齢": "age", "斤量": "carried_weight",
    "騎手": "jockey", "調教師": "trainer", "馬体重": "body_weight", "増減": "body_weight_diff", "単勝オッズ": "odds",
    "ブリンカー": "blinker", "減量": "reduction",
}


def assign_waku(n):
    """頭数 n のときの、馬番 1〜n の枠番（JRAの決め方）。

    8頭以下は馬番と同じ。9〜16頭は、前の枠から1頭ずつ、後ろの枠は2頭ずつ。17頭は8枠が3頭、18頭は7・8枠が3頭ずつ。
    """
    if n <= 8:
        sizes = [1] * n
    elif n <= 16:
        sizes = [1] * (16 - n) + [2] * (n - 8)
    else:
        k = n - 16
        sizes = [2] * (8 - k) + [3] * k
    return [w for w, size in enumerate(sizes, 1) for _ in range(size)]


def load_history():
    """成績の履歴（生のCSV。race_id には年を付ける）"""
    df = pd.concat([pd.read_csv(p, dtype={"race_id": str}, parse_dates=["date"]) for p in T.DATA], ignore_index=True)
    df["race_id"] = df["date"].dt.year.astype(str) + "_" + df["race_id"]
    return df.sort_values(["date", "venue", "race_no", "umaban"]).reset_index(drop=True)


def prepare_history(history):
    """履歴の基本の列を作る（時間のかかる処理。アプリでは1回だけ行って使い回す）"""
    return T.prepare(history)


def last_rows(history):
    """馬ごとの、最後の出走の行（名前から、性別・調教師・年齢などを補うため）"""
    return history.sort_values(["date", "race_no"]).groupby("horse_name").tail(1).set_index("horse_name")


def course_string(surface, turn, inout):
    """画面の選択から、成績CSVの `course` の書き方（例「芝・右・外」「ダート・右」）を作る"""
    if surface == "障害":
        return "芝・ダート"
    parts = [surface, turn] + ([inout] if inout != "内" else [])
    return "・".join(parts)


def horse_defaults(last, name, race_date):
    """履歴にある馬の、性別・年齢（出走日の年に合わせて加算）・調教師・騎手・馬体重・斤量を返す。無ければ None"""
    if name not in last.index:
        return None
    r = last.loc[name]
    return dict(sex=r["sex"], age=int(r["age"]) + (pd.Timestamp(race_date).year - r["date"].year), trainer=r["trainer"],
                jockey=r["jockey"], body_weight=int(r["body_weight"]) if pd.notna(r["body_weight"]) else None,
                carried_weight=float(r["base_weight"]) if pd.notna(r["base_weight"]) else None)


def card_rows(history, race, card):
    """画面で入力した出馬表（日本語の見出し）から、履歴と同じ列をもつ行を作る"""
    c = card.rename(columns=CARD_COLUMNS)
    c = c[c["horse_name"].astype(str).str.strip() != ""].copy()
    reduction = pd.to_numeric(c.pop("reduction"), errors="coerce").fillna(0).reset_index(drop=True) if "reduction" in c else 0.0
    n = len(c)
    rows = pd.DataFrame(index=range(n), columns=history.columns)
    for k, v in c.reset_index(drop=True).items():
        rows[k] = v
    race_date = pd.Timestamp(race["date"])
    fixed = dict(race_id=FUTURE_ID, date=race_date, venue=race["venue"], race_no=int(race["race_no"]), kai=0, nichi=0,
                 weather=race["weather"], going=race["going"], surface=race["surface"], course=race["course"],
                 distance=int(race["distance"]), race_class=race["race_class"], handicap=bool(race["handicap"]),
                 start_time=race["start_time"], num_runners=n)
    for k, v in fixed.items():
        rows[k] = v
    for col in ["waku", "umaban", "age"]:
        rows[col] = pd.to_numeric(rows[col]).astype(int)
    for col in ["carried_weight", "body_weight", "body_weight_diff", "odds"]:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    rows["blinker"] = rows["blinker"].fillna(False).astype(bool)
    for col in history.columns:  # 結果の列など、すべて空欄の数値列は、数値型（NaN）にそろえる（文字列型だと計算できない）
        if rows[col].isna().all() and pd.api.types.is_numeric_dtype(history[col]) and not pd.api.types.is_bool_dtype(history[col]):
            rows[col] = pd.Series(np.nan, index=rows.index, dtype="float64")
    rows["base_weight"] = rows["carried_weight"] + reduction  # 減量騎手のときは、本来の斤量（斤量 + 減量）
    rows["jockey_mark"] = ""
    if rows["odds"].notna().all():
        rows["popularity"] = rows["odds"].rank(method="min")
    return rows


def predict_card(history, race, card, models, prepared=None):
    """出馬表（race: レースの条件、card: 馬ごとの表）を予想し、馬ごとの確率を返す

    history は生の履歴、prepared は prepare_history(history) の結果（渡すと速い）。
    特徴量は、出馬表の馬・騎手・調教師に関係する行だけで作る（全履歴で作った場合と同じ値になる。
    過去成績系の特徴量は、馬・騎手・調教師ごとの履歴だけから決まるため）。
    """
    rows = card_rows(history, race, card)
    race_date = pd.Timestamp(race["date"])
    hp = T.prepare(history) if prepared is None else prepared
    past = hp[hp["date"] < race_date]  # 出馬表の日付より前の履歴だけを使う
    rows = T.prepare(rows)
    rel = (past["horse_name"].isin(rows["horse_name"]) | past["jockey"].isin(rows["jockey"]) | past["trainer"].isin(rows["trainer"]))
    df = T.add_features(pd.concat([past[rel], rows], ignore_index=True), prior=past["top3"].mean())
    fut = df[df["race_id"] == FUTURE_ID].copy()

    use_odds = fut["odds"].notna().all()
    fut["p_ability"] = models["top3_noodds"].predict(fut)                       # オッズなしの3着内確率
    fut["p_top3"] = models["top3_odds"].predict(fut) if use_odds else np.nan     # オッズありの3着内確率
    # 能力勝率: オッズ・他人の評価を使わず、先に決める（合計1）。オッズは、この値を固定したあとの期待値にだけ使う
    pw = models["win_indep"].predict_proba(fut[T.FEATURES_INDEP])[:, 1] ** models["win_sharpen"]
    fut["p_win"] = pw / pw.sum()
    fut["ev_win"] = fut["p_win"] * fut["odds"]                                    # 単勝の期待値（1円あたり）。オッズが無ければ空欄
    if use_odds:
        x = np.c_[np.log(fut["mkt_prob"]), np.log(fut["num_runners"])]
        fut["p_mkt3"] = T.normalize_in_race(fut, models["market_lr"].predict_proba(x)[:, 1])
        fut["p_mkt_win"] = fut["mkt_prob"]
    else:
        fut["p_mkt3"] = np.nan
        fut["p_mkt_win"] = np.nan
    fut["use_odds"] = use_odds
    # 市場込みの勝率（Benter型。オッズが全頭そろっているときだけ）と、複勝の予想払戻（3着以内に入ったときの、100円あたりの払戻）
    if use_odds and "win_benter" in models:
        a, b = models["win_benter"]
        raw = models["top3_odds"].predict_ability(fut)
        score = a * np.log(fut["mkt_prob"].to_numpy()) + b * logit(np.clip(raw, 1e-4, 1 - 1e-4))
        w = np.exp(score - score.max())
        fut["p_win_mix"] = w / w.sum()
    else:
        fut["p_win_mix"] = np.nan
    if "place_pay" in models and use_odds:
        pp = models["place_pay"]
        fut["pay_hat"] = np.exp(pp["slope"] * np.log(fut["odds"]) + pp["icpt"]) * pp["smear"]
    else:
        fut["pay_hat"] = np.nan
    return fut.sort_values("p_top3" if use_odds else "p_ability", ascending=False)
