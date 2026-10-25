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
    # 理論の補正: 無敗馬（1走以上して、すべて1着だった馬）は、市場の勝率より少なく勝つ傾向があったので（2024〜2026年、全無敗馬1,387頭で、市場の0.93倍）、
    # 市場込みの勝率を UNBEATEN_FACTOR 倍して、合計1にそろえ直す
    rec = past.groupby("horse_name")["win"].agg(["sum", "size"])
    unb = fut["horse_name"].map(lambda n: bool(n in rec.index and rec.loc[n, "size"] >= 1 and rec.loc[n, "sum"] == rec.loc[n, "size"]))
    fut["unbeaten"] = unb
    if fut["p_win_mix"].notna().all():
        w = fut["p_win_mix"] * np.where(unb, UNBEATEN_FACTOR, 1.0)
        fut["p_win_theory"] = w / w.sum()
    else:
        fut["p_win_theory"] = np.nan
    if "place_pay" in models and use_odds:
        pp = models["place_pay"]
        fut["pay_hat"] = np.exp(pp["slope"] * np.log(fut["odds"]) + pp["icpt"]) * pp["smear"]
    else:
        fut["pay_hat"] = np.nan
    return fut.sort_values("p_top3" if use_odds else "p_ability", ascending=False)


UNBEATEN_FACTOR = 0.93  # 無敗馬の勝率にかける倍率（検証: 実際の勝ち数 ÷ 市場から期待される勝ち数 ＝ 188 ÷ 203）


# 特徴量の日本語名（画面の「なぜこの勝率になったか」用。無いものは、列名のまま出す）
FEATURE_LABELS = {
    "mean_behind": "過去の、勝ち馬とのタイム差（平均）", "horse_top3_rate": "過去の3着内率", "horse_mean_finish_rate": "過去の着順（平均）",
    "body_weight": "馬体重", "horse_prev_finish_rate": "前走の着順", "jockey_top3_rate": "騎手の3着内率", "trainer_top3_rate": "調教師の3着内率",
    "cond_mean_finish_rate": "今回と同じ条件での過去の着順", "best_finish_rate": "過去の最高の着順", "prev_behind": "前走の、勝ち馬とのタイム差",
    "surface_mean_finish_rate": "同じ芝・ダートでの過去の着順", "horse_runs": "出走数", "days_since_last": "前走からの間隔（日）",
    "carried_weight": "斤量", "age": "年齢", "waku": "枠", "umaban": "馬番", "distance": "距離", "mean_early_pos": "序盤の位置取り（平均）",
    "mean_last_corner_pos": "最終コーナーの位置（平均）", "mean_gain": "直線で抜いた頭数（平均）", "front_rate": "逃げた割合", "prev_early_pos": "前走の序盤の位置",
    "jockey_surf_top3_rate": "騎手の、芝・ダート別の3着内率", "horse_jockey_top3_rate": "この馬と騎手のコンビの3着内率",
    "jockey_trainer_top3_rate": "騎手と調教師の組み合わせの3着内率", "prev_class_diff": "前走からのクラスの昇降", "jockey_change": "乗り替わり",
    "prev_mkt_prob": "前走の、市場からの評価（勝率）", "mean_mkt_prob": "過去の、市場からの評価（平均）",
    "body_weight_diff": "馬体重の増減", "num_runners": "頭数", "class_rank": "クラス", "wet_diff": "道悪での着順の差", "n_front_runners": "先行馬の頭数",
}


def feature_label(f):
    base, z = (f[:-2], True) if f.endswith("_z") else (f, False)
    name = FEATURE_LABELS.get(base, base)
    return f"{name}（メンバー内の相対）" if z else name


def explain_win(res, models, horse_name, top=7):
    """能力勝率が、なぜその値になったかを調べる。

    能力勝率（オッズなし）: 特徴量を1つずつ、そのレースの中央値（カテゴリは最頻値）に置き換えて、その馬の勝率がどれだけ変わるかを見る。
    変化がマイナスなら、その特徴量が、勝率を押し上げている。特徴量どうしの重なりは、分けられない目安。
    市場込みの勝率（オッズあり）: Benter型の式の、市場の項と能力の項。
    戻り値: (特徴量ごとの表, 市場込みの分解の辞書。オッズが無いときは None)
    """
    res = res.reset_index(drop=True)
    i = res.index[res["horse_name"] == horse_name][0]
    m, sh, F = models["win_indep"], models["win_sharpen"], T.FEATURES_INDEP
    X0 = res[F].copy()
    for c in F:
        if str(X0[c].dtype) != "category":
            X0[c] = X0[c].astype(float)
    raw = m.predict_proba(X0)[:, 1] ** sh
    p = raw[i] / raw.sum()
    rows = []
    for f in F:
        X = X0.copy()
        col = X[f]
        val = col.mode().iloc[0] if str(col.dtype) == "category" else col.median()
        X.loc[i, f] = val
        r2 = m.predict_proba(X)[:, 1][i] ** sh
        rows.append((f, X0.loc[i, f], val, r2 / (raw.sum() - raw[i] + r2) - p))
    t = pd.DataFrame(rows, columns=["f", "値", "中央", "d"])
    t = t.reindex(t["d"].abs().sort_values(ascending=False).index).head(top)
    t.insert(0, "特徴量", [feature_label(f) for f in t["f"]])
    t["勝率への影響（ポイント）"] = -t["d"] * 100  # プラス = この特徴量が、勝率を押し上げている
    mix = None
    if res["p_win_mix"].notna().all() and "win_benter" in models:
        a, b = models["win_benter"]
        mkt = res["mkt_prob"].to_numpy()

        def mix_p(raw3):
            sc = a * np.log(mkt) + b * logit(np.clip(raw3, 1e-4, 1 - 1e-4))
            w = np.exp(sc - sc.max())
            return w / w.sum()

        raw3 = models["top3_odds"].predict_ability(res)
        p_mix = mix_p(raw3)[i]
        w0 = np.exp(a * np.log(mkt) - (a * np.log(mkt)).max())
        p_a_only = (w0 / w0.sum())[i]  # 能力の項を0にした（市場の勝率だけを、指数 a で補正した）勝率
        rows2 = []
        for f in T.FEATURES:
            X = res.copy()
            col = X[f]
            X[f] = X[f].astype(float) if str(col.dtype) != "category" else col
            val = col.mode().iloc[0] if str(col.dtype) == "category" else col.astype(float).median()
            X.loc[i, f] = val
            rows2.append((f, res.loc[i, f], val, mix_p(models["top3_odds"].predict_ability(X))[i] - p_mix))
        t2 = pd.DataFrame(rows2, columns=["f", "値", "中央", "d"])
        t2 = t2.reindex(t2["d"].abs().sort_values(ascending=False).index).head(top)
        t2.insert(0, "特徴量", [feature_label(f) for f in t2["f"]])
        t2["市場込みの勝率への影響（ポイント）"] = -t2["d"] * 100
        mix = dict(mkt=float(res["p_mkt_win"][i]), mkt_term=float(a * np.log(mkt[i])), ability3=float(raw3[i]),
                   ability_term=float(b * logit(np.clip(raw3[i], 1e-4, 1 - 1e-4))), p_win_mix=float(p_mix), p_a_only=float(p_a_only),
                   a=float(a), b=float(b), table=t2[["f", "特徴量", "値", "中央", "市場込みの勝率への影響（ポイント）"]].reset_index(drop=True))
    return t[["f", "特徴量", "値", "中央", "勝率への影響（ポイント）"]].reset_index(drop=True), dict(p_ability=float(p), mix=mix)


# ---- 専門用語を使わない説明文 ---------------------------------------------------------------------

# 数字が小さいほど良い特徴量（着順、タイム差）／大きいほど良い特徴量（3着内率など）。どちらでもないものは、「高い・低い」と書く
LOWER_IS_BETTER = {"mean_behind", "prev_behind", "horse_mean_finish_rate", "horse_prev_finish_rate", "cond_mean_finish_rate",
                   "surface_mean_finish_rate", "best_finish_rate"}
HIGHER_IS_BETTER = {"horse_top3_rate", "jockey_top3_rate", "trainer_top3_rate", "jockey_surf_top3_rate", "horse_jockey_top3_rate",
                    "jockey_trainer_top3_rate", "mean_gain", "prev_mkt_prob", "mean_mkt_prob"}
PLAIN_NAME = {"mean_behind": "過去のレースで、勝ち馬につけられた差", "prev_behind": "前走で、勝ち馬につけられた差", "horse_top3_rate": "過去の3着内率",
              "horse_mean_finish_rate": "過去の着順", "horse_prev_finish_rate": "前走の着順", "cond_mean_finish_rate": "今回と同じ条件での過去の着順",
              "surface_mean_finish_rate": "同じ芝・ダートでの過去の着順", "best_finish_rate": "過去の最高の着順", "jockey_top3_rate": "騎手の成績（3着内率）",
              "trainer_top3_rate": "調教師の成績（3着内率）", "jockey_surf_top3_rate": "騎手の、この芝・ダートでの成績", "horse_jockey_top3_rate": "この馬と騎手のコンビの成績",
              "jockey_trainer_top3_rate": "騎手と調教師の組み合わせの成績", "mean_gain": "最後の直線で、他の馬を抜く力", "prev_mkt_prob": "前走での、人気（市場の評価）",
              "mean_mkt_prob": "過去の、人気（市場の評価）", "body_weight": "馬体重", "carried_weight": "背負う斤量", "age": "年齢", "waku": "枠", "umaban": "馬番",
              "mean_early_pos": "レース序盤の位置取り", "horse_runs": "走った回数", "days_since_last": "前走からの間隔", "distance": "距離"}


def _level_word(z, good_is_high):
    """メンバーの中での位置（z）を、言葉にする。good_is_high が None のときは、良し悪しでなく、高い・低いで書く"""
    mag = "飛び抜けて" if abs(z) >= 1.5 else "かなり" if abs(z) >= 1.0 else "やや" if abs(z) >= 0.4 else "ほぼ平均的"
    if mag == "ほぼ平均的":
        return "ほぼ平均的"
    high = z > 0
    if good_is_high is None:
        return f"{mag}{'高い' if high else '低い'}"
    return f"{mag}{'良い' if high == good_is_high else '悪い'}"


RATE_PCT = {"horse_top3_rate", "jockey_top3_rate", "trainer_top3_rate", "jockey_surf_top3_rate", "horse_jockey_top3_rate", "jockey_trainer_top3_rate"}
PLAIN_NAME.update({"prev_dist_diff": "前走からの距離の変化", "carried_weight_diff": "前走からの斤量の変化", "body_weight_diff": "馬体重の増減"})


def plain_reason(row):
    """説明の表の1行（f, 値, 中央）から、専門用語を使わない1つの文を作る。例: 「過去の3着内率が、メンバーの中で飛び抜けて良い」。作れないときは None"""
    f, v, med = row["f"], row["値"], row["中央"]
    base = f[:-2] if f.endswith("_z") else f
    name = PLAIN_NAME.get(base)
    if name is None:
        return None  # 日本語名が無い特徴量は、画面に出さない
    good_is_high = True if base in HIGHER_IS_BETTER else False if base in LOWER_IS_BETTER else None
    if f.endswith("_z") and isinstance(v, (int, float)):
        return f"{name}が、メンバーの中で{_level_word(float(v), good_is_high)}"
    if not (isinstance(v, (int, float)) and isinstance(med, (int, float))):
        return f"{name}が、{v}"
    if base in RATE_PCT:
        return f"{name}が、{v * 100:.0f}%（このレースの中央値は {med * 100:.0f}%）"
    if base in LOWER_IS_BETTER and base != "mean_behind" and base != "prev_behind":  # 着順を、頭数で割った値（0〜1）
        return f"{name}が、出走した馬の上位{max(v, 0.01) * 100:.0f}%（このレースの中央値は 上位{med * 100:.0f}%）"
    if base in ("mean_behind", "prev_behind"):
        return f"{name}が、{v:.1f}秒（このレースの中央値は {med:.1f}秒）"
    if base in ("prev_dist_diff", "carried_weight_diff"):
        unit = "m" if base == "prev_dist_diff" else "kg"
        return f"{name}が、{v:+.0f}{unit}（このレースの中央値は {med:+.0f}{unit}）"
    return f"{name}が、{v:.3g}（このレースの中央値は {med:.3g}）"


def plain_reasons(why, kind, n=4, min_abs=0.5):
    """表から、強み（kind="+"）または弱み（kind="-"）の文を、影響の大きい順に n 個まで作る。同じ特徴量（本人の値と、メンバー内の相対）は、1つにまとめる"""
    col = [c for c in why.columns if "影響" in c][0]
    out, seen = [], set()
    for _, r in why.iterrows():
        d = r[col]
        if (kind == "+" and d < min_abs) or (kind == "-" and d > -min_abs):
            continue
        base = r["f"][:-2] if r["f"].endswith("_z") else r["f"]
        if base in seen:
            continue
        txt = plain_reason(r)
        if txt is None:
            continue
        seen.add(base)
        out.append((txt, abs(float(d))))
        if len(out) >= n:
            break
    return out
