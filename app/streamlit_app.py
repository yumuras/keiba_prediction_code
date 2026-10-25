"""競馬予想モデルの画面（Streamlit）

使い方:
    uv run streamlit run app/streamlit_app.py

- 予想する: これからのレース（出馬表）を入力して、馬ごとの3着内確率を出す
- 過去のレースで確かめる: 過去の月ごとの予測（各月より前のデータだけで学習）を、実際の結果と並べる
- モデルの成績: 市場（オッズ）との比較

モデルは scripts/train_model.py で作る（models/ に保存される）。
"""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import bets as B  # noqa: E402
import jra_card as J  # noqa: E402
import jra_odds as O  # noqa: E402
import predict as P  # noqa: E402
import train as T  # noqa: E402

st.set_page_config(page_title="競馬予想モデル", page_icon="🏇", layout="wide")

# スマホ向けの調整: セレクトボックスの一覧が画面の下に切れないよう、一覧の高さを画面の4割までにし、ページの下に余白を足す。タップしやすいよう、選択肢の高さも広げる
st.markdown("""
<style>
@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; padding-bottom: 50vh; }
  div[data-baseweb="popover"] ul[role="listbox"] { max-height: 40vh !important; }
  div[data-baseweb="popover"] li[role="option"] { padding-top: 0.7rem; padding-bottom: 0.7rem; }
  div[data-baseweb="select"] > div { min-height: 3rem; }
  button[kind="secondary"], button[kind="primary"] { min-height: 3rem; }
}
</style>
""", unsafe_allow_html=True)


# ---- データ・モデルの読み込み（一度だけ） -----------------------------------------------------

@st.cache_resource(show_spinner="成績の履歴を読み込み中…")
def get_history():
    return P.load_history()


@st.cache_resource(show_spinner="履歴を準備中…")
def get_prepared():
    return P.prepare_history(get_history())


@st.cache_resource
def get_last():
    return P.last_rows(get_history())


@st.cache_resource(show_spinner="モデルを読み込み中…")
def get_models():
    return joblib.load(P.MODEL_PATH) if P.MODEL_PATH.exists() else None


@st.cache_data(show_spinner="バックテストを読み込み中…")
def get_backtest():
    return pd.read_pickle(P.BACKTEST_PATH, compression="gzip") if P.BACKTEST_PATH.exists() else None


@st.cache_data(ttl=600, show_spinner="JRA公式から、今週の開催の一覧を取得中…")
def jra_meetings():
    return J.list_meetings()


@st.cache_data(ttl=300, show_spinner="JRA公式から、出馬表を取り込み中…（数秒）")
def jra_card(cname):
    return J.fetch_card(cname)


@st.cache_data(ttl=120, show_spinner="JRA公式から、実際のオッズを取り込み中…（約10秒）")
def jra_real_odds(cname):
    return O.fetch_odds(cname)


def saved_cards():
    return sorted((ROOT / "cards").glob("*.json"), reverse=True)


WEEKDAY = "月火水木金土日"
CARD_COLS = ["馬名", "枠", "馬番", "性別", "年齢", "斤量", "騎手", "調教師", "馬体重", "増減", "単勝オッズ", "ブリンカー", "減量"]


def horses_to_card(horses, n=None):
    """取り込んだ馬の一覧（英語のキー）から、予想に渡す表（日本語の見出し）を作る。枠が無ければ、JRAの決め方で決める"""
    horses = sorted(horses, key=lambda h: h["umaban"])
    waku = P.assign_waku(len(horses))
    return pd.DataFrame([{
        "馬名": h["name"], "枠": h.get("waku") or waku[i], "馬番": h["umaban"], "性別": h["sex"], "年齢": h["age"], "斤量": h["weight"],
        "騎手": h["jockey"], "調教師": h["trainer"], "馬体重": h.get("body_weight"), "増減": h.get("body_weight_diff"),
        "単勝オッズ": h.get("odds"), "ブリンカー": bool(h.get("blinker")), "減量": h.get("reduction", 0.0)} for i, h in enumerate(horses)], columns=CARD_COLS)


def race_dict(day, r, going, weather):
    """レースの条件（予想に渡す形）"""
    return dict(date=day["date"], venue=day["venue"], race_no=r["race_no"], start_time=r["start_time"], surface=r["surface"], distance=r["distance"],
                course=P.course_string(r["surface"], r["turn"], r["inout"]), going=going, weather=weather, race_class=r["race_class"], handicap=r["handicap"],
                odds_cname=r.get("odds_cname"))


def card_from_history(race_id):
    """履歴の1レースを、取り込みと同じ形（条件と馬の一覧）にする（練習・やり直し用）"""
    g = get_history()
    g = g[g["race_id"] == race_id].sort_values("umaban")
    r = g.iloc[0]
    course = r["course"] if isinstance(r["course"], str) else ""
    day = dict(date=r["date"].strftime("%Y-%m-%d"), venue=r["venue"], going=r["going"], weather=r["weather"])
    race = dict(race_no=int(r["race_no"]), start_time=r["start_time"], name=f"{r['race_class']}", race_class=r["race_class"], surface=r["surface"],
                distance=int(r["distance"]), turn=next((x for x in ("右", "左", "直線") if x in course), "右"),
                inout="外内" if "外内" in course else ("外" if "外" in course else "内"), handicap=bool(r["handicap"]),
                horses=[dict(umaban=int(x.umaban), waku=int(x.waku), name=x.horse_name, sex=x.sex, age=int(x.age), weight=float(x.carried_weight),
                             reduction=float(x.base_weight - x.carried_weight), jockey=x.jockey, trainer=x.trainer, odds=x.odds,
                             body_weight=int(x.body_weight) if pd.notna(x.body_weight) else None,
                             body_weight_diff=int(x.body_weight_diff) if pd.notna(x.body_weight_diff) else None, blinker=bool(x.blinker))
                        for x in g.itertuples()])
    return day, race


# ---- サイドバー -------------------------------------------------------------------------------

models = get_models()
with st.sidebar:
    st.title("🏇 競馬予想モデル")
    if models:
        m = models["meta"]
        st.caption(f"学習: {m['date_min']}〜{m['date_max']}（{m['n_races']:,}レース）\n\n特徴量 {m['n_features']}個／作成 {m['trained_at']}")
    st.warning("このモデルは、公開されている成績情報の範囲では**オッズ（市場）を超えていません**。"
               "馬券で利益が出る保証はありません。参考として使ってください。", icon="⚠️")
    st.markdown("結論と限界は [README](https://github.com/yumuras/keiba_prediction#結論と限界) に書いています。")

if models is None:
    st.error("学習済みモデルがありません。ターミナルで `uv run python scripts/train_model.py` を実行するか、下のボタンで学習してください（数分かかります）。")
    if st.button("モデルを学習する"):
        with st.spinner("学習中…（数分）"):
            out = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_model.py")], capture_output=True, text=True)
        st.code(out.stdout[-1500:] + out.stderr[-1500:])
        get_models.clear()
        st.rerun()
    st.stop()

tab_pred, tab_back, tab_score, tab_rules, tab_help = st.tabs(["予想する", "過去のレースで確かめる", "モデルの成績", "予想の前提", "使い方"])

def compute_prediction(race, card):
    """出馬表（日本語の見出しの表）を予想して、結果を session_state に保存する（選択を変えても、結果が消えないように）"""
    card = card.copy()
    card = card[card["馬名"].astype(str).str.strip() != ""].reset_index(drop=True)
    last, problems, filled = get_last(), [], []
    n = len(card)
    for i, r in card.iterrows():
        df_ = P.horse_defaults(last, r["馬名"], race["date"])
        for col, key in [("性別", "sex"), ("年齢", "age"), ("騎手", "jockey"), ("調教師", "trainer"), ("馬体重", "body_weight"), ("斤量", "carried_weight")]:
            if pd.isna(r[col]) or str(r[col]).strip() == "":
                if df_ and df_.get(key) is not None:
                    card.at[i, col] = df_[key]
                    if col == "馬体重":
                        card.at[i, "増減"] = 0
                    filled.append(f"{r['馬名']}の{col}")
                elif col in ("性別", "年齢", "騎手", "調教師", "斤量"):
                    problems.append(f"{r['馬名']}：{col}が空欄です（履歴にない馬は、入力してください）")
        if pd.isna(r["枠"]) and pd.notna(r["馬番"]) and 1 <= int(r["馬番"]) <= n:
            card.at[i, "枠"] = P.assign_waku(n)[int(r["馬番"]) - 1]
    if n < 4:
        problems.append("4頭以上必要です")
    if card["馬番"].isna().any() or card["馬番"].duplicated().any():
        problems.append("馬番が空欄、または重複しています")
    if problems:
        st.error("入力を確認してください：\n\n" + "\n\n".join(f"- {p}" for p in problems))
        return
    with st.spinner("特徴量を作って予想しています…（10〜20秒）"):
        res = P.predict_card(get_history(), race, card, models, prepared=get_prepared())
    st.session_state["pred"] = dict(res=res, filled=filled, race=race)


def render_prediction(pred):
    """保存した予想の結果を表示する（① 能力勝率 ② オッズを使った期待値 ③ 買い目）"""
    res, filled, race = pred["res"], pred["filled"], pred["race"]
    st.markdown(f"**予想したレース：{race['date']} {race['venue']}{race['race_no']}R**（{race['race_class']}・{race['surface']}{race['distance']}m）")
    if filled:
        st.caption("履歴から補った項目：" + "、".join(filled[:12]) + ("…" if len(filled) > 12 else ""))
    use_odds = bool(res["use_odds"].iloc[0])
    style = lambda e: "不明" if pd.isna(e) else "逃げ・先行" if e < 0.25 else "先行" if e < 0.4 else "中団" if e < 0.6 else "差し・追込"

    # ① 能力勝率: オッズ・他人の評価を使わず、先に決める（前提15）
    st.markdown("#### ① 能力勝率（オッズを見る前に決めた値）")
    a1 = res.sort_values("p_win", ascending=False)
    t1 = pd.DataFrame({
        "順位": range(1, len(a1) + 1), "馬番": a1["umaban"].astype(int).to_numpy(), "馬名": a1["horse_name"].to_numpy(),
        "能力勝率": a1["p_win"].to_numpy() * 100, "3着内（オッズなし）": a1["p_ability"].to_numpy() * 100,
        "脚質の目安": a1["mean_early_pos"].map(style).to_numpy(), "過去の出走": a1["horse_runs"].astype(int).to_numpy()})
    st.dataframe(t1, hide_index=True, use_container_width=True, column_config={
        "能力勝率": st.column_config.ProgressColumn(min_value=0.0, max_value=60.0, format="%.1f%%"),
        "3着内（オッズなし）": st.column_config.ProgressColumn(min_value=0.0, max_value=100.0, format="%.0f%%")})
    st.caption("オッズ・人気・過去のオッズ由来の特徴量は使っていません。この値は、オッズを入れる前後で変わりません（固定されています）。")

    # ② オッズで期待値（能力勝率を固定したあとに、オッズを使う）
    st.markdown("#### ② オッズを使った期待値")
    if use_odds:
        a2 = res.sort_values("ev_win", ascending=False)
        t2 = pd.DataFrame({
            "馬番": a2["umaban"].astype(int).to_numpy(), "馬名": a2["horse_name"].to_numpy(), "単勝オッズ": a2["odds"].to_numpy(),
            "能力勝率": a2["p_win"].to_numpy() * 100, "市場の勝率": a2["p_mkt_win"].to_numpy() * 100,
            "単勝の期待値": a2["ev_win"].to_numpy(),
            "3着内（オッズあり）": a2["p_top3"].to_numpy() * 100, "市場の3着内": a2["p_mkt3"].to_numpy() * 100})
        st.dataframe(t2, hide_index=True, use_container_width=True, column_config={
            "能力勝率": st.column_config.NumberColumn(format="%.1f%%"), "市場の勝率": st.column_config.NumberColumn(format="%.1f%%"),
            "単勝の期待値": st.column_config.NumberColumn(format="%.2f", help="能力勝率 × 単勝オッズ。1を超えれば、平均的には買った額より多く戻る計算"),
            "3着内（オッズあり）": st.column_config.NumberColumn(format="%.0f%%"), "市場の3着内": st.column_config.NumberColumn(format="%.0f%%")})
        st.warning(
            "**この期待値は、買い目の根拠にしないでください。** 検証（2026年3〜9月、約2.7万頭）では、能力勝率 × オッズ が1を超えた馬は全体の53%"
            "（1レースあたり約7頭）もいて、その馬の実際の単勝回収率は70%でした。期待値の高さと回収率には関係が見られず（期待値1.2〜1.5の馬は56%、"
            "2.0以上の馬は78%）、オッズを見ずに決めた勝率は、人気薄を市場より高く見積もるため、期待値が大きく出すぎます。"
            "市場との比較は、右の「3着内（オッズあり）」と「市場の3着内」を見てください。", icon="⚠️")
    else:
        st.info("単勝オッズを全頭に入れると、能力勝率にオッズをかけた期待値と、市場との比較が出ます。")

    # なぜこの勝率になったか（単勝の勝率。能力、オッズあり）
    st.markdown("#### なぜこの勝率になったか（単勝の勝率）")
    mixable = bool(use_odds and res["p_win_mix"].notna().all())
    order = res.sort_values("p_win_mix" if mixable else "p_win", ascending=False)
    names_by_no = dict(zip(res["umaban"].astype(int), res["horse_name"]))
    pick = st.pills("馬番を選ぶ（人気の順に並べています）", [int(u) for u in order["umaban"]], selection_mode="single", default=int(order.iloc[0]["umaban"]),
                    key=f"why_{race['date']}_{race['venue']}_{race['race_no']}")
    pick = int(order.iloc[0]["umaban"]) if pick is None else pick
    why, info = P.explain_win(res, models, names_by_no[pick])
    row = res[res["umaban"].astype(int) == pick].iloc[0]
    st.markdown(f"**{pick}番 {names_by_no[pick]}**")
    if bool(res[res["umaban"].astype(int) == pick].iloc[0].get("unbeaten", False)):
        st.info("この馬は無敗（出走した全レースで1着）です。過去の検証（2024〜2026年）では、無敗馬は、市場から期待される勝ち数の0.93倍しか勝っていません（人気の無敗馬では0.83倍）。"
                "確率の基準で「理論の補正」を選ぶと、市場込みの勝率を0.93倍して、買い目の確率に使います（弱い兆候です）。")
    c1, c2, c3 = st.columns(3)
    c1.metric("能力勝率（オッズなし）", f"{row['p_win'] * 100:.1f}%")
    if mixable:
        c2.metric("市場の勝率（オッズから）", f"{row['p_mkt_win'] * 100:.1f}%")
        c3.metric("市場込みの勝率", f"{row['p_win_mix'] * 100:.1f}%")
    st.markdown("**能力勝率（オッズなし）が、その値になった主な理由**")
    show = why.copy()
    show["値"] = show["値"].map(lambda v: f"{v:.3g}" if isinstance(v, (int, float)) else str(v))
    show["中央"] = show["中央"].map(lambda v: f"{v:.3g}" if isinstance(v, (int, float)) else str(v))
    show = show.rename(columns={"中央": "このレースの中央値"})
    st.dataframe(show, hide_index=True, use_container_width=True, column_config={
        "勝率への影響（ポイント）": st.column_config.NumberColumn(format="%+.1f", help="その特徴量を、このレースの中央値に置き換えたとき、勝率が下がる（プラス）／上がる（マイナス）幅")})
    st.caption("特徴量を1つずつ、このレースの中央値に置き換えて、この馬の勝率がどれだけ変わるかを見ています。プラスは、その特徴量が勝率を押し上げているという意味です。"
               "特徴量どうしの重なりは分けられない目安です。「メンバー内の相対」は、同じレースの他の馬と比べた値です。")
    if info["mix"]:
        m_ = info["mix"]
        st.markdown("**市場込みの勝率（オッズあり）が、その値になった理由**")
        d1, d2 = (m_["p_a_only"] - m_["mkt"]) * 100, (m_["p_win_mix"] - m_["p_a_only"]) * 100
        st.markdown(
            f"- 出発点：市場の勝率（単勝オッズから）**{m_['mkt'] * 100:.1f}%**\n"
            f"- ① 補正：式の係数 {m_['a']:.2f}（1より大きいので、人気馬の勝率を少し上げ、人気薄を下げる）で、**{m_['p_a_only'] * 100:.1f}%**（{d1:+.1f}ポイント）\n"
            f"- ② 能力の項：能力モデルの3着内確率 {m_['ability3'] * 100:.0f}% を、係数 ({m_['b']:.2f}) で足して、**{m_['p_win_mix'] * 100:.1f}%**（{d2:+.1f}ポイント）\n"
            f"- 式：`{m_['a']:.2f} × log(市場の勝率) + ({m_['b']:.2f}) × logit(能力の3着内確率)`（市場の項 {m_['mkt_term']:+.2f}、能力の項 {m_['ability_term']:+.2f}）を、レース内で合計1にそろえます。")
        st.markdown("**② の能力の項に効いた特徴量**（市場込みの勝率が、特徴量でどれだけ動くか）")
        t3 = m_["table"].copy()
        t3["値"] = t3["値"].map(lambda v: f"{v:.3g}" if isinstance(v, (int, float)) else str(v))
        t3["中央"] = t3["中央"].map(lambda v: f"{v:.3g}" if isinstance(v, (int, float)) else str(v))
        st.dataframe(t3.rename(columns={"中央": "このレースの中央値"}), hide_index=True, use_container_width=True,
                     column_config={"市場込みの勝率への影響（ポイント）": st.column_config.NumberColumn(format="%+.2f")})
        st.caption("プラスは、その特徴量が、市場込みの勝率を押し上げているという意味です。**能力の項の係数は、小さなマイナス（" + f"{m_['b']:.2f}" + "）なので、"
                   "能力が高いと見える特徴量は、むしろ、市場込みの勝率を少し下げる向きに出ます**（過去の検証で、能力が高く見える馬は、市場がすでに織り込んでいて、"
                   "上乗せにならなかったため）。係数が小さいので、影響は、1ポイント未満で、市場込みの勝率は、市場の勝率にほぼ等しくなります。"
                   "これは、過去のデータに当てはめた式の係数による見かけの理由で、因果関係ではありません。")
    else:
        st.info("単勝オッズを全頭に入れると、市場込みの勝率の理由も出ます。")

    # ③ 買い目（券種別）
    st.markdown("#### ③ 買い目（券種別）")
    mix_ok = bool(use_odds and res["p_win_mix"].notna().all())
    c1, c2, c3 = st.columns([3, 3, 1])
    bases = (["市場込み（オッズも使う）", "市場込み＋理論の補正（無敗馬を割引）"] if mix_ok else []) + ["能力ベース（オッズを見ない）"]
    basis = c1.radio("確率の基準", bases, horizontal=True, key="bet_basis")
    order = c2.radio("並び順", ["的中しやすい順", "期待値の目安が高い順"], horizontal=True, key="bet_order")
    topn = c3.selectbox("表示数", [3, 5, 10], index=1, key="bet_top")
    mix, sort = basis.startswith("市場込み"), ("ev" if order.startswith("期待値") else "prob")
    p_basis = (res["p_win_theory"] if "理論" in basis else res["p_win_mix"] if mix else res["p_win"]).to_numpy()
    umaban, names = res["umaban"].astype(int).to_numpy(), res["horse_name"].tolist()
    p_market = res["p_mkt_win"].to_numpy() if use_odds else None
    pct = lambda: st.column_config.NumberColumn(format="%.1f%%")

    # 実際のオッズ（JRA公式。馬連・ワイド・馬単・3連複・3連単を含む）。ボタンを押したときだけ取り込む（約10秒、6ページ）
    rkey = (race["date"], race["venue"], race["race_no"])
    real = st.session_state.get("real_odds")
    real = real["odds"] if real and real["key"] == rkey else None
    if race.get("odds_cname"):
        b1, b2 = st.columns([2, 5])
        if b1.button("実際のオッズを取り込む" if real is None else "オッズを取り直す"):
            jra_real_odds.clear() if real is not None else None
            st.session_state["real_odds"] = dict(key=rkey, odds=jra_real_odds(race["odds_cname"]))
            st.rerun()
        if real is not None:
            b2.caption(f"実際のオッズ（JRA公式、{real['fetched_at']} 時点）で、期待値を出しています。")
        else:
            b2.caption("今は、「基準の確率 ÷ 市場の確率 × 払戻率」の目安です。実際のオッズ（馬連・3連単など）を取り込むと、確率 × オッズ の期待値になります。")
    pb3 = (res["p_top3"] if mix else res["p_ability"]).to_numpy()
    for tab, bet in zip(st.tabs(B.BETS), B.BETS):
        with tab:
            if real is not None:
                t = B.real_table(umaban, names, p_basis, pb3, real, bet, topn, sort)
                if len(t):
                    t = t.copy()
                    t["確率"] = t["確率"] * 100
                    st.dataframe(t, hide_index=True, use_container_width=True, column_config={"確率": pct()})
                    if sort == "ev":
                        st.caption(f"期待値の順に並べるときは、確率が低すぎる買い目（{B.MIN_PROB[bet]:.1%}未満）を除いています。")
                else:
                    st.info("この券種のオッズが取り込めていません（発売前、または、取消・除外の馬がいます）。")
                continue
            if bet == "単勝":
                t = B.single_table(umaban, names, p_basis, p_market, res["odds"].to_numpy() if use_odds else None, "単勝", topn, sort)
            elif bet == "複勝":
                pb = (res["p_top3"] if mix else res["p_ability"]).to_numpy()
                t = B.single_table(umaban, names, pb, p_market, res["odds"].to_numpy() if use_odds else None, "複勝", topn, sort,
                                   pay_hat=res["pay_hat"].to_numpy(), p_market_top3=res["p_mkt3"].to_numpy())
            else:
                t = B.combo_table(umaban, names, p_basis, p_market, bet, topn, sort)
            t = t.copy()
            for c in ("確率", "市場の確率"):
                if c in t:
                    t[c] = t[c] * 100
            st.dataframe(t, hide_index=True, use_container_width=True, column_config={
                "確率": pct(), "市場の確率": pct(), "期待値の目安": st.column_config.NumberColumn(format="%.2f", help="1を超えれば、平均的には買った額より多く戻る計算（目安）"),
                "単勝オッズ": st.column_config.NumberColumn(format="%.1f")})
            if sort == "ev" and use_odds:
                st.caption(f"期待値の順に並べるときは、確率が低すぎる買い目（{B.MIN_PROB[bet]:.1%}未満）を除いています。払戻率は{B.RATES[bet]:.1%}です。")
    st.warning(
        "**買い目の根拠にはしないでください。** 検証（2026年3〜9月、2,004レース）では、馬連・馬単・3連複・3連単のどれでも、「市場＋能力モデル」の確率は、"
        "市場（オッズ）だけの確率と区別がつきませんでした。単勝では、期待値が1を超えた馬の実際の回収率が約70%で、期待値の高さと回収率に関係が見られませんでした。"
        "実際のオッズを取り込む前の期待値の目安は、「基準の確率 ÷ 市場の確率 × 払戻率」で出した値で、"
        "市場の確率と同じなら、払戻率（72.5〜80%）にしかなりません。実際のオッズで出した期待値（確率 × オッズ）も、確率が市場と同じ程度の精度なので、1を超える買い目があっても、見積もりの誤差による可能性が高いです。組み合わせの確率は、1着・2着・3着の順位を、勝率から出す標準的な方法（Harville）による近似です。", icon="⚠️")

    st.bar_chart(t1.set_index("馬名")[["能力勝率"]])
    new = res[res["horse_runs"] == 0]["horse_name"].tolist()
    if new:
        st.warning("履歴に過去の出走がない馬：" + "、".join(new) + "（新馬、または名前の表記違いかもしれません。過去成績は使えず、予想の精度は下がります）")
    st.caption("能力勝率はレース内で合計100%、3着内確率は合計300%（3着以内の頭数）にそろえています。"
               "映像で確認できていない不利は、モデルは使いません（断定しない）。前提は「予想の前提」のタブに書いています。")




# ---- 予想する ---------------------------------------------------------------------------------

with tab_pred:
    st.subheader("レースを選んで予想する")
    src = st.radio("出馬表の取得元", ["今週の開催（JRA公式から取り込む）", "保存した出馬表", "過去のレース（履歴から）", "手で入力する"], horizontal=True)
    race = card = None

    if src.startswith("今週") or src.startswith("保存"):
        if src.startswith("今週"):
            try:
                meetings = jra_meetings()
            except Exception as e:  # ネットワークやJRAのページの変更
                meetings = []
                st.error(f"JRA公式から取得できませんでした：{e}")
            labels = [f"{m['date']}（{WEEKDAY[pd.Timestamp(m['date']).dayofweek]}） {m['label']}" for m in meetings]
            if meetings:
                c1, c2 = st.columns([3, 1])
                # 一覧から選ぶ形だと、スマホで下が切れるので、ボタンの並び（縦）にする
                i = c1.radio("開催", range(len(meetings)), index=next((k for k, m in enumerate(meetings) if m["date"] >= str(date.today())), len(meetings) - 1), format_func=lambda k: labels[k])
                if c2.button("最新のオッズに更新"):
                    jra_card.clear()
                try:
                    day = jra_card(meetings[i]["cname"])
                except Exception as e:
                    day = None
                    st.error(f"出馬表を取り込めませんでした：{e}")
            else:
                day = None
        else:
            files = saved_cards()
            f = st.selectbox("出馬表のファイル", files, format_func=lambda x: x.name) if files else None
            day = json.loads(f.read_text(encoding="utf-8")) if f else None
        if day:
            races = day["races"]
            # レースは、一覧から選ぶ形だと、スマホで下が切れる／一覧の中でスクロールしにくいので、番号のボタンにして、選んだレースの内容は、下に1行で出す
            j = st.pills("レース", range(len(races)), selection_mode="single", default=0, key=f"race_{day['date']}_{day.get('venue', '')}",
                         format_func=lambda k: f"{races[k]['race_no']}R")
            j = 0 if j is None else j
            st.caption(f"{races[j]['race_no']}R {races[j]['start_time']} {races[j]['name']}（{races[j]['race_class']}・{races[j]['surface']}{races[j]['distance']}m・{len(races[j]['horses'])}頭）")
            c1, c2 = st.columns(2)
            going = c1.selectbox("馬場（当日の発表に合わせて）", P.GOINGS, index=P.GOINGS.index(day.get("going", "良")))
            weather = c2.selectbox("天候", P.WEATHERS, index=P.WEATHERS.index(day.get("weather", "晴")))
            race, card = race_dict(day, races[j], going, weather), horses_to_card(races[j]["horses"])
            if card["単勝オッズ"].isna().any():
                st.info("単勝オッズが出ていない馬がいます。全頭そろうと、オッズありの予想と市場との比較が出ます。")
            if card["馬体重"].isna().all():
                st.caption("馬体重は未発表です（各馬の前走の値で補います）。")

    elif src.startswith("過去"):
        h = get_history()
        dates = sorted(h["date"].dt.strftime("%Y-%m-%d").unique(), reverse=True)
        c1, c2, c3 = st.columns(3)
        d = c1.selectbox("日付", dates, key="pick_date")
        hd = h[h["date"].dt.strftime("%Y-%m-%d") == d]
        v = c2.selectbox("競馬場", sorted(hd["venue"].unique()), key="pick_venue")
        hv = hd[hd["venue"] == v]
        rn = c3.selectbox("レース", sorted(hv["race_no"].unique()), key="pick_race", format_func=lambda x: f"{x}R")
        day, r_ = card_from_history(hv[hv["race_no"] == rn]["race_id"].iloc[0])
        race, card = race_dict(day, r_, day["going"], day["weather"]), horses_to_card(r_["horses"])
        st.info("日付より前の履歴だけで予想します（このレースの結果は使いません）。ただし、最終モデルはこのレースの結果を学習しているので、予想は実際より良く見えます。"
                "公平な確認は「過去のレースで確かめる」のタブです。")

    else:
        st.caption("レースの条件と出馬表を入れます。履歴にある馬は、馬名だけで、性別・年齢・調教師・騎手・馬体重・斤量を補います。")
        c = st.columns(6)
        d_ = c[0].date_input("日付", value=date.today())
        v_ = c[1].selectbox("競馬場", P.VENUES, index=4)
        rn_ = c[2].number_input("R", 1, 12, 11)
        stt = c[3].text_input("発走時刻", "15:30")
        sf = c[4].selectbox("コース", P.SURFACES)
        dist = c[5].number_input("距離（m）", 1000, 3600, 1600, step=100)
        c = st.columns(6)
        turn = c[0].selectbox("回り", ["右", "左", "直線"])
        inout = c[1].selectbox("内回り・外回り", ["内", "外", "外内"])
        going = c[2].selectbox("馬場", P.GOINGS)
        weather = c[3].selectbox("天候", P.WEATHERS)
        cls_ = c[4].selectbox("クラス", P.CLASSES, index=4)
        hc = c[5].checkbox("ハンデ戦")
        empty = pd.DataFrame({"馬名": [""] * 8, "枠": [None] * 8, "馬番": list(range(1, 9)), "性別": [None] * 8, "年齢": [None] * 8, "斤量": [None] * 8,
                              "騎手": [""] * 8, "調教師": [""] * 8, "馬体重": [None] * 8, "増減": [None] * 8, "単勝オッズ": [None] * 8, "ブリンカー": [False] * 8, "減量": [0.0] * 8})
        up = st.file_uploader("CSVから読み込む（見出しは下の表と同じ）", type="csv")
        card = st.data_editor(pd.read_csv(up) if up is not None else empty, num_rows="dynamic", use_container_width=True, hide_index=True, key="editor",
                              column_config={"性別": st.column_config.SelectboxColumn(options=["牡", "牝", "セ"]), "ブリンカー": st.column_config.CheckboxColumn()})
        st.download_button("CSVの見本をダウンロード", empty.head(3).to_csv(index=False).encode("utf-8-sig"), "card_template.csv", "text/csv")
        race = dict(date=d_.strftime("%Y-%m-%d"), venue=v_, race_no=rn_, start_time=stt, surface=sf, distance=dist, course=P.course_string(sf, turn, inout),
                    going=going, weather=weather, race_class=cls_, handicap=hc)

    if card is not None:
        st.markdown(f"**{race['date']} {race['venue']}{race['race_no']}R**　{race['race_class']}・{race['surface']}{race['distance']}m・{race['course']}・{len(card)}頭")
        if src.startswith("今週") or src.startswith("保存") or src.startswith("過去"):
            show = card[["馬番", "枠", "馬名", "性別", "年齢", "斤量", "騎手", "調教師", "単勝オッズ"]]
            st.dataframe(show, hide_index=True, use_container_width=True, height=min(38 * (len(show) + 1), 420))
        if st.button("予想する", type="primary"):
            compute_prediction(race, card)
    if st.session_state.get("pred"):
        render_prediction(st.session_state["pred"])

# ---- 過去のレースで確かめる -----------------------------------------------------------------------

bt = get_backtest()
with tab_back:
    st.subheader("過去のレースで確かめる")
    if bt is None:
        st.info("バックテストがありません。`uv run python scripts/train_model.py` で作れます。")
    else:
        st.caption("2026年3〜9月のレースを、**各月より前のデータだけで学習したモデル**で予測した結果です（未来を見ていない、公平な比較です）。")
        bdates = sorted(bt["date"].dt.strftime("%Y-%m-%d").unique(), reverse=True)
        c1, c2, c3 = st.columns(3)
        d = c1.selectbox("日付", bdates, key="bt_date")
        bd = bt[bt["date"].dt.strftime("%Y-%m-%d") == d]
        v = c2.selectbox("競馬場", sorted(bd["venue"].unique()), key="bt_venue")
        bv = bd[bd["venue"] == v]
        rn = c3.selectbox("レース", sorted(bv["race_no"].unique()), key="bt_race", format_func=lambda x: f"{x}R")
        g = bv[bv["race_no"] == rn].copy()
        r0 = g.iloc[0]
        st.markdown(f"**{d} {v}{rn}R** {r0['race_class']}・{r0['surface']}{int(r0['distance'])}m・{r0['going']}・{int(r0['num_runners'])}頭")
        top_m = g.nlargest(3, "p_top3")
        top_k = g.nlargest(3, "p_mkt3")
        c1, c2 = st.columns(2)
        c1.metric("モデルの上位3頭のうち、3着以内", f"{int(top_m['top3'].sum())}頭")
        c2.metric("市場（オッズ）の上位3頭のうち、3着以内", f"{int(top_k['top3'].sum())}頭")
        show = g.sort_values("finish", na_position="last")
        st.dataframe(pd.DataFrame({
            "着順": show["finish"].map(lambda x: "—" if pd.isna(x) else int(x)), "馬番": show["umaban"].astype(int), "馬名": show["horse_name"],
            "人気": show["popularity"], "単勝オッズ": show["odds"], "市場の3着内": show["p_mkt3"] * 100, "モデル（オッズあり）": show["p_top3"] * 100,
            "モデル（オッズなし）": show["p_ability"] * 100, "複勝払戻": show["place_payout"].map(lambda x: "" if pd.isna(x) or x == 0 else int(x))}),
            hide_index=True, use_container_width=True, column_config={
                c: st.column_config.ProgressColumn(min_value=0.0, max_value=100.0, format="%.0f%%") for c in ["市場の3着内", "モデル（オッズあり）", "モデル（オッズなし）"]})

# ---- モデルの成績 -----------------------------------------------------------------------------

with tab_score:
    st.subheader("モデルの成績（2026年3〜9月、各月より前のデータだけで学習）")
    if bt is None:
        st.info("バックテストがありません。")
    else:
        n = bt["race_id"].nunique()
        rows = []
        for name, col in [("市場（オッズ＋頭数）", "p_mkt3"), ("オッズなしモデル", "p_ability"), ("オッズありモデル", "p_top3")]:
            auc_race = bt.groupby("race_id").apply(lambda x, c=col: roc_auc_score(x["top3"], x[c]) if 0 < x["top3"].sum() < len(x) else np.nan,
                                                   include_groups=False).mean()
            rows.append({"モデル": name, "AUC（全体）": roc_auc_score(bt["top3"], bt[col]), "AUC（レース内）": auc_race,
                         "LogLoss": log_loss(bt["top3"], bt[col].clip(1e-6, 1 - 1e-6))})
        st.caption(f"{n:,}レース・{len(bt):,}頭")
        st.dataframe(pd.DataFrame(rows).style.format({"AUC（全体）": "{:.4f}", "AUC（レース内）": "{:.4f}", "LogLoss": "{:.4f}"}),
                     hide_index=True, use_container_width=True)
        picks = []
        for name, f in [("1番人気", lambda g: g.loc[g["odds"].idxmin()]), ("市場の予測1位", lambda g: g.loc[g["p_mkt3"].idxmax()]),
                        ("オッズなしモデルの1位", lambda g: g.loc[g["p_ability"].idxmax()]), ("オッズありモデルの1位", lambda g: g.loc[g["p_top3"].idxmax()])]:
            p = pd.DataFrame([f(g) for _, g in bt.groupby("race_id")])
            pp = p["place_payout"].fillna(0)
            ok = p["place_payout"].notna()
            picks.append({"予想": name, "勝率": (p["finish"] == 1).mean(), "複勝率": (p["finish"] <= 3).mean(),
                          "単勝回収率": (p["win"] * p["odds"]).mean(), "複勝回収率": pp[ok].mean() / 100})
        st.markdown("**各レースの1位の馬を買った場合**（100円ずつ）")
        st.dataframe(pd.DataFrame(picks).style.format({c: "{:.1%}" for c in ["勝率", "複勝率", "単勝回収率", "複勝回収率"]}),
                     hide_index=True, use_container_width=True)
        st.info("オッズありモデルは、オッズだけのモデルと区別できない精度です。回収率はどれも100%を下回り、このモデルで馬券の利益は出ません。"
                "詳しくは README の「結論と限界」を見てください。")

# ---- 予想の前提 -------------------------------------------------------------------------------

with tab_rules:
    st.markdown((ROOT / "docs" / "principles.md").read_text(encoding="utf-8").replace("# 馬選びの前提（予想で守ること）\n", "", 1))

# ---- 使い方 ---------------------------------------------------------------------------------

with tab_help:
    st.markdown("""
### 使い方
1. **予想する**で、日付・競馬場・レースの条件を入れ、出馬表の表に馬を入れて「予想する」を押します。
2. 履歴にある馬は、馬名だけ入れれば、性別・年齢・調教師・騎手・馬体重・斤量を自動で補います（変えたい項目だけ入れてください）。
3. 履歴にない馬（新馬など）は、性別・年齢・斤量・騎手・調教師も入れてください。
4. **単勝オッズ**を全頭に入れると、市場との比較と、オッズありの予想が出ます。入れなければ、オッズなしの予想だけです。
5. 入力の練習には、「過去のレースの出馬表を読み込む」が使えます。

### 出馬表はどこで調べる？
JRA公式サイトの「出馬表」（レースの前日〜当日）から、馬名・枠・馬番・性齢・斤量・騎手・調教師・馬体重を写せます。自動で取り込む機能は、まだありません。

### 予想の読み方
- **能力勝率**：オッズ・人気・過去のオッズ由来の特徴量を使わずに、先に決める勝率です（前提15）。
- **単勝の期待値**：能力勝率 × 単勝オッズです。能力勝率を固定したあとに、オッズを使って出します。
- **3着内（オッズあり）**：市場の勝率と、オッズなしの予想を組み合わせた確率です。
- **3着内（オッズなし）**：過去成績・騎手・調教師・脚質・馬場などから出した確率で、オッズは使いません。
- **モデル − 市場**：プラスなら、モデルが市場より高く評価した馬です。ただし検証では、市場より高く評価した馬が実際に好成績になる証拠は、はっきり見つかっていません。
- 履歴は **{}** までです。履歴より後に出走したレースの結果は、使われません。
""".format(str(get_history()["date"].max().date())))
