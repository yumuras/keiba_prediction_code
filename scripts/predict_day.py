"""出馬表のJSON（cards/*.json）の全レースを予想して、結果をMarkdownで保存する

使い方:
    python scripts/predict_day.py cards/2026-10-25_kyoto.json      # cards/2026-10-25_kyoto_predictions.md を作る

JSON の形は cards/2026-10-25_kyoto.json を参照（日付・競馬場・馬場・天候と、レースごとの条件・馬の一覧）。
Streamlitの画面と同じ処理（src/predict.py）で、学習済みモデル（models/models.joblib）を使う。
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")
import predict as P  # noqa: E402


def main(path):
    card = json.loads(Path(path).read_text(encoding="utf-8"))
    history = P.load_history()
    prepared = P.prepare_history(history)
    last = P.last_rows(history)
    models = joblib.load(P.MODEL_PATH)
    known = set(last.index)

    out = [f"# {card['date']} {card['venue']} の予想\n",
           f"- {card['note']}",
           f"- 学習・履歴: {models['meta']['date_min']}〜{models['meta']['date_max']}（{models['meta']['n_races']:,}レース）。**履歴にない成績（それ以降のレース）は使っていません。**",
           "- 能力勝率は、オッズ・人気・過去のオッズ由来の特徴量を使わずに決めた勝率です。オッズは、市場の確率と、市場との比較にだけ使っています。",
           "- このモデルは、公開されている成績情報の範囲では、オッズ（市場）を超えていません。**買い目の根拠にはしないでください。**\n"]
    for r in card["races"]:
        n = len(r["horses"])
        waku = P.assign_waku(n)
        rows = []
        for h in sorted(r["horses"], key=lambda x: x["umaban"]):
            d = P.horse_defaults(last, h["name"], card["date"]) or {}
            rows.append({"馬名": h["name"], "枠": waku[h["umaban"] - 1], "馬番": h["umaban"], "性別": h["sex"], "年齢": h["age"], "斤量": h["weight"],
                         "騎手": h["jockey"], "調教師": h["trainer"], "馬体重": d.get("body_weight"), "増減": 0 if d.get("body_weight") else None,
                         "単勝オッズ": h["odds"], "ブリンカー": False, "減量": h["reduction"]})
        race = dict(date=card["date"], venue=card["venue"], race_no=r["race_no"], start_time=r["start_time"], surface=r["surface"],
                    distance=r["distance"], course=P.course_string(r["surface"], r["turn"], r["inout"]), going=card["going"], weather=card["weather"],
                    race_class=r["race_class"], handicap=r["handicap"])
        res = P.predict_card(history, race, pd.DataFrame(rows), models, prepared=prepared)
        res = res.assign(odds_rank=res["odds"].rank(method="min").astype(int)).sort_values("p_top3", ascending=False)
        new = [h["name"] for h in r["horses"] if h["name"] not in known]
        out.append(f"## {r['race_no']}R {r['start_time']} {r['name']}（{r['race_class']}・{r['surface']}{r['distance']}m・{r['turn']}{'・' + r['inout'] if r['inout'] != '内' else ''}・{n}頭）\n")
        out.append("| 順位 | 馬番 | 馬名 | 騎手 | 単勝 | 人気 | 能力勝率 | 3着内（オッズあり） | 市場の3着内 | 3着内（オッズなし） | 過去の出走 |")
        out.append("|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for i, x in enumerate(res.itertuples(), 1):
            out.append(f"| {i} | {x.umaban} | {x.horse_name} | {x.jockey} | {x.odds:.1f} | {x.odds_rank} | {x.p_win:.1%} | {x.p_top3:.0%} | {x.p_mkt3:.0%} | {x.p_ability:.0%} | {int(x.horse_runs)} |")
        top = res.sort_values("p_win", ascending=False).iloc[0]
        out.append(f"\n能力勝率の1位：{top.umaban}番 {top.horse_name}（{top.p_win:.1%}）／ 3着内確率（オッズあり）の上位3頭：" +
                   "、".join(f"{x.umaban}番 {x.horse_name}" for x in res.head(3).itertuples()))
        if new:
            out.append(f"\n履歴に過去の出走がない馬：{'、'.join(new)}（新馬、または名前の表記違い。過去成績は使えない）")
        out.append("")
        print(f"{r['race_no']}R done", flush=True)
    dest = Path(path).with_name(Path(path).stem + "_predictions.md")
    dest.write_text("\n".join(out), encoding="utf-8")
    print("→", dest)


if __name__ == "__main__":
    main(sys.argv[1])
