"""変換後CSVの自己レビュー

PDFに書かれている「答え合わせ用の数字」とCSVを突き合わせる。
    python review_csv.py races.csv PDF1 [PDF2 ...]

チェック内容:
  [PDF照合]  レース数 / 各レースの頭数（「（N頭）」） / 勝ち馬オッズ×100 ≒ 単勝払戻金
  [整合性]   着順の連番 / 着順とタイムの順序 / 馬番の重複・範囲 / 枠番の範囲
  [値の範囲] 斤量・馬体重・年齢・オッズ・距離
  [文字]     外字の取りこぼし（'?'） / 必須列の欠損
終了コード: 0=問題なし, 1=要修正(ERROR) あり
"""
import re
import sys
import pdfplumber
import pandas as pd

sys.path.insert(0, __import__("os").path.dirname(__file__))
from extract_pdf import fix_text, HALVES  # noqa: E402

errors, warns = [], []


def err(m):
    errors.append(m)


def warn(m):
    warns.append(m)


def pdf_facts(path):
    """PDFから レースID → {頭数, 単勝払戻} を読む"""
    facts = {}
    head_re = re.compile(r"(\d{5}) ?\d+月\d+日 .*第\d+競走")
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            for x0, x1 in HALVES:
                x1 = min(x1, p.width)
                t = fix_text(p.crop((x0, 0, x1, p.height)).extract_text() or "")
                cur = None
                for ln in t.split("\n"):
                    m = head_re.search(ln)
                    if m:
                        cur = m.group(1)
                        facts.setdefault(cur, {})
                    if cur is None:
                        continue
                    n = re.search(r"（(\d+)頭）", ln)
                    if n and "runners" not in facts[cur]:
                        facts[cur]["runners"] = int(n.group(1))
                    w = re.search(r"単\s*勝\s+([\d，,]+)円", ln)
                    if w and "win_pay" not in facts[cur]:
                        facts[cur]["win_pay"] = int(re.sub(r"\D", "", w.group(1)))
    return facts


def main(csv_path, pdfs):
    df = pd.read_csv(csv_path, dtype={"race_id": str})
    facts = {}
    for p in pdfs:
        facts.update(pdf_facts(p))

    # ---- PDF照合 ----
    csv_ids, pdf_ids = set(df.race_id), set(facts)
    if pdf_ids - csv_ids:
        err(f"CSVに無いレース: {sorted(pdf_ids - csv_ids)}")
    if csv_ids - pdf_ids:
        warn(f"PDFで確認できないレース: {sorted(csv_ids - pdf_ids)}")
    for rid, f in sorted(facts.items()):
        r = df[df.race_id == rid]
        if r.empty:
            continue
        if "runners" in f and len(r) != f["runners"]:
            err(f"{rid}: 頭数 CSV={len(r)} / PDF={f['runners']}")
        if "win_pay" in f:
            win = r[r.finish == 1]
            if len(win) == 0:
                err(f"{rid}: 1着馬がいない")
            else:
                ok = any(abs(o * 100 - f["win_pay"]) <= 10 for o in win.odds.dropna())
                if not ok:
                    err(f"{rid}: 勝ち馬オッズ {win.odds.tolist()} が単勝払戻 {f['win_pay']}円 と合わない")

    # ---- 整合性 ----
    for rid, r in df.groupby("race_id"):
        fin = r.finish.dropna().astype(int).tolist()
        if fin != list(range(1, len(fin) + 1)):
            warn(f"{rid}: 着順が連番でない {fin}（同着ならOK）")
        t = r.dropna(subset=["finish"]).sort_values("finish").time_sec.tolist()
        if any(b < a for a, b in zip(t, t[1:])):
            err(f"{rid}: 着順が下なのにタイムが速い馬がいる")
        if r.umaban.duplicated().any():
            err(f"{rid}: 馬番が重複 {r.umaban[r.umaban.duplicated()].tolist()}")
        if (r.umaban > len(r) + 2).any() or (r.umaban < 1).any():
            warn(f"{rid}: 馬番が頭数に対して不自然（取消があればOK）")
        nstop = r.finish.isna().sum()
        if nstop:
            warn(f"{rid}: 着順なし {nstop}頭 → PDFで競走中止/除外/失格か確認: {r[r.finish.isna()].horse_name.tolist()}")

    rng = {"waku": (1, 8), "carried_weight": (48, 66), "body_weight": (300, 650),
           "age": (2, 12), "odds": (1.0, 2000), "distance": (1000, 4300)}
    for col, (lo, hi) in rng.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]
        if len(bad):
            err(f"{col} が範囲外 {lo}-{hi}: {bad[['race_id', 'horse_name', col]].values.tolist()[:5]}")

    # ---- 払戻（単勝・複勝）----
    if "place_payout" in df.columns:
        for rid, r in df.groupby("race_id"):
            if r.place_payout.isna().any() or r.win_payout.isna().any():
                warn(f"{rid}: 単勝・複勝の払戻を読めていない（1着同着などで表の文字が重なると読めない。空欄のまま）")
                continue
            n = len(r)
            expected = 3 if n >= 8 else (2 if n >= 5 else 1)  # 複勝は 8頭以上で3着まで、5〜7頭で2着まで
            paid = r[r.place_payout > 0]
            if len(paid) != expected:
                warn(f"{rid}: 複勝の払戻が{len(paid)}頭（{n}頭立てなら{expected}頭のはず。同着ならOK）")
            if (paid.place_payout < 100).any():
                err(f"{rid}: 複勝払戻が100円未満 {paid.place_payout.tolist()}")
            w = r[r.win_payout > 0]
            if len(w) and (w.place_payout > w.win_payout).any():
                warn(f"{rid}: 勝ち馬の複勝払戻が単勝払戻より高い（頭数が少ないレースなどでまれに起きる）")
            if len(w) and not any(abs(o * 100 - p_) <= 10 for o, p_ in zip(w.odds, w.win_payout)):
                err(f"{rid}: 勝ち馬のオッズと単勝払戻が合わない")

    # ---- 文字・欠損 ----
    for col in ["horse_name", "jockey", "trainer", "color", "region"]:
        bad = df[df[col].astype(str).str.contains(r"\?")]
        if len(bad):
            err(f"{col} に外字の取りこぼし '?': {sorted(set(bad[col]))} → extract_pdf.py の KNOWN_FIX に追加")
    must = ["race_id", "date", "surface", "distance", "race_class", "waku", "umaban", "horse_name",
            "sex", "age", "color", "carried_weight", "jockey", "trainer", "odds"]
    for col in must:
        n = df[col].isna().sum()
        if n:
            err(f"必須列 {col} に欠損 {n}件: {df[df[col].isna()][['race_id', 'horse_name']].values.tolist()[:5]}")
    debut = df.race_class.eq("新馬")
    n = df.loc[~debut, "body_weight_diff"].isna().sum()
    if n:
        warn(f"新馬戦以外で馬体重増減が欠損 {n}件")

    # ---- 結果 ----
    print(f"対象: {csv_path}  {len(df)}行 / {df.race_id.nunique()}レース")
    print(f"PDF照合: {len(facts)}レース（頭数・単勝払戻を確認）")
    for m in errors:
        print("ERROR ", m)
    for m in warns:
        print("WARN  ", m)
    print("結果:", "要修正" if errors else ("問題なし（WARNは目視確認）" if warns else "問題なし"))
    return 1 if errors else 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2:]))
