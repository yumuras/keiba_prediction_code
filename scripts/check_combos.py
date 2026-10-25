"""組み合わせ（馬連・馬単・3連複・3連単）の確率の検証

バックテスト（models/backtest.pkl。python scripts/train_model.py で作る）の各レースについて、実際の1〜3着の組み合わせに、
3種類の勝率（市場そのまま／市場（補正あり）／市場＋能力モデル（Benter型））から Harville の方法で出した確率が、
どれだけの値を付けていたか（確率の対数の平均。0に近いほど良い）を比べる。差は、レースを復元抽出した95%区間つき。

使い方:  python scripts/check_combos.py
"""
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
bt = pd.read_pickle(ROOT / "models" / "backtest.pkl", compression="gzip")
print("レース", bt.race_id.nunique(), "頭", len(bt))
rng = np.random.default_rng(0)
cols = {"市場(オッズそのまま)": "p_win_raw", "市場(補正あり)": "p_win_mkt", "市場+能力モデル(Benter)": "p_win_model"}
rows = {k: {"馬連": [], "馬単": [], "3連複": [], "3連単": []} for k in cols}
n_ok = 0
for rid, g in bt.groupby("race_id", sort=False):
    f = g.dropna(subset=["finish"]).sort_values("finish")
    if len(f) < 3 or not (f["finish"].iloc[:3].tolist() == [1, 2, 3]):
        continue
    idx = [g.index.get_loc(i) for i in f.index[:3]]
    n_ok += 1
    for name, c in cols.items():
        p = g[c].to_numpy(); p = p / p.sum()
        a, b, d = p[idx[0]], p[idx[1]], p[idx[2]]
        ex = a * b / (1 - a)                       # 馬単
        qu = ex + b * a / (1 - b)                  # 馬連
        tri = a * b / (1 - a) * d / (1 - a - b)    # 3連単
        trio = sum(x * y / (1 - x) * z / (1 - x - y) for x, y, z in itertools.permutations([a, b, d]))  # 3連複
        r = rows[name]; r["馬連"].append(np.log(qu)); r["馬単"].append(np.log(ex)); r["3連複"].append(np.log(trio)); r["3連単"].append(np.log(tri))
print("評価できたレース", n_ok)
def ci(x):
    x = np.asarray(x); idx = rng.integers(0, len(x), size=(2000, len(x))); m = x[idx].mean(axis=1); return x.mean(), np.percentile(m, [2.5, 97.5])
base = "市場(補正あり)"
print(f"{'券種':6s} {'市場(そのまま)':>12s} {'市場(補正)':>12s} {'Benter':>10s} | Benter−市場(補正) 95%区間")
for t in ["馬連", "馬単", "3連複", "3連単"]:
    v = {k: np.array(rows[k][t]) for k in cols}
    m, (lo, hi) = ci(v["市場+能力モデル(Benter)"] - v[base])
    print(f"{t:6s} {v['市場(オッズそのまま)'].mean():12.4f} {v[base].mean():12.4f} {v['市場+能力モデル(Benter)'].mean():10.4f} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
