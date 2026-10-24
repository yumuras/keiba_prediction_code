"""成績CSVから、馬・レース・騎手・調教師を見られる静的サイトを作る

使い方:
    python scripts/build_site.py              # site/ に出力
    python -m http.server -d site 8000        # http://localhost:8000 で見る（site/index.html を直接開いても動く）

サーバーは要らない静的なHTMLで、検索はブラウザの中で動く（assets/index.js に検索用の索引を入れる）。
生成物（site/）はGit管理外。JRAの成績から作ったデータなので、公開する場合は利用条件を確認すること。
"""
import argparse
import html
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

ROOT = Path(__file__).resolve().parents[1]
DATA = sorted((ROOT / "data").glob("all_*.csv"))
WEATHER = sorted((ROOT / "data").glob("weather_*.csv"))
TEMPLATES = Path(__file__).resolve().parent / "site_templates"
ASSETS = Path(__file__).resolve().parent / "site_assets"

WEEKDAY = "月火水木金土日"
GOING_WET = {"稍重", "重", "不良"}


# ---- データの準備 ---------------------------------------------------------------------

def load():
    df = pd.concat([pd.read_csv(p, dtype={"race_id": str}, parse_dates=["date"]).assign(year=int(p.stem[-4:])) for p in DATA],
                   ignore_index=True)
    df["key"] = df["year"].astype(str) + "_" + df["race_id"]  # race_id は年をまたぐと重複するので、年を付ける
    df = df.sort_values(["date", "venue", "race_no", "umaban"]).reset_index(drop=True)
    df["date_s"] = df["date"].dt.strftime("%Y-%m-%d")

    # 気象（発走時刻の直前の正時）
    if WEATHER:
        w = pd.concat([pd.read_csv(p, parse_dates=["date"]) for p in WEATHER])
        df["hour"] = df["start_time"].str[:2].astype(int)
        df = df.merge(w[["venue", "date", "hour", "temp", "humidity"]], on=["venue", "date", "hour"], how="left")

    # 位置取り（0 = 先頭、1 = 最後方）
    first = df[["corner1", "corner2", "corner3", "corner4"]].bfill(axis=1).iloc[:, 0]
    df["early_pos"] = (first - 1) / (df["num_runners"] - 1)
    df["last_pos"] = (df["corner4"] - 1) / (df["num_runners"] - 1)
    df["is_front"] = (first == 1).astype(float).where(first.notna())
    df["finish_rate"] = df["finish"].fillna(df["num_runners"]) / df["num_runners"]
    df["course_label"] = df["surface"].fillna("") + df["distance"].astype(str) + "m"
    df["race_label"] = df["venue"] + df["race_no"].astype(str) + "R"
    return df


def ids(series):
    """名前 → ページ番号（名前の五十音順。ビルドごとに変わってよい）"""
    names = sorted(series.dropna().unique())
    return {n: i for i, n in enumerate(names)}


def f1(x, nd=1):
    return "" if pd.isna(x) else f"{x:.{nd}f}"


def pct(x):
    return "" if pd.isna(x) else f"{x:.1%}"


def intn(x):
    return "" if pd.isna(x) else str(int(x))


def corner_str(r):
    v = [r.corner1, r.corner2, r.corner3, r.corner4]
    return "-".join(str(int(c)) for c in v if pd.notna(c))


def weight_str(r):
    w = f"{r.carried_weight:g}"
    return (r.jockey_mark if isinstance(r.jockey_mark, str) else "") + w


def bw_str(r):
    if pd.isna(r.body_weight):
        return ""
    d = r.body_weight_diff
    return f"{int(r.body_weight)}" + ("" if pd.isna(d) else f"({'±0' if d == 0 else f'{int(d):+d}'})")


def date_label(d):
    return f"{d.year}年{d.month}月{d.day}日（{WEEKDAY[d.dayofweek]}）"


def stats(g):
    """出走した馬のまとまり（1頭の全レース、1人の騎手の全騎乗など）の成績"""
    runs = len(g)
    fin = g["finish"]
    w, p2, p3 = (fin == 1).sum(), (fin == 2).sum(), (fin == 3).sum()
    ok = g["win_payout"].notna()
    return dict(
        runs=runs, w=int(w), p2=int(p2), p3=int(p3), out=int(runs - w - p2 - p3), top3=int(w + p2 + p3),
        win_rate=pct(w / runs) if runs else "", place_rate=pct((w + p2 + p3) / runs) if runs else "",
        win_roi=pct(g.loc[ok, "win_payout"].sum() / (100 * ok.sum())) if ok.any() else "",
        place_roi=pct(g.loc[ok, "place_payout"].sum() / (100 * ok.sum())) if ok.any() else "",
    )


def breakdown(g, col, order=None, label=None):
    rows = []
    for k, sub in g.groupby(col, observed=True):
        s = stats(sub)
        rows.append(dict(label=label(k) if label else k, **s))
    if order:
        rows.sort(key=lambda r: order.index(r["label"]) if r["label"] in order else 99)
    else:
        rows.sort(key=lambda r: -r["runs"])
    return rows


# ---- ページの生成 ---------------------------------------------------------------------

class Site:
    def __init__(self, out):
        self.out = Path(out)
        self.env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
        self.n_pages = 0

    def write(self, rel, template, root="", **ctx):
        path = self.out / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.env.get_template(template).render(root=root, period=self.period, **ctx), encoding="utf-8")
        self.n_pages += 1


def build(out):
    t0 = time.time()
    df = load()
    site = Site(out)
    site.period = f"{df['date'].min():%Y年%-m月}〜{df['date'].max():%Y年%-m月}"
    if site.out.exists():
        shutil.rmtree(site.out)
    (site.out / "assets").mkdir(parents=True)
    for f in ASSETS.iterdir():
        shutil.copy(f, site.out / "assets" / f.name)

    horse_id, jockey_id, trainer_id = ids(df["horse_name"]), ids(df["jockey"]), ids(df["trainer"])
    url_h = lambda n: f"h/{horse_id[n]}.html"
    url_j = lambda n: f"j/{jockey_id[n]}.html"
    url_t = lambda n: f"t/{trainer_id[n]}.html"
    df["horse_url"] = df["horse_name"].map(url_h)
    df["jockey_url"] = df["jockey"].map(url_j)
    df["trainer_url"] = df["trainer"].map(url_t)
    print(f"読み込み {len(df)}行 {time.time() - t0:.0f}s", flush=True)

    # ---- レース ----
    races = {k: g for k, g in df.groupby("key", sort=False)}
    day_venue_races = {}
    for k, g in races.items():
        r0 = g.iloc[0]
        day_venue_races.setdefault((r0.date_s, r0.venue), []).append((r0.race_no, k))
    for lst in day_venue_races.values():
        lst.sort()

    def race_title(r0):
        return f"{r0.date_s} {r0.venue}{r0.race_no}R {r0.race_class}"

    for k, g in races.items():
        r0 = g.iloc[0]
        g = g.assign(_o=g["finish"].fillna(99)).sort_values(["_o", "umaban"])
        rows = [dict(
            finish=int(r.finish) if pd.notna(r.finish) else None, waku=int(r.waku), umaban=int(r.umaban), horse=r.horse_name,
            horse_url=r.horse_url, sexage=f"{r.sex}{int(r.age)}", weight=weight_str(r), jockey=r.jockey, jockey_url=r.jockey_url,
            trainer=r.trainer, trainer_url=r.trainer_url, bw=bw_str(r), time=r.time if isinstance(r.time, str) else "",
            ninki=intn(r.popularity), odds=f1(r.odds), corners=corner_str(r),
            place_pay=intn(r.place_payout) if pd.notna(r.place_payout) and r.place_payout > 0 else "",
        ) for r in g.itertuples()]
        siblings = day_venue_races[(r0.date_s, r0.venue)]
        i = [kk for _, kk in siblings].index(k)
        nav = lambda kk: dict(key=kk, label=f"{races[kk].iloc[0].race_no}R") if kk else None
        chips = [f"{r0.kai}回{r0.nichi}日", r0.race_class, f"{r0.course}・{r0.distance}m" if isinstance(r0.course, str) else f"{r0.distance}m",
                 f"馬場 {r0.going}", f"天候 {r0.weather}", f"{int(r0.num_runners)}頭立て", f"発走 {r0.start_time}"]
        if pd.notna(getattr(r0, "temp", np.nan)):
            chips.append(f"気温 {r0.temp:.1f}℃・湿度 {int(r0.humidity) if pd.notna(r0.humidity) else '—'}%")
        if r0.handicap:
            chips.append("ハンデ戦")
        win = g[g["win_payout"] > 0]
        place = g[g["place_payout"] > 0].sort_values("finish")
        corners = [c for c in ("corner1", "corner2", "corner3", "corner4") if g[c].notna().any()]
        site.write(
            f"r/{k}.html", "race.html", root="../", title=race_title(r0), date=r0.date_s, date_label=date_label(r0.date),
            prev=nav(siblings[i - 1][1] if i > 0 else None), next=nav(siblings[i + 1][1] if i + 1 < len(siblings) else None),
            chips=chips, rows=rows,
            win_pay=intn(win["win_payout"].iloc[0]) if len(win) else "", win_horse=win["horse_name"].iloc[0] if len(win) else "",
            place_pays=[dict(horse=r.horse_name, pay=int(r.place_payout)) for r in place.itertuples()],
            corner_note="通過は、各コーナーを何番目に通過したか（最終コーナーまで）です。" if corners else "",
        )
    print(f"レース {len(races)} {time.time() - t0:.0f}s", flush=True)

    # ---- 開催日 ----
    dates = sorted(df["date_s"].unique())
    by_date = {d: g for d, g in df.groupby("date_s")}
    date_meta = {}
    for d, g in by_date.items():
        venues = sorted(g["venue"].unique())
        date_meta[d] = dict(date=d, label=date_label(g["date"].iloc[0]), venues="・".join(venues), races=g["key"].nunique())
    for j, d in enumerate(dates):
        g = by_date[d]
        blocks = []
        for v in sorted(g["venue"].unique()):
            gv = g[g["venue"] == v]
            rs = []
            for rn, kk in day_venue_races[(d, v)]:
                gg = races[kk]
                r0 = gg.iloc[0]
                top = gg.dropna(subset=["finish"]).sort_values("finish")
                nm = lambda i, col: top[col].iloc[i] if len(top) > i else ""
                rs.append(dict(
                    key=kk, no=rn, start=r0.start_time, cls=r0.race_class, course=f"{r0.course_label}", going=r0.going, n=int(r0.num_runners),
                    w1=nm(0, "horse_name"), w1_url=nm(0, "horse_url"), w1_jockey=nm(0, "jockey"),
                    win_pay=intn(gg["win_payout"].max()) if gg["win_payout"].notna().any() else "",
                    w2=nm(1, "horse_name"), w2_url=nm(1, "horse_url"), w3=nm(2, "horse_name"), w3_url=nm(2, "horse_url"),
                ))
            blocks.append(dict(venue=v, kai=f"{gv['kai'].iloc[0]}回{gv['nichi'].iloc[0]}日", races=rs))
        site.write(f"d/{d}.html", "date.html", root="../", label=date_meta[d]["label"], venues=date_meta[d]["venues"],
                   venue_blocks=blocks, prev=date_meta[dates[j - 1]] if j else None,
                   next=date_meta[dates[j + 1]] if j + 1 < len(dates) else None)

    months = {}
    for d in dates:
        months.setdefault(d[:7], []).append(date_meta[d])
    month_list = [dict(label=f"{m[:4]}年{int(m[5:])}月", days=v) for m, v in sorted(months.items(), reverse=True)]
    for m in month_list:
        m["days"] = list(reversed(m["days"]))
    site.write("calendar.html", "calendar.html", months=month_list)
    print(f"開催日 {len(dates)} {time.time() - t0:.0f}s", flush=True)

    # ---- 馬 ----
    surface_order = ["芝", "ダート", "障害"]
    df["going_group"] = np.where(df["going"].isin(GOING_WET), "道悪（稍重・重・不良）", "良")
    for name, g in df.groupby("horse_name"):
        g = g.sort_values(["date", "race_no"])
        last = g.iloc[-1]
        chips = [f"{last.sex}", f"{int(last.age)}歳（{last.date.year}年時点）", f"毛色 {last.color}"]
        if isinstance(last.region, str) and last.region:
            chips.append(f"産地 {last.region}")
        chips.append(f"調教師 {last.trainer}")
        st = None
        if g["early_pos"].notna().any():
            e = g["early_pos"].mean()
            label = "逃げ・先行" if (e < 0.25 or g["is_front"].mean() >= 0.4) else "先行" if e < 0.4 else "中団" if e < 0.6 else "差し・追込"
            st = dict(label=label, early=f1(e, 2), front_rate=pct(g["is_front"].mean()), last=f1(g["last_pos"].mean(), 2))
        spark = None
        if len(g) >= 2:
            w, h, pad = 600, 90, 10
            xs = np.linspace(pad, w - pad, len(g))
            ys = pad + g["finish_rate"].to_numpy() * (h - 2 * pad)
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
            dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{4 if f == 1 else 2.5}" fill="{"#d99a00" if f == 1 else "var(--accent)"}"/>'
                           for x, y, f in zip(xs, ys, g["finish"]))
            spark = f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none"><polyline points="{pts}" fill="none" stroke="var(--accent)" stroke-width="1.5" opacity=".6"/>{dots}</svg>'
        hist = [dict(
            date=r.date_s, key=r.key, race=f"{r.race_label}", cls=r.race_class, course=r.course_label, going=r.going, n=int(r.num_runners),
            waku=int(r.waku), umaban=int(r.umaban), finish=int(r.finish) if pd.notna(r.finish) else None, ninki=intn(r.popularity),
            odds=f1(r.odds), jockey=r.jockey, jockey_url=r.jockey_url, weight=weight_str(r), bw=bw_str(r),
            time=r.time if isinstance(r.time, str) else "", corners=corner_str(r),
        ) for r in g.sort_values(["date", "race_no"], ascending=False).itertuples()]
        site.write(f"h/{horse_id[name]}.html", "horse.html", root="../", name=name, chips=chips, stats=stats(g), style=st, spark=spark,
                   by_surface=breakdown(g, "surface", surface_order), by_going=breakdown(g, "going_group"), history=hist)
    print(f"馬 {len(horse_id)} {time.time() - t0:.0f}s", flush=True)

    # ---- 騎手・調教師 ----
    def person_pages(col, idmap, url, prefix, role, partner_col_name, partner_title, partner_label, partner_url, other_col, other_key, other_title):
        for name, g in df.groupby(col):
            g = g.sort_values(["date", "race_no"])
            partners = []
            for pn, sub in g.groupby(partner_col_name):
                s = stats(sub)
                partners.append(dict(label=pn, url=partner_url(pn), **s))
            partners = sorted(partners, key=lambda r: -r["runs"])[:25]
            recent = [dict(
                date=r.date_s, key=r.key, race=r.race_label, horse=r.horse_name, horse_url=r.horse_url, finish=int(r.finish) if pd.notna(r.finish) else None,
                ninki=intn(r.popularity), odds=f1(r.odds), other=getattr(r, other_key),
            ) for r in g.iloc[::-1].head(40).itertuples()]
            site.write(f"{prefix}/{idmap[name]}.html", "person.html", root="../", name=name, role=role, stats=stats(g),
                       by_year=breakdown(g, "year", label=lambda y: f"{y}年"), by_surface=breakdown(g, "surface", surface_order),
                       by_venue=breakdown(g, "venue"), partners=partners, partner_title=partner_title, partner_col=partner_label,
                       recent=recent, recent_word=other_title, other_col=other_col)

    person_pages("jockey", jockey_id, url_j, "j", "騎手", "horse_name", "よく乗った馬", "馬", url_h, "調教師", "trainer", "騎乗")
    person_pages("trainer", trainer_id, url_t, "t", "調教師", "horse_name", "管理した馬", "馬", url_h, "騎手", "jockey", "出走")
    print(f"騎手・調教師 {len(jockey_id)}+{len(trainer_id)} {time.time() - t0:.0f}s", flush=True)

    # ---- ランキング ----
    def rank_rows(col, url, min_runs=1, key="w", n=30, limit=None):
        rows = []
        for name, g in df.groupby(col):
            s = stats(g)
            if s["runs"] >= min_runs:
                rows.append(dict(label=name, url=url(name), wr=s["w"] / s["runs"], pr=s["top3"] / s["runs"], **s))
        rows.sort(key=lambda r: (-r["w"], -r["runs"]) if key == "w" else (-r["wr"], -r["runs"]) if key == "wr" else (-r["pr"], -r["runs"]))
        return rows[:n]

    blocks = [
        dict(title="馬：勝利数", col="馬", rows=rank_rows("horse_name", url_h)),
        dict(title="騎手：勝利数", col="騎手", rows=rank_rows("jockey", url_j)),
        dict(title="騎手：勝率（200騎乗以上）", col="騎手", rows=rank_rows("jockey", url_j, 200, "wr")),
        dict(title="騎手：複勝率（200騎乗以上）", col="騎手", rows=rank_rows("jockey", url_j, 200, "pr")),
        dict(title="調教師：勝利数", col="調教師", rows=rank_rows("trainer", url_t)),
        dict(title="調教師：勝率（200出走以上）", col="調教師", rows=rank_rows("trainer", url_t, 200, "wr")),
    ]
    site.write("rankings.html", "rankings.html", blocks=blocks)

    # ---- コース別の傾向 ----
    courses = []
    d2 = df[df["surface"] != "障害"]
    for (venue, course, dist), g in d2.groupby(["venue", "course", "distance"]):
        nr = g["key"].nunique()
        if nr < 5:
            continue
        fav = g[g["popularity"] == 1]
        win = g[g["finish"] == 1]
        rate = lambda m: (g.loc[m, "finish"] <= 3).mean() if m.any() else np.nan
        courses.append(dict(
            venue=venue, course=course if isinstance(course, str) else "", distance=int(dist), races=nr, n=f1(g.groupby("key").size().mean()),
            fav_win=pct((fav["finish"] == 1).mean()), fav_place=pct((fav["finish"] <= 3).mean()),
            front=pct(rate(g["early_pos"] <= 1 / 3)), back=pct(rate(g["early_pos"] >= 2 / 3)),
            inner=pct(rate(g["waku"] <= 4)), outer=pct(rate(g["waku"] >= 5)),
            time=f"{int(win['time_sec'].mean() // 60)}:{win['time_sec'].mean() % 60:04.1f}", time_sec=f"{win['time_sec'].mean():.1f}",
        ))
    courses.sort(key=lambda c: (c["venue"], c["course"], c["distance"]))
    site.write("courses.html", "courses.html", courses=courses)

    # ---- トップ・検索 ----
    latest = [date_meta[d] for d in reversed(dates[-8:])]
    venues = [dict(venue=v, races=g["key"].nunique(), runs=len(g), days=g["date_s"].nunique())
              for v, g in df.groupby("venue")]
    site.write("index.html", "index.html", n_races=f"{df['key'].nunique():,}", n_runs=f"{len(df):,}", n_horses=f"{len(horse_id):,}",
               n_jockeys=len(jockey_id), n_trainers=len(trainer_id), n_days=len(dates), latest=latest, venues=venues)
    site.write("search.html", "search.html")

    index = [["h", n, url_h(n), f"{g['sex'].iloc[-1]}{int(g['age'].iloc[-1])} {len(g)}戦"] for n, g in df.groupby("horse_name")]
    index += [["j", n, url_j(n), f"{len(g)}騎乗"] for n, g in df.groupby("jockey")]
    index += [["t", n, url_t(n), f"{len(g)}出走"] for n, g in df.groupby("trainer")]
    index += [["d", f"{d} {date_meta[d]['label']}", f"d/{d}.html", date_meta[d]["venues"]] for d in dates]
    (site.out / "assets" / "index.js").write_text("window.SEARCH_INDEX=" + json.dumps(index, ensure_ascii=False, separators=(",", ":")) + ";",
                                                    encoding="utf-8")
    size = sum(f.stat().st_size for f in site.out.rglob("*") if f.is_file()) / 1e6
    print(f"完了: {site.n_pages} ページ / {size:.0f}MB / {time.time() - t0:.0f}s → {site.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=ROOT / "site")
    build(ap.parse_args().out)


if __name__ == "__main__":
    main()
