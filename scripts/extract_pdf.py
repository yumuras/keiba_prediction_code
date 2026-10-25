"""JRA 年度別全成績PDF → CSV 変換スクリプト

使い方:
    python extract_pdf.py 2026-1kyoto1.pdf [他のPDF ...] -o races.csv

PDFは左右2段組。x座標の列位置はページ間で共通（文字サイズのみ変化）なので、
座標で列を切り分けて1頭1行に変換する。
"""
import re
import unicodedata
import argparse
import pdfplumber
import pandas as pd

HALVES = [(0, 420), (420, 1e9)]  # 左段・右段のx範囲


def detect_offset(chars, x0):
    """枠番の数字(相対x=21)と馬名先頭(相対x=49.5)が同じ行に並ぶ位置から、段の列オフセットを推定"""
    from collections import Counter
    cnt = Counter()
    # 枠番は段の左端付近にしかない（右側の票数欄などの数字を拾うと推定がずれる）
    digs = [c for c in chars if ct(c).isdigit() and c["x0"] - x0 < 40]
    for d in digs:
        for c in chars:
            if abs(c["top"] - d["top"]) < 2 and abs(c["x0"] - d["x0"] - 28.5) < 0.4 and not ct(c).isdigit():
                cnt[round(d["x0"] - 21, 1)] += 1
                break
    return cnt.most_common(1)[0][0] if cnt else x0
MARKS = "▲△☆◇★"
KNOWN_FIX = {"?島克駿": "鮫島克駿", "?島良太": "鮫島良太", "?野泰之": "辻野泰之", "洞?湖": "洞爺湖", "?哲英": "辻哲英", "?城": "茨城", "?島一歩": "鮫島一歩", "井?一也": "井樋一也", "?島克駿ド": "鮫島克駿", "?島克也": "鮫島克也"}  # PDF内で外字になっている人名・地名
CLASSES = ["3勝クラス", "2勝クラス", "1勝クラス", "オープン", "新馬", "未勝利"]


def ct(c):
    """CID数字を変換（(cid:9872)=0 … (cid:9881)=9）。それ以外の外字は '?'"""
    m = re.match(r"\(cid:(\d+)\)", c["text"])
    if not m:
        return c["text"]
    n = int(m.group(1))
    return str(n - 9872) if 9872 <= n <= 9881 else "?"


def txt(cs):
    return "".join(ct(c) for c in sorted(cs, key=lambda c: c["x0"]))


def digits(cs):
    return "".join(ch for ch in txt(cs) if ch.isdigit())


def z2h(s):
    return s.translate(str.maketrans("０１２３４５６７８９．，：", "0123456789.,:"))


def fix_text(t):
    return re.sub(r"\(cid:(\d+)\)",
                  lambda m: str(int(m.group(1)) - 9872) if 9872 <= int(m.group(1)) <= 9881 else "", t)


def clean_race_name(rest):
    """見出し行の「第N競走」より後ろから、レース名を取り出す。

    例「 万葉ステークス 3，000」→「万葉ステークス」、「第87回菊花賞（Ｇ Ⅰ） 3，000」→「菊花賞」。
    距離・グレード・回数・（国際）などの注記を除き、全角の英数字・括弧は半角にそろえる（例「ＮＨＫマイルカップ」→「NHKマイルカップ」）。
    """
    name = re.sub(r"\s+", "", rest)
    name = re.split(r"\d[，,]\d{3}", name)[0]  # 距離（3，000 など）より前
    name = re.sub(r"[（(]((Ｊ|J)[・･]?)?(Ｇ|G)[ⅠⅡⅢ][）)]|[（(][ＬL][）)]", "", name)  # グレード
    name = re.sub(r"[（(](国際|特指|指定|混合)[）)]", "", name)
    name = unicodedata.normalize("NFKC", name)
    return re.sub(r"^第\d+回", "", name)


def parse_headers(text):
    head_re = re.compile(r"(\d{5}) ?(\d+)月(\d+)日 (晴|曇|小雨|雨|小雪|雪) ?(\S+?) ?（(\d{4})年(\d+)(\S+?)）第(\d+)日 第(\d+)競走")
    lines = text.split("\n")
    out = []
    for i, ln in enumerate(lines):
        m = head_re.search(ln)
        if not m:
            continue
        block = "\n".join(lines[i:i + 8])
        nxt = next((j for j in range(i + 1, len(lines)) if head_re.search(lines[j])), len(lines))
        seg = z2h("\n".join(lines[i:nxt]))
        after = ln[m.end():] + "\n" + "\n".join(lines[i + 1:i + 4])
        dist = re.search(r"(\d)，(\d{3})", after)
        going = m.group(5)
        jg = re.search(r"（(Ｊ|J)・(Ｇ|G)([ⅠⅡⅢ])）", ln)
        # 障害戦: 馬場に「（ダート）良」のような種別が付く／「障害」「J・G」の表記がある
        jump = going.startswith("（") or "障害" in ln or "ジャンプ" in ln or jg is not None
        if jump:
            # 障害戦は芝・ダートの馬場が別記される。芝の馬場（前の行の「（芝）良」）を優先
            tg = re.search(r"（\s*芝\s*）\s*(\S+)", lines[i - 1]) if i > 0 else None
            going = tg.group(1) if tg else re.sub(r"（[^）]*）", "", going)
            surf = re.search(r"（((芝|ダート)[^）]*)）", "\n".join(lines[i + 1:i + 3]).replace(" ", ""))
        else:
            surf = re.search(r"（((芝|ダート|障害)[^）]*)）", block.replace(" ", ""))
        blk = block.replace("除未出走馬および未勝利馬", "")
        cls = next((k for k in CLASSES if k in blk), None)
        g = re.search(r"（(Ｇ|G)([ⅠⅡⅢ])）", block)
        if g:
            cls = "G" + g.group(2)
        if jg:
            cls = "J・G" + jg.group(3)
        elif cls is None and re.search(r"（(Ｌ|L)）", ln):
            cls = "L"  # リステッド競走
        if cls is None:
            cls = "オープン"  # 特別名のみでクラス表記のないオープン競走
        am = re.search(r"([2-9])歳(以上)?", ln[m.end():] + " " + " ".join(lines[i + 1:i + 4]))
        hm = re.search(r"ハロンタイム[ \t]*([\d.―]+)", seg)
        um = re.search(r"上り\s*4F\s*([\d.]+)\s*―\s*3F\s*([\d.]+)", seg)
        out.append(dict(
            age_min=int(am.group(1)) if am else None, age_open=int(bool(am and am.group(2))) if am else None,
            laps=hm.group(1).strip().replace(" ", "") if hm else None,
            last3f=float(um.group(2)) if um else None,
            race_id=m.group(1),
            date=f"{m.group(6)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}",
            venue=m.group(8), kai=int(m.group(7)), nichi=int(m.group(9)),
            race_no=int(m.group(10)), weather=m.group(4), going=going,
            surface="障害" if jump else (surf.group(2) if surf else None),
            course=surf.group(1) if surf else None,
            distance=int(dist.group(1) + dist.group(2)) if dist else None,
            race_class=cls, race_name=clean_race_name(ln[m.end():]), handicap="ハンデキャップ" in block,
            start_time=(lambda t: f"{int(t.group(1)):02d}:{t.group(2)}" if t else None)(re.search(r"発走(\d+)時(\d+)分", block)),
        ))
    return out


def header_ys(chars, off):
    """各レースの『発走○時○分』行（左端 x≈20.5）の y座標を上から順に返す"""
    ys = []
    for c in chars:
        if c["text"] == "発" and 19.5 <= c["x0"] - off < 22:
            nxt = [d for d in chars if abs(d["top"] - c["top"]) < 1.5 and 0 < d["x0"] - c["x0"] < 9]
            if any(d["text"] == "走" for d in nxt):
                ys.append(c["top"])
    return sorted(ys)


def parse_corners(chars, off, lo, hi, raw=False):
    """『通過順位』欄を読み、コーナーごとの馬番の並び（先頭から）を、書かれている順のリストで返す

    例「1  7−6（9，8）3，2 …」→ [[7, 6, 9, 8, 3, 2, …], …]
    （ ）は併走、− は差があることを表すが、ここでは先頭からの並び順だけを使う。
    1行に左右2列で「1 …」「2 …」のように並び、長い場合は次の行に続く。
    コーナー番号は読まない。2周目は丸数字（③④など）で書かれ、その文字コードがフォントごとに
    異なって数字に戻せないため。代わりに、書かれている順番からコーナーを決める（parse_pdf 参照）。
    """
    rel = lambda c: c["x0"] - off
    labels = [c for c in chars if c["text"] == "順" and lo < c["top"] < hi and 35 < rel(c) < 42]
    labels = [c for c in labels if any(d["text"] == "位" and abs(d["top"] - c["top"]) < 1.5 for d in chars)]
    if not labels:
        return []
    y0 = labels[0]["top"]
    # 行ごとにまとめる。丸数字のコーナー番号は本文より少し上にずれるので、3pt以内は同じ行とみなす。
    # 馬番の上に付く「・」は2桁の馬番の間に入ると数字を分断するので除く
    cand = sorted((c for c in chars if y0 - 3 < c["top"] < y0 + 40 and rel(c) > 55 and c["text"] != "・"),
                  key=lambda c: c["top"])
    rows = {}
    for c in cand:
        key = next((k for k in rows if c["top"] - k <= 3), c["top"])
        rows.setdefault(key, []).append(c)
    seqs, raws, current = [], [], {}
    for y in sorted(rows):
        rc = sorted(rows[y], key=lambda c: c["x0"])
        text = "".join(ct(c) for c in rc)
        if re.search(r"[ぁ-んァ-ヴ一-龥]", text):
            break  # 次の欄（勝馬の紹介など）に入った
        if not re.search(r"\d", text):
            continue
        for col, cs in enumerate([[c for c in rc if rel(c) < 225], [c for c in rc if rel(c) >= 225]]):
            if not cs:
                continue
            # コーナー番号は、すぐ後ろの文字から離れて置かれた1文字
            if len(cs) > 1 and cs[1]["x0"] - cs[0]["x0"] > 8:
                seqs.append([])
                raws.append("")
                current[col] = len(seqs) - 1
                cs = cs[1:]
            elif col not in current:
                continue
            nums = re.sub(r"\D+", " ", "".join(ct(c) for c in cs)).split()
            raws[current[col]] += "".join(ct(c) for c in cs)
            seqs[current[col]].extend(int(n) for n in nums)  # 番号が無い行は前の行からの続き
    return raws if raw else seqs


def corner_lateral(raw):
    """通過順位の文字列から、馬番 → (かたまりの頭数, かたまりの中の位置)。

    （ ）の中は、並走している馬のかたまりで、内側の馬から外側の馬の順に書かれている。
    位置は 0（いちばん内）〜 1（いちばん外）。かたまりが無い馬（1頭だけで書かれた馬）は、頭数1・位置は空欄。
    「（）」は、「）（」（かたまりが続く）が、1つの記号として読まれたもの。
    """
    raw = raw.replace("（）", "）（").replace("＝", "−")
    out, group = {}, None
    for tok in re.findall(r"\d+|（|）|[^\d（）]", raw):
        if tok == "（":
            group = []
        elif tok == "）":
            if group:
                for k, u in enumerate(group):
                    out[u] = (len(group), k / (len(group) - 1) if len(group) > 1 else None)
            group = None
        elif tok.isdigit():
            if group is None:
                out[int(tok)] = (1, None)
            else:
                group.append(int(tok))
    return out


def parse_payouts(text):
    """払戻金の欄から、レースごとの単勝・複勝の払戻（100円あたりの円）を読む。

    例「単勝 870円 複勝 300円 240円 210円 枠連 …」→ win=[870], place=[300, 240, 210]。
    複勝の馬番は丸数字の図形で書かれていて文字として読めないが、払戻は着順（1着、2着、3着）の並びで書かれている。
    """
    head_re = re.compile(r"(\d{5}) ?\d+月\d+日 .*第\d+競走")
    pay_re = re.compile(r"単\s*勝\s+((?:[\d，,]+円\s*)+)複\s*勝\s+(.*?)(?:枠\s*連|$)")
    num = lambda t: [int(re.sub(r"\D", "", x)) for x in re.findall(r"([\d，,]+)円", t)]
    out, cur = {}, None
    for ln in text.split("\n"):
        m = head_re.search(ln)
        if m:
            cur = m.group(1)
        if cur is None or cur in out:
            continue
        m = pay_re.search(ln)
        if m:
            out[cur] = dict(win=num(m.group(1)), place=num(m.group(2)))
    return out


def to_sec(t):
    if not t:
        return None
    mm, ss = t.split(":")
    return int(mm) * 60 + float(ss)


def parse_row(rc, y, off):
    rel = lambda c: c["x0"] - off
    g = lambda a, b, tol=2.5: [c for c in rc if a <= rel(c) < b and abs(c["top"] - y) < tol]

    waku = digits(g(18, 26))
    umaban = digits(g(26, 46))
    name = txt(g(46, 110.5)).replace(" ", "")
    sexc = g(110.5, 116)
    sex = ct(sexc[0]) if sexc else "?"
    if sex not in ("牡", "牝"):
        sex = "セ"
    age = digits(g(110.5, 121))  # 10歳以上は十の位が性別の列にかかる
    color = txt(g(121, 132.5, 3))
    color = "芦" if "?" in color else color  # 芦毛の「芦」は外字

    groups = {}
    for c in g(132.5, 148, 4):
        if ct(c).isdigit():
            groups.setdefault(round(c["top"] - y), []).append(c)
    wv = []
    for k in groups:
        s = digits(groups[k])
        if len(s) >= 2:
            wv.append(float(s[:2] + ("." + s[2:] if len(s) > 2 else "")))

    jraw = txt(g(147, 192.5))
    mark = "".join(ch for ch in jraw if ch in MARKS)
    jockey = re.sub(f"[{MARKS} ]", "", jraw)
    jockey = re.sub(r"\d+$", "", jockey)  # 隣の馬主欄の先頭の数字（「79組合」など）がかかることがある
    jockey = KNOWN_FIX.get(jockey, jockey).replace("．", ".")
    trainer = txt([c for c in rc if 234 <= rel(c) < 262 and -1.5 <= c["top"] - y < 3]).replace(" ", "")
    trainer = KNOWN_FIX.get(trainer, trainer).replace("．", ".")
    region = txt([c for c in rc if 262 <= rel(c) < 275 and -1.5 <= c["top"] - y < 3]).replace(" ", "")
    region = re.sub(r"[ァ-ヴー]+$", "", region)  # 隣の生産者欄の先頭のカタカナ（「トーア牧場」など）を除く
    region = KNOWN_FIX.get(region, region)
    blinker = any(ct(c) == "B" for c in g(299, 308.5))
    bw = digits(g(308.5, 320))
    dcs = g(320, 334)
    ds = txt(dcs)
    dn = digits(dcs)
    if dn:
        body_diff = -int(dn) if any(ch in ds for ch in "−－") else int(dn)
    else:
        body_diff = 0 if "±" in ds else None
    tm = z2h(txt(g(334, 356)))
    odds = z2h(txt(g(380, 402))).strip()
    return dict(waku=waku, umaban=umaban, name=name, sex=sex, age=age, color=color, wv=wv,
                mark=mark, jockey=jockey, trainer=trainer, region=region, blinker=blinker,
                bw=bw, body_diff=body_diff, tm=tm, odds=odds)


def parse_pdf(path):
    out = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            for x0, x1 in HALVES:
                chars = [c for c in p.chars if x0 <= c["x0"] < min(x1, p.width)]
                if not chars:
                    continue
                off = detect_offset(chars, x0)
                text = fix_text(p.crop((x0, 0, min(x1, p.width), p.height)).extract_text() or "")
                heads = parse_headers(text)
                pays = parse_payouts(text)
                hys = header_ys(chars, off)
                if len(hys) != len(heads):
                    print(f"警告: ヘッダー位置の数が一致しません p{p.page_number} x{x0}: {len(heads)} vs {len(hys)}")
                    hys = hys[:len(heads)]
                bounds = list(zip(hys, hys[1:] + [1e9]))

                # 馬の行 = 枠番(x≈21)の数字 と 馬名先頭(x≈49.5) が同じ高さにある行
                rel = lambda c: c["x0"] - off
                starts = [c for c in chars if 49 <= rel(c) < 50.5 and not ct(c).isdigit()]
                row_ys = []
                for s in sorted(starts, key=lambda c: c["top"]):
                    if any(20.5 <= rel(c) < 22 and ct(c).isdigit() and abs(c["top"] - s["top"]) < 2 for c in chars):
                        if not row_ys or abs(s["top"] - row_ys[-1]) > 2:
                            row_ys.append(s["top"])

                for h, (lo, hi) in zip(heads, bounds):
                    prev_t, pos = None, 0
                    # 馬番 → コーナーごとの順位（何番目に通過したか）。最後の並びを corner4（最終コーナー）とし、
                    # 後ろから詰める（3・4コーナーだけのコースは corner3, corner4 に入る。2周するコースは後半の周回）
                    pay = pays.get(h["race_id"])
                    seqs = parse_corners(chars, off, lo, hi)[-4:]
                    cpos = {4 - len(seqs) + 1 + i: {u: j + 1 for j, u in enumerate(seq)} for i, seq in enumerate(seqs)}
                    raws = parse_corners(chars, off, lo, hi, raw=True)[-4:]
                    clat = {4 - len(raws) + 1 + i: corner_lateral(r) for i, r in enumerate(raws)}
                    for y in row_ys:
                        if not (lo < y < hi):
                            continue
                        rc = [c for c in chars if -5 < c["top"] - y < 5]
                        rowtxt = txt([c for c in rc if abs(c["top"] - y) < 2.5])
                        if "除外" in rowtxt or "取消" in rowtxt:
                            continue  # 競走除外・出走取消は出走していないので除く
                        r = parse_row(rc, y, off)
                        if not r["umaban"]:
                            continue  # 負担重量の注記（「1増（ただし…）」など）が枠番の位置にかかった行
                        # 1分未満の走破タイム（1000m戦など）は「57.8」のように分が付かないので 0: を補う
                        m = re.match(r"(\d:)?\d\d\.\d", r["tm"])
                        if "〃" in r["tm"]:
                            t = prev_t
                        elif m:
                            t = m.group(0) if m.group(1) else "0:" + m.group(0)
                        else:
                            t = None
                        stopped = t is None
                        prev_t = t
                        if not stopped:
                            pos += 1
                        out.append(dict(
                            **h,
                            finish=None if stopped else pos,
                            waku=int(r["waku"]) if r["waku"] else None,
                            umaban=int(r["umaban"]) if r["umaban"] else None,
                            horse_name=r["name"], sex=r["sex"],
                            age=int(r["age"]) if r["age"] else None, color=r["color"],
                            carried_weight=min(r["wv"]) if r["wv"] else None,
                            base_weight=max(r["wv"]) if r["wv"] else None,
                            jockey_mark=r["mark"], jockey=r["jockey"], trainer=r["trainer"],
                            region=r["region"], blinker=r["blinker"],
                            body_weight=int(r["bw"]) if r["bw"] else None,
                            body_weight_diff=r["body_diff"],
                            time=t, time_sec=to_sec(t),
                            odds=float(r["odds"]) if re.fullmatch(r"\d+\.\d", r["odds"]) else None,
                            **{f"corner{k}": cpos.get(k, {}).get(int(r["umaban"])) for k in (1, 2, 3, 4)},
                            **{f"corner{k}_{n}": clat.get(k, {}).get(int(r["umaban"]), (None, None))[i]
                               for k in (1, 2, 3, 4) for i, n in enumerate(("grp", "lat"))},
                            # 払戻（100円あたり）。着順に対応させる。払戻が無い馬は0、払戻を読めなかったレースは空欄
                            win_payout=None if pay is None else (pay["win"][0] if pos == 1 and pay["win"] else 0),
                            place_payout=None if pay is None else (pay["place"][pos - 1] if 0 < pos <= len(pay["place"]) else 0),
                        ))
    df = pd.DataFrame(out)
    if df.empty:
        return df
    df["popularity"] = df.groupby("race_id")["odds"].rank(method="min")
    df["num_runners"] = df.groupby("race_id")["horse_name"].transform("count")
    df["win"] = (df["finish"] == 1).astype(int)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("-o", "--out", default="races.csv")
    a = ap.parse_args()
    df = pd.concat([parse_pdf(p) for p in a.pdfs], ignore_index=True)
    df.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"{len(df)} 行 / {df['race_id'].nunique()} レース → {a.out}")
