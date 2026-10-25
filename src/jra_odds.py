"""JRA公式サイトの「オッズ」ページから、1レースの実際のオッズ（各券種）を取り込む

使い方:
    odds = fetch_odds(cname)     # cname は jra_card.fetch_card() が返す、各レースの "odds_cname"

取り込む券種: 単勝・複勝・馬連・ワイド・馬単・3連複・3連単（枠連は取り込まない）。
1レースにつき、ページを6回取得する（単複・馬連・ワイド・馬単・3連複・3連単。3連単のページは約1MB）。1秒間隔で、約8秒かかる。
個人の利用の範囲で使うこと（JRAの「ご利用に際して」: 掲載内容の二次利用は、私的使用・引用の範囲を超える場合に許諾が必要）。

返す値のキーは、馬番（インデックスではない）。オッズは「倍」（100円が何円になるか ÷ 100）。
複勝・ワイドは、下限と上限（lo, hi）。発売されていない組み合わせ（取消・除外の馬を含む）は、含まれない。
"""
import re
import time

import requests
from lxml import html as LH

URL = "https://www.jra.go.jp/JRADB/accessO.html"
HEADERS = {"User-Agent": "keiba_prediction personal research (https://github.com/yumuras/keiba_prediction)"}
INTERVAL = 1.0
TABS = {"馬連": "quinella", "ワイド": "wide", "馬単": "exacta", "3連複": "trio", "3連単": "trifecta"}

_session = requests.Session()
_session.headers.update(HEADERS)


def _post(cname):
    r = _session.post(URL, data={"cname": cname}, timeout=60)
    r.raise_for_status()
    r.encoding = "shift_jis"
    time.sleep(INTERVAL)
    return r.text


def _num(s):
    s = (s or "").strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _range(td):
    """「5.0-9.7」のような範囲（または、span の min / max）を (下限, 上限) にする"""
    lo, hi = td.xpath('.//span[@class="min"]'), td.xpath('.//span[@class="max"]')
    if lo and hi:
        return _num(lo[0].text_content()), _num(hi[0].text_content())
    m = re.match(r"\s*([\d.,]+)\s*-\s*([\d.,]+)", td.text_content())
    return (_num(m.group(1)), _num(m.group(2))) if m else (None, None)


def _caption(t):
    c = t.find("caption")
    return re.sub(r"\s+", "", c.text_content()) if c is not None else ""


def parse_win_place(doc):
    win, place = {}, {}
    for tr in doc.xpath('//table[contains(@class,"tanpuku")]//tbody/tr'):
        num = tr.xpath('./td[@class="num"]')
        if not num or not num[0].text_content().strip().isdigit():
            continue
        u = int(num[0].text_content().strip())
        tan = tr.xpath('./td[@class="odds_tan"]')
        fuku = tr.xpath('./td[@class="odds_fuku"]')
        if tan and _num(tan[0].text_content()) is not None:
            win[u] = _num(tan[0].text_content())
        if fuku:
            lo, hi = _range(fuku[0])
            if lo is not None:
                place[u] = (lo, hi)
    return win, place


def _rows(t, ranged=False):
    """表の行（組み合わせの残りの馬番, オッズ）。オッズが空欄の行（同じ馬など）は除く"""
    for tr in t.xpath(".//tbody/tr"):
        th = tr.xpath("./th")
        td = tr.xpath("./td")
        if not th or not td or not th[0].text_content().strip().isdigit():
            continue
        v = _range(td[0]) if ranged else _num(td[0].text_content())
        if (v[0] if ranged else v) is not None:
            yield int(th[0].text_content().strip()), v


def parse_pairs(doc, kind, ordered):
    """馬連・ワイド・馬単: 表のタイトルが、軸の馬番（馬単は1着の馬番）。行が、相手の馬番"""
    out = {}
    for t in doc.xpath(f'//table[contains(@class,"{kind}")]'):
        axis = _caption(t)
        if not axis.isdigit():
            continue
        for j, v in _rows(t, ranged=(kind == "wide")):
            out[(int(axis), j) if ordered else tuple(sorted((int(axis), j)))] = v
    return out


def parse_trio(doc):
    out = {}
    for t in doc.xpath('//table[contains(@class,"fuku3")]'):
        m = re.match(r"(\d+)-(\d+)$", _caption(t))
        if not m:
            continue
        for k, v in _rows(t):
            out[tuple(sorted((int(m.group(1)), int(m.group(2)), k)))] = v
    return out


def parse_trifecta(doc):
    out = {}
    for li in doc.xpath('//ul[contains(@class,"tan3_list")]/li'):
        nums = [n.text_content().strip() for n in li.xpath('.//div[@class="num"]')]
        t = li.xpath('.//table[contains(@class,"tan3")]')
        if len(nums) < 2 or not t or not all(n.isdigit() for n in nums[:2]):
            continue
        for k, v in _rows(t[0]):
            out[(int(nums[0]), int(nums[1]), k)] = v
    return out


def fetch_odds(cname):
    """レースの、全券種のオッズを取り込む。cname は、単勝・複勝のオッズページの cname（"pw151ou..."）"""
    main = LH.fromstring(_post(cname))
    win, place = parse_win_place(main)
    html_tabs = {}
    for a in main.xpath("//a[@onclick]"):
        label = re.sub(r"\s+", "", a.text_content())
        m = re.search(r"'(pw15\w+/[0-9A-F]{2})'", a.get("onclick", ""))
        if label in TABS and m and label not in html_tabs:
            html_tabs[label] = m.group(1)
    out = dict(win=win, place=place, quinella={}, wide={}, exacta={}, trio={}, trifecta={}, fetched_at=time.strftime("%H:%M:%S"))
    for label, key in TABS.items():
        if label not in html_tabs:
            continue
        doc = LH.fromstring(_post(html_tabs[label]))
        out[key] = {"馬連": lambda: parse_pairs(doc, "umaren", False), "ワイド": lambda: parse_pairs(doc, "wide", False),
                    "馬単": lambda: parse_pairs(doc, "umatan", True), "3連複": lambda: parse_trio(doc),
                    "3連単": lambda: parse_trifecta(doc)}[label]()
    return out
