"""JRA公式サイトの「出馬表」から、これからのレースの出馬表を取り込む

使い方（Streamlitの画面から呼ばれる。コードからも使える）:
    meetings = list_meetings()                 # 今週の開催の一覧（日付・競馬場・回・日）
    card = fetch_card(meetings[0]["cname"])    # その開催の全レースの出馬表（cards/*.json と同じ形）

JRAのページは、ボタンを押すと、フォームに cname という値を入れて POST する仕組み。ここでも同じ POST を行う。
1回の取り込みで、2〜3回のアクセスだけにして、間隔（1秒）を空ける。個人の利用の範囲で使うこと
（JRAの「ご利用に際して」: 掲載内容の二次利用は、私的使用・引用の範囲を超える場合に許諾が必要）。
"""
import re
import time

import requests
from lxml import html as LH

URL = "https://www.jra.go.jp/JRADB/accessD.html"
HEADERS = {"User-Agent": "keiba_prediction personal research (https://github.com/yumuras/keiba_prediction)"}
LIST_CNAME = "pw01dli00/F3"  # 「出馬表」（今週の開催の一覧）
INTERVAL = 1.0
VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]

_session = requests.Session()
_session.headers.update(HEADERS)


def _post(cname):
    r = _session.post(URL, data={"cname": cname}, timeout=30)
    r.raise_for_status()
    r.encoding = "shift_jis"
    time.sleep(INTERVAL)
    return r.text


def list_meetings():
    """今週の開催の一覧: [{date, label, venue, cname}]（日付の順）"""
    text = _post(LIST_CNAME)
    out = []
    for block in re.split(r'<h3 class="sub_header">', text)[1:]:
        m = re.match(r"(\d+)月(\d+)日", block)
        if not m:
            continue
        for cname, label in re.findall(r"doAction\('/JRADB/accessD\.html',\s*'(pw01drl[^']+)'\);\"[^>]*>.*?</i>([^<]+)<", block, flags=re.S):
            ymd = re.search(r"(\d{8})/", cname)
            venue = next((v for v in VENUES if v in label), "")
            if ymd and venue:
                d = ymd.group(1)
                out.append(dict(date=f"{d[:4]}-{d[4:6]}-{d[6:]}", label=label.strip(), venue=venue, cname=cname))
    return out


def _text(el):
    return re.sub(r"\s+", " ", el.text_content()).strip() if el is not None else ""


def _cls(el, name):
    """class 属性に name を含む最初の要素（例: class="cell course" → "course"）"""
    r = el.xpath(f'.//*[contains(concat(" ", normalize-space(@class), " "), " {name} ")]')
    return r[0] if r else None


def _race_class(cls, grade_alt, name):
    if "Ⅰ" in grade_alt or "Ⅱ" in grade_alt or "Ⅲ" in grade_alt:
        return "J・" + ("GⅠ" if "Ⅰ" in grade_alt else "GⅡ" if "Ⅱ" in grade_alt else "GⅢ") if "障害" in name else \
            ("GⅠ" if "Ⅰ" in grade_alt else "GⅡ" if "Ⅱ" in grade_alt else "GⅢ")
    if grade_alt.strip() in ("L", "Ｌ"):
        return "L"
    return {"新馬": "新馬", "未勝利": "未勝利", "1勝クラス": "1勝クラス", "2勝クラス": "2勝クラス", "3勝クラス": "3勝クラス", "オープン": "オープン"}.get(cls, cls)


def _parse_race(li):
    start = re.search(r"(\d+)時(\d+)分", _text(_cls(li, "time")))
    name = _text(_cls(li, "race_name"))
    cls = _text(_cls(li, "class"))
    rule = _text(_cls(li, "weight"))
    course = _text(_cls(li, "course"))
    grade = " ".join(li.xpath('.//h2//span[contains(@class,"grade_icon")]//img/@alt'))
    race_no = int(re.search(r"syutsuba_(\d+)R", li.get("id")).group(1))
    dist = int(re.search(r"([\d,]+)\s*メートル", course).group(1).replace(",", ""))
    detail = re.search(r"（([^）]*)）", course).group(1)  # 例「芝・右 外」「ダート・右」「芝・ダート」
    surface = "障害" if ("障害" in name or "障害" in cls or "芝・ダート" in detail) else ("芝" if detail.startswith("芝") else "ダート")
    turn = next((x for x in ("右", "左", "直線") if x in detail), "右")
    inout = "外内" if "外内" in detail else ("外" if "外" in detail else "内")
    horses = []
    for tr in li.xpath(".//tbody/tr"):
        num = tr.find('.//td[@class="num"]')
        if num is None or not re.search(r"\d", _text(num)):
            continue
        age = _text(tr.find('.//td[@class="age"]'))
        jockey = _text(tr.find('.//td[@class="jockey"]'))
        waku = re.search(r"枠(\d)", "".join(tr.xpath('.//td[@class="waku"]//img/@alt')))
        w = _text(tr.find('.//td[@class="weight"]'))
        odds = re.search(r"[\d.]+", _text(tr.find('.//td[@class="odds"]')))
        bw = re.search(r"(\d+)(?:\(([+\-±]?\d+)\))?", _text(tr.find('.//td[@class="h_weight"]')))
        horses.append(dict(
            umaban=int(re.search(r"\d+", _text(num)).group()), waku=int(waku.group(1)) if waku else None,
            name=_text(tr.find('.//td[@class="horse"]')), sex={"せ": "セ"}.get(age[:1], age[:1]) if not age.startswith("せん") else "セ",
            age=int(re.search(r"\d+", age).group()), weight=float(re.search(r"[\d.]+", w).group()) if w else None,
            reduction={"☆": 1.0, "▲": 3.0, "△": 2.0, "◇": 2.0, "★": 3.0}.get(next((c for c in jockey if c in "☆▲△◇★"), ""), 0.0),
            jockey=re.sub(r"[☆▲△◇★\s]", "", jockey), trainer=re.sub(r"\s", "", _text(tr.find('.//td[@class="trainer"]'))),
            odds=float(odds.group()) if odds else None, body_weight=int(bw.group(1)) if bw and bw.group(1) and int(bw.group(1)) > 200 else None,
            body_weight_diff=int(bw.group(2).replace("±", "0")) if bw and bw.group(2) else None,
            blinker=bool(tr.xpath('.//img[contains(@alt,"ブリンカー")]'))))
    return dict(race_no=race_no, start_time=f"{int(start.group(1)):02d}:{start.group(2)}" if start else "", name=name,
                race_class=_race_class(cls, grade, name), surface=surface, distance=dist, turn=turn, inout=inout,
                handicap=(rule == "ハンデ"), category=_text(_cls(li, "category")), horses=horses)


def fetch_card(cname):
    """開催（cname）の全レースの出馬表を取り込む。cards/*.json と同じ形（馬場・天候は未設定）"""
    page = _post(cname)
    m = re.search(r"doAction\('/JRADB/accessD\.html',\s*'(pw01des01[^']+)'\)", page)
    if not m:
        raise RuntimeError("「全てのレースを表示」が見つかりません（出馬表がまだ公開されていない可能性があります）")
    doc = LH.fromstring(_post(m.group(1)))
    head = _text(_cls(doc, "date"))
    d = re.search(r"(\d{4})年(\d+)月(\d+)日", head)
    races = [_parse_race(li) for li in doc.xpath('//ul[contains(@class,"syutsuba_unit_list")]/li[starts-with(@id,"syutsuba_")]')]
    venue = next((v for v in VENUES if v in head), "")
    return dict(date=f"{d.group(1)}-{int(d.group(2)):02d}-{int(d.group(3)):02d}", venue=venue, label=head, going="良", weather="晴",
                note=f"JRA公式の出馬表（{head}）。単勝オッズは取得時点の値。", races=races)
