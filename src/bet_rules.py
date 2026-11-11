"""買う・見送りの判定（予想表と画面で使う）

過去の検証（scripts/check_roi_patterns.py、README「回収率の構造的な偏りの検証」「強い馬の深掘り」）で、
回収率が全体より明らかに高かった条件だけを「買う」にし、それ以外は「見送り」にする。
あわせて、回収率が明らかに悪かった買い方には、警告を出す。

数字はどれも最終オッズでの過去の結果。買う時点のオッズでの成績は、scripts/record_odds.py と scripts/paper_bets.py で記録して確かめる。
"""
import pandas as pd

# 本命の単勝オッズがこれ未満なら、2番人気の複勝を買う（最終オッズ）
# 2024〜2026年で見つけた条件（150件・115%）。見つけたあとで足した2023年では 81% だった。4年で188件・109%（95%区間 87〜132%）
BUY = 1.3
# 能力モデル（オッズなし）の3着内確率 ÷ 市場の3着内確率 がこれ以上の馬は、「モデルだけが推す馬」
# （2つのバックテストの上位10%の境目。上位10%の複勝回収率は 53%・45%）
SOLO_RATIO = 2.9


def judge(res, last=None):
    """1レースの予想（predict.predict_card の結果）から、判定を返す。

    res: 馬ごとの行。odds（単勝オッズ）・p_ability・p_mkt3・umaban・horse_name を使う
    last: 馬名 → 前走の行（predict.last_rows）。前走の人気・着順の警告に使う（無ければ省く）
    戻り値: dict(verdict="買う"|"見送り"|"判定なし", bet=文字列 or None, reason=文字列, warnings=[文字列])
    """
    odds = res["odds"]
    if odds.isna().any() or len(res) < 5:
        return dict(verdict="判定なし", bet=None, reason="オッズが全頭そろっていないため", warnings=[])
    r = res.assign(_rank=odds.rank(method="first")).sort_values("_rank")
    fav, second = r.iloc[0], r.iloc[1]
    warnings = []
    if fav["odds"] < 1.5:
        warnings.append(f"本命（{int(fav.umaban)}番 {fav.horse_name}）を外した3連複ボックスは避ける（過去の回収率 13%・52%）")
    if "p_ability" in r and "p_mkt3" in r:
        solo = r[(r["p_ability"] / r["p_mkt3"] >= SOLO_RATIO)]
        if len(solo):
            warnings.append("モデルだけが高く評価している馬：" + "、".join(f"{int(x.umaban)}番 {x.horse_name}" for x in solo.itertuples())
                            + "（市場より大きく高く評価した馬の複勝回収率は 約50%。残す根拠にしない）")
    if last is not None:
        beaten = []
        for x in r.itertuples():
            if x.horse_name in last.index:
                p = last.loc[x.horse_name]
                if pd.notna(p.get("popularity")) and p["popularity"] <= 2 and pd.notna(p.get("finish")) and p["finish"] > 3:
                    beaten.append(f"{int(x.umaban)}番 {x.horse_name}（前走{int(p['popularity'])}番人気{int(p['finish'])}着）")
        if beaten:
            warnings.append("前走で人気を裏切った馬：" + "、".join(beaten) + "（巻き返し狙いは不利。次走は市場の見込みより1.3ポイント悪い）")
    bet = f"2番人気 {int(second.umaban)}番 {second.horse_name} の複勝"
    if fav["odds"] < BUY:
        return dict(verdict="買う", bet=bet, reason=f"本命 {int(fav.umaban)}番 {fav.horse_name} が {fav['odds']:.1f}倍（{BUY}倍未満。2023〜2026年の回収率 109%、まだ検証中）", warnings=warnings)
    return dict(verdict="見送り", bet=None, reason=f"本命が {fav['odds']:.1f}倍で、回収率の高い条件に当てはまらない", warnings=warnings)


def as_markdown(j):
    head = {"買う": "🟢 **買う**", "見送り": "⚪ 見送り", "判定なし": "⚪ 判定なし"}[j["verdict"]]
    lines = [f"**判定**：{head}" + (f" ― {j['bet']}" if j["bet"] else "") + f"（{j['reason']}）"]
    lines += [f"- ⚠️ {w}" for w in j["warnings"]]
    return "\n".join(lines) + "\n"
