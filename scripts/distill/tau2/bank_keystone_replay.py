# -*- coding: utf-8 -*-
"""reference-filter keystone 오프라인 REPLAY 정량 (2026-07-14·HANDOFF §0·[[08]]/[[05]]).

bank_filter_repro(유일식별 *가능성* 체크)와 다름: 여기선 **실제 A2 reference_filter 규칙**을
`t2_compute.apply_op(op=filter)` 엔진으로 궤적의 ⋈ 오선택 케이스마다 replay해 "잘못 고른 ref 중
X% 결정론 교정"을 산출한다. A2 스펙(match/criteria/on_ambiguous)은 banking_knowledge.gate.json서 로드
(엔진=일반 filter op·규칙=A2·[[05]] 리터럴0). criteria는 gold record서 파생(=perfect-formalize 천장).

★정직성([[08]] handoff §0-caveat): criteria를 gold record서 파생 = perfect-formalize 천장.
  merchant=gold.description서 리딩 브랜드 토큰 추출=근사(formalize half=LLM·미측정). → 2수치 보고:
    (a) date+type만       = 완전 결정론·formalize 불요 = 교정 *하한*
    (b) date+type+merchant = perfect-formalize          = 교정 *천장*
  전체 교정률 = formalize정확도 × filter ∈ [하한, 천장].
"""
import json, glob, os, sys
from collections import Counter
import bank_filter_repro as B
from t2_compute import apply_op

GATE = os.path.join(os.path.dirname(__file__), "a2", "banking_knowledge.gate.json")

# merchant 리딩 브랜드 토큰 근사: description 앞머리의 generic-prefix 스킵 후 첫 실질 토큰.
# (user 발화가 실제 주는 형태 — "the CityFit charge"·"my Starbucks purchase". 전체 description 아님.)
_PREFIX = {"DIRECT", "DEPOSIT", "ATM", "WITHDRAWAL", "WITHDRAWL", "POS", "TRANSFER",
           "FROM", "TO", "WIRE", "ACH", "PAYMENT", "PURCHASE", "DECLINED", "INTEREST",
           "CREDIT", "DEBIT", "CARD", "-"}


def merchant_token(desc):
    """gold.description → 리딩 브랜드 토큰(단일). 숫자토큰·generic-prefix 스킵. 없으면 None."""
    if not desc:
        return None
    for tok in str(desc).split():
        t = tok.strip("-").upper()
        if not t or tok.startswith("#") or any(c.isdigit() for c in tok):
            continue
        if t in _PREFIX:
            continue
        return tok            # 원형(대소문자 유지·contains는 op가 lower로 비교)
    return None


def load_filter_spec():
    """A2 reference_filter → apply_op(op=filter) 스펙. 엔진은 이 스펙만 dispatch(리터럴0)."""
    a2 = json.load(open(GATE, encoding="utf-8"))
    rf = a2["reference_filter"]
    return {
        "op": "filter",
        "over": "records",
        "match": rf["match"],                     # date eq·description contains·type eq
        "return": rf["key_field"],                # transaction_id
        "on_ambiguous": rf["on_ambiguous"],       # none
    }, rf


def iter_cross_cases():
    """bank_filter_repro와 동일한 hard-task ⋈ 케이스 추출 → (grec, chosen_id, recs) yield.
    (동일 모집단이라 83% 유일식별 수치와 직접 비교 가능.)"""
    per = {}
    data = {}
    for f in glob.glob("C:/tmp/traj/*_banking.json"):
        try:
            d = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        data[f] = d
        for s in d["simulations"]:
            r = (s.get("reward_info") or {}).get("reward")
            if r is None:
                continue
            t = str(s["task_id"]); per.setdefault(t, [0, 0]); per[t][1] += 1
            if r == 1.0:
                per[t][0] += 1
    hard = {t for t, p in per.items() if p[1] >= 10 and p[0] / p[1] <= 0.10}

    for f, d in data.items():
        for s in d["simulations"]:
            if str(s["task_id"]) not in hard:
                continue
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue
            msgs = s.get("messages") or []
            recs = B.gathered_records(msgs)
            if len(recs) < 2:
                continue
            byid = {r.get("transaction_id"): r for r in recs}
            cl = [(tc.get("name"), B.nd(tc.get("arguments")))
                  for m in msgs for tc in (m.get("tool_calls") or [])]
            for ac in (ri.get("action_checks") or []):
                if ac.get("action_match"):
                    continue
                a = ac.get("action") or {}
                if a.get("name") != "call_discoverable_agent_tool":
                    continue
                g = B.nd(a.get("arguments")); gt = g.get("agent_tool_name"); gn = B.nd(g.get("arguments"))
                same = [B.nd(ar.get("arguments")) for nm, ar in cl
                        if nm == "call_discoverable_agent_tool" and str(ar.get("agent_tool_name")) == str(gt)]
                if not same:
                    continue
                an = same[0]
                gid = str(gn.get("transaction_id") or "")
                chosen = str(an.get("transaction_id") or "")
                if not gid or gid == chosen:
                    continue                              # transaction_id ⋈ 아님
                grec = byid.get(gid)
                if not grec:
                    continue                              # gold record 미파싱
                yield grec, chosen, recs


def build_spec(rf, use_amount):
    match = list(rf["match"])
    if use_amount:
        match = match + [{"field": "amount", "eq": "criteria.amount"}]
    return {"op": "filter", "over": "records", "match": match,
            "return": rf["key_field"], "on_ambiguous": rf["on_ambiguous"]}


def run(rf, recs, grec, use_merchant, use_amount=False):
    """A2 filter op을 criteria(gold서 파생=perfect-formalize) 위에서 replay → 반환 id."""
    crit = {"date": grec.get("date"), "transaction_type": grec.get("type")}
    if use_merchant:
        crit["merchant"] = merchant_token(grec.get("description"))
    if use_amount:
        crit["amount"] = grec.get("amount")
    ctx = {"records": recs, "criteria": crit}
    return apply_op(build_spec(rf, use_amount), ctx)


def is_true_dup(grec, recs):
    """gold와 全식별필드(date·amount·type·description) 동일 record ≥2 = 진짜중복(on_ambiguous=none 정당 abstain)."""
    fields = ["date", "amount", "type", "description"]
    return sum(1 for r in recs if all(r.get(fl) == grec.get(fl) for fl in fields)) >= 2


def main():
    spec, rf = load_filter_spec()
    print("=== A2 reference_filter 스펙 (banking_knowledge.gate.json) ===")
    print("  criteria_fields:", rf["criteria_fields"], "· on_ambiguous:", rf["on_ambiguous"])
    print("  match:", [f"{c['field']} {'eq' if 'eq' in c else 'contains'} {c.get('eq') or c.get('contains')}" for c in rf["match"]])

    cases = list(iter_cross_cases())
    n = len(cases)
    dup = sum(1 for grec, _, recs in cases if is_true_dup(grec, recs))
    dec = n - dup
    print("\n=== transaction_id ⋈ 오선택 케이스 n=%d (17 frontier·hard-core) ===" % n)
    print("    진짜중복 %d = %.1f%% (date/amount/type/description 全동일 ≥2·on_ambiguous=none 정당abstain·완벽에이전트도 못맞힘)" % (dup, 100 * dup / max(n, 1)))
    print("    결정가능부 = %d (진짜중복 제외)\n" % dec)

    # 각 variant = 어느 필드까지 perfect-formalize 가정하는가. 전부 gold record서 파생.
    variants = [
        ("(a) date+type            [merchant 미사용·하한]", False, False),
        ("(b) date+type+merchant   [브랜드토큰 formalize·천장]", True, False),
        ("(c) date+type+amount     [amount formalize·결정가능부 천장]", False, True),
    ]
    for label, use_m, use_a in variants:
        c = Counter()
        for grec, chosen, recs in cases:
            res = run(rf, recs, grec, use_m, use_a)
            gid = grec.get("transaction_id")
            if res is None:
                c["③미해결(0매칭/모호→none)"] += 1
                if is_true_dup(grec, recs):
                    c["└─그중 진짜중복(정당abstain)"] += 1
            elif str(res) == str(gid):
                c["①교정성공(filter==gold)"] += 1
            else:
                c["②filter오답(≠gold)"] += 1
        ok = c["①교정성공(filter==gold)"]
        print("── %s" % label)
        for k in ["①교정성공(filter==gold)", "②filter오답(≠gold)",
                  "③미해결(0매칭/모호→none)", "└─그중 진짜중복(정당abstain)"]:
            print("     %-30s %5d (%.1f%%)" % (k, c.get(k, 0), 100 * c.get(k, 0) / max(n, 1)))
        print("     ★교정률: 전체 %.1f%% · 결정가능부 %.1f%%\n" % (100 * ok / max(n, 1), 100 * ok / max(dec, 1)))

    print("─" * 68)
    print("★정직성 캐비엇 ([[08]]·과대주장 방지):")
    print("  1. ②filter오답=0은 *구조적*이다 — criteria를 gold record서 파생하므로 gold는 항상")
    print("     자기 기준을 만족→매칭셋에 gold 포함→유일매칭이면 필연 gold. 이 replay는 필터의")
    print("     *reach 천장*을 재지, imperfect-formalize 하의 Δspurious(오치환율)를 재지 않는다.")
    print("     Δspurious는 별도 게이트(설계 §8-2)·미측정.")
    print("  2. 전 variant = perfect-formalize 천장(기준을 gold서 파생). 실제 레버 = formalize정확도")
    print("     × 이 천장. 단 date/type/amount = 저모호 구조적 추출(날짜·enum·금액)이라 formalize")
    print("     쉬움; merchant(어려운 NER)는 거의 무관((a)만으로 결정가능부 91.6%)→천장이 쉬운 formalize에 기댐.")
    print("  3. 미해결(비-중복)의 정체 = 다중매칭(merchant 브랜드토큰 조잡 예: 'PREMIUM'). amount 추가")
    print("     시 결정가능부 100% 해소·0오답. ⇒ 비-중복 ⋈는 전부 결정론 filter로 gold 도달 가능.")
    print("  4. 모집단 = 파싱신뢰 필터 미적용(설계 §5d는 798 부분표본). 파싱갭은 미해결↑(보수적)·")
    print("     허위교정↑ 아님(gold 항상 매칭셋).")


if __name__ == "__main__":
    main()
