#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x29: 식별표 정적 시뮬 (설계서 rev2 §6-2·리뷰 세부 7·무료·결정론·gold 0).
실 A2 `policy_group_rows` + 실 카탈로그로 4조건 검정:
 ⓐ C276★① 진짜 6종이 표 멤버십으로 회수  ⓑ 019(ThredUp→Thrive Market) 차단
 ⓒ 케이스 11(LinkedIn Learning→LinkedIn Ads) 차단  ⓓ 비선행 범주어가 named 경로서 전부 조회-실패
사용: py -3 x29_table_sim.py [<domain 경로>]"""
import json, os, re, sys, types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
for m, a in (("tau2", {}), ("tau2.data_model", {}),
             ("tau2.data_model.message", {"UserMessage": object, "ToolMessage": object,
                                          "MultiToolMessage": object}),
             ("tau2.agent", {}), ("tau2.agent.llm_agent", {})):
    if m not in sys.modules:
        mod = types.ModuleType(m)
        for k, v in a.items():
            setattr(mod, k, v)
        sys.modules[m] = mod
import t2_scaffold_get as SG  # noqa: E402

DOM = sys.argv[1] if len(sys.argv) > 1 else "/home/woori/scratch/tau2-bench/data/tau2/domains/banking_knowledge"
A2 = json.load(open(os.path.join(HERE, "a2", "banking_knowledge.specific.json"), encoding="utf-8"))
ISO = next(t for t in A2["scaffold_get_tools"]
           if t["name"] == "get_reward_discrepancies")["variants"]["ratefix"]["isolate"]
QP = ISO["quote_pin"]
TBL = QP["policy_group_rows"]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-z]+", " ", str(s).lower())).strip()


merch = set()


def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "merchant_name" and isinstance(v, str):
                merch.add(v.strip())
            else:
                walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)


walk(json.load(open(os.path.join(DOM, "db.json"), encoding="utf-8")))
corpus = " ".join(norm(json.load(open(f, encoding="utf-8")).get("content") or "")
                  for f in sorted(__import__("glob").glob(os.path.join(DOM, "documents", "*.json"))))

OK = True


def chk(c, m):
    global OK
    OK &= bool(c)
    print(("  ✓ " if c else "  ✗ ") + m)


def V(pin, merchant, kind="named_merchant"):
    """quote는 표 키를 담은 실 코퍼스라고 두고(축자 검증은 x28/단위테스트 담당) 멤버십만 시뮬."""
    return SG._quote_pin_check(QP, {"q": pin, "p": pin, "k": kind}, {"m": merchant},
                               "q", 0, norm(pin))[0]


print(f"표 엔트리 {len(TBL)} (비공집합 {sum(1 for v in TBL.values() if v)}) · 카탈로그 {len(merch)}")
print("\nⓐ C276★① 진짜 6종 회수:")
for policy, row in [("Target", "Target - Eco Collection"), ("Microsoft", "Microsoft 365"),
                    ("Apple", "Apple Store"), ("Dell", "Dell Technologies"),
                    ("Zoom", "Zoom Video"), ("Slack", "Slack Technologies")]:
    chk(V(policy, row) == "pass", f"{policy!r} → {row!r} 통과")

print("\nⓑ 019 차단 (named·category 양 경로):")
chk(V("ThredUp", "Thrive Market") == "reject_member", "'ThredUp' → 'Thrive Market' 차단")
chk(V("Thrift and Resale Markets", "Thrive Market", "category") == "reject_member",
    "그룹 'Thrift and Resale Markets' → 'Thrive Market' 차단(구성원 {ThredUp})")

print("\nⓒ 케이스 11 차단:")
chk(V("LinkedIn Learning", "LinkedIn Ads") == "reject_member",
    "'LinkedIn Learning'(=[]·판단된 무대응) → 'LinkedIn Ads' 차단")
chk(V("LinkedIn", "LinkedIn Ads") == "lookup_missing", "조각-핀 'LinkedIn' → 조회 실패(재질의)")

print("\nⓓ 비선행 범주어 핀이 named 경로서 전부 조회-실패:")
gen, bad = [], []
for m in sorted(merch):
    toks = norm(m).split()
    for t in toks[1:]:
        if len(t) >= 4 and t in corpus:
            gen.append((m, t))
            if V(t, m) != "lookup_missing":
                bad.append((m, t, V(t, m)))
            break
print(f"  비선행 토큰 핀 후보 {len(gen)}종")
chk(not bad, f"전부 lookup_missing (예외 {len(bad)}: {bad[:4]})")

print("\nⓓ-2 ★허위 3종 비회수 (2차 리뷰 조건 4 — 표가 앵커보다 옳은 지점):")
print("   C276★①이 정정한 허위 false-abstain 3종. 앵커는 이들을 *잘못* 회수했고 표는 회수하지 않아야 한다.")
for pin, row, why in [("home", "Home Depot", "'home-sharing' 파생 합성어"),
                      ("Electronics", "Electronics Express Miami", "범주 제목(열거형 비-구성원)"),
                      ("LinkedIn Learning", "LinkedIn Ads", "타-상인 명명")]:
    vd = V(pin, row)
    chk(vd != "pass", f"{pin!r} ↛ {row!r} ({why}) — verdict={vd}")

print("\nⓔ 부수: 접두-동형 타업체 오통과 없음:")
for pin, row in [("Dell", "Delta Airlines"), ("Dell", "Delta Sky Club"),
                 ("Target", "Targeting Solutions Inc"), ("Amazon", "AWS")]:
    chk(V(pin, row) == "reject_member", f"{pin!r} ↛ {row!r}")

print("\n%s" % ("PASS — 표-시뮬 4조건 충족" if OK else "FAIL"))
sys.exit(0 if OK else 1)
