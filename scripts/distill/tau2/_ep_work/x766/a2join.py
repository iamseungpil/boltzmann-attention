# -*- coding: utf-8 -*-
"""x766-d: 사이드카 발화 태그 <-> A2 키 **직접 조인**.

원리: 발화 문구가 A2 값 안에 축자로 들어 있으면 그 태그의 출처는 **그 키**다(근접 추정 아님).
엔진 리터럴에만 있는 태그는 별도 표시. 읽기 전용.
"""
import json
import os
import re
import sys

ENG = r"C:\workspace\ba-frft\scripts\distill\tau2"
A2 = os.path.join(ENG, "a2")
TAGS = json.load(open(os.path.join(ENG, "_ep_work", "x766", "want_tags.json"), encoding="utf-8"))
EXTRA = ["G1_AUTH_FIRST", "G2_CONFIRM_WRITE", "G4_TRANSFER_MSG", "G5_STATUS_PRECONDITION",
         "G6_SELECT_CONFIRM", "G7_OP_CONSTRAINTS", "G_EXHAUST",
         "GB1_VERIFY_BEFORE_ACCOUNT_ACCESS", "GB2_NOTICE_BEFORE_TRANSFER", "G1_USER_ID_PROVIDED"]
TAGS = TAGS + EXTRA

DOMS = {"banking": "banking_knowledge.gate.json",
        "retail": "retail.gate.json",
        "airline": "airline.gate.json"}


def walk(o, path):
    yield path, o
    if isinstance(o, dict):
        for k, v in o.items():
            for r in walk(v, path + [str(k)]):
                yield r
    elif isinstance(o, list):
        for i, v in enumerate(o):
            for r in walk(v, path + ["[%d]" % i]):
                yield r


out = {}
for dom, fn in DOMS.items():
    d = json.load(open(os.path.join(A2, fn), encoding="utf-8"))
    hits = {}
    for topkey, val in d.items():
        if topkey.startswith("_"):
            continue
        blob = json.dumps(val, ensure_ascii=False)
        for t in TAGS:
            if ("[%s]" % t) in blob or (t in EXTRA and ('"%s"' % t) in blob):
                hits.setdefault(t, []).append(topkey)
    out[dom] = hits

json.dump(out, open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for dom in DOMS:
    print("### %s : %d tags authored in A2" % (dom, len(out[dom])))
    for t in sorted(out[dom]):
        print("   %-34s <- %s" % (t, out[dom][t]))
    print()
print("### tags NOT authored in any A2 (engine-literal only):")
allt = set()
for dom in out:
    allt |= set(out[dom])
print(sorted(set(TAGS) - allt))
