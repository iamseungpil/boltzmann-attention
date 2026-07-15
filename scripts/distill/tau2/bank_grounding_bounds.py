# -*- coding: utf-8 -*-
"""그라운딩 오탐(enum-schema) 교정: 2 bound.
 - LIBERAL: 값이 전 맥락(user+tool·schema/KB 포함) 존재 = 25.9%(오염·상한).
 - CONSERV: 값이 USER 발화에만 존재 = 고객이 직접 명시(schema 무오염·하한).
 - 필드 유형 분리: enum(정규화 필요) vs data(id/amount/date).
"""
import json, glob, re, sys, io
sys.path.insert(0, "C:/workspace/ba-frft/scripts/distill/tau2")
import bank_perstep_decomp as P
import bank_frontier_mechanism as M
abox = json.load(open("C:/workspace/ba-frft/scripts/distill/tau2/a2/banking_knowledge.gate.json", encoding="utf-8"))
cmap = P.load_compute_fields(abox)

ENUM_HINT = re.compile(r"(reason|category|type|action|status|option|design|method|resolution|class|compromised|possession|filed|provided)", re.I)

def present(val, ctx):
    v = str(val).strip().lower()
    if not v or v in ("none", "null", "true", "false"): return None
    if v in ctx: return True
    toks = [t for t in re.split(r"[_\s]+", v) if len(t) >= 4]
    if toks and sum(1 for t in toks if t in ctx) >= max(1, len(toks) - 1): return True
    return False

from collections import Counter
cnt = Counter()
for f in sorted(glob.glob("C:/tmp/traj/*_banking.json")):
    d = json.load(open(f, encoding="utf-8"))
    for s in d.get("simulations", []):
        ri = s.get("reward_info") or {}
        if ri.get("reward") in (None, 1.0): continue
        if tuple(ri.get("reward_basis") or []) != ("DB",): continue
        if str(s.get("termination_reason")) == "too_many_errors": continue
        ctx_all = " ".join(str(m.get("content")) for m in (s.get("messages") or []) if m.get("role") in ("user", "tool")).lower()
        ctx_user = " ".join(str(m.get("content")) for m in (s.get("messages") or []) if m.get("role") == "user").lower()
        for (tf, field, val) in M.ga_wrong_fields(s, abox, cmap):
            if M._COMPUTE_LIKE.search(field):
                cnt["compute-like(결정론·규칙추가)"] += 1; continue
            kind = "enum" if ENUM_HINT.search(field) else "data"
            pa = present(val, ctx_all); pu = present(val, ctx_user)
            if pa is None: cnt["bool/빈값"] += 1; continue
            cnt[("%s|LIB=%s|USER=%s" % (kind, bool(pa), bool(pu)))] += 1

tot = sum(v for k, v in cnt.items() if "|" in k)
print("=== GATHER-ASK 틀린 필드 그라운딩 2-bound (enum vs data·%d 판정) ===" % tot)
# data 필드
print("\n[data 필드 (id/amount/date) — literal 그라운딩 유효]")
for k in sorted(cnt):
    if k.startswith("data|"): print("  %-28s %5d" % (k, cnt[k]))
print("\n[enum 필드 (NL→정규화 필요) — literal=schema 오염 주의]")
for k in sorted(cnt):
    if k.startswith("enum|"): print("  %-28s %5d" % (k, cnt[k]))
print("\n[기타]")
for k in ("compute-like(결정론·규칙추가)", "bool/빈값"):
    print("  %-28s %5d" % (k, cnt[k]))
# 요약
data_user = sum(v for k, v in cnt.items() if k.startswith("data|") and "USER=True" in k)
data_tot = sum(v for k, v in cnt.items() if k.startswith("data|"))
enum_user = sum(v for k, v in cnt.items() if k.startswith("enum|") and "USER=True" in k)
enum_tot = sum(v for k, v in cnt.items() if k.startswith("enum|"))
print("\n★ data 필드 USER-명시(conserv 그라운딩): %d/%d" % (data_user, data_tot))
print("★ enum 필드 USER-명시: %d/%d (나머지=NL→정규화=의미매핑 필요·literal 상한 오염)" % (enum_user, enum_tot))
