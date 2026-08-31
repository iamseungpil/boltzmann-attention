# -*- coding: utf-8 -*-
"""x613 — t7391_reg12 task 24(retail) 게이트·후보요약 격리 재현 ([[78]] 격리→배선 규율).

재료는 (a) 궤적 축자 (b) `a2/retail.gate.json` 선언에서만 읽는다.
프롬프트 저작 0 · 모델 호출 0 · gold 무참조(진단 인용은 보고서에 별도 표기).

물음 ①: msg 13 `cancel_pending_order(#W3561391)` 시점에서 retail 게이트는 무엇을 답하는가?
물음 ②: 반사실 — `last_user` 가 msg[1](최초 요청)이었다면?
물음 ③: 반사실 — `last_user` 가 msg[16](손님의 **후회** 발화)이었다면? (사후 반증용)
물음 ④: 리졸버가 살아 있으면 G5_STATUS_PRECONDITION·G6_SELECT_CONFIRM 이 이 write 를 잡는가?
물음 ⑤: `T2_PRESENT_READS=1` 이었다면 msg 7 의 `get_user_details` 꼬리에 무엇이 붙었을 것인가?
        (재료 가용성 확인 — 모델 반응은 이 프로브가 답하지 못한다)

반증 조건:
  ①에서 allowed=False 가 나오면 "게이트가 'Sure' 로 열렸다"는 주장은 거짓이다.
  ⑤에서 None 이 나오면 "후보요약 재료가 있었다"는 주장은 거짓이다.
"""
import gzip, json, sys

TAU2 = r"C:\workspace\ba-frft\scripts\distill\tau2"
sys.path.insert(0, TAU2)
sys.stdout.reconfigure(encoding="utf-8")

from gate_interpreter import (GateInterpreter, load_domain_a2, CONFIRM_RE,
                              candidate_summary, nested_candidate_summary)

RES = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\t7391_reg12.results.json.gz"
d = json.load(gzip.open(RES, "rt", encoding="utf-8"))
sim = [x for x in d["simulations"] if str(x.get("task_id")) == "24"][0]
MS = sim["messages"]

# ── 궤적이 실제로 반환한 read 결과만으로 resolver 구성 (msg 13 write **이전** 상태만) ──
RECORDS, USER = {}, None
for m in MS[:13]:
    if m.get("role") != "tool":
        continue
    try:
        o = json.loads(m.get("content") or "")
    except Exception:
        continue
    if isinstance(o, dict) and o.get("order_id"):
        RECORDS[o["order_id"]] = o
    if isinstance(o, dict) and o.get("user_id") and o.get("orders"):
        USER = o


def resolve_field(path, args):
    target_arg, producer, field = path[0], path[1], path[2]
    val = args.get(target_arg)
    if not val:
        return None
    if producer == "get_order_details":
        return (RECORDS.get(val) or {}).get(field)
    if producer == "get_user_details":
        return (USER or {}).get(field)
    return None


def fetch_record(producer, id_arg, id_val):
    return RECORDS.get(id_val) if producer == "get_order_details" else None


LIVE_RESOLVERS = {"resolve_field": resolve_field, "fetch_record": fetch_record,
                  "resolve_owner": resolve_field}

A2 = load_domain_a2("retail")
GATES = A2["gates"]
print("gates:", [g["id"] for g in GATES])
print("읽힌 레코드:", sorted(RECORDS), "· user.orders =", (USER or {}).get("orders"))

U1, U3, U16 = MS[1]["content"], MS[3]["content"], MS[16]["content"]
assert not any(m.get("role") == "user" for m in MS[4:13]), "msg4~12 에 user 턴이 있으면 전제가 깨진다"

WRITE = ("cancel_pending_order", {"order_id": "#W3561391", "reason": "no longer needed"})

print("\n===== ① 라이브 재현 — last_user = msg[3] (인증 턴) =====")
print("msg[3] 축자:", repr(U3))
m = CONFIRM_RE.search(U3)
print("CONFIRM_RE.search(msg3) →", (m.group(0), m.span()) if m else None)
for label, resolvers in [("resolvers=살아있음", LIVE_RESOLVERS), ("resolvers=죽음(빈 dict)", {})]:
    gi = GateInterpreter(GATES, resolvers=resolvers)
    gi.auth_user = "sofia_hernandez_5364"      # msg 4/5 find_user_id_by_name_zip 성공
    ok, gid, why = gi.check(WRITE[0], WRITE[1], last_user_msg=U3, transfer_msg_sent=None)
    print("-- %-22s msg13 allowed=%-5s gate=%s" % (label, ok, gid))
    if why:
        print("     why=%s" % why[:160])

print("\n===== ② 반사실 — last_user = msg[1] (최초 취소 요청) =====")
m1 = CONFIRM_RE.search(U1)
print("msg[1] 축자:", repr(U1))
print("CONFIRM_RE.search(msg1) →", (m1.group(0), m1.span()) if m1 else None)
for label, resolvers in [("resolvers=살아있음", LIVE_RESOLVERS), ("resolvers=죽음", {})]:
    gi = GateInterpreter(GATES, resolvers=resolvers)
    gi.auth_user = "sofia_hernandez_5364"
    ok, gid, why = gi.check(WRITE[0], WRITE[1], last_user_msg=U1, transfer_msg_sent=None)
    print("-- %-22s allowed=%-5s gate=%s" % (label, ok, gid))
    if why:
        print("     why=%s" % why[:300])

print("\n===== ③ 반사실 — last_user = msg[16] (손님의 후회 발화) =====")
m16 = CONFIRM_RE.search(U16)
print("msg[16] 앞 90자:", repr(U16[:90]))
print("CONFIRM_RE.search(msg16) →", (m16.group(0), m16.span()) if m16 else None)
gi = GateInterpreter(GATES, resolvers={})
gi.auth_user = "sofia_hernandez_5364"
print("   allowed=%s gate=%s" % gi.check(WRITE[0], WRITE[1],
                                         last_user_msg=U16, transfer_msg_sent=None)[:2])

print("\n===== ④ G5 전제조건 — #W3561391 의 status 는 무엇인가 =====")
print("   resolve_field(status) =", resolve_field(["order_id", "get_order_details", "status"],
                                                  WRITE[1]))
print("   (G5 allow=['pending'] → 상태 축으로는 이 write 가 **적법**하다. 즉 G5 는 이 칸을 못 막는다)")

print("\n===== ⑤ T2_PRESENT_READS=1 이었다면 msg 7 꼬리에 붙었을 문면 =====")
g6 = [g for g in GATES if g["id"] == "G6_SELECT_CONFIRM"][0]
cs = candidate_summary(LIVE_RESOLVERS, g6, "sofia_hernandez_5364")
print("   candidate_summary is None?", cs is None)
if cs:
    print(cs[:1600])
    print("   ...(전체 %d자)" % len(cs))
    # 닫힌 술어: 각 주문의 T-Shirt 품목 수 — 요약 문면만으로 셀 수 있는가
    print("\n   [닫힌 술어 검산] 주문별 T-Shirt 라인 수 (요약에 실린 items 로만 계산):")
    for oid in (USER or {}).get("orders", []):
        rec = RECORDS.get(oid) or {}
        n = sum(1 for it in (rec.get("items") or []) if it.get("name") == "T-Shirt")
        mats = [it.get("options", {}).get("material") for it in (rec.get("items") or [])
                if it.get("name") == "T-Shirt"]
        print("     %s  T-Shirt=%d  materials=%s" % (oid, n, mats))

print("\n===== ⑥ T2_PRESENT_NESTED=1 이었다면 msg 11(#W9609649) 꼬리에 붙었을 문면 =====")
spec = [s for s in A2.get("present_specs", []) if s.get("trigger_tool") == "get_order_details"][0]
ns = nested_candidate_summary(RECORDS.get("#W9609649"), spec)
print("   nested_candidate_summary is None?", ns is None)
if ns:
    print(ns[:1400])

print("\n===== ⑦ calc_specs 에 '주문별 품목명 개수' 스펙이 있는가 =====")
for c in A2.get("calc_specs", []):
    print("   trigger=%-22s op=%-14s nested=%-10s cond=%s label=%s"
          % (c.get("trigger_tool"), c.get("op"), c.get("nested_field"),
             c.get("cond_field"), c.get("label")))
print("   → get_order_details 위의 op 는 'sum(price)' 하나뿐 · count_where 는 "
      "get_product_details/variants 에만 선언 = **선언 결손**")
