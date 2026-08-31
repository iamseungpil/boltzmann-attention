# -*- coding: utf-8 -*-
"""x611 — t7391_reg12 task 12(retail) 게이트 격리 재현 ([[78]] 격리→배선 규율).

재료는 (a) 궤적 축자 (b) `a2/retail.*.json` 선언에서만 읽는다.
프롬프트 저작 0 · 모델 호출 0 · gold 무참조(진단 인용은 보고서에 별도 표기).

물음 ①: msg 12/14/16 의 `return_delivered_order_items` 시점에서 `GateInterpreter(retail gates)`
        는 무엇을 답하는가? — 특히 `G2_CONFIRM_WRITE`.
물음 ②: 리졸버가 **살아 있을 때**와 **죽어 있을 때** 게이트 답이 갈리는가?
        (라이브 t7391 은 G1/G3/G5/G6/G7 마커가 전 sim 0 = 리졸버 死 가설)

반증 조건: ①에서 allowed=False 가 나오면 "게이트가 'Sure' 로 열렸다"는 주장은 거짓이다.
"""
import gzip, json, sys

TAU2 = r"C:\workspace\ba-frft\scripts\distill\tau2"
sys.path.insert(0, TAU2)
sys.stdout.reconfigure(encoding="utf-8")

from gate_interpreter import GateInterpreter, load_domain_a2, CONFIRM_RE

RES = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\t7391_reg12.results.json.gz"
d = json.load(gzip.open(RES, "rt", encoding="utf-8"))
sim = [x for x in d["simulations"] if str(x.get("task_id")) == "12"][0]
MS = sim["messages"]

# ── 궤적이 실제로 반환한 read 결과만으로 resolver 구성 (msg 12 write **이전** 상태만) ──
RECORDS, USER = {}, None
for m in MS[:12]:
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
print("gates(우선순위 정렬 전):", [g["id"] for g in GATES])
print("읽힌 레코드:", sorted(RECORDS), "· user.orders =", (USER or {}).get("orders"))

# ── 궤적 축자: write 시점의 last_user_msg 는 msg[3] 하나뿐(4~17 에 user 턴 없음) ──
U1, U3, U19 = MS[1]["content"], MS[3]["content"], MS[19]["content"]
assert not any(m.get("role") == "user" for m in MS[4:12]), "msg4~11 에 user 턴이 있으면 전제가 깨진다"

WRITES = [
    (12, {"order_id": "#W5490111",
          "item_ids": ["4579334072", "1421289881", "4947717507"],
          "payment_method_id": "paypal_9497703"}),
    (14, {"order_id": "#W7387996", "item_ids": ["5796612084"],
          "payment_method_id": "paypal_9497703"}),
    (16, {"order_id": "#W5490111",
          "item_ids": ["4579334072", "1421289881", "4947717507"],
          "payment_method_id": "credit_card_3124723"}),
]

print("\n===== ① 라이브 재현 — last_user = msg[3] (인증 턴) =====")
print("msg[3] 축자:", repr(U3))
m = CONFIRM_RE.search(U3)
print("CONFIRM_RE.search(msg3) →", (m.group(0), m.span()) if m else None)
for label, resolvers in [("resolvers=살아있음", LIVE_RESOLVERS), ("resolvers=죽음(빈 dict)", {})]:
    gi = GateInterpreter(GATES, resolvers=resolvers)
    gi.auth_user = "mia_garcia_4516"          # msg 4/5 find_user_id_by_email 성공
    print("\n-- %s" % label)
    for i, args in WRITES:
        ok, gid, why = gi.check("return_delivered_order_items", args,
                                last_user_msg=U3, transfer_msg_sent=None)
        print("   msg%-3d allowed=%-5s gate=%s" % (i, ok, gid))
        if why:
            print("          why=%s" % why[:120])

print("\n===== ② 반사실 — last_user 를 msg[1](최초 요청)로 바꾸면 =====")
m1 = CONFIRM_RE.search(U1)
print("CONFIRM_RE.search(msg1) →", m1.group(0) if m1 else None)
gi = GateInterpreter(GATES, resolvers={})
gi.auth_user = "mia_garcia_4516"
ok, gid, why = gi.check("return_delivered_order_items", WRITES[0][1],
                        last_user_msg=U1, transfer_msg_sent=None)
print("   allowed=%s gate=%s" % (ok, gid))
print("   why=%s" % (why or "")[:200])

print("\n===== ③ msg[19] — 손님이 **거부**한 발화도 게이트를 여는가 =====")
m19 = CONFIRM_RE.search(U19)
print("msg[19] 앞 60자:", repr(U19[:60]))
print("CONFIRM_RE.search(msg19) →", (m19.group(0), m19.span()) if m19 else None)
gi = GateInterpreter(GATES, resolvers={})
gi.auth_user = "mia_garcia_4516"
print("   allowed=%s" % gi.check("return_delivered_order_items", WRITES[0][1],
                                 last_user_msg=U19, transfer_msg_sent=None)[0])

print("\n===== ④ msg[20] 이 진짜 tool_call 이었다면 — G4_TRANSFER_MSG =====")
gi = GateInterpreter(GATES, resolvers={})
gi.auth_user = "mia_garcia_4516"
ok, gid, why = gi.check("transfer_to_human_agents", {"summary": "…"},
                        last_user_msg=U19, transfer_msg_sent=False)
print("   allowed=%s gate=%s" % (ok, gid))
print("   why=%s" % (why or "")[:220])
