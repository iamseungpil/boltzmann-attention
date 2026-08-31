# -*- coding: utf-8 -*-
"""x603 — t7391_reg12 task 9 게이트 격리 재현 ([[78]] 격리→배선 규율).

재료는 전부 (a) 궤적 축자 (b) `a2/retail.settings.json` 선언에서만 읽는다.
프롬프트 저작 0 · 모델 호출 0 · gold 무참조(진단 인용은 별도 표기).

물음: msg 15 의 `exchange_delivered_order_items` 호출 시점에서
      `GateInterpreter(retail gates)` 는 무엇을 답하는가?
"""
import gzip, json, os, sys
TAU2 = r"C:\workspace\ba-frft\scripts\distill\tau2"
sys.path.insert(0, TAU2)
sys.stdout.reconfigure(encoding="utf-8")

from gate_interpreter import GateInterpreter, load_domain_a2, CONFIRM_RE

RES = r"C:\workspace\ba-frft\reports\facet_rft_2026\sim_results\t7391_reg12.results.json.gz"
d = json.load(gzip.open(RES, "rt", encoding="utf-8"))
sim = [x for x in d["simulations"] if str(x.get("task_id")) == "9"][0]
MS = sim["messages"]

# ── 궤적이 실제로 반환한 read 결과만으로 resolver 구성 (env 대체·읽기 전용) ──
RECORDS = {}          # order_id -> dict
USER = None
for i, m in enumerate(MS[:15]):     # ★msg15 write **이전** 상태만 (msg16 은 write 후 레코드)
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
        rec = RECORDS.get(val) or {}
        return rec.get(field) if field != "address1" else (rec.get("address") or {}).get("address1")
    if producer == "get_user_details":
        return (USER or {}).get(field)
    return None

def fetch_record(producer, id_arg, id_val):
    if producer == "get_order_details":
        return RECORDS.get(id_val)
    return None

RESOLVERS = {"resolve_field": resolve_field, "fetch_record": fetch_record,
             "resolve_owner": resolve_field}

A2 = load_domain_a2("retail")
GATES = A2["gates"]
print("gates(선언 순서):", [g["id"] for g in GATES])
gi = GateInterpreter(GATES, resolvers=RESOLVERS)
print("gates(엔진 정렬·_KIND_PRIORITY):", [g["id"] for g in gi.gates])
print()

# ── 궤적을 msg 15 직전까지 그대로 관찰 ────────────────────────────────
pending = {}
for i, m in enumerate(MS[:15]):
    if m.get("role") == "assistant":
        for tc in (m.get("tool_calls") or []):
            pending[tc.get("id")] = (tc.get("name"), tc.get("arguments") or {})
    elif m.get("role") == "tool":
        nm_args = pending.get(m.get("id"))
        if nm_args:
            gi.observe(nm_args[0], nm_args[1], m.get("content"), ok=True)
print("observe 후 auth_user =", repr(gi.auth_user))

# ── last_user_msg = 뒤에서 처음 만나는 user 메시지 (t2_gate_patch._regen_last_user 동형) ──
last_user, last_i = None, None
for j in range(14, -1, -1):
    if MS[j].get("role") == "user" and MS[j].get("content"):
        last_user, last_i = MS[j]["content"], j
        break
print("last_user_msg = msg[%d] %r" % (last_i, last_user))
mm = CONFIRM_RE.search(last_user or "")
print("CONFIRM_RE ->", bool(mm), (mm.group(0), mm.span()) if mm else "")
print()

call = (MS[15]["tool_calls"] or [])[0]
name, args = call["name"], call["arguments"]
print("검사 대상 호출:", name, json.dumps(args, ensure_ascii=False))
ok, gid, why = gi.check(name, args, last_user_msg=last_user, transfer_msg_sent=None)
print("check() ->", ok, gid)
print("reason:", (why or "")[:400])
print()

# ── 반증용: 확인 발화가 아니었다면? (극성만 뒤집은 동일 호출) ──
for probe in ["I'm not sure which email I used for the order.",
              "My name is Mei Kovacs, zip 28236."]:
    ok2, gid2, _ = gi2 = None, None, None
    g2 = GateInterpreter(GATES, resolvers=RESOLVERS)
    for i, m in enumerate(MS[:15]):
        if m.get("role") == "assistant":
            for tc in (m.get("tool_calls") or []):
                pending[tc.get("id")] = (tc.get("name"), tc.get("arguments") or {})
        elif m.get("role") == "tool":
            na = pending.get(m.get("id"))
            if na:
                g2.observe(na[0], na[1], m.get("content"), ok=True)
    ok2, gid2, why2 = g2.check(name, args, last_user_msg=probe, transfer_msg_sent=None)
    print("[반증] last_user=%r -> allowed=%s gate=%s" % (probe[:45], ok2, gid2))

# ── §2. 프롬프트 토큰 회계 (턴-국소 피드백 버퍼 지문 · TASK_1 §3⒞ 동형) ──────────
print()
print("=== 프롬프트 토큰 회계 ===")
def _chars(m):
    n = len(m.get("content") or "")
    for tc in (m.get("tool_calls") or []):
        n += len(json.dumps(tc.get("arguments"), ensure_ascii=False)) + len(tc.get("name") or "")
    return n
gens = [(i, (m.get("raw_data") or {}).get("usage", {}).get("prompt_tokens"))
        for i, m in enumerate(MS) if m.get("role") == "assistant" and m.get("raw_data")]
print("%-9s %-8s %-7s %-9s %-9s %s" % ("구간", "추가자수", "Δpt", "tok/자", "기대(0.45/0.30)", "잔차"))
for (a, pa), (b, pb) in zip(gens, gens[1:]):
    add = sum(_chars(MS[k]) for k in range(a, b))
    json_heavy = any(MS[k].get("role") == "tool" for k in range(a, b))
    exp = add * (0.45 if json_heavy else 0.30)
    print("%-9s %-8d %-7d %-9.3f %-9.0f %+d"
          % ("%d→%d" % (a, b), add, pb - pa, (pb - pa) / add if add else 0, exp, (pb - pa) - exp))
print("※ 0.45 tok/자 = 같은 sim 의 8→12 구간(주문 JSON 3,755자 → 1,695 tok)으로 보정한 값.")
g6 = next(g for g in GATES if g.get("kind") == "select_confirm")
_gi2 = GateInterpreter(GATES, resolvers=RESOLVERS); _gi2.auth_user = "mei_kovacs_8020"
_m6 = _gi2._present_candidates(g6)
print("※ G6 DISAMBIGUATION 메시지 실측 = %d 자 (≈%.0f tok @0.45) — 12→15 잔차 %+d 와 대조하라"
      % (len(_m6), len(_m6) * 0.45, 1203))
