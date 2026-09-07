# -*- coding: utf-8 -*-
"""F1 — follow_up 장전 조건이 「불렸나」가 아니라 「비지 않은 결과를 냈나」인지 (x808 §7-1).

실물 재현: `bank_p1_task_024` seed 626729 — 도구가 축자 "No transaction ... needs a dispute"
를 반환했는데 follow_up 이 "you found reward discrepancies" 를 발화해 give 가 나갔고
그 한 줄이 db_match 를 깼다. ⚠단위통과≠라이브발화([[30]]).
"""
import io, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_gate_patch as G

A2 = json.load(io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
DECL = [t for t in A2["scaffold_get_tools"] if t["name"] == "get_reward_discrepancies"][0]

class TC:
    def __init__(s, name, id): s.name, s.id, s.requestor = name, id, "assistant"
class M:
    def __init__(s, role, content="", id=None, tool_calls=None, error=False):
        s.role, s.content, s.id, s.tool_calls, s.error = role, content, id, tool_calls or [], error

EMPTY_BASE  = DECL["return_template_empty"]
EMPTY_RATE  = DECL["variants"]["ratefix"]["return_template_empty"]
NONEMPTY    = ("Transactions whose recorded reward does NOT match the expected reward "
               "(these require a cash back dispute): txn_a1, txn_b2")

def call(cid="c1"):
    return M("assistant", tool_calls=[TC("get_reward_discrepancies", cid)])

ok = True
def check(label, got, want):
    global ok
    if got != want: ok = False
    print("  %s %-56s got=%s want=%s" % ("ok  " if got == want else "!!  ", label, got, want))

print("### 기본 변이 (T2_A2_VARIANT 미설정)")
os.environ.pop("T2_A2_VARIANT", None)
check("빈 결과 → 장전 안 함",        G._sg_produced_findings([call(), M("tool", EMPTY_BASE, "c1")], DECL), False)
check("비지 않은 결과 → 장전",       G._sg_produced_findings([call(), M("tool", NONEMPTY,  "c1")], DECL), True)
check("호출만 있고 반환 없음 → 안 함", G._sg_produced_findings([call()], DECL), False)
check("에러 반환 → 안 함",           G._sg_produced_findings([call(), M("tool", NONEMPTY, "c1", error=True)], DECL), False)
check("빈 1 + 비지않음 1 → 장전",     G._sg_produced_findings(
        [call("c1"), M("tool", EMPTY_BASE, "c1"), call("c2"), M("tool", NONEMPTY, "c2")], DECL), True)
check("다른 도구의 출력은 무시",      G._sg_produced_findings(
        [M("assistant", tool_calls=[TC("verify_identity", "z1")]), M("tool", NONEMPTY, "z1")], DECL), False)

print("### ratefix 변이 (라이브가 쓰는 것)")
os.environ["T2_A2_VARIANT"] = "ratefix"
check("ratefix 빈 결과 → 장전 안 함", G._sg_produced_findings([call(), M("tool", EMPTY_RATE, "c1")], DECL), False)
check("ratefix 비지 않음 → 장전",     G._sg_produced_findings([call(), M("tool", NONEMPTY,   "c1")], DECL), True)
os.environ.pop("T2_A2_VARIANT", None)

print("### empty 선언이 없는 도구 = 종전 거동(항상 참)")
check("선언 없음 → True", G._sg_produced_findings(
        [M("assistant", tool_calls=[TC("x", "q1")]), M("tool", "whatever", "q1")],
        {"name": "x"}), True)

print()
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
