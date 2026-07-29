#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C214(day9 재발사 전 보강 E1~E4) 오프라인 검증. ⚠단위통과≠라이브발화([[30]])."""
import io, json, os, sys, inspect
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import t2_gate_patch as GP
OK = True
def chk(c, m):
    global OK; OK &= bool(c); print(("  ✓ " if c else "  ✗ ") + m)
class M:
    def __init__(self, role, content=None, tool_calls=None, mid=None, error=False):
        self.role, self.content, self.tool_calls, self.id, self.error = role, content, tool_calls, mid, error
class T:
    def __init__(self, name, cid="c", args=None, requestor="assistant"):
        self.name, self.id, self.arguments, self.requestor = name, cid, (args or {}), requestor

FIT_UNV = ("{'eligible': [...], 'unverified': [{'card': 'Silver Rewards Card', "
           "'undocumented': ['fx_fee depends on premium_subscriber — confirm it']}]}")
FIT_OK = "{'eligible': [...], 'unverified': []}"

def test_e1():
    print("[E1] unverified 재호출 미이행 검출")
    c1 = M("assistant", tool_calls=[T("check_card_application_fit", "f1")])
    r1 = M("tool", FIT_UNV, mid="f1")
    p = GP._unverified_pending([c1, r1])
    chk(p is not None and p[0] == "check_card_application_fit", "unverified 행 → pending")
    chk(p and "Silver Rewards Card" in p[1], "요약에 미검증 항목 포함")
    c2 = M("assistant", tool_calls=[T("check_card_application_fit", "f2")])
    chk(GP._unverified_pending([c1, r1, c2, M("tool", FIT_OK, mid="f2")]) is None,
        "같은 도구 재호출 → 해소(003 정답 수순)")
    chk(GP._unverified_pending([c1, M("tool", FIT_OK, mid="f1")]) is None, "unverified 빈 목록 → 미발화")

def test_sites():
    print("[E1/E2/E3] 발화 지점·비강제 확인")
    s = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    for flag, mark in [("T2_UNVERIFIED_FOLLOWUP", "T2_UNVERIFIED_FU] fired"),
                       ("T2_GIVE_EXEC_NUDGE", "T2_GIVE_EXEC] nudge"),
                       ("T2_SEARCH_EXHAUST_NUDGE", "T2_SEARCH_EXHAUST] nudge")]:
        chk(flag in s and mark in s, "%s 분기·마크 실재" % flag)
    for tag in ('"unverifiedfu")', '"givexec")', '"searchexhaust")'):
        seg = s[max(0, s.index(tag) - 1200):s.index(tag) + 40]
        chk("tool_choice" not in seg.split("_ap_regen")[-1], "%s = 비강제 넛지(경계정본 §3-2)" % tag)
    chk('_given - _ran' in s, "E2 술어=성사 give − user 실행분")
    chk('"[DUPLICATE-READ]" in _m6.content' in s, "E3 술어=엔진 자기 스텁 계수")
    chk("chain suppressed" in s, "E4 진단 마크(체인 억제 사유 계측)")

def test_e4_a2():
    print("[E4] A2 dispute→update 체인 임계")
    A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
    ch = [c for c in A2["follow_up_chains"] if c.get("after") == ["submit_cash_back_dispute"]]
    chk(len(ch) == 1, "체인 1건 실재(중복 추가 없음)")
    chk(ch[0].get("resign_th") == 1, "resign_th=1 하향")
    chk("update_transaction_rewards" in ch[0]["requires"], "requires 보존")
    chk("do not update" in ch[0]["feedback"], "양방향 문구 보존(조기 갱신 Δspurious 방지)")

def test_flags():
    print("[flags] go_stack C214")
    s = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()
    for f in ("T2_UNVERIFIED_FOLLOWUP=1", "T2_GIVE_EXEC_NUDGE=1", "T2_SEARCH_EXHAUST_NUDGE=1"):
        chk(f in s, f)

if __name__ == "__main__":
    for fn in (test_e1, test_sites, test_e4_a2, test_flags):
        fn(); print()
    print("RESULT: %s" % ("ALL PASS" if OK else "FAIL"))
    sys.exit(0 if OK else 1)
