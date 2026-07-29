#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C213/day9 처방(G1·W1·N1·T1) 오프라인 검증. 경계정본 §4·DAY9 설계서 표.
⚠단위통과≠라이브발화([[30]])."""
import io, json, os, sys, inspect
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
import gate_interpreter as GI
import t2_gate_patch as GP
import t2_eplan_patch as EP
import t2_compliance as TC2
OK = True
def chk(c, m):
    global OK; OK &= bool(c); print(("  ✓ " if c else "  ✗ ") + m)

NT = "TRANSFER NOTICE: I have checked the knowledge base and there are no further actions I can take for this request. Would you like to be transferred to a human agent?"

def test_g1():
    print("[G1] notice 공용 정규화 술어")
    chk(GI.notice_sent_in([NT], NT) is True, "정본 문구 그대로 → 인정")
    sofia = NT.replace("human agent?", "human agent, Sofia?")
    chk(GI.notice_sent_in([sofia], NT) is True, "', Sofia' 개인화 변형 → 인정(032 [S] 직격)")
    chk(GI.notice_sent_in(["  transfer   NOTICE:  I have checked the knowledge base and there are no further actions I can take for this request..."], NT) is True, "공백/대소문 변형 → 인정")
    chk(GI.notice_sent_in(["I will transfer you to a human agent now."], NT) is False, "무관 발화 → 불인정(오인정 0)")
    chk(GI.notice_sent_in(["x"], "") is None, "notice_text 부재 → None")
    src = inspect.getsource(GI.notice_norm)
    chk("re.sub" in src and "similar" not in src.lower(), "닫힌 연산만(유사도 없음)")
    for mod, fn in [(GP, "_transfer_msg_sent"), (GP, "_regen_transfer_sent")]:
        chk("notice_sent_in" in inspect.getsource(getattr(mod, fn)), f"{fn} 공용 술어 배선")
    chk("notice_sent_in" in inspect.getsource(EP._terminal_grant_check), "EPLAN ⓐ 공용 술어 배선(48자 원시-prefix 제거)")
    chk("notice_sent_in" in io.open(os.path.join(HERE, "t2_compliance.py"), encoding="utf-8").read(), "compliance 측정층 배선(스코프 (a))")

def test_w1():
    print("[W1] walk 강제-보류 강등")
    esrc = io.open(os.path.join(HERE, "t2_eplan_patch.py"), encoding="utf-8").read()
    chk("T2_EPLAN_WALK_HOLD" in esrc, "보류=opt-in 플래그화(기본 표면화만)")
    chk("walk gap surfaced only" in esrc, "gap 표면화 마크 실재")
    chk('requestor", "assistant") == "user"' in esrc, "user-실행 write exec 가산(001 오계상 교정)")
    i_soft = esrc.index("walk gap surfaced only"); i_hold = esrc.index("drive_decision(_drives")
    chk(i_soft < i_hold, "표면화 분기가 보류 결정보다 선행")

def test_n1():
    print("[N1] 무관-give 넛지")
    gsrc = io.open(os.path.join(HERE, "t2_gate_patch.py"), encoding="utf-8").read()
    chk("T2_GIVE_RELEVANCE_NUDGE" in gsrc, "플래그 분기 실재")
    chk("T2_GIVE_RELEVANCE] nudge" in gsrc, "마크 실재(정당/무관 분류 계측 anchor)")
    seg = gsrc[gsrc.index("T2_GIVE_RELEVANCE_NUDGE"):gsrc.index("T2_GIVE_RELEVANCE_NUDGE")+3000]
    chk('"giverel")' in seg and "tool_choice" not in seg.split('"giverel")')[0].split("_ap_regen")[-1], "넛지=비강제 regen(tool_choice 없음)")
    chk("give it again" in seg, "정당 선제-give 재발행 허용 문구(오탐 안전변)")
    # rev1(day9 스모크 오탐 [S]): 선행 성공 give 존재할 때만 발화
    chk("_prior_gives" in seg, "선행-give 수집 로직 실재")
    chk("(_prior_gives - {_tgt})" in seg, "술어=원장 미등장 ∧ 다른 give 기성사(단독 give 미발화)")
    i_led = seg.index("_ledger_txt = "); i_pri = seg.index("_prior_gives, _gid2n")
    chk(i_led < i_pri < seg.index("for _tc3 in"), "술어 계산이 발화 판정보다 선행")
    chk('getattr(_m3, "error", False)' in seg, "성사=비-에러 tool 결과만 계상")

def test_t1():
    print("[T1] 접지 선행 확인(코드 변경 0)")
    A2 = json.load(io.open(os.path.join(HERE, "a2", "banking_knowledge.gate.json"), encoding="utf-8"))
    reb = next(t for t in A2["scaffold_get_tools"] if "rebate" in t["name"])
    g = (reb.get("ground") or {}).get("scalar_fields") or []
    chk(any(f.get("param") == "monthly_threshold" and "kb" in (f.get("corpus") or []) for f in g),
        "monthly_threshold ground 기선언 확인(편입 불요·존재-검사 한계는 §2 명기)")

def test_flags():
    print("[flags] go_stack C213")
    s = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()
    chk("T2_GIVE_RELEVANCE_NUDGE=1" in s, "N1 ON")
    chk("T2_EPLAN_WALK_HOLD=1" not in s.replace("T2_EPLAN_WALK_HOLD=1로만", "").replace("T2_EPLAN_WALK_HOLD=1(격리", ""), "W1 보류 기본 OFF(주석 외 미설정)")

if __name__ == "__main__":
    for fn in (test_g1, test_w1, test_n1, test_t1, test_flags):
        fn(); print()
    print("RESULT: %s" % ("ALL PASS" if OK else "FAIL"))
    sys.exit(0 if OK else 1)
