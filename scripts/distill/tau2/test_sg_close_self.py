# -*- coding: utf-8 -*-
r"""`T2_SG_CLOSE_SELF` 래칫 — 마감 문면과 배선을 **초 단위**로 고정 (모델 0 · 무료).

## 무엇을 고정하나

1. **꺼져 있으면 문면이 종전과 바이트 동일**하다(거동 불변 — 이 검정이 그것의 유일한 증거다).
2. 켜면 문면이 **격리가 이긴 형태**다 — `instructions → 원장 → 필드계약 → 형식`.
   그 순서는 `x525`(072 chk_538bfb9cba 인출 10 · `N_wire` 9/10 ↔ `Q_wirefresh` **10/10** · n=4)
   가 고른 것이고, 여기 적힌 순서가 그 팔과 다르면 **격리와 라이브가 갈린다**([[78]]).
3. 엔진이 **새 문장을 쓰지 않는다** — 선언(`instructions`·`answer_format`)과 넘어온 두 토막뿐.
4. 소스에 **배선 두 갈래가 실재**한다(`msgs = [_um2]` ↔ `list(msgs) + [_um2]`) — 死배선 방지([[24]]).

사용: PYTHONPATH=. py -3 test_sg_close_self.py
"""
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_scaffold_get as SG                                        # noqa: E402

FAIL = []
ISO = {"instructions": "INSTR-TEXT", "answer_format": "FMT-TEXT"}
PB = "\ntransactions: PARAMS-TEXT"
RECS = "Record ID: r1\ntype: atm_withdrawal"


def chk(cond, what, extra=""):
    print("  %-4s %s%s" % ("ok" if cond else "FAIL", what, ("  %s" % extra) if extra else ""))
    if not cond:
        FAIL.append(what)


print("① 꺼져 있으면 종전 문면과 바이트 동일")
old = ("=== FIELD CONTRACT ===" + PB + "\n\n"
       + "=== RECORDS ===\n" + RECS + "\n\n"
       + ISO["answer_format"])
got = SG.close_text(ISO, PB, RECS, False)
chk(got == old, "구판 문면 재현(거동 불변)", repr(got[:40]))
chk("INSTR-TEXT" not in got, "구판에는 instructions 가 **없다**(그것이 결손이었다)")

print("")
print("② 켜면 격리가 이긴 순서")
new = SG.close_text(ISO, PB, RECS, True)
chk(new.startswith("INSTR-TEXT"), "instructions 가 **맨 앞**")
chk(new.index("=== RECORDS ===") < new.index("=== FIELD CONTRACT ==="),
    "원장이 **필드 계약보다 앞**")
chk(new.rstrip().endswith("FMT-TEXT"), "형식이 **맨 뒤**")
chk(new == ("INSTR-TEXT\n\n=== RECORDS ===\n" + RECS + "\n\n"
            + "=== FIELD CONTRACT ===" + PB + "\n\n" + "FMT-TEXT"),
    "x525 `Q_wirefresh` 와 **바이트 동일**", repr(new[:60]))

print("")
print("③ 엔진이 새 문장을 쓰지 않는다")
resid = new
for tok in ("INSTR-TEXT", "FMT-TEXT", PB, RECS,
            "=== RECORDS ===", "=== FIELD CONTRACT ==="):
    resid = resid.replace(tok, "")
chk(resid.strip() == "", "선언·머리말·넘어온 토막 말고 남는 글자가 없다", repr(resid[:40]))

print("")
print("④ 빈 칸에서 안 죽는다")
chk(SG.close_text(ISO, "", "", True).strip() != "", "계약·원장이 비어도 문면이 난다")
chk(SG.close_text({}, PB, RECS, True).strip() != "", "선언이 비어도 안 죽는다")

print("")
print("⑤ 배선 두 갈래가 소스에 실재한다([[24]] 死배선 방지)")
src = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
chk('os.environ.get("T2_SG_CLOSE_SELF") == "1"' in src, "플래그 검사가 소스에 있다")
chk("msgs = [_um2] if _self2 else (list(msgs) + [_um2])" in src,
    "켜면 앞 턴을 버리고, 꺼지면 종전대로 덧붙인다")
chk("close_text(iso, _pb2, _recs2, _self2)" in src, "마감 문면이 이 순수함수를 지난다")
dec = io.open(os.path.join(HERE, "go_stack.sh"), encoding="utf-8").read()
chk("export T2_SG_CLOSE_SELF=" in dec, "go_stack 에 선언돼 있다(test_flag_registry)")

print("")
print("RESULT: %s" % ("PASS" if not FAIL else "FAIL (%d) %s" % (len(FAIL), FAIL[:3])))
sys.exit(0 if not FAIL else 1)
