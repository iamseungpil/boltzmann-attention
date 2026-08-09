# -*- coding: utf-8 -*-
"""정박 슬롯 회귀 — 결정 블록 조립과 D1c 불일치 탐지가 규격서대로인지 (순수함수·GPU 0).

정본: `reports/facet_rft_2026/ANCHOR_SLOT_SPEC_2026_08_09.md` §3 규칙 · §4 게이트.
근거 원장: C367(파레토 지배) · C370(`B_rank`) · C371(블록 사다리) · C372(D1c·부정통제).

실행: py -3 test_anchor_slot.py
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_ledger as LG                                          # noqa: E402
from gate_interpreter import load_domain_a2                      # noqa: E402

FAIL = []


def ok(cond, label, extra=""):
    print(("  OK   " if cond else "  FAIL ") + label + (("  — " + str(extra)) if extra else ""))
    if not cond:
        FAIL.append(label)


# ── 합성 고정물 (도메인 무관·엔진만 시험한다) ────────────────────────────────
ROWS = [("Alpha", ["gain=100", "floor=10"]),
        ("Bravo", ["gain=300", "floor=90"]),
        ("Charlie", ["gain=200", "floor=50"])]
GAIN = {"Alpha": ("100", "docA"), "Bravo": ("300", "docB"), "Charlie": ("200", "docC")}
SPEC = {"decided_text": "\nIt answers: {choice}.\nfigures: {operands}\nrow: {row}\nnext: {runners}",
        "reask_prompt": "has {axis}={chosen}, highest {axis} is {best}. Answer again."}

print("\n§1 결정 블록 조립 (decided_text)")
blk = LG.decided_text(SPEC, "Bravo", ROWS, "deposit = 30000", "gain", GAIN)
ok("It answers: Bravo." in blk, "지목이 한 번 나온다")
ok(blk.count("It answers:") == 1, "지목 예산 ≤ 1 (R1)")
ok("deposit = 30000" in blk, "피연산자가 실린다")
ok("Bravo: gain=300, floor=90" in blk, "선택된 행이 표 생성기와 같은 형식으로 실린다")
ok("Charlie (gain=200)" in blk and "Alpha (gain=100)" in blk, "순위가 내림차순으로 실린다")
ok("Bravo" not in blk.split("next:")[1], "순위 목록에서 지목 자신은 빠진다")

print("\n§2 축이 없으면 순위 없이 나간다 (모르는 것을 지어내지 않는다)")
blk2 = LG.decided_text(SPEC, "Bravo", ROWS, "x = 1", None, None)
ok("next: " in blk2 and blk2.rstrip().endswith("next:"), "runners 가 빈다", repr(blk2[-24:]))
ok("row: Bravo" in blk2, "선택된 행은 축 없이도 실린다")

print("\n§3 선언이 없으면 블록을 안 만든다 (침묵)")
ok(LG.decided_text({}, "Bravo", ROWS, "", "gain", GAIN) == "", "decided_text 미선언 → 빈 문자열")
ok(LG.decided_text(SPEC, None, ROWS, "", "gain", GAIN) == "", "지목 없음 → 빈 문자열")

print("\n§4 D1c — 불일치 탐지는 값만 돌려준다 (이름 반환 금지·[[05]] Q2)")
mm = LG.mismatch_value(ROWS, GAIN, "Alpha")
ok(mm == (100.0, 300.0), "낮은 것을 고르면 (고른 값, 최댓값)", mm)
ok(all(not isinstance(v, str) for v in (mm or ())), "반환에 문자열(=이름)이 없다")
ok(LG.mismatch_value(ROWS, GAIN, "Bravo") is None, "최댓값을 고르면 None(재질의 안 함)")
ok(LG.mismatch_value(ROWS, GAIN, "Zulu") is None, "목록 밖 이름 → None(판정 불가)")
ok(LG.mismatch_value([], GAIN, "Alpha") is None, "빈 통과 집합 → None")
ok(LG.mismatch_value(ROWS, {}, "Alpha") is None, "축 값이 없으면 → None")

print("\n§5 재질의 문구에 이름이 들어가지 않는다 (D1b 보존)")
rq = SPEC["reask_prompt"].format(axis="gain", chosen=LG._num(mm[0]), best=LG._num(mm[1]))
ok(not any(n in rq for n, _b in ROWS), "후보 이름이 하나도 안 들어간다", rq)
ok("300" in rq and "100" in rq, "값은 들어간다")

print("\n§6 통과 집합의 구조체 반환이 문자열판과 일치한다 (파싱 제거·[[59]])")
a2 = load_domain_a2("banking_knowledge")
spec = next(s for s in a2["ledger_metrics"] if s.get("eligible_text"))
import t2_factdag as FD                                          # noqa: E402
maps = {ax: FD._a3_map(a2["policy_ontology"]["rows"], {"axis": ax})
        for ax in spec["eligible"]["show_axes"]}
txt = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000})
rws = LG.eligible_text(730, {}, maps, spec, {"qualifying_deposit_usd": 30000}, as_rows=True)
body = [l for l in txt.splitlines() if l.startswith("  ") and ":" in l]
ok(len(body) == len(rws), "행 수가 같다", "%d vs %d" % (len(body), len(rws)))
ok(all(LG._row_line(s, b) == l for (s, b), l in zip(rws, body)), "줄 생성이 유일 지점을 지난다")

print("\n§7 A2 정본·gate 두 층이 같은 템플릿을 갖는다 ([[24]])")
HERE = os.path.dirname(os.path.abspath(__file__))


def grab(fn):
    d = json.load(io.open(os.path.join(HERE, "a2", fn), encoding="utf-8"))
    for x in d.get("ledger_metrics", []):
        if x.get("decided_text"):
            return tuple(x.get(k) for k in ("decided_text", "reask_prompt",
                                            "objective_axis_prompt"))
    return None


g1, g2 = grab("banking_knowledge.settings.json"), grab("banking_knowledge.gate.json")
ok(g1 is not None, "정본(settings)에 선언이 있다")
ok(g1 == g2, "정본과 gate 가 바이트 동일")
ok(all(v for v in (g1 or ())), "세 템플릿이 다 비어 있지 않다")

print("\n§8 라이브 spec 이 그 템플릿을 실제로 병합해 들고 있다 (死코드 방지)")
ok(bool(spec.get("decided_text")), "병합된 spec 에 decided_text 가 있다")
ok(bool(spec.get("reask_prompt")), "병합된 spec 에 reask_prompt 가 있다")
ok(bool(spec.get("objective_axis_prompt")), "병합된 spec 에 objective_axis_prompt 가 있다")
ok("{choice}" in spec["decided_text"] and "{runners}" in spec["decided_text"],
   "decided_text 에 필요한 자리표시자가 있다")
ok(all(("{%s}" % k) in spec["reask_prompt"] for k in ("axis", "chosen", "best")),
   "reask_prompt 에 필요한 자리표시자가 있다")

print("\n" + ("FAIL %d: %s" % (len(FAIL), FAIL) if FAIL else "PASS  (0 실패)"))
sys.exit(1 if FAIL else 0)
