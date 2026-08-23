# -*- coding: utf-8 -*-
"""우리 층이 낸 문장을 정본 하나가 판정하는가 (R10 · 2026-08-24).

`t2_gap.py:59` 가 우리-층 문면 표지를 **다섯 문자열짜리 사본**으로 들고 있었다([[67]] 위반).
사본은 조용히 갈라진다 — t7346(2026-08-22·40 sim·tool 메시지 993)에서 실제로 **양쪽으로**
갈라져 있었다:

    사본만 잡던 것 47   (`could not be verified` 33 · `GROUNDING WARNING` 14)
    정본만 잡던 것 45   ← 사본이 놓치던 것. I3_RIVAL 단이 경쟁 문구가 있는데도 침묵했다
    둘 다 잡던 것 51   (`NOT_VERIFIED`)

그래서 `deny_kind` 로 그냥 갈아끼우지 않았다(그러면 47 을 잃는다 — 주석·통지는 거절이 아니다).
정본에 `ours_text` + `our_notice_ledger` 를 **추가**하고 사본을 지웠다.

이 검정이 잡는 것:
  ① 사본이 되살아나지 않는다(`t2_gap` 소스에 손 목록 0 · 정본 호출만).
  ② 원장은 **유도**된다 — 손 목록이 아니라 파일에서 오고, 항목마다 출처가 붙는다.
  ③ 원장이 **인쇄 자리**를 출처로 댄다(주석·독스트링 언급이 아니라).
  ④ A2 의 `_note_` 출처 주석은 원장에 안 들어온다(나가는 문장이 아니다).
  ⑤ `deny_kind` 의 눈금은 **한 칸도 안 움직였다** — R1 이 방금 교정한 자[尺]다.
"""

import ast
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                              # noqa: E402

fail = []


def check(name, ok, extra=""):
    print("%s %s%s" % ("PASS" if ok else "FAIL", name, (" | " + extra) if extra else ""))
    if not ok:
        fail.append(name)


# ── ① 사본이 없다 ────────────────────────────────────────────────────────────
GAP = io.open(os.path.join(HERE, "t2_gap.py"), encoding="utf-8").read()
check("t2_gap 에 손 목록 `OURS` 가 없다", "OURS = (" not in GAP)
check("t2_gap 이 정본 `ours_text` 를 부른다", "F.ours_text(" in GAP)
_gaptree = ast.parse(GAP)
_rival = next((n for n in ast.walk(_gaptree)
               if isinstance(n, ast.FunctionDef) and n.name == "rival_text"), None)
_rbody = (ast.get_source_segment(GAP, _rival) or "") if _rival is not None else ""
_code = " ".join(l for l in _rbody.splitlines() if not l.lstrip().startswith("#"))
check("`rival_text` 본문에 표지 리터럴 0",
      "NOT_VERIFIED" not in _code and "GROUNDING WARNING" not in _code)

# ── ② 원장은 유도된다 · 출처가 붙는다 ────────────────────────────────────────
L = F.our_notice_ledger()
check("통지 원장이 비어 있지 않다", len(L) > 0, "size=%d" % len(L))
check("모든 항목에 출처 파일이 붙는다", all(bool(v) for v in L.values()))
check("출처는 우리 파일이다(.py 또는 a2 .json)",
      all(v.endswith(".py") or v.endswith(".json") for v in L.values()),
      str(sorted({v for v in L.values() if not (v.endswith(".py") or v.endswith(".json"))})))

# ── ③ 인쇄 자리를 출처로 댄다 ────────────────────────────────────────────────
src_gw = L.get("[GROUNDING WARNING]")
check("`[GROUNDING WARNING]` 이 원장에 있다", src_gw is not None)
check("그 출처가 **인쇄 자리**(t2_scaffold_get.py)다",
      src_gw == "t2_scaffold_get.py", "got=%r" % (src_gw,))
_sg = io.open(os.path.join(HERE, "t2_scaffold_get.py"), encoding="utf-8").read()
check("그 파일이 실제로 그 표지를 인쇄한다", '"[GROUNDING WARNING]' in _sg)

# ── ④ A2 출처 주석은 안 들어온다 ─────────────────────────────────────────────
check("메모리 링크(`[[NN]]`)가 원장에 없다", not any(k.startswith("[[") for k in L),
      str([k for k in L if k.startswith("[[")])[:120])

# ── ⑤ `ours_text` 판정 ───────────────────────────────────────────────────────
POS = [
    ("공백 표지 주석", "[GROUNDING WARNING] 2 input value(s) could not be verified against the ledger"),
    ("A2 통지(본문 중간)", "Rows: 3. could not be verified from the reward-rate policy documents; "
                           "those rows remain UNVERIFIED."),
    ("A2 거절 템플릿", "NOT_VERIFIED - only 1 of the required 2 identity values match so far"),
    ("동적 표지 게이트", "Error: [POLICY GATE G1_AUTH_FIRST] you must verify identity first"),
    ("저작 거절 접두", "Error: [OPERATOR-PROVENANCE] tool name 'x_1822' was not discovered from any "
                       "prior search/listing result"),
]
for nm, body in POS:
    check("ours_text 양성 — %s" % nm, F.ours_text(body) is True, body[:48])

NEG = [
    ("env 성공 출력", '{"accounts": [{"account_id": "a1", "balance": 100.0}]}'),
    ("env 오류", "Error: user_id not found"),
    ("빈 본문", ""),
    ("손님 산문", "Sure, I can help you with that. Which account did you mean?"),
]
for nm, body in NEG:
    check("ours_text 음성 — %s" % nm, F.ours_text(body) is False, body[:48])

# ── ⑥ `deny_kind` 눈금 불변 ──────────────────────────────────────────────────
RULER = [
    ("Error: [POLICY GATE G1_AUTH_FIRST] verify first", "ours"),
    ("Error: [DUPLICATE-READ] you already read this", "ours"),
    ("NOT_VERIFIED - only 1 of 2 match", "ours"),
    ("Error: user_id not found", "env"),
    ('{"ok": true}', ""),
    ("[GROUNDING WARNING] 2 value(s) unverified. Result: {\"ok\": true}", ""),
]
for body, want in RULER:
    got = F.deny_kind(body)[0]
    check("deny_kind 불변 — %r" % body[:34], got == want, "want=%r got=%r" % (want, got))

print("")
print("RESULT: %s" % ("ALL PASS" if not fail else "FAIL %s" % fail))
sys.exit(1 if fail else 0)
