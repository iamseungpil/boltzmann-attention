# -*- coding: utf-8 -*-
"""ACTION-INDEX 1회 표면화 검정 — A3 선언 · 3층 동기 · 엔진은 나열만.

근거(x319·n=24·블록 8·8·8 — 잡음 바닥 ±4 밖):
    도움 없음 **10/24** → **action 문서 제목 43줄 24/24** · 도구 설명 91종 23/24 ·
    이름만 91종 16/24
⇒ 표면화가 열고, **의미를 담은 것이 이름보다 낫다**. 그리고 가장 싼 것이 가장 좋다 —
  698 문서도 91 설명도 아니라 **43줄**(사용자 지시 *"비용이 최소가 되게"*).

고정하는 것:
  1. 두 A2 층의 `action_index` 가 **바이트 동일**([[24]] — 한쪽만 고치면 死코드/옛값 사용)
  2. 출처가 **기계 도출**임(문서 title + 그 문서가 대는 레지스트리 도구명) — 저작 흔적 0
  3. 엔진은 **고르지 않는다**: 발화문에 순위·최댓값·"정답은 X" 가 없다([[62]] ④)
  4. 선언이 없으면 **무발화**(거동 보존)
"""
import io
import json
import os
import re
import sys

try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_search as S                                              # noqa: E402

FAIL = []
LAYERS = ("a2/banking_knowledge.specific.json", "a2/banking_knowledge.gate.json")


def chk(c, m):
    if not c:
        FAIL.append(m)
    print("  %s %s" % ("ok  " if c else "FAIL", m))


def load(f):
    return json.load(io.open(os.path.join(HERE, f), encoding="utf-8"))


def main():
    print("[선언·동기]")
    idxs = []
    for f in LAYERS:
        d = load(f)
        po = d.get("policy_ontology") or {}
        idxs.append(po.get("action_index"))
        chk(bool(po.get("action_index")), "%s 에 action_index 선언" % os.path.basename(f))
    chk(idxs[0] == idxs[1], "두 층 action_index 바이트 동일([[24]])")

    rows = idxs[0] or []
    chk(len(rows) >= 20, "행 %d (문서 전수가 아니라 action 문서만)" % len(rows))
    chk(all(r.get("title") and r.get("tools") for r in rows), "모든 행에 제목과 도구가 있다")
    chk(all(re.match(r"^[a-z][a-z_]+_\d{4}$", t) for r in rows for t in r["tools"]),
        "도구명이 레지스트리 형식(기계 추출 흔적)")

    print("[발화]")
    a2 = load(LAYERS[0])
    note = S.action_index_note(a2)
    chk(bool(note) and note.count("\n") >= len(rows), "43줄 형태로 발화(%d자)" % len(note))
    chk(all(r["title"][:30] in note for r in rows[:5]), "선언한 제목이 그대로 나온다")
    low = note.lower()
    chk(not any(w in low for w in ("best", "most likely", "you should call", "correct tool")),
        "지목·순위 문구 없음([[62]] ④ — 엔진은 고르지 않는다)")
    chk(S.action_index_note({}) == "", "선언 없으면 무발화(거동 보존)")
    chk(S.action_index_note({"policy_ontology": {"action_index": rows}}) == "",
        "문구(action_index_text) 없으면 무발화 — 엔진 리터럴 0([[05]])")

    print("\n%s (%d fail)" % ("FAIL" if FAIL else "PASS", len(FAIL)))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
