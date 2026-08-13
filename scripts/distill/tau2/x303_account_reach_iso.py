# -*- coding: utf-8 -*-
r"""x303 — 계좌-id 도달 축 격리 (FORENSIC_SYNTHESIS §2-B·[[18]] A_minimal vs B_fullctx).

실물(087 `bank_t7286_a_20260814h`·전 메시지 정독으로 컷 확정·[[08]]):
  msg22  `get_debit_cards_by_account_id_7823(account_id="mt35a7c9d2")` — **user_id 를 account_id
         자리에** → env not found. (이름 미노출 구간의 오표적)
  msg31  KB 7위가 이름을 **명시**: "Use get_all_user_accounts_by_user_id_3847 to retrieve the
         bank accounts (checkings, savings) for a customer."
  msg33  손님 축자: 계좌 ID 를 자기 쪽에서 못 구한다 · **"look up my Blue Account card ...
         on your side"** — 에이전트-측 조회를 명시 요구.
  msg34~ 그런데 모델은 재검색 → give_card_last_4 루프 복귀. 도구 호출 0. → 이관.

질문([[62]] — 레버 전에 결손 측정): 이름이 노출된 뒤의 그 결정점에서, 모델은 계좌 목록
도구를 **부를 수 있는가**?
  A_MIN   정보-맞춘 최소 문맥(검증된 신원·손님 요구·이름을 담은 KB 줄 — 전부 라이브 축자)
  B_FULL  라이브 궤적 그대로 msg34 직전까지(+사이드카 축자)

판정(사전 고정·n=8):
  A≥6 ∧ B≤2  → 부하(load) — 레버는 **전달**(문맥 축소·배치)뿐. 계산·지목 대행 금지.
  A≤2 ∧ B≤2  → 능력 결손 후보 — 그때 U2 문면(deny 에 획득 경로 명시·[[64]])을 **측정 후** 검토.
  A≥6 ∧ B≥6  → 격리는 되는데 라이브만 실패 = 우리층 간섭 재수사([[55]]).
성공 = tool_calls 에 unlock/call × get_all_user_accounts_by_user_id_3847. 재검색·give·텍스트는
각각 따로 센다(라이브 재현 형태).

실행(리모트·8141): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
  /home/woori/venvs/seka_env/bin/python x303_account_reach_iso.py [N]
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from x216_read_and_offset import chat                             # noqa: E402
import x238_action_forensic as X                                  # noqa: E402
import x241_uncalled_unlock_probe as U                            # noqa: E402
import x283_discovery_reach_probe as P                            # noqa: E402
import x291_checking_pick_iso as B                                # noqa: E402

TAG = "bank_t7286_a_20260814h"
TASK = "task_087"
TARGET = "get_all_user_accounts_by_user_id_3847"
WRONGBIND = "get_debit_cards_by_account_id_7823"


def cut_of(sim):
    """컷 = 손님의 '그쪽에서 찾아 달라' 발화 다음 자리(= 라이브가 재검색으로 이탈한 결정점).

    수기 인덱스 박제 대신 규칙: TARGET 이름이 tool 결과에 처음 등장한 뒤에 오는 첫 user 메시지
    바로 다음. (087 실물에선 msg31 노출 → msg33 user → 컷 34.)
    """
    msgs = sim["messages"]
    seen = None
    for i, m in enumerate(msgs):
        if seen is None and m.get("role") == "tool" and TARGET in str(m.get("content") or ""):
            seen = i
        elif seen is not None and m.get("role") == "user":
            return i + 1
    return None


def a_minimal(sim, cut):
    """정보-맞춘 최소 문맥 — 전 줄이 라이브 축자([[03b]])·요약/의역 0."""
    msgs = sim["messages"]
    # ①검증 결과 축자(라이브 tool 줄) ②이름 노출 KB 줄 축자 ③손님 마지막 요구 축자
    verify_line = next(" ".join(str(m.get("content")).split()) for m in msgs
                       if m.get("role") == "tool"
                       and "Verification logged successfully" in str(m.get("content") or ""))
    kb_msg = next(str(m.get("content")) for m in msgs
                  if m.get("role") == "tool" and TARGET in str(m.get("content") or ""))
    i = kb_msg.find(TARGET)
    j = kb_msg.rfind("Internal:", 0, i)
    kb_line = " ".join(kb_msg[max(0, j if j >= 0 else i - 200):i + 260].split())
    last_user = next(" ".join(str(m.get("content")).split()) for m in
                     reversed(msgs[:cut]) if m.get("role") == "user")
    return "\n".join([
        "[user] " + " ".join(str(msgs[1].get("content") or "").split())[:600],
        "[tool] " + verify_line[:400],
        "[tool] " + kb_line,
        "[user] " + last_user[:700],
    ])


def classify(r):
    blob = " ".join(str(t) for t in (r.get("tool_calls") or []))
    if TARGET in blob:
        return "target"
    if WRONGBIND in blob:
        m = re.search(r'"account_id"\s*:\s*"([^"]+)"', blob)
        return "wrongbind:%s" % (m.group(1)[:24] if m else "?")
    for t in (r.get("tool_calls") or []):
        n = str(t.get("name") or "")
        if n.startswith("KB_search"):
            return "research"
        if n:
            return n[:32]
    return "(text)" if r.get("content") else "(empty)"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sim = next(s for s in X.load(TAG) if s["task_id"] == TASK
               and s.get("reward_info") is not None)
    cut = cut_of(sim)
    if cut is None:
        print("컷 없음 — TARGET 노출 후 user 발화가 없다")
        return
    tools = U.tools_of(sim)
    P.TAG = TAG
    ours = P.our_lines(sim)
    b_full = B.render(sim["messages"][:cut], ours)
    a_min = a_minimal(sim, cut)
    print("x303 cut=%d · target=%s · n=%d · URL=%s" % (
        cut, TARGET, n, os.environ.get("T2_PROBE_URL", "8140(기본⚠)")))
    print("A_MIN %d자 · B_FULL %d자\n" % (len(a_min), len(b_full)))
    for label, body in (("A_MIN", a_min), ("B_FULL", b_full)):
        hit = 0
        cnt = collections.Counter()
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 500)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            k = classify(r)
            hit += k == "target"
            cnt[k] += 1
        print("%-7s target %d/%d · %s" % (label, hit, n, dict(cnt)))
    print("\n※ 판정(사전 고정): A≥6∧B≤2=부하(레버=전달만) · A≤2∧B≤2=능력 후보(그때 U2 문면 측정)"
          " · A≥6∧B≥6=우리층 간섭 재수사")


if __name__ == "__main__":
    main()
