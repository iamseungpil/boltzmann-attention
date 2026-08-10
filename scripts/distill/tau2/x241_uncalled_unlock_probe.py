# -*- coding: utf-8 -*-
r"""x241 — **"열고 안 쓴다"를 닫는 것은 말인가 뺄셈인가** (격리 · 유료 0 · 로컬 LLM · 새 엔진 0).

## 왜 (x238 → 이 프로브 · ⛔0 ①②)

099 의 실패 시행은 발견 도구를 **unlock 만 하고 호출하지 않았다**(= gold `099_2`). 그래서 선행
요건이 안 닫혔고, 우리 결정 재료는 `[ORDER] 아직 못 한다` 단계에 갇혀 **한 번도 나가지 못했다**
(통과 시행 2건은 같은 턴에 `[ACTION]` 으로 넘어가며 블록을 실어 보냈다).

우리 층에는 이미 그 자리를 겨눈 레버가 있고(`T2_UNCALLED_UNLOCK`) 라이브에서 **발화했다**
— 그런데 안 들었다. C404 가 예측하는 모양이다: *말해 주는 것으로는 안 된다.*

⇒ **전달 팔을 먼저 재고**(⛔0 ②), 그것이 실패할 때에만 뺄셈이 정당해진다.

## 팔 (n=8 · 계기 = 다음 도구 호출 하나)

  A_LIVE       실제 문맥 + 전체 도구            ← 부정 통제(그냥 되면 레버 불요)
  B_TELL       + 우리 `[UNLOCKED-NOT-CALLED]` 축자   ← **전달 팔**(현행 라이브)
  C_MINUS      검색 도구를 **뺀** 도구 목록          ← 뺄셈 팔(x236 형)
  D_BOTH       뺄셈 + 문구
  E_ISO        **격리 문맥**(손님 발화 + 잠금 해제 결과 축자) + 전체 도구  ← 액션 에이전트가 이긴 형태
  F_ISO_MINUS  격리 + 검색 도구 제거

★E/F 를 넣은 이유 (사용자 지적 2026-08-10): *"action 에이전트 격리로도 안 되나?"* — 격리는
  기전이 아니라 **문맥 축소**라 배제 판정에도 발화 생성에도 아닌 이 자리에 그대로 붙일 수 있다.
  x228(액션)·C397(진단)이 같은 축소로 각각 6/6·100% 를 냈으므로 **재기 전에는 배제할 수 없다**.
  E 가 높으면 처방은 뺄셈이 아니라 **격리 서브**이고, 로스터에 넷째 기능이 서는 것이 아니라
  기존 서브의 문맥 규약이 하나 더 확인되는 것이다.

통과 = 다음 호출이 `call_discoverable_agent_tool` 이고 인자가 **그 잠긴 도구 이름**인 것.

⚠문구는 코드에서 축자로 가져온다(재작성 0). ⚠도구 스키마는 궤적이 실제로 부른 이름만 쓴다.
⚠뺄셈 팔은 *도구를 지우는 것*이지 답을 주는 것이 아니다 — 무엇을 부를지는 끝까지 모델이 고른다.

실행(리모트): python x241_uncalled_unlock_probe.py [N]
"""
import collections
import json
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

TAG = "bank_asubON_20260810"
TASK = "task_099"
LOCKED = "get_all_user_accounts_by_user_id_3847"
ASK = "What is your next step? Make exactly one tool call."

SEARCH = {"KB_search_dense", "KB_search_bm25"}


def tell_text():
    """우리 레버의 문구를 **코드에서** 축자로 가져온다(두 벌이 되면 갈린다·[[03b]])."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_gate_patch.py"),
               encoding="utf-8").read()
    i = src.find("Error: [UNLOCKED-NOT-CALLED]")
    chunk = src[i:i + 700]
    parts = re.findall(r'"([^"]*)"', chunk)
    body = "".join(parts[:4])
    return body.replace("%s", LOCKED)


def tools_of(sim, drop=()):
    """궤적이 실제로 부른 도구만으로 스키마를 만든다 — 새 도구를 발명하지 않는다."""
    names = []
    for m in sim["messages"]:
        for tc in (m.get("tool_calls") or []):
            n = (tc.get("function") or {}).get("name") or tc.get("name")
            if n and n not in names and m.get("role") == "assistant":
                names.append(n)
    out = []
    for n in names:
        if n in drop:
            continue
        out.append({"type": "function", "function": {
            "name": n, "description": "",
            "parameters": {"type": "object",
                           "properties": {"agent_tool_name": {"type": "string"},
                                          "arguments": {"type": "string"},
                                          "query": {"type": "string"},
                                          "user_id": {"type": "string"}}}}})
    return out


def context(sim, upto):
    out = []
    for m in sim["messages"][:upto]:
        r, c = m.get("role"), " ".join(str(m.get("content") or "").split())
        tcs = [(tc.get("function") or {}).get("name") or tc.get("name")
               for tc in (m.get("tool_calls") or [])]
        if tcs:
            out.append("[%s calls] %s" % (r, ", ".join(tcs)))
        if c:
            out.append("[%s] %s" % (r, c[:700]))
    return "\n".join(out)


def iso_context(sim, upto):
    """격리 문맥 — **손님 발화**와 **잠금 해제가 돌려준 축자**만. 대화 잔여물은 전부 뺀다.

    x228·C397 이 이긴 구성과 같은 규약이다: 판단에 필요한 것만 남기고 궤적은 지운다.
    도구 설명은 우리가 짓지 않는다 — `unlock_discoverable_agent_tool` 의 **출력 그대로**다.
    """
    users, unlocked = [], []
    for m in sim["messages"][:upto]:
        c = " ".join(str(m.get("content") or "").split())
        if m.get("role") == "user" and c:
            users.append("[user] %s" % c[:700])
        if m.get("role") == "tool" and c.startswith("Tool unlocked:"):
            unlocked.append(c[:700])
    return "\n".join(users + [""] + unlocked)


def scored(msg):
    for tc in (msg.get("tool_calls") or []):
        f = tc.get("function") or {}
        if (f.get("name") or "") == "call_discoverable_agent_tool" and LOCKED in str(
                f.get("arguments") or ""):
            return "HIT"
    names = [(tc.get("function") or {}).get("name") for tc in (msg.get("tool_calls") or [])]
    return names[0] if names else "(발화만)"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sim = [s for s in X.load(TAG)
           if s["task_id"] == TASK and (s.get("reward_info") or {}).get("reward") != 1][0]
    # 잠근 뒤 처음으로 우리 레버가 울 수 있었던 자리 = unlock 직후의 assistant 턴
    cut = None
    for i, m in enumerate(sim["messages"]):
        for tc in (m.get("tool_calls") or []):
            nm = (tc.get("function") or {}).get("name") or tc.get("name")
            if nm == "unlock_discoverable_agent_tool":
                cut = i + 2
    ctx = context(sim, cut)
    tell = tell_text()
    print("문맥 %d자 (unlock 직후 턴 %d) · 문구 %d자" % (len(ctx), cut, len(tell)))
    print("문구 축자: %s\n" % tell[:200])
    iso = iso_context(sim, cut)
    print("격리 문맥 %d자\n" % len(iso))
    arms = [("A_LIVE", ctx, ()),
            ("B_TELL", ctx + "\n\n" + tell, ()),
            ("C_MINUS", ctx, SEARCH),
            ("D_BOTH", ctx + "\n\n" + tell, SEARCH),
            ("E_ISO", iso, ()),
            ("F_ISO_MINUS", iso, SEARCH)]
    for name, body, drop in arms:
        tools = tools_of(sim, drop)
        c = collections.Counter()
        for i in range(n):
            try:
                r = chat(body + "\n\n" + ASK, tools, 0.0 if i == 0 else 0.7, 200)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            c[scored(r)] += 1
        print("  %-9s 도구 %d개 · HIT %d/%d   %s"
              % (name, len(tools), c["HIT"], n, c.most_common(3)))
    print("\n※ 읽는 법 — B 가 낮고 C/D 가 높으면 **말이 아니라 뺄셈**이다(C404 의 일곱 번째 사례)."
          "\n  B 가 이미 높으면 라이브 실패는 문구가 아니라 **발화 자리**의 문제다(1회 캡·resign 조건).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
