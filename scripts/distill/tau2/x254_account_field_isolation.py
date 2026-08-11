# -*- coding: utf-8 -*-
r"""x254 — **타입/클래스 두 칸을 채우는 일은 격리로 닫히는가** (071·유료 0·8141 전용).

## 왜 (사용자 지적 2026-08-11: *"정보를 알고도 안 고르는 문제는 대부분 격리 문제 같다"*)

라이브 071 t0 이 낸 인자와 gold:

    gold   account_type="business_checking"   account_class="Sky Blue"
    라이브 account_type="checking"            account_class="Sky Blue Business Checking"

**두 칸이 같이 틀렸다** — *business* 를 타입에서 빼서 클래스에 붙였다. 접미사 모양 문제가 아니라
**필드 혼동**이다. 그리고 환경은 이미 축자로 말하고 있다:

    account_type: Must be one of: 'checking' (personal checking), 'savings',
                  'business_checking', 'business_savings'
    account_class: The full official account class name

⇒ *몰라서*가 아니라 **선언을 눈앞에 두고 틀렸다**. C404 가 예측하는 자리(더 말해도 안 듣는다)이고,
C411 의 경계로 보면 **격리 쪽**이다 — 이 판단의 재료는 *대화 밖 사실*(도구 선언·고른 이름·손님이
말한 종류)이지 *대화 진행 상태*가 아니다.

⚠과일반화 금지: 오늘 x241 의 `E_ISO` 는 **2/8** 로 졌다(그 판단은 선행 단계 완료를 요구했다).
  그래서 격리를 **1순위 가설로 두되 재고** 판정한다.

## 팔 (n=8 · 계기 = 두 칸을 **정확히** 채우는가)

  A_LIVE     실제 궤적 + 우리 층 발화(사이드카 축자)   ← 라이브 재현(낮아야 한다)
  B_TELL     A + 타입/클래스 구분을 **문장으로**        ← 전달 팔(C404 예측: 안 듣는다)
  C_ISO      **격리**: 손님 발화 + 도구 선언 축자 + 결정된 클래스 이름
  D_ISO_BARE 격리에서 **클래스 이름 없이**              ← 부정 통제(이름 없이도 되면 결정이 불요)

읽는 법 — C 가 A/B 를 이기면 격리가 처방이다. C 도 낮으면 격리 문제가 아니다(x241 형).
D 가 이미 높으면 우리가 이름을 실어 줄 필요가 없다.

⚠도구 선언은 **환경 축자**를 그대로 쓴다(우리가 새로 쓰지 않는다).
⚠8141 전용([[30]]·사용자 지시). 8140 은 유료 런 자리다.

실행: T2_PROBE_URL=http://localhost:8141/v1/chat/completions python x254_account_field_isolation.py [N]
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

from x216_read_and_offset import chat, URL                        # noqa: E402
import x238_action_forensic as X                                  # noqa: E402
from x241_uncalled_unlock_probe import ctx_with_ours              # noqa: E402

RUN, TASK, TRIAL = "bank_sa_20260811d", "task_071", 0
TOOL = "open_bank_account_4821"
GOLD_TYPE, GOLD_CLASS = "business_checking", "Sky Blue"
SRC_TOOLS = ("/home/woori/scratch/tau2-bench/src/tau2/domains/"
             "banking_knowledge/tools.py")

TOOLS = [{"type": "function", "function": {
    "name": "call_discoverable_user_tool", "description": "Run an unlocked discoverable tool.",
    "parameters": {"type": "object", "properties": {
        "discoverable_tool_name": {"type": "string"},
        "arguments": {"type": "string"}}}}}]

ASK = ("The customer is ready to open the account. Issue the tool call now, "
       "with `discoverable_tool_name` set to '%s' and `arguments` a JSON string." % TOOL)


def tool_decl():
    """도구 선언을 **환경 축자**로 — 우리가 다시 쓰지 않는다([[03b]])."""
    s = open(SRC_TOOLS, encoding="utf-8").read()
    i = s.find("def %s(" % TOOL)
    doc = s[i:i + 1200]
    j, k = doc.find('"""'), doc.find('"""', doc.find('"""') + 3)
    body = doc[j + 3:k] if j >= 0 and k > j else doc[:600]
    return "Tool on record — %s\n%s" % (TOOL, " ".join(body.split())[:700])


def user_turns(sim, upto=None):
    out = []
    for m in (sim["messages"][:upto] if upto else sim["messages"]):
        if m.get("role") == "user":
            c = " ".join(str(m.get("content") or "").split())
            if c and "###" not in c:
                out.append("[user] " + c[:700])
    return "\n".join(out[:6])


def scored(msg):
    """두 칸을 **정확히** 채웠는가 — 부분 점수도 함께 센다."""
    for tc in (msg.get("tool_calls") or []):
        f = tc.get("function") or {}
        raw = str(f.get("arguments") or "")
        t = re.search(r'"account_type"\s*:\s*\\?"([^"\\]+)', raw)
        c = re.search(r'"account_class"\s*:\s*\\?"([^"\\]+)', raw)
        tv, cv = (t.group(1) if t else None), (c.group(1) if c else None)
        if tv == GOLD_TYPE and cv == GOLD_CLASS:
            return "BOTH", (tv, cv)
        if tv == GOLD_TYPE:
            return "TYPE만", (tv, cv)
        if cv == GOLD_CLASS:
            return "CLASS만", (tv, cv)
        return "둘 다 틀림", (tv, cv)
    return "호출 없음", (None, None)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    if ":8141" not in URL:
        print("중단: 8141 이 아니다 (%s)" % URL)
        return 1
    sim = [s for s in X.load(RUN)
           if s["task_id"] == TASK and s.get("trial") == TRIAL][0]
    cut = None
    for i, m in enumerate(sim["messages"]):
        for tc in (m.get("tool_calls") or []):
            if TOOL in str((tc.get("function") or {}).get("arguments") or ""):
                cut = i
                break
        if cut is not None:
            break
    cut = cut or len(sim["messages"]) - 1
    live = ctx_with_ours(sim, cut, tag=RUN)
    decl = tool_decl()
    users = user_turns(sim, cut)
    tell = ("Note on the two fields: `account_type` is the ACCOUNT CATEGORY from the tool's own "
            "list, and `account_class` is only the product name — the category must not be "
            "repeated inside the class name.")
    print("URL %s · 결정 턴 %d · 라이브 %d자 · 격리 %d자" % (URL, cut, len(live), len(users)))
    print("선언 축자: %s\n" % decl[:220])
    arms = [("A_LIVE", live),
            ("B_TELL", live + "\n\n[system] " + tell),
            ("C_ISO", users + "\n\n" + decl + "\n\nThe account class decided on record: %s"
             % GOLD_CLASS),
            ("D_ISO_BARE", users + "\n\n" + decl)]
    for name, body in arms:
        c, shown = collections.Counter(), []
        for i in range(n):
            try:
                r = chat(body + "\n\n" + ASK, TOOLS, 0.0 if i == 0 else 0.7, 220)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            k, vals = scored(r)
            c[k] += 1
            if len(shown) < 2:
                shown.append(str(vals))
        print("  %-11s BOTH %d/%d   %s   예: %s"
              % (name, c["BOTH"], n, dict(c), " · ".join(shown)))
    print("\n※ C 가 A/B 를 이기면 **격리가 처방**이다. C 도 낮으면 격리 문제가 아니다(x241 형)."
          "\n  D 가 이미 높으면 결정된 이름을 실어 줄 필요가 없다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
