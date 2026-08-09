# -*- coding: utf-8 -*-
"""x156 — **무엇이 문맥을 더럽히는가**. 범주별 격리(제거 방향 + 추가 방향). 유료 0·로컬 32B.

사용자 지시(2026-08-09): *"깨끗한 문맥인지가 더 중요한 것 같다. 문맥이 서로 상충하면서 LLM 이
계산이나 필터를 못 하는가. 깨끗한 문맥을 만들기 위해 혼잡스럽게 만드는 문구가 뭔지 격리하라."*

여기까지 확정된 것: 같은 표를 궤적 안에 실으면 0/5, 깨끗한 문맥에 실으면 5/5(x149·x151·x154).
그러면 남는 질문은 **궤적의 어느 부분이 그 차이를 만드는가**다. 두 방향으로 잰다 —
제거(무엇을 빼면 살아나나)와 추가(무엇만 넣어도 죽나). 한 방향만 보면 필요·충분을 못 가른다.

⚠**우리 층 문구는 궤적에 없다**(비커밋 뷰-채널). 그러므로 오염원 후보는 셋뿐이다:
  ⒜ 에이전트 자기 발화 ⒝ 도구 출력 ⒞ 손님 발화.
⚠범주는 **구조로만** 가른다(role·도구 이름·숫자 유무) — 내용으로 고르면 우리가 답을 심는 것이다.

실행: py -3 x156_context_dirt.py [TAG] [N] [TASK]
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

import x149_choice_isolation as X                              # noqa: E402
import x150_choice_ablation as Y                               # noqa: E402


def bucket(m):
    """구조만 본다: 역할 · (도구 메시지면) 어느 도구 · (에이전트면) 숫자를 담았나."""
    role = m.get("role")
    if role == "user":
        return "user"
    if role == "tool":
        c = str(m.get("content") or "")
        if "Accounts for user" in c:
            return "tool:accounts"
        if "in 'referrals'" in c:
            return "tool:referrals"
        if len(c) > 1200:
            return "tool:docs"          # KB 회수문서(길이로만 가른다)
        return "tool:other"
    if role == "assistant":
        c = str(m.get("content") or "")
        if not c.strip():
            return "assistant:callonly"
        return "assistant:numbers" if re.search(r"\d", c) else "assistant:prose"
    return "other"


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "bank_elig_20260809i"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    task = sys.argv[3] if len(sys.argv) > 3 else "task_099"
    ms = Y.msgs_of(tag, task)
    facts, gold, Q = X.FACTS[task], X.GOLD[task], X.QUESTION
    cnt = collections.Counter(bucket(m) for m in ms)
    print("%s · 메시지 %d · 범주 %s\n" % (task, len(ms), dict(cnt)))
    cats = [c for c, k in cnt.most_common() if k and c != "other"]

    head = "Here is a customer-service conversation so far.\n\n"

    def build(keep):
        sel = [m for m in ms if bucket(m) in keep]
        return (head + Y.render(sel) + "\n\n" + facts + "\n\n" + Q) if sel \
            else (facts + "\n\n" + Q)

    arms = collections.OrderedDict()
    arms["FULL"] = build(set(cats))
    arms["NONE(clean)"] = facts + "\n\n" + Q
    for c in cats:                       # 제거 방향 — 하나씩 뺀다
        arms["−%s" % c] = build(set(cats) - {c})
    for c in cats:                       # 추가 방향 — 하나만 넣는다
        arms["only %s" % c] = build({c})

    for label, prompt in arms.items():
        ans = [X.ask(prompt, 0.0 if i == 0 else 0.7) for i in range(n)]
        hit = sum(1 for a in ans if gold.lower() in str(a).lower())
        print("  %-24s %d/%d   %s" % (label, hit, n,
              collections.Counter(re.sub(r"\s+", " ", a)[:22] for a in ans).most_common(2)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
