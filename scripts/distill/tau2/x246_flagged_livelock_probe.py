# -*- coding: utf-8 -*-
r"""x246 — **접힌 문구가 라이브락의 원인인가 결과인가** (격리 · 유료 0 · 로컬 LLM · 엔진 0).

## 왜 (C413 · ⛔0 ①·[[55]])

`resolve the flagged call` 이 한 sim 에 **3회 이상** 나온 6건은 **6/6 전부 실패**했고, 세 런을
가로지르므로 레버와 무관하다. 자리도 찾았다 — `t2_gate_patch.py:7194` `_FB_GENERIC`: deny 가
`admit()` 에 접힐 때 **본문을 일반 문구로 대체**해 *무엇을 고칠지*를 지운다.

그러나 **상관은 인과가 아니다**. 반복이 막힘의 *원인*인지, 이미 막힌 대화의 *결과*인지 모른다.
같은 문맥 위에서 그 문구만 바꿔 다음 한 수를 재면 갈린다.

## 팔 (n=8 · 계기 = 다음 한 수가 **정체**인가 **진전**인가)

  A_GENERIC   문맥 + `_FB_GENERIC` **3회**            ← 라이브 재현(정체가 재현돼야 한다)
  B_ORIGINAL  문맥 + **원본 deny 본문**(같은 sim 축자) ← 처방 후보 ①: 이름을 되살린다
  C_NONE      문맥 + **아무 문구 없음**                ← 처방 후보 ②: 아예 안 내보낸다(C404 빼기)
  D_ONCE      문맥 + `_FB_GENERIC` **1회**            ← 반복이 문제인지 문구가 문제인지 가른다

정체 = 답변이 다시 *"resolve/flagged"* 를 말하거나 같은 이관 호출을 되풀이하는 것.
진전 = **다른 도구를 부르거나**, 손님에게 **재제출을 넘기는 것**(도구 이름 + 값).

읽는 법 — A 가 정체를 재현하고 B/C 가 진전을 내면 **반복이 원인**이고 처방이 정당해진다.
넷 다 정체면 원인은 다른 곳이고 이 문구는 **결과**다(그러면 C413 처방을 짓지 않는다).

⚠문구는 전부 **코드·궤적 축자**다(재작성 0). ⚠계기는 문자열 대리라 [[M]]·팔마다 실물 2건 인쇄.

실행(리모트·GPU1): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
                   python x246_flagged_livelock_probe.py [N]
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
from x241_uncalled_unlock_probe import ctx_with_ours, tools_of     # noqa: E402

RUN = "bank_uq_20260811"
TASK, TRIAL = "task_010", 2
GOLD = "Platinum Rewards Card"
HAND = "submit_referral"
STALL = re.compile(r"resolv\w*\s+(?:the\s+)?flag|flagged call", re.I)


def generic():
    """접힘이 쓰는 일반 문구 — **코드에서** 축자로 가져온다."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "t2_gate_patch.py"),
               encoding="utf-8").read()
    m = re.search(r'_FB_GENERIC = "([^"]+)"', src)
    return m.group(1) if m else ""


def original_body(sim, tag):
    """같은 sim 에서 **접히기 전에 나갔던 구체 본문** — 사이드카 축자에서 가장 긴 deny 를 쓴다."""
    import t2_fbsidecar as FB

    class _M(object):
        def __init__(s, r, c):
            s.role, s.content = r, c

    k = FB._sim_key([_M(m.get("role"), m.get("content")) for m in sim["messages"]])
    best = ""
    for ln in open("/home/woori/scratch/logs/fb_%s.jsonl" % tag, encoding="utf-8",
                   errors="replace"):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        if o.get("sim") != k:
            continue
        t = " ".join((o.get("text") or "").split())
        if t.startswith("Error: [") and not STALL.search(t) and len(t) > len(best):
            best = t
    return best[:900]


def scored(msg):
    txt = str(msg.get("content") or "")
    names = [(tc.get("function") or {}).get("name") or tc.get("name")
             for tc in (msg.get("tool_calls") or [])]
    if HAND in txt and GOLD in txt:
        return "진전:넘김"
    if names and not STALL.search(txt):
        return "진전:도구(%s)" % names[0]
    if STALL.search(txt):
        return "정체:되뇜"
    return "정체:발화만"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sim = [s for s in X.load(RUN)
           if s["task_id"] == TASK and s.get("trial") == TRIAL][0]
    # 라이브락이 시작된 자리 = 그 문구가 처음 나온 뒤의 답변 턴
    cut = None
    for i, m in enumerate(sim["messages"]):
        if m.get("role") == "assistant" and STALL.search(str(m.get("content") or "")):
            cut = i
            break
    if cut is None:
        cut = min(36, len(sim["messages"]) - 1)
    gen = generic()
    orig = original_body(sim, RUN)
    ctx = ctx_with_ours(sim, cut, tag=RUN)
    tools = tools_of(sim)
    print("문맥 %d자 (라이브락 시작 턴 %d) · 도구 %d개" % (len(ctx), cut, len(tools)))
    print("일반 문구: %s" % gen)
    print("원본 본문 %d자: %s\n" % (len(orig), orig[:200]))
    arms = [("A_GENERIC", ctx + "\n\n" + "\n".join(["[system] " + gen] * 3)),
            ("B_ORIGINAL", ctx + "\n\n[system] " + (orig or gen)),
            ("C_NONE", ctx),
            ("D_ONCE", ctx + "\n\n[system] " + gen)]
    for name, body in arms:
        c, shown = collections.Counter(), []
        for i in range(n):
            try:
                r = chat(body, tools, 0.0 if i == 0 else 0.7, 220)
            except Exception as e:
                r = {"content": "ERR %s" % type(e).__name__}
            c[scored(r)] += 1
            if len(shown) < 2:
                shown.append(" ".join(str(r.get("content") or "").split())[:130])
        prog = sum(v for k, v in c.items() if k.startswith("진전"))
        print("  %-11s 진전 %d/%d   %s" % (name, prog, n, dict(c)))
        for s in shown:
            print("        · %s" % s)
    print("\n※ A 가 정체를 재현하고 B/C 가 진전을 내면 **반복이 원인**이다."
          "\n  넷 다 정체면 이 문구는 결과이고 C413 처방을 짓지 않는다."
          "\n  D(1회) 가 A(3회)와 같으면 문제는 **반복이 아니라 문구 자체**다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
