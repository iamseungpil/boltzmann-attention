# -*- coding: utf-8 -*-
r"""x245 — **상태 분해가 그 턴에 닿으면 010 이 달라지는가** (격리 · 유료 0 · 로컬 LLM · 엔진 0).

## 왜 (C409 → 이 프로브 · ⛔0 ①②)

010 의 원인은 확정됐다: 재료는 **생산**됐고(원장 전사 4/4 · 블록 6~9회) **배급**에서 죽는다.
계기가 두 관문을 짚었다 — `resolve_cap`(정체 상한) · `other_lever`(그 턴에 다른 레버가 이미
울면 재료 경로를 통째로 건너뛴다). 그리고 사이드카가 상관을 보여 준다:

    t1(통과)  턴24 상태 분해 발화 → 손님이 **Platinum Rewards Card** 로 재제출 → pass
    t0(실패)  턴34 (10턴 늦게·이미 이관 사슬)
    t2(실패)  **없음**

⚠이것은 **n=3 상관**이다. 관문을 고치기 전에 *그 문장이 닿으면 실제로 달라지는가*를 재야 한다
(⛔0 ②: 전달로 되면 레버는 전달뿐이고, 전달로도 안 되면 관문 수리는 헛수고다).

## 팔 (n=8 · 계기 = 답변이 **거절된 카드 이름**을 말하는가)

  A_LIVE      궤적 + **우리 층이 실제로 넣은 문장**(사이드카 축자)   ← 라이브 재현(낮아야 한다)
  B_BREAKDOWN A + 상태 분해 문장 한 줄                              ← **전달 팔**(관문 수리의 값)
  C_ISO       격리(손님 발화 + 원장 도구 출력 축자) + 상태 분해      ← 문맥 축소까지 했을 때
  D_ISO_BARE  격리 + 원장 출력만(상태 분해 없음)                     ← 부정 통제(원장만으로 되면 불요)

읽는 법 — B 가 A 보다 높으면 **관문 수리가 값을 산다**. B≈A 면 그 문장은 원인이 아니고
관문을 고쳐도 소용없다(그때는 이 인쇄물이 그 사실의 기록이다). D 가 높으면 우리 문장 자체가 불요다.

⚠문자열 대리지표다([[M]]) — 팔마다 실물 2건을 인쇄한다.

실행(리모트·GPU1): T2_PROBE_URL=http://localhost:8141/v1/chat/completions \
                   python x245_breakdown_delivery_probe.py [N]
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
from x241_uncalled_unlock_probe import ctx_with_ours, context     # noqa: E402

TAG = "bank_asubON_20260810"
TASK = "task_010"
GOLD = "Platinum Rewards Card"
ASK = "Now write your reply to the customer."
# 상태 분해는 **우리 층이 라이브에서 실제로 낸 문장**이다 — 다시 쓰지 않고 사이드카에서 가져온다.
SIG = "grouped by the status each record carries"


def breakdown_text(tag):
    """사이드카에서 상태 분해 문장 축자 하나를 꺼낸다(재작성 0·[[03b]])."""
    p = "/home/woori/scratch/logs/fb_%s.jsonl" % tag
    for ln in open(p, encoding="utf-8", errors="replace"):
        try:
            o = json.loads(ln)
        except Exception:
            continue
        t = " ".join((o.get("text") or "").split())
        if SIG in t and GOLD in t:
            return t
    return ""


def ledger_out(sim, upto):
    """원장 도구가 돌려준 축자 — 격리 팔의 유일한 사실 재료."""
    out = []
    for m in sim["messages"][:upto]:
        c = " ".join(str(m.get("content") or "").split())
        if m.get("role") == "tool" and "referrals" in c and "Record ID" in c:
            out.append(c[:1500])
    return "\n".join(out[:1])


def users(sim, upto):
    return "\n".join("[user] %s" % " ".join(str(m.get("content") or "").split())[:700]
                     for m in sim["messages"][:upto] if m.get("role") == "user")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sims = [s for s in X.load(TAG) if s["task_id"] == TASK
            and (s.get("reward_info") or {}).get("reward") != 1]
    bd = breakdown_text(TAG)
    if not bd:
        print("상태 분해 축자를 못 찾았다 — 중단(지어내지 않는다)")
        return 1
    print("상태 분해 축자 %d자: %s\n" % (len(bd), bd[:190]))
    for sim in sims:
        # 손님이 **구체를 요구한 직후**의 답변 자리 — 그 턴이 t1 에서 통과를 만든 자리다
        cut = None
        for i, m in enumerate(sim["messages"]):
            c = str(m.get("content") or "")
            if m.get("role") == "user" and re.search(r"exactly which|specific|which one|criteria",
                                                     c, re.I):
                cut = i + 1
                break
        if cut is None:
            print("t%s — 요구 턴을 못 찾았다. 건너뛴다." % sim.get("trial"))
            continue
        live = ctx_with_ours(sim, cut, tag=TAG)
        plain = context(sim, cut)
        iso = users(sim, cut) + "\n\n" + ledger_out(sim, cut)
        arms = [("A_LIVE", live),
                ("B_BREAKDOWN", live + "\n\n[system] " + bd),
                ("C_ISO", iso + "\n\n" + bd),
                ("D_ISO_BARE", iso)]
        print("=" * 96)
        print("%s t%s · 요구 턴 %d · 라이브 %d자 · 궤적만 %d자 · 격리 %d자"
              % (TASK, sim.get("trial"), cut, len(live), len(plain), len(iso)))
        for name, body in arms:
            hit, shown = 0, []
            for i in range(n):
                try:
                    t = chat(body + "\n\n" + ASK, None, 0.0 if i == 0 else 0.7,
                             260).get("content", "") or ""
                except Exception as e:
                    t = "ERR %s" % type(e).__name__
                if re.search(r"(?<![A-Za-z])%s" % re.escape(GOLD), t):
                    hit += 1
                if len(shown) < 2:
                    shown.append(" ".join(t.split())[:150])
            print("  %-12s 거절 카드 이름 %d/%d" % (name, hit, n))
            for s in shown:
                print("        · %s" % s)
    print("\n※ B > A 면 관문 수리가 값을 산다. B ≈ A 면 그 문장은 원인이 아니다."
          "\n  D 가 높으면 우리 문장 없이 원장만으로 되는 것이라 레버가 불요다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
