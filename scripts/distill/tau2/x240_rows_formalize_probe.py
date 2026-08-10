# -*- coding: utf-8 -*-
r"""x240 — **상태 분해가 왜 침묵했나**: 원장 전사(`formalize_rows`)를 궤적 위에서 반복 재생한다
(유료 0 · 로컬 LLM · 새 엔진 0).

## 왜 (x238 → 이 프로브)

010 은 세 시행 모두 `[ACTION]` 단계까지 올라갔는데, **통과한 시행에서만** 상태 분해 문장이 나갔다:

    t1(통과)  턴24  "Of the 4 record(s) … REJECTED 1 — Platinum Rewards Card"   ← 손님이 그 카드로 재제출
    t0(실패)  턴34  같은 문장 — **10턴 늦게**(이미 이관 사슬로 들어간 뒤)
    t2(실패)  **한 번도 없음**

그 문장은 `status_breakdown(_e2["rows"], …)` 가 만들고, `rows` 는 **LLM 이 전사한 원장 행**이다
(`formalize_rows` — 엔진은 텍스트를 읽지 않는다·[[59]]). 전사가 실패하면 `rows=[]` 이고
**아무 말 없이** 문장이 사라진다. ⇒ 가설 H2: *전사가 간헐적으로 실패한다.*

## 어떻게 재는가

같은 함수를 같은 프롬프트로 **같은 도구 출력** 위에서 n 회 돌린다(온도는 라이브와 같은 0.0 과,
변동을 보기 위한 0.7 둘 다). 실패율이 0 이면 H2 는 죽고 원인은 다른 칸이다.

⚠이 프로브는 **우리 층의 신뢰성**을 재는 것이다([[25]] — 우리 도구는 100% 정답 의무).
⚠재료는 궤적에 실제로 있던 `get_referrals_by_user` 출력 축자다. gold 는 채점에도 안 쓴다.

실행(리모트): python x240_rows_formalize_probe.py [N]
"""
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_ledger as LG                                            # noqa: E402
import x238_action_forensic as X                                  # noqa: E402
from x216_read_and_offset import chat                             # noqa: E402
from x239_intent_gate_probe import _Agent, _LA, _UM               # noqa: E402


def specs():
    """A2 가 선언한 원장 스펙 — 프롬프트도 키도 전부 거기서 온다(엔진 어휘 0)."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "a2", "banking_knowledge.gate.json")
    a2 = json.load(open(p, encoding="utf-8"))
    # 원장 선언은 **도구 이름**으로 붙는다 — 이 계열의 원장 도구는 `get_referrals_by_user` 다.
    return {(s.get("trigger_tool") or "?"): s
            for s in LG.specs_for(a2, "get_referrals_by_user")}


def ledger_texts(tag, task):
    """그 태스크의 각 시행에서 원장 도구가 실제로 돌려준 축자."""
    out = []
    for s in X.load(tag):
        if s["task_id"] != task:
            continue
        got = []
        for m in s["messages"]:
            c = str(m.get("content") or "")
            if m.get("role") == "tool" and "referrals" in c and "Record ID" in c:
                got.append(c)
        out.append((s.get("trial"), (s.get("reward_info") or {}).get("reward"), got))
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    sp_all = specs()
    print("선언된 원장 스펙: %s" % list(sp_all))
    for task in ("task_010", "task_099"):
        print("\n" + "=" * 92)
        for trial, rew, texts in sorted(ledger_texts("bank_asubON_20260810", task)):
            if not texts:
                print("%s t%s rew=%s — 원장 출력 없음" % (task, trial, rew))
                continue
            txt = texts[0]
            print("%s t%s rew=%s · 원장 출력 %d자 · 재생 %d회" % (task, trial, rew, len(txt), n))
            for name, sp in sp_all.items():
                if not (sp.get("row_keys") and sp.get("formalize_prompt")):
                    continue
                got = collections.Counter()
                sample = None
                for i in range(n):
                    rows = LG.formalize_rows(_Agent(), _LA, _UM, txt, sp)
                    got[len(rows)] += 1
                    if rows and sample is None:
                        sample = rows[0]
                print("   %-22s 행수 분포 %s   예시 %s"
                      % (name, dict(got), json.dumps(sample, ensure_ascii=False)[:110]))
    print("\n※ 읽는 법 — 행수 0 이 섞이면 H2(전사 간헐 실패)가 살고, 처방은 새 레버가 아니라"
          "\n  **우리 층의 재시도·검산**이다([[25]]). 0 이 없으면 원인은 다른 칸이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
