# -*- coding: utf-8 -*-
"""x133 — 발췌 규칙이 추출을 죽이는가 (유료 0·로컬 vllm).

x132(단계 0 게이트)가 `dp_p/task_100 sim1`에서 `doc_minimums = 모델 0종`을 냈다. 그 코퍼스에는
문턱 문장(*"minimum relationship duration of 60 days as a checking account holder is required"*)이
**5개 항목에** 있는데 전부 오프셋 5,901~9,147 지점이고, 항목 길이는 12,077~14,429다.
현행/§1a 발췌는 **항목당 3,000자**라 다섯 군데를 전부 자른다 — 그런데 총량은 19,448자로 예산
90,000의 22%만 쓴다(탈락 항목 0). **예산이 남는데 상한이 필요한 문장만 버린다.**

이 프로브가 가르는 것(같은 모델·같은 프롬프트·발췌만 다름 = 단일변수):
  A `per3000`  현행 규칙                → 예상 0종
  B `fill`     예산을 채우는 규칙        → 나오면 원인 = **발췌**, 안 나오면 원인 = 모델/프롬프트
[[55]] 진단 순서(배관→문구→계기→모델)에서 *배관*을 먼저 배제하는 자리다.

usage: x133_excerpt_budget_probe.py --dir bank_stack_dp_20260808p --task task_100 --sim 1 \
         --prompts threshold_prompt,limit_prompt --base http://localhost:8140/v1
"""

import argparse
import gzip
import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import t2_ledger as LG                       # noqa: E402
from gate_interpreter import load_domain_a2   # noqa: E402

FIELD = {"threshold_prompt": "min_days", "limit_prompt": "limit"}


def per_item(items, per=3000, budget=90000):
    """현행 규칙 — 항목당 상한을 **항상** 건다(`t2_ledger.py:245~253`·§1a 초판)."""
    sel, used, dropped = [], 0, 0
    for t in reversed(list(items)):
        s = str(t)[:per]
        if used + len(s) > budget:
            dropped += 1
            continue
        sel.append(s)
        used += len(s)
    sel.reverse()
    return sel, dropped


def fill(items, budget=90000, floor=3000):
    """예산을 **채우는** 규칙 — 항목당 몫을 예산에서 나눠 갖고, 남으면 더 가져간다.

    최신부터 담되 상한은 `max(floor, 남은 예산 / 남은 항목 수)`. 항목이 하나면 예산 전부.
    위치로만 고르고 내용은 보지 않는 것은 그대로다([[59]]).
    """
    items = list(items)
    sel, used = [], 0
    rest = list(reversed(items))
    for i, t in enumerate(rest):
        left_items = len(rest) - i
        cap = max(floor, (budget - used) // max(1, left_items))
        s = str(t)[:cap]
        if used + len(s) > budget:
            s = s[:max(0, budget - used)]
        if not s:
            break
        sel.append(s)
        used += len(s)
    sel.reverse()
    return sel, len(items) - len(sel)


class _Agent(object):
    def __init__(self, model, base):
        self.llm = model if model.startswith("openai/") else "openai/" + model
        self.llm_args = {"temperature": 0.0, "api_base": base, "api_key": "dummy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--sim", type=int, default=0)
    ap.add_argument("--prompts", default="threshold_prompt,limit_prompt")
    ap.add_argument("--domain", default="banking_knowledge")
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--needle", default="minimum relationship duration of 60 days")
    a = ap.parse_args()

    import tau2.agent.llm_agent as la
    from tau2.data_model.message import UserMessage

    p = os.path.join(REPO, "reports", "facet_rft_2026", "sim_results", a.dir + ".json.gz")
    data = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
    sims = [s for s in (data.get("simulations") or []) if s.get("task_id") == a.task]
    sim = sims[a.sim]
    texts = []
    for m in sim.get("messages") or []:
        if m.get("role") not in ("tool", "user"):
            continue
        c = m.get("content")
        if isinstance(c, list):
            c = "\n".join(str(x) for x in c)
        texts.append(str(c or ""))
    hay = " ".join("\n".join(texts).split())
    a2 = load_domain_a2(a.domain)
    specs = list(a2.get("ledger_metrics") or [])
    agent = _Agent(a.model, a.base)

    print("%s %s sim%d · corpus %d항목 %dchars · needle 보유 항목 %d"
          % (a.dir, a.task, a.sim, len(texts), sum(len(t) for t in texts),
             sum(1 for t in texts if a.needle in t)))

    for pname in [x.strip() for x in a.prompts.split(",") if x.strip()]:
        spec = next((s for s in specs if s.get(pname)), None)
        if spec is None:
            print("  %s: 선언 없음" % pname)
            continue
        for mode, fn in (("per3000(현행)", per_item), ("fill(예산충전)", fill)):
            sel, dropped = fn(texts)
            joined = "\n---\n".join(sel)
            raw = ""
            try:
                try:
                    um = UserMessage(role="user", content=spec[pname].format(text=joined))
                except TypeError:
                    um = UserMessage(content=spec[pname].format(text=joined))
                kw = {k: v for k, v in dict(agent.llm_args).items() if "tool" not in k}
                sub = la.generate(model=agent.llm, tools=None, messages=[um],
                                  call_name="x133", **kw)
                raw = getattr(sub, "content", None) or ""
            except Exception as e:
                print("  %-14s %-16s 호출 실패 %r" % (pname, mode, e))
                continue
            got, rej, given = LG.parse_pairs(raw, FIELD.get(pname, "value"), hay)
            print("  %-16s %-16s 입력 %6dchars(탈락 %d·needle %s) → 모델 %d종·채택 %d·거절 %d %s"
                  % (pname, mode, len(joined), dropped,
                     "있음" if a.needle in joined else "없음",
                     given, len(got), rej,
                     ", ".join("%s=%s" % (k, v[0]) for k, v in sorted(got.items())) or ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
