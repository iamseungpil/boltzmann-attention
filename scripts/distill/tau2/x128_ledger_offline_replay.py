# -*- coding: utf-8 -*-
"""x128 — 고친 원장 경로가 **실제 궤적의 도구 출력 위에서** 무엇을 내놓는가 (유료 0).

왜 필요한가: `T2_LEDGER`는 라이브에서 네 번 다 `NameError`로 죽었고(win_20260807i),
고친 뒤에도 *"고쳤으니 될 것"* 은 궤적이 아니다. 이 프로브는 그 런의 **진짜 도구 출력**을 꺼내
`formalize_rows`(모델 전사) → `ledger_facts`(엔진 산수)를 그대로 태운다. 다른 것은 전부 같고
user-sim만 없다 — 그래서 무료이고, 그래서 실패하면 라이브에서도 실패한다.

가르는 것:
  · 전사가 0행이면      → 모델이 이 출력 형식을 못 읽는다(프롬프트 문제)
  · 행은 나오는데 무음  → 스펙 매칭·필드명 문제(엔진 문제)
  · 블록이 나오면       → 그 수가 기계 진리와 맞는지 눈으로 대조

usage: x128_ledger_offline_replay.py --dir bank_stack_win_20260807i --tasks task_100,task_101
       [--base http://localhost:8140/v1] [--model Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8]
"""

import argparse
import glob
import gzip
import io
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import t2_ledger as LG           # noqa: E402
import t2_offload as OFF         # noqa: E402
from gate_interpreter import load_domain_a2   # noqa: E402


def _load(dirname):
    """sim_results 의 gz 또는 tau2 의 raw results.json — 있는 쪽을 쓴다."""
    cands = [os.path.join(REPO, "reports", "facet_rft_2026", "sim_results", dirname + ".json.gz")]
    cands += glob.glob(os.path.join(os.path.expanduser("~"), "scratch", "tau2-bench",
                                    "data", "simulations", dirname, "results.json"))
    for p in cands:
        if not os.path.exists(p):
            continue
        op = gzip.open if p.endswith(".gz") else open
        with op(p, "rt", encoding="utf-8", errors="replace") as fh:
            return json.load(fh), p
    raise SystemExit("no results found for %r" % dirname)


def _tool_outputs(sim):
    """(호출 이름, 그 호출의 결과 본문) 쌍 — 이름은 디스패처면 내부 이름으로 푼다."""
    by_id, calls = {}, {}
    for m in sim.get("messages") or []:
        for tc in (m.get("tool_calls") or []):
            args = tc.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            inner = (args or {}).get("agent_tool_name") or ""
            nm = str(tc.get("name") or "")
            calls[tc.get("id")] = inner if (nm.startswith("call_") and inner) else nm
        if m.get("role") == "tool":
            by_id[m.get("id")] = m
    out = []
    for cid, name in calls.items():
        msg = by_id.get(cid)
        if msg is None or msg.get("error"):
            continue
        c = msg.get("content")
        if isinstance(c, list):
            c = "\n".join(str(x) for x in c)
        out.append((re.sub(r"_\d+$", "", str(name)), str(c or "")))
    return out


class _Agent(object):
    """`formalize_*`가 만지는 것만 — 모델 이름과 호출 인자."""

    def __init__(self, model, base):
        # litellm은 provider 접두사로 라우팅한다 — 없으면 호출이 **모델에 닿기 전에** 죽고,
        # `formalize_rows`의 `except`가 그것을 삼켜 "전사 0행"으로 보인다(1차 실행이 그랬다).
        # 라이브 러너와 같은 형태(`openai/<served-name>` + api_base)로 맞춘다.
        self.llm = model if "/" in model.split("/")[0] or model.startswith("openai/") \
            else "openai/" + model
        self.llm_args = {"temperature": 0.0, "api_base": base, "api_key": "dummy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tasks", default="task_100,task_101")
    ap.add_argument("--domain", default="banking_knowledge")
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    a = ap.parse_args()

    import tau2.agent.llm_agent as la
    from tau2.data_model.message import UserMessage

    data, src = _load(a.dir)
    a2 = load_domain_a2(a.domain)
    specs = (a2 or {}).get("ledger_metrics") or []
    print("source: %s\nledger_metrics 선언: %d개 (%s)\n"
          % (src, len(specs), ", ".join(s.get("trigger_tool", "?") for s in specs)))

    want = set(t.strip() for t in a.tasks.split(",") if t.strip())
    agent = _Agent(a.model, a.base)
    for sim in data.get("simulations") or []:
        tid = sim.get("task_id")
        if want and tid not in want:
            continue
        print("=" * 96)
        print("== %s ==" % tid)
        outs = _tool_outputs(sim)
        # ★라이브와 **같은 집합**을 먹여야 한다. 라이브 훅의 `_tx`는 role ∈ {tool, user}이고,
        #   이 환경이 "오늘"을 말하는 문장은 도구 출력이 아니라 **user 메시지**다 — 도구 출력만
        #   모으면 그 문장이 통째로 빠져 프로브가 라이브보다 *가난한* 입력으로 판정하게 된다.
        texts = []
        for m in sim.get("messages") or []:
            if m.get("role") not in ("tool", "user"):
                continue
            c = m.get("content")
            if isinstance(c, list):
                c = "\n".join(str(x) for x in c)
            texts.append(str(c or ""))
        hit = 0
        for name, content in outs:
            for spec in LG.specs_for(a2, name):
                hit += 1
                rows = LG.formalize_rows(agent, la, UserMessage, content, spec)
                print("  %-42s 전사 %d행" % (name, len(rows)))
                if not rows:
                    print("     ⚠0행 — 전사 실패(프롬프트 축). 원문 앞머리: %s"
                          % " ".join(content.split())[:120])
                    continue
                now = LG.formalize_now(agent, la, UserMessage, texts, spec)
                blk = OFF.ledger_facts(rows, spec, now=now)
                print("     now=%s" % now)
                print("     %s" % ("(블록 없음 — 엔진 산수가 낼 것이 없다)" if not blk
                                   else "\n     ".join(blk.strip().splitlines())))
        if not hit:
            print("  ⚠이 sim의 어떤 호출도 선언과 매칭되지 않았다 — 호출 이름: %s"
                  % ", ".join(sorted({n for n, _c in outs})))


if __name__ == "__main__":
    main()
