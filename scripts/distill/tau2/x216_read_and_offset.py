# -*- coding: utf-8 -*-
r"""x216 — ⓐ계좌 미조회 표의 **상쇄** · ⓑ**격리하면 읽으라는 말을 듣는가** (유료 0 · 엔진 0).

## 왜 (2026-08-10 부검 · 사용자 지시)

유료 런 099 t0 은 계좌를 **안 읽어서** 표가 아예 안 만들어졌다. 트레이스: 읽으라는 요구
`T2_DEMANDED_STEP` **12회** · 호출 가능한 실제 이름까지 변환해 제시 `T2_CALLABLE_FRONTIER`
**12회** · 그런데도 미호출. **더 말해서 될 문제가 아니다.**

남는 정당한 길 둘을 여기서 잰다 —

### §1 상쇄 — 표가 그 읽기에 의존하지 않게 하면?

`eligible_text` 는 피연산자 하나만 있어도 발화한다(회귀 계약). 계좌를 안 읽으면 재직기간
(`days`)이 없을 뿐 예치 기준으로는 거를 수 있다. **그런데 공짜가 아니다** — 100 은 **재직 함정**
이라(65일 · World Blue 90 · Hunter Green 60) `days` 가 없으면 World Blue 를 못 뺀다.
*"모름 ≠ 탈락"* 대로 남기면 100 이 틀린다. ⇒ **099 를 여는 것이 100 을 깨는지** 를 잰다.

  DAYS      재직 있음 (지금 거동)
  NODAYS    재직 없음 (계좌 미조회 상태의 표)

### §2 읽기 순종 — 격리하면 부르나 (사용자 가설)

> *"격리해서 읽으라고 하면 읽을 수도 있을 것 같다."*

궤적에서 12회 무시된 요구가 **깨끗한 문맥**에서는 먹히는지 본다. 먹힌다면 부하/혼잡은
추론뿐 아니라 **순종**에도 걸리는 것이고, 그건 새 축이다.

  ISO       손님의 말 + 우리 요구 + 도구 스키마만 (깨끗)
  TRAJ      실패 궤적의 실제 문맥 + 같은 요구 + 같은 스키마
  ISO_NO    요구 없이 손님의 말만 (부정 통제 — 요구 없이도 부르면 요구의 공로가 아니다)

⚠도구 호출 여부만 센다. 어떤 인자를 주는지는 여기서 판정하지 않는다.

실행: python x216_read_and_offset.py [N]
"""
import collections
import json
import os
import re
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_ledger as LG                                          # noqa: E402
import t2_factdag as FD                                         # noqa: E402
from gate_interpreter import load_domain_a2                      # noqa: E402
from x207_rederive_objective import CASE                         # noqa: E402

URL = os.environ.get("T2_PROBE_URL", "http://localhost:8140/v1/chat/completions")
MODEL = os.environ.get("T2_PROBE_MODEL", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
TARGET = "get_all_user_accounts_by_user_id_3847"
# 우리 층이 실제로 보내는 요구 문구 (트레이스 축자 형태)
DEMAND = ("Before advising on a referral, the eligibility check on record requires reading this "
          "customer's accounts. The callable name for it is %s." % TARGET)
SAID = {"task_099": ("My friend is starting a new business and will deposit about $30,000. I want "
                     "to refer her. Which account gives me the biggest referral bonus?"),
        "task_100": ("My former business partner just incorporated her LLC and will deposit about "
                     "$31,000. I want to refer her so I get the biggest referral bonus.")}
TOOLS = [
    {"type": "function", "function": {
        "name": "unlock_discoverable_agent_tool", "description": "Unlock a discoverable tool.",
        "parameters": {"type": "object", "properties": {"agent_tool_name": {"type": "string"}},
                       "required": ["agent_tool_name"]}}},
    {"type": "function", "function": {
        "name": "call_discoverable_agent_tool", "description": "Call an unlocked discoverable tool.",
        "parameters": {"type": "object",
                       "properties": {"agent_tool_name": {"type": "string"},
                                      "arguments": {"type": "string"}},
                       "required": ["agent_tool_name", "arguments"]}}},
    {"type": "function", "function": {
        "name": "get_referrals_by_user", "description": "List this user's referral records.",
        "parameters": {"type": "object", "properties": {"user_id": {"type": "string"}},
                       "required": ["user_id"]}}},
    {"type": "function", "function": {
        "name": "KB_search_bm25", "description": "Search the knowledge base.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
]


def chat(prompt, tools=None, temp=0.0, mx=200, choices=None):
    body = {"model": MODEL, "temperature": temp, "max_tokens": mx,
            "messages": [{"role": "user", "content": prompt}]}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if choices:
        body["guided_choice"] = list(choices)
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["choices"][0]["message"]


def called_target(msg):
    for tc in (msg.get("tool_calls") or []):
        f = tc.get("function") or {}
        blob = str(f.get("name", "")) + str(f.get("arguments", ""))
        if TARGET in blob or "get_all_user_accounts" in blob:
            return True
    return False


def traj_context():
    """실패한 099 궤적에서 **요구가 나갔던 시점까지**의 문맥 (없으면 빈 문자열)."""
    import glob
    import gzip
    for p in sorted(glob.glob("/home/woori/scratch/tau2-bench/data/simulations/*/results.json")) + \
            sorted(glob.glob("/home/woori/workspace_common/boltzmann-attention-pi/reports/"
                             "facet_rft_2026/sim_results/*.json.gz")):
        try:
            f = gzip.open(p, "rt", encoding="utf-8") if p.endswith(".gz") else open(p, encoding="utf-8")
            d = json.load(f)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        for s in d.get("simulations") or []:
            if not isinstance(s, dict) or s.get("task_id") != "task_099":
                continue
            if (s.get("reward_info") or {}).get("reward") == 1:
                continue
            msgs = s.get("messages") or []
            if not any("get_referrals_by_user" in str((tc.get("function") or {}).get("name"))
                       for m in msgs for tc in (m.get("tool_calls") or [])):
                continue
            parts = []
            for m in msgs:
                c = " ".join(str(m.get("content") or "").split())
                for tc in (m.get("tool_calls") or []):
                    fn = tc.get("function") or tc
                    parts.append("[%s calls %s]" % (m.get("role"), fn.get("name")))
                if c:
                    parts.append("[%s] %s" % (m.get("role"), c))
                if len(parts) > 40:
                    break
            return "\n".join(parts)[:60000]
    return ""


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8
    a2 = load_domain_a2("banking_knowledge")
    sp = next(x for x in a2["ledger_metrics"] if x.get("eligible_text"))
    cfg = sp["eligible"]
    rows = (a2.get("policy_ontology") or {}).get("rows") or []
    kb = LG.subject_kinds(rows, cfg.get("kind_field") or "kind")
    maps0 = {ax: FD._a3_map(rows, {"axis": ax}) for ax in (cfg.get("show_axes") or [])}
    tpl = sp["rederive_prompt"]
    out = {}

    print("=" * 92)
    print("§1 상쇄 — 계좌를 안 읽어 `days` 가 없는 표는 무엇을 낳나")
    print("=" * 92)
    for task in ("task_099", "task_100"):
        c = CASE[task]
        maps, _d = LG.restrict_to_kind(maps0, kb, c["kind"])
        for arm, days in (("DAYS", c["days"]), ("NODAYS", None)):
            tbl = (LG.eligible_text(days, c.get("tally") or {}, maps, sp, c["stated"]) or "").strip()
            names = [l.strip().split(":")[0].strip() for l in tbl.splitlines()
                     if l.startswith("  ") and ":" in l]
            if not names:
                print("  %-9s %-6s 표가 비었다 — 침묵" % (task, arm))
                continue
            facts = "\n".join((["days since the earliest account was opened = %d" % days]
                               if days is not None else [])
                              + ["%s = %s" % (k, LG._num(v)) for k, v in sorted(c["stated"].items())])
            cnt = collections.Counter()
            for i in range(n):
                p = tpl.format(table=tbl, facts=facts)
                try:
                    cnt[chat(p, None, 0.0 if i == 0 else 0.7, 24, names).get("content", "").strip()] += 1
                except Exception as e:
                    cnt["ERR %s" % type(e).__name__] += 1
            hit = sum(v for k, v in cnt.items() if str(k).strip() == c["gold"])
            out["%s/%s" % (task, arm)] = [hit, n]
            print("  %-9s %-6s 표 %d행(%s) → gold %d/%d  %s"
                  % (task, arm, len(names), ", ".join(names)[:60], hit, n, cnt.most_common(1)))

    print("\n" + "=" * 92)
    print("§2 읽기 순종 — 궤적에서 12회 무시된 요구가 격리에서는 먹히나")
    print("=" * 92)
    tj = traj_context()
    print("  궤적 문맥 %d자%s" % (len(tj), "" if tj else "  ⚠못 찾음 — TRAJ 팔 생략"))
    arms = [("ISO", SAID["task_099"] + "\n\n" + DEMAND),
            ("ISO_NO", SAID["task_099"])]
    if tj:
        arms.append(("TRAJ", tj + "\n\n" + DEMAND))
    for arm, body in arms:
        c = collections.Counter()
        for i in range(n):
            try:
                m = chat(body + "\n\nWhat do you do next?", TOOLS, 0.0 if i == 0 else 0.7)
            except Exception as e:
                c["ERR"] += 1
                continue
            c["호출O" if called_target(m) else "호출X"] += 1
            for tc in (m.get("tool_calls") or []):
                c["도구:" + str((tc.get("function") or {}).get("name"))] += 1
        out["read/" + arm] = [c["호출O"], n]
        print("  %-7s 계좌읽기 호출 %d/%d   %s"
              % (arm, c["호출O"], n, [x for x in c.most_common(4) if str(x[0]).startswith("도구")]))
    json.dump(out, open(os.environ.get("T2_X216_OUT", "x216_out.json"), "w"), indent=1)
    print("\n※ §1 에서 NODAYS 가 099 를 열고 100 을 깨면 그것이 상쇄다 — 켜려면 조건을 갈라야 한다."
          "\n  §2 에서 ISO 가 TRAJ 보다 높으면 **순종에도 부하가 걸린다** — 읽기도 격리 대상이 된다."
          "\n  ISO_NO 가 ISO 와 같으면 요구의 공로가 아니라 문맥의 공로다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
