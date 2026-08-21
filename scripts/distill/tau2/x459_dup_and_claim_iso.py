# -*- coding: utf-8 -*-
r"""x459 — DUP 과 완료-사칭을 **격리로 가른다** (2026-08-21·무료·LLM 로컬)

## 왜
`x458`(C583)이 기전을 지목했다. 지목은 관측이지 인과가 아니므로 [[62]] 순서대로 격리로 잰다.

    ⒜ DUP  — 085 의 `log_verification` 반복 4건 중 3건이 **우리 문구 직후**다:
             축자 *"VERIFIED — 2 identity values match … you may now call log_verification"*.
             가설: **우리가 그 말을 다시 하니 모델이 다시 부른다**([[55]]·[[64]] 계보).
    ⒝ 사칭 — 074 두 sim 이 도구를 **한 번도 안 부르고** 끝말이 *"The corrections have been
             applied"* 다. 가설: 결정점에서 **행동 없이 완료를 선언**한다([[46]] knowing-doing).

## 팔 (한 변수만 다르다 · 재료는 실제 궤적에서 자른다)
⒜ 두 번째 호출 **직전까지**의 문맥으로 *"다음 도구 호출"* 을 묻는다:
    A_asis     궤적 그대로 (우리 문구가 다시 들어 있다)
    B_strip    **우리 문구의 재발화만** 중립 문장으로 치환 (다른 것은 전부 동일)
    C_state    우리 문구를 남기되 **이미 기록됨**을 함께 말한다([[64]] 무엇을 하면 되는지)
  채점(닫힌): 다음 호출이 `log_verification` 인가.
  ⇒ A 가 부르고 B 가 안 부르면 **원인은 우리 문구**다. C 가 안 부르면 **문구 수리로 닫힌다**.

⒝ 마지막 발화 **직전까지**의 문맥으로 *"다음에 무엇을 하나"* 를 묻는다:
    A_asis     궤적 그대로
    N_neg      **부정통제**([[57]]) — 무내용 재촉("Please continue.")만 덧붙임
    D_name     "완료를 말하려면 그렇게 만든 도구 호출을 대라"([[64]] 이름을 대는 거부)
  채점(닫힌): 답이 **도구 호출**인가 **완료 선언**인가(도구 0회인데 done 이면 사칭).
  ⛔[[46]]: 이 축은 라이브 A/B 가 **null**(1/16↔1/16)이었다. 격리서 갈려도 승격 금지.

⛔처방을 만들지 않는다. 갈리는지만 본다.

사용: (리모트·cwd=tau2 · PYTHONPATH=src:…) py x459_dup_and_claim_iso.py --port 8141
"""
import argparse
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                 # noqa: E402  정본 로더·래퍼 해제
import x431_spec_selects as X           # noqa: E402  ask 정본(사본 금지·[[67]])

REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
NLC = chr(10)
DUP_TOOL = "log_verification"
CLAIM_TASKS = ("task_074",)

# 우리 층이 그 자리에서 내보내는 문구의 **머리**(엔진이 스스로 찍는 문면·도메인 어휘 0)
OURS_HEAD = "VERIFIED"
NEUTRAL = "(identity check completed earlier in this conversation)"

SYS_NEXT = (
    "You are continuing a customer-service session. Given the transcript so far, decide the ONE "
    "next step. Reply with ONE JSON object only:\n"
    '{"next": "tool" or "final", "tool": "<tool name, if next is tool>", '
    '"why": "<one short sentence>"}\n'
    "Answer only from the transcript. Do not invent tools that do not appear in it."
)


def _role(m):
    return str(m.get("role") or "") if isinstance(m, dict) else ""


def _content(m):
    return str((m.get("content") if isinstance(m, dict) else None) or "")


def render(msgs, strip_ours=False, state_done=False, append=None):
    """궤적 조각을 한 덩이 텍스트로. **우리 문구만** 갈아 끼울 수 있게 한다."""
    out = []
    for m in msgs:
        r = _role(m)
        c = " ".join(_content(m).split())
        if r == "tool" and c.startswith(OURS_HEAD):
            if strip_ours:
                c = NEUTRAL
            elif state_done:
                c = c + " NOTE: this verification has already been logged; it does not need logging again."
        if r == "user" and c:
            out.append("CUSTOMER: " + c[:600])
        elif r == "assistant":
            tcs = [F.nameof(t) for t in (m.get("tool_calls") or [])]
            if tcs:
                out.append("AGENT calls: " + ", ".join(tcs))
            elif c:
                out.append("AGENT: " + c[:400])
        elif r == "tool" and c:
            out.append("TOOL RESULT: " + c[:400])
    if append:
        out.append(append)
    return NLC.join(out)


def dup_cases(sims):
    """두 번째 `log_verification` 직전까지를 자른다 — 그 자리가 재발화의 결정점이다."""
    cases = []
    for s in sims:
        msgs = s.get("messages") or []
        hits = [i for i, m in enumerate(msgs)
                for tc in (m.get("tool_calls") or []) if F.nameof(tc) == DUP_TOOL]
        for k in range(1, len(hits)):
            pre = msgs[:hits[k]]
            if not any(_role(m) == "tool" and _content(m).strip().startswith(OURS_HEAD) for m in pre):
                continue                        # 우리 문구가 없는 반복은 이 가설의 대상이 아니다
            cases.append({"task": F.task_id(s), "sim": F.sim_key(s), "at": hits[k], "pre": pre})
    return cases


def claim_cases(sims):
    """**gold 가 요구한 변이가 남아 있는 채** 말로 끝낸 sim — 마지막 발화 직전까지.

    ⚠1차 판은 *"성공 변이가 하나도 없는 sim"* 으로 걸렀는데 074 는 **다른 도구는 성공시켰고**
      `apply_checking_account_credit` 만 안 했다 ⇒ 결정점 **0개**로 프로브가 아예 안 돌았다.
      술어를 *"MISSING 이 남아 있다"* 로 바꾼다 — 그것이 *"할 일이 남았는데 말로 끝냈나"* 다.
    """
    cases = []
    for s in sims:
        d = F.mutation_diff(s)
        if not d["missing"]:
            continue                            # 남은 gold 변이가 없으면 사칭할 것도 없다
        msgs = s.get("messages") or []
        last = None
        for i, m in enumerate(msgs):
            if _role(m) == "assistant" and _content(m).strip() and not (m.get("tool_calls") or []):
                last = i
        if last is None:
            continue
        cases.append({"task": F.task_id(s), "sim": F.sim_key(s), "at": last,
                      "pre": msgs[:last], "said": _content(msgs[last])[:200],
                      "missing": sorted({m["name"] for m in d["missing"]})[:4]})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8141)
    ap.add_argument("--tags", default="")
    ap.add_argument("--n", type=int, default=3, help="같은 문맥을 몇 번 물을지(greedy 라 변동은 작다)")
    ap.add_argument("--claim-tasks", default="072,073,074",
                    help="빈 문자열이면 1단계 전수")
    ap.add_argument("--max-claim", type=int, default=12)
    ap.add_argument("--out", default="x459_dup_claim_iso.json")
    a = ap.parse_args()

    tags = [t for t in a.tags.split(",") if t.strip()] or \
        [F.tag_of_file(p) for p in F.all_result_files() if "t7328" in os.path.basename(p)]
    sims = []
    for t in sorted(set(tags)):
        try:
            sims += list(F.sims(t))
        except Exception:
            pass
    print("=" * 96)
    print("x459 · sim %d · 태그 %s" % (len(sims), ", ".join(sorted(set(tags)))))
    print("=" * 96)

    rows = []

    # ── ⒜ DUP ────────────────────────────────────────────────────────────────
    dc = dup_cases(sims)
    print(NLC + "⒜ DUP 결정점 %d개 (우리 문구가 문맥에 있는 재호출만)" % len(dc))
    agg = collections.Counter()
    for c in dc:
        for arm, kw in (("A_asis", {}), ("B_strip", {"strip_ours": True}),
                        ("C_state", {"state_done": True})):
            body = render(c["pre"], **kw) + NLC + NLC + "What is the ONE next step?"
            calls = 0
            for _ in range(a.n):
                got = X.ask(a.port, SYS_NEXT, body, maxtok=200) or {}
                if str(got.get("tool") or "") == DUP_TOOL:
                    calls += 1
            agg[(arm, "again")] += calls
            agg[(arm, "n")] += a.n
            rows.append({"probe": "dup", "task": c["task"], "sim": c["sim"],
                         "arm": arm, "again": calls, "n": a.n})
            print("   %-9s %-8s log_verification 재호출 %d/%d" % (c["task"], arm, calls, a.n))
    print("   [합] " + " · ".join("%s %d/%d" % (arm, agg[(arm, "again")], agg[(arm, "n")])
                                  for arm in ("A_asis", "B_strip", "C_state")))

    # ── ⒝ 완료 사칭 ──────────────────────────────────────────────────────────
    cc = claim_cases(sims)
    if a.claim_tasks:
        want = {"task_" + x.strip() for x in a.claim_tasks.split(",") if x.strip()}
        cc = [c for c in cc if c["task"] in want]
    cc = cc[:a.max_claim]
    print(NLC + "⒝ 완료-사칭 결정점 %d개 (성공 변이 0 인 sim 의 마지막 발화 직전)" % len(cc))
    agg2 = collections.Counter()
    for c in cc:
        print("   %-9s 남은 gold: %s" % (c["task"], ", ".join(c["missing"])[:70]))
        print("      실제 끝말: %s" % c["said"][:110].replace(NLC, " "))
        for arm, ap_ in (("A_asis", None), ("N_neg", "CUSTOMER: Please continue."),
                         ("D_name", "SYSTEM: If you are about to say something is done, you must "
                                    "first name the tool call that did it. If no such call was "
                                    "made, make it now.")):
            body = render(c["pre"], append=ap_) + NLC + NLC + "What is the ONE next step?"
            tool_n = 0
            for _ in range(a.n):
                got = X.ask(a.port, SYS_NEXT, body, maxtok=200) or {}
                if str(got.get("next") or "") == "tool" and got.get("tool"):
                    tool_n += 1
            agg2[(arm, "tool")] += tool_n
            agg2[(arm, "n")] += a.n
            rows.append({"probe": "claim", "task": c["task"], "sim": c["sim"],
                         "arm": arm, "tool": tool_n, "n": a.n})
            print("      %-8s 도구를 고름 %d/%d (나머지는 완료 선언)" % (arm, tool_n, a.n))
    print("   [합] " + " · ".join("%s %d/%d" % (arm, agg2[(arm, "tool")], agg2[(arm, "n")])
                                  for arm in ("A_asis", "N_neg", "D_name")))

    p = os.path.join(REP, a.out)
    with io.open(p, "w", encoding="utf-8") as f:
        json.dump({"tags": sorted(set(tags)), "rows": rows,
                   "dup_summary": {a2: [agg[(a2, "again")], agg[(a2, "n")]]
                                   for a2 in ("A_asis", "B_strip", "C_state")},
                   "claim_summary": {a2: [agg2[(a2, "tool")], agg2[(a2, "n")]]
                                     for a2 in ("A_asis", "N_neg", "D_name")}},
                  f, ensure_ascii=False, indent=1)
    print(NLC + "→ %s" % p)
    print(NLC + "판정: ⒜ A 높고 B 낮으면 **원인은 우리 문구** · C 도 낮으면 문구 수리로 닫힌다.")
    print("      ⒝ N_neg 이 A 와 같아야 부정통제 성립 — D_name 만 오르면 그때가 신호다.")
    print("      ⛔[[46]]: ⒝ 는 라이브 A/B 가 null 이었다. 격리서 갈려도 승격 금지.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
