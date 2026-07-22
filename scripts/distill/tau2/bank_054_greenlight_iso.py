#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""★054 stall 격리 replay (2026-07-22·사용자 지시 "격리 실험"·[[08]]/[[05]]).

원인 격리: r13 054 = verify 후 게이트가 "you may now call log_verification" 초록불을 줬는데
에이전트가 `get_current_time`(시각 도구)을 안 불러 시각을 못 얻고 → 고객에게 시각 물으며 temp=0 루프.
(r11 성공=get_current_time 부름·r12 부름-but-날조. 격리는 궤적 forensic으로 이미 확인.)

이 프로브 = **결정점 재생**(givemiss_replay 동형·하네스 헬퍼 재사용·손문맥 금지):
  r13 054 궤적을 greenlight tool 메시지 *직후*에서 얼려 다음 에이전트 턴을 재샘플.
단일변수 = greenlight 메시지 내용:
  A  있는 그대로(스캐폴드 초록불)                      → 재현 + get_current_time 도달성(temp0.7)
  B  초록불 + " The current time is <T>."(해법#2 주입) → 시각 표면화가 루프를 닫나
  D  초록불 + " (call get_current_time first ...)"     → 전제 명명(대안 해법)
판정: A에서 get_current_time 0/n(temp0.7)이면 문맥이 하드-억제(스캐폴드 효과)·>0이면 변수/lockin.
      B/D가 log_verification(옳은 시각) 올리면 해법 인과 확정.

Run(리모트): python3 bank_054_greenlight_iso.py --base http://localhost:8140/v1 --n 8
"""
import argparse, gzip, json, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SIMDIR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
from bank_fab_probes import to_openai, load_tools, post, AGENT_INSTRUCTION  # noqa: E402

GOLD_TIME = "2025-11-14 03:40:00 EST"   # get_current_time이 실제 반환하는 값(r11 turn27 실측)


def load_sim(tag, tid):
    with gzip.open(os.path.join(SIMDIR, f"{tag}.results.json.gz"), "rt", encoding="utf-8") as f:
        sims = json.load(f)["simulations"]
    return next(s for s in sims if s["task_id"] == tid)


def freeze_at_greenlight(msgs):
    """greenlight tool 메시지(포함)까지 자른 원시 메시지 리스트 반환."""
    for i, m in enumerate(msgs):
        if m.get("role") == "tool" and "you may now call log_verification" in str(m.get("content", "")).lower():
            return msgs[: i + 1], i
    raise RuntimeError("greenlight not found")


def classify(msg):
    tcs = msg.get("tool_calls") or []
    if not tcs:
        c = str(msg.get("content", "")).lower()
        if "current time" in c or "timestamp" in c:
            return "산문:시각-요청"
        return "산문:기타"
    names = [t["function"]["name"] for t in tcs]
    if "get_current_time" in names:
        return "★get_current_time"
    if "log_verification" in names:
        tc = next(t for t in tcs if t["function"]["name"] == "log_verification")
        try:
            args = json.loads(tc["function"].get("arguments") or "{}")
        except Exception:
            args = {}
        tv = str(args.get("time_verified", ""))
        return "log_verification(옳은시각)" if "2025-11-14 03:40" in tv else f"log_verification(틀린시각:{tv[:22]})"
    if "verify_identity" in names:
        return "verify_identity(재확인)"
    return "다른도구:" + names[0]


def run_arm(base, model, conv, tools, temp, n, max_tokens=400):
    c = Counter()
    for _ in range(n):
        payload = {"model": model, "messages": conv, "tools": tools,
                   "temperature": temp, "max_tokens": max_tokens, "n": 1}
        try:
            r = post(base, payload, timeout=420)
            c[classify(r["choices"][0]["message"])] += 1
        except Exception as e:
            c["ERR:" + repr(e)[:40]] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8140/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--tag", default="bank_rall13b_20260722")
    a = ap.parse_args()

    tools, policy, a2 = load_tools()
    sysmsg = [{"role": "system", "content": AGENT_INSTRUCTION + "\n\n<policy>\n" + policy + "\n</policy>"}]
    s = load_sim(a.tag, "task_054")
    frozen, gi = freeze_at_greenlight(s["messages"])
    base_conv = sysmsg + to_openai(frozen)
    gl_text = frozen[-1]["content"]
    print(f"frozen at greenlight idx={gi}, {len(frozen)} msgs. greenlight='{gl_text[:90]}'", flush=True)

    # arm 변형 = 마지막 tool 메시지 content 치환 (단일변수)
    def with_gl(extra):
        conv = [dict(m) for m in base_conv]
        conv[-1] = dict(conv[-1]); conv[-1]["content"] = gl_text + extra
        return conv

    # ★E arm = 실제 shipped A2 met_template(ledger variant) 렌더 = 배포 문구 그대로 검증([[03b]])
    a2met = None
    try:
        mt = a2["scaffold_get_tools"]
        vt = [d for d in mt if d.get("name") == "verify_identity"][0]
        tmpl = vt["variants"]["ledger"]["op"]["met_template"]
        a2met = tmpl.format(count=3, matched="address, phone_number, email")
    except Exception as e:
        print("A2 met_template load failed:", e)

    def replace_gl(newtext):
        conv = [dict(m) for m in base_conv]
        conv[-1] = dict(conv[-1]); conv[-1]["content"] = newtext
        return conv

    arms = {
        "A_asis":  base_conv,
        "B_time":  with_gl(f" The current time is {GOLD_TIME}."),
        "D_hint":  with_gl(" If you need the timestamp for time_verified, call get_current_time first, then call log_verification."),
    }
    if a2met:
        arms["E_a2met_shipped"] = replace_gl(a2met)
    for name, conv in arms.items():
        c0 = run_arm(a.base, a.model, conv, tools, 0.0, 1)
        c7 = run_arm(a.base, a.model, conv, tools, 0.7, a.n)
        print(f"\n[{name}]  temp0(1)={dict(c0)}\n         temp0.7(n={a.n})={dict(c7)}", flush=True)
    print("\n판정: A_asis get_current_time율=하드억제 여부 · B/D log_verification(옳은시각)율=해법 인과", flush=True)


if __name__ == "__main__":
    main()
