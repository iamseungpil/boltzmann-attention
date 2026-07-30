#!/usr/bin/env python3
"""C241 U3' 측정 게이트 — REF_ISO 서브콜의 **페르소나 명사 제거**가 행동을 바꾸는지 (2026-07-30).

U3'는 `t2_gate_patch.py`의 REF_ISO 서브콜 프롬프트를
  base   "You are a precise **banking** assistant."
  treat  "You are a precise assistant."
로 바꿨다. 프롬프트 변경이므로 [[03b]] 규율상 **행동 불변을 주장할 수 없다**. 이 프로브가 그
게이트다: 같은 입력에 두 페르소나를 돌려 **선택 일치율**을 재고, 불일치 시 어느 쪽이 gold에
가까운지 본다.

★프롬프트는 엔진 코드(`t2_gate_patch.py` REF_ISO 블록)의 구성을 **축자 복제**한다 — 재구성이
다르면 측정이 무의미하다. 입력은 day6~9c 실 궤적에서 A2 `ref_iso` 스펙이 적용되는 케이스를
추출한다(합성 아님).

⚠**프록시 한계**: 라이브 REF_ISO는 에이전트 모델(32B·8140)을 쓰는데 그 서버는 Y1 본런이
점유 중이다. 이 프로브는 **7B 프록시**이므로 "7B에서 불일치 없음"이 32B 불변을 함의하지 않는다.
32B arm은 8140 해제 후 별건.

용법: py -3 x9_refiso_persona_probe.py --model 7B --seeds 3 --temp 0.0
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys
import urllib.request
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.abspath(os.path.join(_HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
PORTS = {"7B": ("http://localhost:8142/v1", "Qwen/Qwen2.5-7B-Instruct"),
         "32B": ("http://localhost:8140/v1", "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8")}

PERSONA_BASE = "You are a precise banking assistant."
PERSONA_TREAT = "You are a precise assistant."


def build_prompt(persona, utext, listing, others, pn, match_hint):
    """★`t2_gate_patch.py` REF_ISO 블록의 프롬프트 구성 축자 복제."""
    return (persona + "\n\n"
            "=== CUSTOMER MESSAGES (verbatim) ===\n" + utext
            + "\n\n=== RECORD LISTING (tool output) ===\n" + listing[:20000]
            + "\n\n=== ACTION BEING FILED ===\n"
            + json.dumps(others, default=str)[:800]
            + "\n\nWhich single '" + str(pn) + "' value from the RECORD LISTING does this "
              "action refer to, based on the customer's messages? If the customer listed "
              "several items, first match EVERY listed item to its record"
            + ((" (" + str(match_hint) + ")") if match_hint else "")
            + ", then answer for THIS action only. Answer with EXACTLY "
              "one value copied from the listing, or UNSURE.")


def load_cases(spec, limit=0):
    """day6~9c 궤적에서 REF_ISO 적용 케이스 추출 (엔진과 동일 술어)."""
    prod = set(spec.get("producer_tools") or [])
    applies_to = spec.get("applies_to")
    aw = spec.get("applies_when") or {}
    arg_key, prefix = aw.get("arg"), aw.get("prefix")
    pn = spec.get("param")
    cases = []
    for path in sorted(glob.glob(os.path.join(_SIM, "bank_day*front*.results.json.gz"))):
        try:
            data = json.load(gzip.open(path, "rt", encoding="utf-8"))
        except Exception:
            continue
        for sim in data.get("simulations") or []:
            msgs = sim.get("messages") or []
            pids = {tc.get("id") for m in msgs for tc in (m.get("tool_calls") or [])
                    if tc.get("name") in prod}
            listing = ""
            for m in msgs:
                if (m.get("role") == "tool" and m.get("id") in pids and not m.get("error")):
                    c = m.get("content")
                    if isinstance(c, str) and len(c) > len(listing):
                        listing = c
            if not listing:
                continue
            utext = "\n".join(str(m.get("content") or "") for m in msgs
                              if m.get("role") == "user")[:6000]
            for m in msgs:
                for tc in (m.get("tool_calls") or []):
                    if tc.get("name") != applies_to:
                        continue
                    a = tc.get("arguments") or {}
                    if prefix and not str(a.get(arg_key, "")).startswith(prefix):
                        continue
                    nested = a.get("arguments")
                    if isinstance(nested, str):
                        try:
                            nested = json.loads(nested)
                        except Exception:
                            nested = None
                    nd = nested if isinstance(nested, dict) else {}
                    cur = str(nd.get(pn) or a.get(pn) or "")
                    if not cur:
                        continue
                    cases.append({
                        "file": os.path.basename(path), "sim": sim.get("id"),
                        "agent_chose": cur,
                        "others": {k: v for k, v in nd.items() if k != pn},
                        "utext": utext, "listing": listing,
                    })
    # 중복 (agent_chose, listing 앞부분) 제거 — 같은 상황 반복 호출 축약
    seen, out = set(), []
    for c in cases:
        k = (c["agent_chose"], c["listing"][:200])
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out[:limit] if limit else out


def call(base, model, prompt, seed, temp):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 40, "temperature": temp, "seed": seed}
    req = urllib.request.Request(base + "/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return (json.loads(r.read())["choices"][0]["message"]["content"] or "").strip()


def norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip().strip('"\'`.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7B")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    # ★2026-07-30 야간: 8140은 Y1 본런이 점유 중이라 32B arm은 GPU1의 8141로 돌린다.
    #   모델 id는 PORTS 그대로(동일 체크포인트·동일 서빙 스펙) — 포트만 갈아끼운다.
    ap.add_argument("--base_url", default="", help="PORTS 기본 endpoint 덮어쓰기(예: http://localhost:8141/v1)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    out_path = args.out or os.path.join(_SIM, f"x9_refiso_persona_{args.model}.jsonl")

    a2 = json.load(open(os.path.join(_HERE, "a2", "banking_knowledge.gate.json"),
                        encoding="utf-8"))
    specs = a2.get("ref_iso") or []
    if not specs:
        sys.exit("A2 ref_iso 없음")
    spec = specs[0]
    pn, hint = spec.get("param"), spec.get("match_hint")
    cases = load_cases(spec, args.limit)
    print(f"REF_ISO 적용 케이스 {len(cases)}건 · param={pn} · model={args.model} "
          f"seeds={args.seeds} temp={args.temp}")
    if not cases:
        sys.exit("케이스 0 — 추출 술어 확인")

    base, model = PORTS[args.model]
    if args.base_url:
        base = args.base_url.rstrip("/")
    print(f"endpoint={base} model={model}")
    rows, agree, diff_rows = [], 0, []
    for i, c in enumerate(cases):
        for seed in range(args.seeds):
            try:
                pb = build_prompt(PERSONA_BASE, c["utext"], c["listing"], c["others"], pn, hint)
                pt = build_prompt(PERSONA_TREAT, c["utext"], c["listing"], c["others"], pn, hint)
                rb, rt = (call(base, model, pb, seed, args.temp),
                          call(base, model, pt, seed, args.temp))
            except Exception as e:
                rows.append({"i": i, "seed": seed, "error": str(e)[:150]})
                continue
            same = norm(rb) == norm(rt)
            agree += same
            r = {"i": i, "seed": seed, "sim": c["sim"], "file": c["file"],
                 "agent_chose": c["agent_chose"], "base": norm(rb), "treat": norm(rt),
                 "same": same, "endpoint": base, "model": model}
            rows.append(r)
            if not same:
                diff_rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"  ...{i + 1}/{len(cases)}")

    ok = [r for r in rows if "error" not in r]
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print()
    print("=" * 72)
    print("U3' 측정 게이트 — 페르소나 명사 제거의 행동 영향")
    print("=" * 72)
    n = len(ok)
    print(f"비교 {n}쌍 (케이스 {len(cases)} × 시드 {args.seeds}) · 에러 {len(rows) - n}")
    if n:
        print(f"**선택 일치율 = {agree}/{n} = {100.0 * agree / n:.1f}%**")
    # 에이전트가 고른 값과의 일치(양쪽 각각)
    for k in ("base", "treat"):
        m = sum(1 for r in ok if norm(r[k]) == norm(r["agent_chose"]))
        print(f"  {k:5s}가 에이전트 원선택과 동일: {m}/{n}")
    uns = {k: sum(1 for r in ok if r[k].upper().startswith("UNSURE")) for k in ("base", "treat")}
    print(f"  UNSURE 빈도: base {uns['base']} · treat {uns['treat']}")
    if diff_rows:
        print()
        print(f"불일치 {len(diff_rows)}건 (정독 대상·[[08]]):")
        for r in diff_rows[:12]:
            print(f"  sim={str(r['sim'])[:14]} seed={r['seed']}")
            print(f"    base : {r['base'][:70]}")
            print(f"    treat: {r['treat'][:70]}")
            print(f"    agent: {r['agent_chose'][:70]}")
    print()
    print("판정 규칙: 일치율 100%면 U3' 행동-불변 [M]. 불일치가 있으면 건별 정독 후")
    print("  ①treat가 더 정확 → GO ②base가 더 정확 → A2 `ref_iso[].persona` 신설로 후퇴")
    print("  ③혼재 → 32B arm(8140 해제 후)까지 보류.")
    print(f"⚠프록시 한계: 라이브 REF_ISO는 에이전트 모델(32B). 이 결과는 {args.model} 기준.")
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
