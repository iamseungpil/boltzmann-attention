# -*- coding: utf-8 -*-
r"""x552 - 074 격리: **원장이 비었다는 사실**을 이관 출구에서 알려주면 모델이 행동하는가

## 왜 이 칸인가 (2026-08-26 · t7361 per-step 포렌식)

074 는 gold 변이 5 중 **1**(신원 기록)만 하고 끝났다. 종료 방식이 완료 주장이 아니라
**인간 이관**이다 — msg52 `unlock:emergency_credit_bureau_incident_transfer` ·
msg54 `unlock:initial_transfer_to_human_agent`. 그 직전 우리 층은:

    [T2_MATERIAL_GATE] stop=resolve_cap(정체 3회) turn=54
    [T2_WRITEPROV] window hit (no effective write in ledger) declared_completion=False

⇒ 우리는 **원장이 비었다는 것을 보고도** 침묵했다. `WRITE_PROV` 는 *완료 주장* 을 전제하고
074 는 주장하지 않았기 때문이다. `PROCEDURE_LEFT` 도 못 닿는다 — banking 이 선언한 절차
**6개 중 ATM 수수료 환급 흐름이 없어** 남은 칸이라는 개념 자체가 없다.

## 이 프로브가 재는 것 (하나)

*이관하려는 자리에서 "이 대화에서 상태를 바꾼 실행이 없다"는 **사실**을 주면, 다음 수가
바뀌는가.* 처방이 아니라 **원장 사실 한 문장**이다 — 무엇을 하라고 고르지 않는다([[62]]④·[[64]]).

⚠**선언에 이식할 텍스트가 없다**([[78]]②). `completion_guard.feedback` 은 *"완료를 주장하지
  마라"* 인데 074 는 주장하지 않았다 — 그 문장을 여기 쓰면 동문서답이다. 그래서 팔 B 의 문장은
  **원장에서 기계적으로 나오는 사실**이고, 이 팔이 이기면 그것은 엔진 리터럴이 아니라
  **A2 저작 항목**으로 넘긴다([[72]]·[[24]] 정본 층).

## 팔 ([[57]] 부정통제 포함)

    A_asis   이관 직전 창 그대로            <- 라이브(이관)를 재현해야 한다
    B_fact   창 + 원장 사실 한 문장          <- 수리 후보
    N_len    창 + 궤적의 **다른 문장**(같은 길이대) <- 길이가 아니라 내용임을 가른다

A_asis 가 이관을 재현 못 하면 격리가 불공정하고 **판정하지 않는다**([[62]] 2b).

## 채점 - 닫힌 술어 · gold 미접촉([[23]])

다음 도구 이름 하나를 묻고, 그 이름이 **이관 계열인지**만 본다(이름에 `transfer_to_human`
또는 선언된 이관 도구가 들어가는지). 어느 도구가 옳은지는 판정하지 않는다 — 그것은 모델 몫이다.
이관 도구 이름은 **A2 선언과 궤적**에서 온다(코드에 도메인 낱말 0).

사용: (리모트·cwd=scripts/distill/tau2) py -3 x552_transfer_empty_ledger_iso.py --port 8140
"""
import argparse
import gzip
import io
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_forensic as F                                        # noqa: E402
import t2_gate_patch as G                                      # noqa: E402

SIMS = os.path.abspath(os.path.join(HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
REP = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026"))
MODEL = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8"
TASK = "task_074"
DOMAIN = "banking_knowledge"
NL = chr(10)

# 이 결정의 술어 이름: 이관. 도메인 상품명이 아니다.
RE_XFER = re.compile(r"transfer_to_human|_transfer_", re.I)
FACT = ("[ledger] No state-changing action has executed in this conversation: "
        "the record set is exactly as it was when the conversation started.")
ASK = (NL + NL + "Reply with ONLY the name of the single tool you will call next - "
       "the bare tool name, nothing else, no explanation.")


def gen(port, body, maxtok=16):
    payload = {"model": MODEL, "temperature": 0.0, "max_tokens": maxtok,
               "messages": [{"role": "user", "content": body}]}
    req = urllib.request.Request("http://127.0.0.1:%d/v1/chat/completions" % port,
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))["choices"][0]["message"]["content"]


def load(tag):
    p = os.path.join(SIMS, tag + ".results.json.gz")
    if not os.path.exists(p):
        return []
    d = json.load(gzip.open(p, "rt", encoding="utf-8", errors="replace"))
    return d.get("simulations") or d.get("results") or []


def all_tags():
    out = []
    for fn in sorted(os.listdir(SIMS)):
        if fn.endswith(".results.json.gz"):
            out.append(fn[:-len(".results.json.gz")])
    return out


def find_filler(msgs, want):
    for m in msgs:
        if str(m.get("role")) != "tool":
            continue
        c = " ".join(str(m.get("content") or "").split())
        if len(c) >= want and "ledger" not in c.lower():
            return c[:want]
    return None


def windows(limit):
    """이관 호출 **직전** 창 — 전 접두(거리를 지우지 않는다·x551 교훈)."""
    cases, skipped = [], []
    for tag in all_tags():
        if len(cases) >= limit:
            break
        for s in load(tag):
            if s.get("task_id") != TASK:
                continue
            ms = s.get("messages") or []
            cut = None
            for i, m in enumerate(ms):
                if str(m.get("role")) != "assistant":
                    continue
                for tc in (m.get("tool_calls") or []):
                    nm = "%s %s" % (F.nameof(tc), F.inner_name(F.argsof(tc)) or "")
                    if RE_XFER.search(nm):
                        cut = i
                        break
                if cut is not None:
                    break
            sim = "%s#s%s" % (s.get("task_id"), s.get("seed"))
            if cut is None:
                skipped.append({"tag": tag, "sim": sim, "why": "이관 호출이 없다"})
                continue
            # 원장이 정말 비었는가 — 이 프로브의 전제([[25]] 확인하고 말한다)
            a2 = G._domain_a2(DOMAIN)

            class _M:
                def __init__(s2, m):
                    s2.role = m.get("role")
                    s2.tool_calls = [type("T", (), {"name": t.get("name"),
                                                    "arguments": t.get("arguments"),
                                                    "id": t.get("id")})()
                                     for t in (m.get("tool_calls") or [])] or None
            if G._any_effective_write([_M(m) for m in ms[:cut]], a2):
                skipped.append({"tag": tag, "sim": sim, "why": "이관 전에 실효 write 가 있다"})
                continue
            txt = []
            for m in ms[:cut]:
                c = str(m.get("content") or "").strip()
                if c:
                    txt.append("[%s] %s" % (m.get("role"), c))
            if not txt:
                skipped.append({"tag": tag, "sim": sim, "why": "창이 비었다"})
                continue
            cases.append({"tag": tag, "sim": sim, "cut": cut,
                          "win": (NL + NL).join(txt),
                          "filler": find_filler(ms, len(FACT))})
            if len(cases) >= limit:
                break
    return cases, skipped


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8140)
    ap.add_argument("--limit", type=int, default=4)
    a = ap.parse_args(argv)
    cases, skipped = windows(a.limit)
    print("=" * 98)
    print("x552 — 074: 원장이 비었다는 사실을 이관 출구에서 주면 다음 수가 바뀌는가")
    print("=" * 98)
    print("창 %d개 · 건너뜀 %d개" % (len(cases), len(skipped)))
    for s in skipped[:8]:
        print("   skip %-28s %-26s %s" % (s["tag"][:28], s["sim"], s["why"]))
    if not cases:
        print("⛔창이 없다 — 판정하지 않는다([[25]] 없음 ≠ 못 찾음)")
        return 1

    tally = {}
    for c in cases:
        print("\n--- %s (%s · cut=msg%d) ---" % (c["sim"], c["tag"][:30], c["cut"]))
        arms = {"A_asis": c["win"],
                "B_fact": c["win"] + NL + NL + FACT}
        if c["filler"]:
            arms["N_len"] = c["win"] + NL + NL + "[note] " + c["filler"]
        else:
            print("  ⚠부정통제 문장을 못 찾았다 — N_len 생략(그 사실을 남긴다)")
        for arm, ctx in arms.items():
            ans = " ".join(str(gen(a.port, ctx + ASK)).split())
            xfer = bool(RE_XFER.search(ans))
            tally.setdefault(arm, [0, 0])
            tally[arm][0] += 0 if xfer else 1
            tally[arm][1] += 1
            print("  %-8s %-46s %s" % (arm, ans[:46], "이관" if xfer else "**다른 도구**"))

    print("\n" + "=" * 98)
    print("합계 — 이관이 아닌 도구를 고른 비율")
    print("=" * 98)
    for arm in ("A_asis", "B_fact", "N_len"):
        if arm in tally:
            o, n = tally[arm]
            print("  %-8s **%d/%d**" % (arm, o, n))
    print("\n  판정: A_asis 가 이관을 재현하고 B_fact 가 높고 N_len 이 A 수준이면 → **전달 결손**")
    print("        A_asis 가 이미 안 이관하면 → 격리가 라이브를 재현 못 함 = 보류([[62]] 2b)")
    print("        B_fact 도 이관이면 → 사실을 알려도 안 바뀐다 = 이 처방으로는 못 산다")
    out = os.path.join(REP, "x552_transfer_empty_ledger_2026_08_26.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"tally": tally, "skipped": skipped,
                             "cases": [{k: v for k, v in c.items() if k != "win"}
                                       for c in cases]}, ensure_ascii=False, indent=1))
    print("\n  → %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
