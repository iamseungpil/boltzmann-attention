# -*- coding: utf-8 -*-
r"""x385 — **정확한 이름을 더미 속에 주면 모델이 그것을 쓰는가**(050 후속 격리 · [[65]] 표적).

## 왜 (t7315 050 궤적)

t34 에 우리 층이 **정확한 도구 이름을 축자로** 배달했다 —
`[CLAIM-PROVENANCE] … (tool: approve_credit_limit_increase_5847)` + `[VERDICT] … Decide with a
TOOL CALL: approve_credit_limit_increase`. 그런데 **바로 다음 턴(t35)** 에 모델은
`approve_credit_limit_increase_**7890**` 으로 접미사를 **지어냈다**.

레버 부재가 아니다 — `PROV`·`UNLOCK_PROV`·`CLAIMPROV`·`UNAVAIL`·`DISCOVERY_STEP2` 다섯이 잡았고
문구도 *"추측하지 말고 KB 를 검색하라"* 까지 말했다. 다른 것은 **전달 형태**다:
그 턴의 배달은 **`CLAIMPROV` 10줄 + 1,504~1,593자 더미**였고, 정답 문자열은 그 안의 괄호였다.

x378 에서는 이름이 **지시 한 줄**일 때 모델이 20/25 로 그대로 따랐다. 그 차이를 여기서 잰다.
[[65]](메인엔 답만·재료를 올리는 것 자체가 부하)의 직접 시험이다.

## 셀 (문맥은 라이브 축자 · 바뀌는 것은 **그 턴의 배달**뿐)

    A_PILE  그 턴에 우리가 실제로 보낸 **전량**(축자·라이브 그대로)    ← 현행
    B_ONE   그중 **이름이 든 한 줄만**                                ← [[65]] 처방
    D_NEG   B_ONE 의 이름을 **다른 실재 도구**로 바꾼 한 줄            ← 계기(이름을 읽나)

## 채점 (결정론 · gold 무참조)

  · `exact`   — 방출된 호출이 **그 정확한 이름**(접미사까지)을 담았는가
  · `fabric`  — 같은 어간에 **다른 접미사**를 지어냈는가(`_\d+` 가 다름)
  · `other`   — 그 어간이 아예 안 나옴

## 판정 (사전 고정)

    B_ONE.exact > A_PILE.exact (과반 컷)  → **더미가 원인** ⇒ 처방 = 결정 턴 배달을 답 한 줄로
    A ≈ B 이고 둘 다 낮음                  → 전달 형태가 아니다 ⇒ 이름 채택 자체가 결손([[62]]③)
    D_NEG 가 심은 이름을 그대로 씀         → 채널 생존 확증(모델이 한 줄은 읽는다)
    세 팔 전부 동일                        → **계기 무효**

⚠오프라인·8140/8141 중 **비어 있는 쪽**에서 돈다. 엔진 수정 0.
사용: T2_PROBE_URL=http://localhost:8140/v1/chat/completions python x385_name_delivery_iso.py
"""
import collections
import io
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/woori/scratch/tau2-bench/src")
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import x370_infomatched_give_iso as G                      # noqa: E402

ROOT = "/home/woori/scratch/tau2-bench/data/simulations"
LOGS = "/home/woori/scratch/logs"
TAGS = ["bank_t7315_050treat_20260818k", "bank_t7314_treat_20260818j",
        "bank_t7314_ctl_20260818j", "bank_t7313_treat_20260818h"]
NAME_RX = re.compile(r"\(tool: ([a-z][a-z_]*_\d{3,4})\)")
ASK = ("You are the bank's support agent. Continue this conversation and make the tool call "
       "you would make now.")
MAXCUT = 10


def sims_of(tag):
    p = os.path.join(ROOT, tag, "results.json")
    if not os.path.exists(p):
        return {}
    doc = json.load(io.open(p, encoding="utf-8"))
    return {str(s.get("task_id")): s for s in (doc.get("simulations") or doc.get("results") or [])}


def sidecar(tag):
    p = os.path.join(LOGS, "fb_%s.jsonl" % tag)
    if not os.path.exists(p):
        return []
    out = []
    for ln in io.open(p, encoding="utf-8", errors="replace"):
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out


def upto_index(sim, turn):
    msgs = sim.get("messages") or []
    for i, m in enumerate(msgs):
        ti = m.get("turn_idx")
        if ti is not None and int(ti) >= int(turn):
            return i
    return len(msgs)


def emitted(msg):
    for tc in ((msg or {}).get("tool_calls") or ()):
        f = tc.get("function") or tc
        return str(f.get("name") or "") + " " + str(f.get("arguments") or "")
    return ""


def main():
    tools = G.agent_tool_specs()
    reg = sorted(t["function"]["name"] for t in tools)
    print("=" * 104)
    print("x385 · 이름 배달 격리 · 도구 %d개" % len(tools))
    print("판정(사전 고정): B_ONE.exact > A_PILE.exact 과반 → 더미가 원인(처방=답 한 줄) · "
          "둘 다 낮으면 전달 형태 아님 · 세 팔 동일 → 계기 무효")
    print("=" * 104)

    cuts = []
    for tag in TAGS:
        sims = sims_of(tag)
        rows = [r for r in sidecar(tag) if isinstance(r.get("turn"), int)]
        by_turn = collections.defaultdict(list)
        for r in rows:
            by_turn[(str(r.get("simtag", "")).split("#")[0], r["turn"])].append(r)
        for (task, turn), rs in sorted(by_turn.items()):
            hit = [(r, NAME_RX.search(str(r.get("text") or ""))) for r in rs]
            hit = [(r, m) for r, m in hit if m]
            if not hit or task not in sims:
                continue
            target = hit[0][1].group(1)
            cuts.append({"tag": tag, "task": task, "turn": turn, "target": target,
                         "sim": sims[task], "rows": rs, "one": hit[0][0]})
    # 컷이 많으면 태스크 다양성 우선으로 자른다(같은 sim 반복은 독립 표본이 아니다)
    seen, sel = set(), []
    for c in cuts:
        k = (c["tag"], c["task"])
        if k in seen:
            continue
        seen.add(k)
        sel.append(c)
        if len(sel) >= MAXCUT:
            break
    print("컷 %d개 (전체 후보 %d)" % (len(sel), len(cuts)))
    print("")

    agg = collections.Counter()
    rows_out = []
    for c in sel:
        base = G.convo(c["sim"], upto_index(c["sim"], c["turn"]))
        if not base:
            continue
        pile = "\n\n".join(" ".join(str(r.get("text") or "").split()) for r in c["rows"])
        one = " ".join(str(c["one"].get("text") or "").split())
        stem = re.sub(r"_\d+$", "", c["target"])
        alt = next((n for n in reg if n != c["target"] and not n.startswith(stem)), reg[0])
        neg = one.replace(c["target"], alt)
        got = {}
        for an, add in (("A_PILE", pile), ("B_ONE", one), ("D_NEG", neg)):
            msg, det = G.det(base + "\n\ntool: " + add + "\n\n" + ASK, tools, 520)
            blob = emitted(msg)
            exact = c["target"] in blob
            fab = bool(re.search(re.escape(stem) + r"_\d+", blob)) and not exact
            got[an] = {"exact": int(exact), "fab": int(fab),
                       "neg": int(alt in blob), "det": det}
            agg[(an, "exact")] += int(exact)
            agg[(an, "fab")] += int(fab)
            agg[(an, "neg")] += int(alt in blob)
        rows_out.append({"tag": c["tag"].split("_")[1], "task": c["task"], "turn": c["turn"],
                         "target": c["target"], "pile_len": len(pile), "one_len": len(one),
                         "got": got})
        print("  %-9s %-6s t%-3s %-38s 더미 %5d자 → 한줄 %4d자 | A=%s B=%s D=%s"
              % (c["task"], c["tag"].split("_")[1], c["turn"], c["target"],
                 len(pile), len(one),
                 "정확" if got["A_PILE"]["exact"] else ("날조" if got["A_PILE"]["fab"] else "무관"),
                 "정확" if got["B_ONE"]["exact"] else ("날조" if got["B_ONE"]["fab"] else "무관"),
                 "심은이름" if got["D_NEG"]["neg"] else "무관"))

    n = len(rows_out)
    print("")
    print("## 집계 n=%d" % n)
    for an in ("A_PILE", "B_ONE", "D_NEG"):
        print("  %-7s exact %d · fabricate %d · 심은이름 %d"
              % (an, agg[(an, "exact")], agg[(an, "fab")], agg[(an, "neg")]))
    a, b = agg[("A_PILE", "exact")], agg[("B_ONE", "exact")]
    if n == 0:
        v = "⛔컷 0 — 계기 결함"
    elif b > a and (b - a) * 2 >= n:
        v = "**더미가 원인** ⇒ 처방 = 결정 턴 배달을 **답 한 줄**로([[65]])"
    elif a == b and a * 2 < n:
        v = "전달 형태가 아니다 — 한 줄로 줘도 안 쓴다 ⇒ [[62]]③ 그 단계 결손"
    else:
        v = "혼합 — 컷별 표를 읽는다([[08]])"
    print("판정: %s" % v)
    out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "..", "reports", "facet_rft_2026",
                                        "x385_name_delivery.json"))
    io.open(out, "w", encoding="utf-8").write(json.dumps(
        {"rows": rows_out, "n": n, "verdict": v}, ensure_ascii=False, indent=1))
    print("원자료: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
