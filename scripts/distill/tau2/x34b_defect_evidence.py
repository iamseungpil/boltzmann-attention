#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x34b: x34가 뽑은 **우리 저작 문구** 후보 5종의 정밀 증거 (per-step·폭주 궤적 전수).

x34는 랭킹만 낸다. 여기서는 후보마다 "이 문구가 왜 결함인가"를 **반증 가능한 수치**로 만든다.
분류(모순/부재/불가능/허위)는 사람이 원문 읽고 하되, 근거는 아래 계수다.

D1 `no_record_template`  : 전제(조회 성공)가 성립 불가한데 "call this tool again"을 지시 →
                           같은 sim에서 조회가 끝내 성공했는가? 직후 `log_verification`(미검증 기록)은?
D2 `[DUPLICATE-READ]`    : "다음 단계로 가라"만 하고 수단은 안 준다 → 직후 완전-동일 재발행률 ·
                           그 sim이 원한 discoverable 도구에 디스패처로 도달했는가?
D3 `[GROUNDING WARNING]` : "레코드에서 정확한 값을 다시 읽어라" → 그 값이 레코드에 **실재하는가**
                           (없으면 이행 불가능 = C275 ⑤ 동형).
D4 `return_template`     : 목록이 `(none)`인데 "표시된 정확한 값으로 갱신하라"를 함께 말하는가(모순).
D5 `Failed to log …`     : 다음 행동 지시가 있는가 · 직후 무엇을 하는가.
"""
import collections
import glob
import gzip
import json
import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026", "sim_results")
CALLS_HI = 30

SIMS = []
for f in sorted(glob.glob(os.path.join(DIR, "*.results.json.gz"))):
    try:
        d = json.load(gzip.open(f, "rt", encoding="utf-8"))
    except Exception:
        continue
    tag = os.path.basename(f).replace(".results.json.gz", "")
    for s in d.get("simulations", []):
        msgs = s.get("messages") or []
        byid = {m.get("id"): str(m.get("content") or "")
                for m in msgs if (m.get("role") or "") == "tool"}
        calls = []
        for m in msgs:
            if (m.get("role") or "") != "assistant":
                continue
            for tc in (m.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
                a = fn.get("arguments", tc.get("arguments"))
                if not isinstance(a, str):
                    a = json.dumps(a, sort_keys=True, ensure_ascii=False)
                calls.append({"name": str(fn.get("name") or ""), "args": a[:4000],
                              "out": byid.get(tc.get("id"), ""), "say": str(m.get("content") or "")})
        if len(calls) > CALLS_HI:
            SIMS.append({"tag": tag, "task": s.get("task_id"), "trial": s.get("trial"),
                         "term": s.get("termination_reason"), "calls": calls})
print("폭주 시뮬 %d건\n" % len(SIMS))


def after(calls, i):
    return calls[i + 1] if i + 1 < len(calls) else None


# ── D1 ────────────────────────────────────────────────────────────────────────
print("=" * 90)
print("D1  `no_record_template` (NOT_VERIFIED — 'call this tool again')")
# ⚠계측기 자기결함(2026-08-01 적발): 이전 판은 per-sim 카운터를 `(tag,task)` **유일쌍 집합** 크기로
#   나눠 "42/42 전건 성공"이라는 거짓을 냈다(008은 조회 0성공·verify 0통과인데도). 분모=시뮬 수로 고침.
#   ⇒ C276★① 교훈의 4번째 적용: 수치가 전건이면 계측기를 먼저 의심하라.
d1_hits, d1_nsim, d1_lookup_ok, d1_log_after, d1_verify_ok = 0, 0, 0, 0, 0
d1_stuck = []
for s in SIMS:
    hit = False
    for i, c in enumerate(s["calls"]):
        if "the account record has not been fetched yet" not in (c["out"] or ""):
            continue
        hit = True
        d1_hits += 1
        n = after(s["calls"], i)
        if n and n["name"] == "log_verification":
            d1_log_after += 1
    if hit:
        d1_nsim += 1
        ok = any(("Found" in (c["out"] or "") and c["name"].startswith("get_user_information"))
                 for c in s["calls"])
        vok = any((c["name"] == "verify_identity" and "NOT_VERIFIED" not in (c["out"] or "")
                   and "VERIFIED" in (c["out"] or "")) for c in s["calls"])
        d1_lookup_ok += ok
        d1_verify_ok += vok
        if not ok:
            d1_stuck.append("%s/%s t%s (%s)" % (s["tag"][:26], s["task"], s["trial"], s["term"]))
print("  발화 %d회 · 시뮬 %d개" % (d1_hits, d1_nsim))
print("  그 시뮬에서 조회가 **끝내 성공** %d/%d · verify가 **끝내 통과** %d/%d"
      % (d1_lookup_ok, d1_nsim, d1_verify_ok, d1_nsim))
print("  ★조회가 끝내 실패한 시뮬(=문구가 이행 불가능) %d개: %s"
      % (len(d1_stuck), "; ".join(d1_stuck[:6])))
print("  ★직후 `log_verification`(=미검증인데 검증 기록 시도) %d회" % d1_log_after)

# ── D2 ────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("D2  `[DUPLICATE-READ]` ('Do NOT repeat' + '다음 단계로 가라' — 수단 미제시)")
d2_hits = d2_same = d2_sims = 0
want_disc, used_disp = 0, 0
DISC_RE = re.compile(r"\b([a-z_]+_\d{3,4})\b")
for s in SIMS:
    hit = False
    for i, c in enumerate(s["calls"]):
        if "[DUPLICATE-READ]" not in (c["out"] or ""):
            continue
        hit = True
        d2_hits += 1
        n = after(s["calls"], i)
        if n and (n["name"], n["args"]) == (c["name"], c["args"]):
            d2_same += 1
    if hit:
        d2_sims += 1
        says = " ".join(c["say"] for c in s["calls"])
        if DISC_RE.search(says):
            want_disc += 1
            if any(c["name"] in ("call_discoverable_agent_tool", "unlock_discoverable_agent_tool")
                   for c in s["calls"]):
                used_disp += 1
print("  발화 %d회 · 시뮬 %d개 · ★직후 **완전 동일** 재발행 %d회 (%.0f%%)"
      % (d2_hits, d2_sims, d2_same, 100.0 * d2_same / max(d2_hits, 1)))
print("  본문에서 discoverable 도구를 지목한 시뮬 %d개 중 디스패처를 실제로 쓴 것 %d개"
      % (want_disc, used_disp))

# ── D3 ────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("D3  `[GROUNDING WARNING]` ('re-read the exact value from the records')")
d3_hits, d3_after_same, d3_recovered = 0, 0, 0
dropped_names = collections.Counter()
for s in SIMS:
    for i, c in enumerate(s["calls"]):
        o = c["out"] or ""
        if "[GROUNDING WARNING]" not in o:
            continue
        d3_hits += 1
        for m in re.finditer(r"(\w+)=([^\s;,()]+)", o[:900]):
            dropped_names[m.group(1)] += 1
        n = after(s["calls"], i)
        if n and (n["name"], n["args"]) == (c["name"], c["args"]):
            d3_after_same += 1
        # 회복 = 같은 도구를 다시 부르고 이번엔 경고가 없다
        for j in range(i + 1, len(s["calls"])):
            if s["calls"][j]["name"] == c["name"]:
                if "[GROUNDING WARNING]" not in (s["calls"][j]["out"] or ""):
                    d3_recovered += 1
                break
print("  발화 %d회 · 직후 완전 동일 재호출 %d · **이후 같은 도구가 경고 없이 성공 %d**"
      % (d3_hits, d3_after_same, d3_recovered))
print("  드롭된 인자 이름 top: %s"
      % ", ".join("%s×%d" % (k, v) for k, v in dropped_names.most_common(8)))

# ── D4 ────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("D4  `return_template` — 목록이 `(none)`인데 '표시된 정확한 값으로 갱신하라'를 동반하는가")
d4_hits = d4_none = 0
d4_sims = set()
for s in SIMS:
    for c in s["calls"]:
        o = c["out"] or ""
        if "update that transaction's rewards to EXACTLY the correct value shown" not in o:
            continue
        d4_hits += 1
        seg = o[o.find("update that transaction's rewards"):][:400]
        if "(none)" in seg or re.search(r"shown:\s*\(none\)", o):
            d4_none += 1
            d4_sims.add((s["tag"], s["task"]))
print("  발화 %d회 · 그중 **목록이 (none)인데 갱신 지시 동반 = %d회** · 시뮬 %d개"
      % (d4_hits, d4_none, len(d4_sims)))
print("  사례: %s" % ", ".join("%s/%s" % t for t in sorted(d4_sims)[:5]))

# ── D5 ────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("D5  `Failed to log verification: Record may already exist.`")
d5_hits = 0
nxt = collections.Counter()
for s in SIMS:
    for i, c in enumerate(s["calls"]):
        if "Failed to log verification" not in (c["out"] or ""):
            continue
        d5_hits += 1
        n = after(s["calls"], i)
        if n:
            nxt[n["name"]] += 1
print("  발화 %d회 · 직후 도구: %s"
      % (d5_hits, ", ".join("%s×%d" % (k, v) for k, v in nxt.most_common(6))))
print("  (문구에 다음 행동 지시 없음 — 검증이 충족된 것인지 아닌지도 말하지 않는다)")
