#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x34c: x34b 결론의 **리뷰 반증** (2026-08-01·설계 리뷰·무료·전수 재집계).

x34b는 D2를 **도구 이름**으로 갈라 "검색 26% / 비-검색 61%"를 냈고, 설계서 초판은 그 위에
"비-검색은 분기가 아예 없다 ⇒ 대안을 주자(2e-1)"를 세웠다. 그런데 스텁 문구는 런 시점마다
다르다 — redirect(C114③·2026-07-23)와 **escalation(C194·2026-07-26)** 은 후기 런에만 있다.
⇒ 층화 기준을 이름이 아니라 **출력 문자열 실재**로 바꾸면 결론이 뒤집힌다.

측정 4종:
  (1) D2 층화   : [DUPLICATE-READ] 발화 × (검색 여부 · redirect 실재 · escalation 실재)
  (2) 표적 크기 : 직후-동일 재발행이 난 **시뮬 수**(백분율이 가린 것) · D1∩D2비검색
  (3) D3 분해   : "미회복"을 재시도-또경고 / **재시도 안 함**으로 가른다
  (4) D5 결합   : D5 발화가 선행 D1 스텁의 하류인가(2c의 무료 사후지표 성립 여부)

Run: python3 x34c_defect_strat.py
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
REDIR = "Do NOT repeat this exact search"          # C114③ redirect 축자 앵커
ESC = "IDENTICAL call"                             # C194 escalation 축자 앵커
D1 = "the account record has not been fetched yet"  # A2 no_record_template 축자 앵커
D5 = "Failed to log verification"

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
                              "out": byid.get(tc.get("id"), "")})
        if len(calls) > CALLS_HI:
            SIMS.append({"key": (tag, s.get("task_id"), s.get("trial")), "calls": calls})
print("폭주 시뮬 %d건 (호출>%d)\n" % (len(SIMS), CALLS_HI))


def is_search(c):
    dn = (c["name"] + " " + c["args"]).lower()
    return ("search" in dn or "bm25" in dn or "kb_" in dn or "grep" in dn)


def nxt_same(cs, i):
    n = cs[i + 1] if i + 1 < len(cs) else None
    return bool(n and (n["name"], n["args"]) == (cs[i]["name"], cs[i]["args"]))


# ── (1) D2 층화 ───────────────────────────────────────────────────────────────
print("=" * 92)
print("(1) [DUPLICATE-READ] 층화 — 도구 이름이 아니라 **스텁이 실제로 담은 문구**로")
cnt, same = collections.Counter(), collections.Counter()
bytool, bytool_same = collections.Counter(), collections.Counter()
for s in SIMS:
    cs = s["calls"]
    for i, c in enumerate(cs):
        o = c["out"] or ""
        if "[DUPLICATE-READ]" not in o:
            continue
        k = ("검색" if is_search(c) else "비검색",
             "redir+" if REDIR in o else "redir-",
             "esc+" if ESC in o else "esc-")
        r = nxt_same(cs, i)
        cnt[k] += 1
        same[k] += r
        bytool[c["name"]] += 1
        bytool_same[c["name"]] += r
print("  %-6s %-7s %-5s | %6s %8s %6s" % ("부류", "redirect", "esc", "발화", "직후동일", "비율"))
for k in sorted(cnt, key=lambda x: -cnt[x]):
    print("  %-6s %-7s %-5s | %6d %8d %5.0f%%"
          % (k[0], k[1], k[2], cnt[k], same[k], 100.0 * same[k] / cnt[k]))
print("  합계 발화 %d · 직후동일 %d (%.0f%%)"
      % (sum(cnt.values()), sum(same.values()),
         100.0 * sum(same.values()) / max(sum(cnt.values()), 1)))
print("  ★esc 문구 축자에 이미 있는 대안 = 'act on the information you already have, "
      "or ask the customer'\n    ⇒ 2e-1이 넣으려던 '말하기/묻기'는 **이미 발화되고 있었다**.")
print("  도구별:")
for n, v in bytool.most_common(8):
    print("    %-40s 발화 %4d · 직후동일 %4d (%.0f%%)"
          % (n, v, bytool_same[n], 100.0 * bytool_same[n] / v))

# ── (2) 표적 크기(시뮬 수) ────────────────────────────────────────────────────
print("\n" + "=" * 92)
print("(2) 표적 크기 — 백분율이 가린 **시뮬 수** ([[08]])")
srch_sims, nons_sims, d1_sims, d2ns_sims = set(), set(), set(), set()
for s in SIMS:
    cs = s["calls"]
    if any(D1 in (c["out"] or "") for c in cs):
        d1_sims.add(s["key"])
    if any("[DUPLICATE-READ]" in (c["out"] or "")
           and c["name"].startswith("get_user_information") for c in cs):
        d2ns_sims.add(s["key"])
    for i, c in enumerate(cs):
        if "[DUPLICATE-READ]" not in (c["out"] or "") or not nxt_same(cs, i):
            continue
        (srch_sims if is_search(c) else nons_sims).add(s["key"])
print("  직후-동일 재발행이 1회 이상 난 시뮬: **검색 %d개 · 비검색 %d개**"
      % (len(srch_sims), len(nons_sims)))
print("  D1 시뮬 %d · D2-비검색 시뮬 %d · **교집합 %d** (D2비검색−D1 = %d)"
      % (len(d1_sims), len(d2ns_sims), len(d1_sims & d2ns_sims), len(d2ns_sims - d1_sims)))
print("  ⇒ 2e-1의 표적은 4시뮬이고 전부 D1 안 = §2f가 기각한 '008형 전용 특수처리'와 같은 크기.")
print("  ⇒ 반복 캡(2e-2)의 표적이 오히려 %d시뮬 = 문구/캡 우선순위가 뒤집힌다." % len(srch_sims))

# ── (3) D3 분해 ───────────────────────────────────────────────────────────────
print("\n" + "=" * 92)
print("(3) [GROUNDING WARNING] '미회복'의 분해 — '끝내 회복 못 함'은 라벨 오류")
hits = ok = warn = never = 0
dropped = collections.Counter()
for s in SIMS:
    cs = s["calls"]
    for i, c in enumerate(cs):
        o = c["out"] or ""
        if "[GROUNDING WARNING]" not in o:
            continue
        hits += 1
        for m in re.finditer(r"(\w+)=([^\s;,()]+)", o[:900]):
            dropped[m.group(1)] += 1
        found = False
        for j in range(i + 1, len(cs)):
            if cs[j]["name"] == c["name"]:
                found = True
                if "[GROUNDING WARNING]" in (cs[j]["out"] or ""):
                    warn += 1
                else:
                    ok += 1
                break
        if not found:
            never += 1
print("  발화 %d = 회복 %d(%.0f%%) + 재시도-또경고 %d(%.0f%%) + **재시도 안 함 %d(%.0f%%)**"
      % (hits, ok, 100.0*ok/max(hits, 1), warn, 100.0*warn/max(hits, 1),
         never, 100.0*never/max(hits, 1)))
print("  ⇒ '재시도 안 함'은 정당한 중단일 수 있다. 2d의 상한 = 재시도-또경고 %d 발화." % warn)
print("  드롭된 인자 이름 top: %s"
      % ", ".join("%s×%d" % (k, v) for k, v in dropped.most_common(8)))
print("  ⚠2d의 전제(드롭 필드가 sub-유래)는 이 이름들이 A2 `row_fields`에 없어야 성립 — 미검증.")

# ── (4) D5 결합 ───────────────────────────────────────────────────────────────
print("\n" + "=" * 92)
print("(4) D5(`Failed to log verification`)가 D1b의 하류인가 — 2c의 무료 사후지표")
d5_hits, d5_prior, d5_sims = 0, 0, set()
after = collections.Counter()
for s in SIMS:
    cs = s["calls"]
    for i, c in enumerate(cs):
        if D5 not in (c["out"] or ""):
            continue
        d5_hits += 1
        d5_sims.add(s["key"])
        if any(D1 in (cs[j]["out"] or "") for j in range(i)):
            d5_prior += 1
        if i + 1 < len(cs):
            after[cs[i + 1]["name"]] += 1
print("  발화 %d회 · 시뮬 %d개 · **선행 D1 스텁 있음 %d회 (%.0f%%)** · 시뮬 교집합 %d/%d"
      % (d5_hits, len(d5_sims), d5_prior, 100.0*d5_prior/max(d5_hits, 1),
         len(d5_sims & d1_sims), len(d5_sims)))
print("  직후 도구: %s" % ", ".join("%s×%d" % (k, v) for k, v in after.most_common(6)))
print("  ⇒ D5는 env 소유라 못 고치지만 **2c가 먹으면 노출이 줄어야 한다** = 사후지표(엔진 순증 0).")

# ── (5) D4 술어 확인 ──────────────────────────────────────────────────────────
print("\n" + "=" * 92)
print("(5) D4 — '(none)'을 만드는 변수 확인 (엔진 술어 선택 근거)")
d4_none, d4_nonempty, d4_zero_total, d4_none_gap = 0, 0, 0, 0
cov_ex = []
for s in SIMS:
    for c in s["calls"]:
        o = c["out"] or ""
        if "[coverage] 0 of 0 rows were checked" in o:
            d4_zero_total += 1
        if "update that transaction's rewards to EXACTLY the correct value shown" not in o:
            continue
        m = re.search(r"correct value shown:\s*(.*?)(?:\n\[coverage\]|\Z)", o, re.S)
        det = (m.group(1).strip() if m else "?")
        cv = re.search(r"\[coverage\] (\d+) of (\d+) rows were checked \((\d+)", o)
        if det.startswith("(none)"):
            d4_none += 1
            if cv and cv.group(1) != cv.group(2):
                d4_none_gap += 1
                if len(cov_ex) < 5:
                    cov_ex.append("judged=%s/%s skipped=%s" % cv.groups())
        else:
            d4_nonempty += 1
print("  모순 발화(details=(none)) %d · 정상(details 비공집합) %d" % (d4_none, d4_nonempty))
print("  ★(none)인데 **coverage 결손** = %d/%d 발화 — 케이스#2는 가설이 아니라 실측"
      % (d4_none_gap, d4_none))
print("    사례: %s" % ("; ".join(cov_ex) or "없음"))
print("  '0 of 0 rows were checked' 발화 %d회" % d4_zero_total)
print("  ⇒ '(none)'은 `_res`(ids)가 아니라 **`_sg_details`** 가 만든다(t2_scaffold_get.py:1526).")
print("     현행 select_discrepant는 둘을 같은 분기에서 채워 동치지만(t2_compute.py:646) 그건")
print("     그 op의 우연 — 엔진 술어는 details 기준으로 잡고 동치를 단위테스트로 못박을 것.")
