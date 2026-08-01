#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""x33: **궤적-레벨 폭주** 전수 포렌식 (무료·로컬·영속 데이터만·GPU 0).

동기(핸드오프 §6c-12-4 · C271 · C289):
  `max_tokens` 캡은 **토큰-레벨**이라 한 응답의 길이만 막는다. 022 t0(65 호출·`call_discoverable
  _user_tool` ×25)와 P2 008(185 호출)은 **호출 수가 폭주**해 context를 넘겼다 — 캡이 못 막는 축이다.
  이 축이 지금 임계 경로 위에 있다(022 t1의 결정성이 여기 걸려 있음).

집계에서 결론 직행 금지([[08]]) — 세는 것은 "몇 번 불렀나"가 아니라 **반복의 구조**다:
  R1 같은 (도구, 인자) 정확 재호출이 몇 번인가 = **무진전 루프**
  R2 같은 도구를 인자만 바꿔 반복인가 = **탐색**(coverage 부하)
  R3 재호출 사이에 도구 **출력이 달라졌나**(=진전 있음) / 같은 출력을 받고도 또 부르나
  R4 재호출 직전에 **엔진 피드백**(표면화 메시지)이 있었나 = 지시 무시(C275형)
채점·gold 미열람. 입력 = `sim_results/*.results.json.gz` 영속본.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              "..", "..", "..", "reports", "facet_rft_2026",
                                              "sim_results"))
ap.add_argument("--glob", default="bank_y2c*.results.json.gz")
ap.add_argument("--top", type=int, default=12)
ap.add_argument("--calls_hi", type=int, default=30, help="폭주 후보 임계(호출 수)")
ap.add_argument("--out", default="")
A = ap.parse_args()

FILES = sorted(glob.glob(os.path.join(A.dir, A.glob)))
if not FILES:
    sys.exit("입력 없음: %s" % os.path.join(A.dir, A.glob))
print("입력 %d파일" % len(FILES))


def tool_calls(msg):
    """assistant 메시지의 tool_calls를 (name, args_json) 목록으로."""
    out = []
    for tc in (msg.get("tool_calls") or []):
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        name = fn.get("name") or tc.get("name")
        args = fn.get("arguments", tc.get("arguments"))
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        out.append((str(name), json.dumps(args, sort_keys=True, ensure_ascii=False)[:4000],
                    tc.get("id")))
    return out


SIMS = []
for f in FILES:
    try:
        d = json.load(gzip.open(f, "rt", encoding="utf-8"))
    except Exception as e:
        print("  ⚠읽기 실패 %s: %r" % (os.path.basename(f), e))
        continue
    tag = os.path.basename(f).replace(".results.json.gz", "")
    for s in d.get("simulations", []):
        msgs = s.get("messages") or []
        calls, byid = [], {}
        for m in msgs:
            if (m.get("role") or "") == "assistant":
                for nm, ar, cid in tool_calls(m):
                    calls.append({"name": nm, "args": ar, "id": cid, "turn": m.get("turn_idx")})
            elif (m.get("role") or "") == "tool":
                byid[m.get("id") or m.get("tool_call_id")] = str(m.get("content") or "")
        for c in calls:
            c["out"] = byid.get(c["id"], "")
        SIMS.append({
            "tag": tag, "task": s.get("task_id"), "trial": s.get("trial"),
            "term": s.get("termination_reason"), "dur": s.get("duration"),
            "nmsg": len(msgs), "ncall": len(calls), "calls": calls,
            "reward": ((s.get("reward_info") or {}).get("reward")),
        })

print("시뮬 %d건\n" % len(SIMS))

# ── 분포: 호출 수 ─────────────────────────────────────────────────────────────
SIMS.sort(key=lambda x: -x["ncall"])
print("=" * 88)
print("[호출 수 상위 %d]  (task/trial · 종료 · 메시지 · 호출 · reward)" % A.top)
for s in SIMS[:A.top]:
    print("  %-22s %-9s t%-2s %-24s msg%4d call%4d r=%s"
          % (s["tag"][:22], s["task"], s["trial"], str(s["term"])[:24],
             s["nmsg"], s["ncall"], s["reward"]))

nc = [s["ncall"] for s in SIMS]
nc_sorted = sorted(nc)
med = nc_sorted[len(nc_sorted) // 2]
print("\n  호출 수: 중앙 %d · 평균 %.1f · 최대 %d · **%d초과 = %d/%d건**"
      % (med, sum(nc) / len(nc), max(nc), A.calls_hi,
         sum(1 for x in nc if x > A.calls_hi), len(nc)))

term_by = collections.Counter((s["term"], s["ncall"] > A.calls_hi) for s in SIMS)
print("\n[종료사유 × 폭주여부]")
for (t, hi), n in sorted(term_by.items(), key=lambda kv: -kv[1]):
    print("  %-26s %-9s %3d" % (str(t), "폭주" if hi else "정상", n))

# ── 반복 구조 (R1~R4) ─────────────────────────────────────────────────────────
print("\n" + "=" * 88)
print("[반복 구조]  폭주 후보(호출>%d) 전수" % A.calls_hi)
FEEDBACK_MARK = ("[coverage]", "could not be verified", "call again", "[T2_", "★FEEDBACK",
                 "[quote-pin]", "[GUIDANCE]")
rows = []
for s in [x for x in SIMS if x["ncall"] > A.calls_hi]:
    byname = collections.Counter(c["name"] for c in s["calls"])
    bypair = collections.Counter((c["name"], c["args"]) for c in s["calls"])
    r1 = sum(n - 1 for n in bypair.values() if n > 1)          # 정확 재호출 초과분
    r2 = sum(n - 1 for n in byname.values() if n > 1) - r1     # 인자만 다른 반복
    # R3: 정확 재호출 쌍에서 출력이 같았나
    same_out = diff_out = 0
    seen = {}
    for c in s["calls"]:
        k = (c["name"], c["args"])
        if k in seen:
            if seen[k] == c["out"]:
                same_out += 1
            else:
                diff_out += 1
        seen[k] = c["out"]
    # R4: 정확 재호출 직전 도구 출력에 엔진 피드백 마크가 있었나
    ignored = 0
    prev_out = ""
    seen2 = set()
    for c in s["calls"]:
        k = (c["name"], c["args"])
        if k in seen2 and any(m in prev_out for m in FEEDBACK_MARK):
            ignored += 1
        seen2.add(k)
        prev_out = c["out"]
    # R5 **집행 검정**: 중복-읽기 스텁이 몇 번 떴고, 그 뒤로 몇 번을 더 불렀나 (=무시량).
    #   ★008 정독(2026-08-01)이 R3 라벨의 오분류를 잡았다 — 같은 인자인데 출력 문구만 달라
    #   ('logged successfully'→'Failed: record may already exist') "진전"으로 셌다. 진짜 구조는
    #   **고정 사이클 반복**이므로 아래 주기 검출로 대체한다(집계 라벨 신뢰 금지·[[08]]).
    dup = sum(1 for c in s["calls"] if "[DUPLICATE-READ]" in (c["out"] or ""))
    first_dup = next((i for i, c in enumerate(s["calls"])
                      if "[DUPLICATE-READ]" in (c["out"] or "")), None)
    after_dup = (len(s["calls"]) - first_dup - 1) if first_dup is not None else 0
    # R6 **사이클 주기**: 이름 시퀀스 꼬리에서 p 주기가 3회 이상 반복되는 최소 p.
    names = [c["name"] for c in s["calls"]]
    period = reps = 0
    for p in range(1, 13):
        if len(names) < 3 * p:
            break
        tail = names[-3 * p:]
        if tail[:p] == tail[p:2 * p] == tail[2 * p:]:
            period = p
            k = 0
            while len(names) >= (k + 1) * p and names[-(k + 1) * p:len(names) - k * p] == tail[:p]:
                k += 1
            reps = k
            break
    top = byname.most_common(3)
    rows.append((s, r1, r2, same_out, diff_out, ignored, top, dup, after_dup, period, reps))

if not rows:
    print("  없음 — 이 데이터셋에 폭주 후보 0건")
for s, r1, r2, same_out, diff_out, ignored, top, dup, after_dup, period, reps in rows:
    print("\n  ── %s %s t%s (%s · msg%d · call%d · r=%s)"
          % (s["tag"], s["task"], s["trial"], s["term"], s["nmsg"], s["ncall"], s["reward"]))
    print("     R1 정확-재호출 초과 %3d  |  R2 인자만-다른 반복 %3d" % (r1, r2))
    print("     R6 **사이클** 주기 %s · 반복 %s  ⇒ %s"
          % (period or "—", reps or "—",
             "고정 사이클 폭주" if period and reps >= 3 else "사이클 아님"))
    print("     R5 **집행 검정**: [DUPLICATE-READ] 발화 %3d · 그 뒤로 더 낸 호출 %3d  ⇒ %s"
          % (dup, after_dup, "스텁 무시(soft 실패·[[07]])" if dup and after_dup > dup else "—"))
    print("     R4 엔진 피드백 직후 동일-재호출 %3d" % ignored)
    print("     최다 도구: %s" % ", ".join("%s×%d" % (n, c) for n, c in top))

# ── 도구별 총계 ───────────────────────────────────────────────────────────────
print("\n" + "=" * 88)
print("[폭주 시뮬에서 반복된 도구 총계]")
agg = collections.Counter()
for s in [x[0] for x in rows]:
    for c in s["calls"]:
        agg[c["name"]] += 1
for n, c in agg.most_common(10):
    print("  %-42s %4d" % (n, c))

if A.out:
    json.dump([{k: v for k, v in s.items() if k != "calls"} for s in SIMS],
              open(A.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n→ %s" % A.out)
