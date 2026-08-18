# -*- coding: utf-8 -*-
r"""x380 — **resolve_cap 래치가 pass 를 깎나, 시간만 태우나**(원장 C536ⓔ · 무료 · GPU 0 · LLM 0).

## 왜 (C530 이 미확정으로 남긴 자리)

`_resolve_cap_ok`(`t2_gate_patch.py:3513`)는 *정체*(발화 이후 새 실행 도구 0)에만 과금하고 캡(3)에
걸리면 **계약 경로 전체가 침묵**한다. 그 침묵이 *"지금 X 를 하라"* 를 없애므로 새 실행이 안 생기고,
새 실행이 없으니 **리셋 조건이 안 온다** — 래치. t7313 `task_040` 은 그 뒤로 **48턴**을 태웠다.

C530 은 *"손해 미확정 ⇒ 관측만 수리"* 로 닫았다. 이 스크립트가 그 손해를 **영속 데이터만으로** 잰다.

## 무엇을 세나 (전부 결정론 · gold 무참조)

  · `stops`      — `[T2_MATERIAL_GATE] stop=resolve_cap` (이 줄이 찍힌다 = **이미 캡에 걸렸다**)
  · `resets`     — `[T2_RESOLVE_CAP] 리셋(...)` (실효 리셋일 때만 찍는다·C530ⓔ)
  · `first`      — 첫 stop 의 turn
  · `after_new`  — 첫 stop **이후** 새로 성공 실행된 도구 이름 수(엔진과 **같은 판정**:
                   `_executed_tool_names` 의 실패 표지 규칙을 그대로 쓴다 — 사본 금지 [[67]])
  · `after_turns`— 첫 stop 이후 남은 메시지 수(= 태운 분량)
  · `reward`     — pass 는 `reward` 로만(C486)

## 판정 (사전 고정 · 결과보다 먼저 인쇄)

    래치 sim 의 **과반이 `after_new` = 0**            → 래치 뒤는 **순수 낭비** ⇒ 처방 = 조기 종료
                                                        (비용 노브 · pass 손실 없음)
    래치 sim 의 상당수가 `after_new` > 0 ∧ reward > 0 → 조기 종료는 **pass 를 판다** ⇒ 처방은
                                                        종료가 아니라 **리셋 술어 수정**([[57]] 인자 변화)
    래치/비래치 pass 차이                              → **교란됨**(래치는 부진의 *결과*이기도 하다)
                                                        ⇒ 인과 주장 금지 · [D] 로만 적는다

사용: py -3 x380_latch_damage.py [태그 ...]
"""
import collections
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import t2_forensic as F                                    # noqa: E402

SUF = ".results.json.gz"
DEFAULT_TAGS = ["bank_t7308_ctl_20260818c", "bank_t7308_treat_20260818c",
                "bank_t7310_ctl_20260818e", "bank_t7310_treat_20260818e",
                "bank_t7312_ctl_20260818g", "bank_t7312_treat_20260818g"]
# A2 가 선언한 실패 표지 — 엔진(`_executed_tool_names`)과 **같은 규칙**을 쓴다.
FAILURE_MARKS = ("NOT_VERIFIED", "Failed to ", "Error:")


def executed_names(msgs, upto=None, frm=0):
    """성공 실행된 도구 이름 집합 — 엔진 규칙 축자(실패는 실행이 아니다·`:2314`)."""
    ok, pending = set(), {}
    for i, m in enumerate(msgs or []):
        if upto is not None and i >= upto:
            break
        for tc in (m.get("tool_calls") or []):
            pending[tc.get("id")] = F.nameof(tc)
        if m.get("role") != "tool":
            continue
        nm = pending.get(m.get("id") or m.get("tool_call_id"))
        txt = str(m.get("content") or "").lstrip()
        failed = bool(m.get("error")) or any(txt.startswith(k) for k in FAILURE_MARKS)
        if nm and not failed and i >= frm:
            ok.add(nm)
    return ok


def main():
    tags = [a for a in sys.argv[1:] if not a.startswith("--")] or DEFAULT_TAGS
    print("=" * 100)
    print("x380 · resolve_cap 래치 손해 · 태그 %d개" % len(tags))
    print("판정(사전 고정): 래치 sim 과반이 after_new=0 → 순수 낭비(처방=조기 종료) · "
          "after_new>0 ∧ reward>0 다수 → 리셋 술어 수정 · 래치/비래치 pass 차 = 교란(인과 금지)")
    print("=" * 100)

    rows, tot_stops, tot_resets = [], 0, 0
    for tag in tags:
        stops = F.turns_of(tag, r"stop=resolve_cap")
        resets = F.by_sim(tag, r"\[T2_RESOLVE_CAP\] 리셋")
        tot_resets += sum(len(v) for v in resets.values())
        for s in F.scored(tag, SUF):
            key = F.simtag(s)
            st = [t for t in (stops.get(key) or []) if t is not None]
            n_stop = len(stops.get(key) or [])
            tot_stops += n_stop
            msgs = s.get("messages") or []
            first = min(st) if st else None
            idx = min(first, len(msgs)) if first is not None else None
            before = executed_names(msgs, upto=idx) if idx is not None else set()
            allnames = executed_names(msgs)
            rows.append({
                "tag": tag, "arm": ("treat" if "treat" in tag else "ctl"),
                "task": F.task_id(s), "reward": (s.get("reward_info") or {}).get("reward"),
                "term": F.term_reason(s), "steps": len(msgs),
                "stops": n_stop, "resets": len(resets.get(key) or []),
                "first": first,
                "after_turns": (len(msgs) - idx) if idx is not None else None,
                "after_new": (len(allnames - before)) if idx is not None else None,
                "new_names": sorted(allnames - before)[:4] if idx is not None else [],
            })

    lat = [r for r in rows if r["stops"] > 0]
    non = [r for r in rows if r["stops"] == 0]
    print("sim %d · 래치 %d · 비래치 %d · stop 총 %d · **실효 리셋 총 %d**"
          % (len(rows), len(lat), len(non), tot_stops, tot_resets))
    print("")

    print("## 래치 sim (첫 stop 이후 무엇이 있었나)")
    hdr = "%-9s %-6s %-5s %5s %6s %6s %7s %9s %6s  %s"
    print(hdr % ("task", "tag", "arm", "stops", "first", "steps", "after", "after_new",
                 "reward", "새 도구(≤4)"))
    print("-" * 100)
    for r in sorted(lat, key=lambda x: (-(x["after_turns"] or 0), x["task"])):
        print(hdr % (r["task"], r["tag"].split("_")[1], r["arm"], r["stops"],
                     str(r["first"]), r["steps"], str(r["after_turns"]),
                     str(r["after_new"]), str(r["reward"]),
                     ",".join(n[:22] for n in r["new_names"]) or "-"))

    zero = [r for r in lat if (r["after_new"] or 0) == 0]
    recov = [r for r in lat if (r["after_new"] or 0) > 0]
    recov_pass = [r for r in recov if (r["reward"] or 0) > 0]
    burn = sum(r["after_turns"] or 0 for r in lat)
    print("")
    print("## 집계")
    print("  래치 %d · 그중 이후 새 실행 0 = **%d (%.0f%%)** · 새 실행 있음 %d(그중 pass %d)"
          % (len(lat), len(zero), 100.0 * len(zero) / max(1, len(lat)), len(recov),
             len(recov_pass)))
    print("  래치 뒤에 태운 메시지 총 **%d** (sim 당 중앙 %s)"
          % (burn, sorted(r["after_turns"] or 0 for r in lat)[len(lat) // 2] if lat else "-"))

    def rate(rs):
        p = [r for r in rs if (r["reward"] or 0) > 0]
        return "%d/%d = %.0f%%" % (len(p), len(rs), 100.0 * len(p) / max(1, len(rs)))
    print("  pass — 래치 %s · 비래치 %s   ⚠교란(래치는 부진의 결과이기도 하다)"
          % (rate(lat), rate(non)))
    tm = collections.Counter(r["term"] for r in lat)
    print("  래치 종료사유: " + " · ".join("%s %d" % (k, v) for k, v in tm.most_common(5)))

    if not lat:
        v = "래치 0 — 이 데이터로는 못 잰다"
    elif len(zero) * 2 > len(lat):
        v = ("**래치 뒤는 순수 낭비** — 과반이 새 실행 0 ⇒ 처방 = **조기 종료**(비용 노브)."
             " 단 %d sim 은 회복했고 그중 %d 이 pass 다 — 종료 조건은 그것을 안 죽이게 짜야 한다."
             % (len(recov), len(recov_pass)))
    else:
        v = ("**조기 종료는 pass 를 판다** — 래치 뒤 회복이 과반(%d/%d·pass %d) ⇒ 처방은 종료가 "
             "아니라 **리셋 술어 수정**([[57]] 인자 변화 기준)" % (len(recov), len(lat),
                                                          len(recov_pass)))
    print("")
    print("판정: %s" % v)
    out = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "..", "reports", "facet_rft_2026",
                                        "x380_latch_damage.json"))
    io.open(out, "w", encoding="utf-8").write(json.dumps(
        {"rows": rows, "latched": len(lat), "zero_after": len(zero), "recovered": len(recov),
         "recovered_pass": len(recov_pass), "total_stops": tot_stops,
         "total_resets": tot_resets, "verdict": v}, ensure_ascii=False, indent=1))
    print("원자료: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
