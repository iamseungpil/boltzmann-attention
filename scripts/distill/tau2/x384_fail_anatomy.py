# -*- coding: utf-8 -*-
r"""x384 — **실패 해부**: 20 태스크의 fail 을 *gold 액션 × turn × 레버 × 응답*으로 편다
(사용자 지시 2026-08-18: *"각 fail 의 실패를 per step per lever per 응답 별로 구분해서 정밀하게"*).

## 왜 이 모양인가

`x375` 는 sim 한 줄에 reward 하나를 준다 — 그 줄로는 *왜* 를 못 읽는다([[08]]). 여기서는 채점의
단위인 **gold 액션마다** ⑴그 도구를 아예 안 불렀는가 ⑵불렀는데 인자가 어긋났는가를 가르고,
그 자리의 **turn**·**우리 층 레버**·**모델 응답**·**우리가 넣은 문장**을 나란히 붙인다.

⚠`action_match` 는 소수점 표기로 무너진다(C486) ⇒ **미매치 = 원인 후보**이지 확정이 아니다.
  그래서 *도구를 아예 안 불렀는가* 를 궤적에서 따로 확인해 두 경우를 갈라 인쇄한다.
⚠원인 코드는 **사전 고정 우선순위**로만 붙인다(첫 매치). 새 이름 만들지 않는다([[48]]).

## 원인 코드 (순서대로 · 첫 매치 · 근거 열이 항상 같이 인쇄된다)

    READ_MISS   미매치 gold 중 **read/generic** 도구를 아예 안 불렀다
    WRITE_MISS  미매치 gold **write** 도구를 아예 안 불렀다
    ARG_MISS    그 도구는 불렀는데 **인자가 어긋났다**(표기 문제 가능·C486)
    LOOP        `stop=resolve_cap` ≥10 ∧ steps ≥60 (창 순환이 궤적을 먹었다)
    USER_LEFT   마지막 2턴에 도구 호출 0 ∧ 종료 `user_stop`(말만 하다 끝)
    OTHER       위 어디에도 안 걸림

사용: py -3 x384_fail_anatomy.py <tag> [<tag> …]   (기본 = t7313 두 팔 + t7312 두 팔)
"""
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

import t2_forensic as F  # noqa: E402

SUF = ".results.json.gz"
DEFAULT = ["bank_t7313_ctl_20260818h", "bank_t7313_treat_20260818h",
           "bank_t7312_ctl_20260818g", "bank_t7312_treat_20260818g"]
NOISE = {"T2_LEVER", "T2_SG_TRACE", "T2_A2_VARIANT", "T2_AXIS", "T2_FB_VIEW"}


def gold_rows(sim):
    """gold 액션마다 (이름, tool_type, action_match)."""
    out = []
    for ck in ((sim.get("reward_info") or {}).get("action_checks") or []):
        a = ck.get("action") or {}
        ar = a.get("arguments") or {}
        nm = (ar.get("agent_tool_name") or ar.get("user_tool_name")
              or ar.get("discoverable_tool_name") or a.get("name") or "?")
        out.append({"name": str(nm), "type": ck.get("tool_type"),
                    "match": bool(ck.get("action_match")), "id": a.get("action_id")})
    return out


def called(sim):
    """실행된 이름 → [turn] (디스패처는 안쪽 이름까지)."""
    out = collections.defaultdict(list)
    for i, m in enumerate(sim.get("messages") or []):
        t = m.get("turn_idx", i)
        for tc in (m.get("tool_calls") or []):
            a = F.argsof(tc)
            out[str(F.inner_name(a) or F.nameof(tc))].append(t)
            out[F.nameof(tc)].append(t)
    return out


def last_turns(sim, k=2):
    """마지막 k 턴에 도구 호출이 있었나 + 마지막 assistant 본문."""
    msgs = sim.get("messages") or []
    tail = msgs[-6:]
    has = any(m.get("tool_calls") for m in tail[-k * 2:])
    txt = ""
    for m in reversed(msgs):
        if m.get("role") == "assistant" and (m.get("content") or ""):
            txt = " ".join(str(m["content"]).split())
            break
    return has, txt


def main():
    tags = [a for a in sys.argv[1:] if not a.startswith("--")] or DEFAULT
    print("=" * 118)
    print("x384 · 실패 해부 · 태그 %s" % ", ".join(t.split("_")[1] + "/" +
                                                 ("treat" if "treat" in t else "ctl") for t in tags)),
    print("원인 코드(사전 고정·첫 매치): READ_MISS → WRITE_MISS → ARG_MISS → LOOP → USER_LEFT → OTHER")
    print("=" * 118)

    causes = collections.Counter()
    detail = []
    for tag in tags:
        arm = "treat" if "treat" in tag else "ctl"
        stops = {k.split("#")[0]: len([x for x in v])
                 for k, v in F.turns_of(tag, r"stop=resolve_cap").items()}
        marks = F.turns_of(tag, r"\[T2_[A-Z_]+\]")          # sim -> [turn…] (마커 전량)
        rows_by_sim = collections.defaultdict(list)
        for d in (F.trace(tag) or []):
            rows_by_sim[str(d.get("sim") or "").split("#")[0]].append(d)
        side = collections.defaultdict(list)
        for r in F.sidecar_rows(tag):
            side[str(r.get("simtag", "")).split("#")[0]].append(r)

        for sim in F.scored(tag, SUF):
            rw = (sim.get("reward_info") or {}).get("reward")
            if (rw or 0) >= 1.0:
                continue
            task = F.task_id(sim)
            g = gold_rows(sim)
            cl = called(sim)
            miss = [x for x in g if not x["match"]]
            never = [x for x in miss if x["name"] not in cl]
            steps = len(sim.get("messages") or [])
            st = stops.get(task, 0)
            has_call, last_txt = last_turns(sim)

            reads = [x for x in never if x["type"] != "write"]
            writes = [x for x in never if x["type"] == "write"]
            argm = [x for x in miss if x["name"] in cl]
            if reads:
                code, why = "READ_MISS", reads[0]["name"]
            elif writes:
                code, why = "WRITE_MISS", writes[0]["name"]
            elif argm:
                code, why = "ARG_MISS", argm[0]["name"]
            elif st >= 10 and steps >= 60:
                code, why = "LOOP", "stop=%d steps=%d" % (st, steps)
            elif not has_call and F.term_reason(sim) == "user_stop":
                code, why = "USER_LEFT", "마지막 2턴 호출 0"
            else:
                code, why = "OTHER", ""
            causes[(code, arm)] += 1

            # 결정 turn = 문제 gold 도구가 불렸으면 그 첫 turn · 아니면 마지막 호출 turn
            dturn = None
            if why in cl:
                dturn = min(cl[why])
            else:
                allt = [t for v in cl.values() for t in v]
                dturn = max(allt) if allt else None
            # 그 turn ±1 에 뜬 우리 층 마커
            lev = []
            for d in rows_by_sim.get(task, []):
                t = d.get("turn")
                if not isinstance(t, int) or dturn is None or abs(t - dturn) > 1:
                    continue
                ln = str(d.get("line") or "")
                i, j = ln.find("[T2_"), ln.find("]")
                if i >= 0 and j > i:
                    nm = ln[i + 1:j]
                    if nm not in NOISE and nm not in lev:
                        lev.append(nm)
            fb = [r for r in side.get(task, []) if isinstance(r.get("turn"), int)
                  and dturn is not None and abs(r["turn"] - dturn) <= 1]
            detail.append({"task": task, "arm": arm, "code": code, "why": why, "dturn": dturn,
                           "gold": len(g), "miss": len(miss), "never": len(never),
                           "stops": st, "steps": steps, "lev": lev[:5],
                           "fb": [(r.get("channel"), " ".join(str(r.get("text", "")).split())[:60])
                                  for r in fb[:2]],
                           "last": last_txt[:70]})

    hdr = "%-9s %-5s %-10s %-30s %5s %5s %5s %5s"
    print(hdr % ("task", "arm", "원인", "표적(도구/근거)", "turn", "gold", "미매", "stop"))
    print("-" * 118)
    for d in sorted(detail, key=lambda x: (x["task"], x["arm"])):
        print(hdr % (d["task"], d["arm"], d["code"], d["why"][:30], str(d["dturn"]),
                     d["gold"], d["miss"], d["stops"]))

    print("")
    print("## 원인 × 팔")
    arms = sorted({d["arm"] for d in detail})
    print("%-11s %s" % ("", " ".join("%-6s" % a for a in arms)))
    for code in ("READ_MISS", "WRITE_MISS", "ARG_MISS", "LOOP", "USER_LEFT", "OTHER"):
        if sum(causes[(code, a)] for a in arms):
            print("%-11s %s" % (code, " ".join("%-6d" % causes[(code, a)] for a in arms)))

    print("")
    print("## 결정 turn 의 레버·문장·응답 (per step)")
    for d in sorted(detail, key=lambda x: (x["code"], x["task"], x["arm"])):
        print("  %-9s %-5s %-10s turn=%-4s 레버=%s"
              % (d["task"], d["arm"], d["code"], str(d["dturn"]),
                 ",".join(x.replace("T2_", "") for x in d["lev"]) or "(없음)"))
        for ch, tx in d["fb"]:
            print("        우리 문장[%s] %s" % (ch, tx))
        if d["last"]:
            print("        마지막 응답: %s" % d["last"])
    out = os.path.normpath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                        "x384_fail_anatomy.json"))
    io.open(out, "w", encoding="utf-8").write(json.dumps(detail, ensure_ascii=False, indent=1))
    print("")
    print("원자료: %s" % out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
