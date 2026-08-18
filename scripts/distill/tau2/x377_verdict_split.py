#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""x377 — **`VIOLATES 0` 을 가른다**(런북 STEP 2 · 무료 · GPU 0 · 사용자 승인 2026-08-18).

## 무엇을 묻나
`T2_VERDICT_CARRY` 는 발화율 100% 인데 판정 줄의 절반이 `VIOLATES 0` 이다(§5⑴). 그것이
  ⒜ **옳은 침묵** — 손님 요구가 실제로 아무 후보도 배제하지 않는다, 인가
  ⒝ **무력** — 배제해야 하는데 판정이 안 갈린다, 인가
는 아직 안 갈렸다. 이 스크립트는 **기계적으로 확인 가능한 부분만** 낸다:

  ① 판정 줄 전수(사이드카 `kind=verdict-lines` 축자) — 태스크·turn·군·OK/VIOLATES
  ② **표적 유무** — 그 군의 후보 이름이 **gold 액션 인자에 실재하는가**(문자열 실재확인만·
     정규식 0·[[59]]). 표적이 없으면 그 발화는 *애초에 고를 것이 없는 자리*다.
  ③ 무정보(`VIOLATES 0`) × 표적유무 교차표 — **판정 기준은 사전 고정**(핸드오프 §0-2 STEP 2):
     *무정보 줄의 과반에서 표적(gold 후보)이 OK 집합에 그냥 섞여 있으면 = 무력*.

⚠gold 는 **판정용 조회**다 — 여기서 읽은 것은 레버 재료로 넘어가지 않는다([[23]]).
⚠엔진 코드가 아니다(오프라인 분석). 처방은 격리 뒤에만([[62]]).

사용: py -3 x377_verdict_split.py <tag> [<tag> ...]
"""
import collections
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t2_forensic as F  # noqa: E402

SUF = ".results.json.gz"
TAGS = [a for a in sys.argv[1:] if not a.startswith("--")]


def parse_lines(text):
    """판정 줄 → [(표시명, 판정)] . 줄 형식은 A2 `verdict_line_template` 축자."""
    out = []
    for ln in str(text or "").splitlines():
        ln = ln.strip()
        if not ln.startswith("- ") or ":" not in ln:
            continue
        name, rest = ln[2:].split(":", 1)
        head = rest.strip().split(" ")[0].strip().upper()
        out.append((name.strip(), head if head in ("OK", "VIOLATES", "UNCLEAR") else "?"))
    return out


def gold_strings(sim):
    """gold 액션의 **문자열 인자값** 전부(이름 포함) — 실재확인용 말뭉치."""
    buf = []
    for a in F.gold_actions(sim):
        act = a.get("action") or a
        buf.append(str(act.get("name") or ""))
        for v in (act.get("arguments") or {}).values():
            if isinstance(v, str):
                buf.append(v)
            elif isinstance(v, (list, dict)):
                buf.append(json.dumps(v, ensure_ascii=False))
    return " || ".join(buf)


def gold_names(sim):
    out = []
    for a in F.gold_actions(sim):
        act = a.get("action") or a
        ar = act.get("arguments") or {}
        inner = (ar.get("agent_tool_name") or ar.get("user_tool_name")
                 or ar.get("discoverable_tool_name") or "")
        out.append(str(inner or act.get("name") or "?"))
    return out


def main():
    rows = []
    for tag in TAGS:
        by_task = {}
        for s in F.scored(tag, SUF):
            by_task[F.simtag(s)] = s
        for r in F.sidecar_rows(tag):
            if r.get("kind") != "verdict-lines":
                continue
            st = r.get("simtag") or ""
            sim = by_task.get(st)
            lines = parse_lines(r.get("text"))
            ok = [n for n, v in lines if v == "OK"]
            vio = [n for n, v in lines if v == "VIOLATES"]
            gs = gold_strings(sim) if sim else ""
            hit = [n for n, _v in lines if n and n.split(" (")[0].lower() in gs.lower()]
            rows.append({
                "tag": tag, "task": st.split("#")[0], "turn": r.get("turn"),
                "group": r.get("group"), "n": len(lines), "ok": len(ok), "vio": len(vio),
                "reward": (sim.get("reward_info") or {}).get("reward") if sim else None,
                "term": F.term_reason(sim) if sim else "?",
                "gold_tools": gold_names(sim) if sim else [],
                "hit": hit,
                "hit_verdict": [v for n, v in lines if n in hit],
            })

    print("=" * 108)
    print("x377  VIOLATES 0 가르기 —  tags: %s" % ", ".join(TAGS))
    print("=" * 108)
    hdr = "%-9s %-6s %-4s %-22s %3s %3s %4s  %-6s %-11s %s"
    print(hdr % ("task", "tag", "turn", "group", "n", "OK", "VIO", "rw", "표적", "gold 안 후보"))
    print("-" * 108)
    for r in sorted(rows, key=lambda x: (x["task"], x["tag"], x["turn"] or 0)):
        tg = "있음" if r["hit"] else "없음"
        print(hdr % (r["task"], r["tag"].split("_")[1], str(r["turn"]), r["group"],
                     r["n"], r["ok"], r["vio"], str(r["reward"]), tg,
                     ", ".join("%s=%s" % (n, v) for n, v in zip(r["hit"], r["hit_verdict"]))
                     or "-"))

    print()
    print("## 교차표 — 무정보(VIOLATES 0) × 표적유무")
    tab = collections.Counter()
    for r in rows:
        tab[("무정보" if r["vio"] == 0 else "갈림", "표적있음" if r["hit"] else "표적없음")] += 1
    print("%-8s %-10s %-10s" % ("", "표적있음", "표적없음"))
    for a in ("무정보", "갈림"):
        print("%-8s %-10d %-10d" % (a, tab[(a, "표적있음")], tab[(a, "표적없음")]))

    ninfo = [r for r in rows if r["vio"] == 0]
    tgt = [r for r in ninfo if r["hit"]]
    print()
    print("무정보 줄 %d 중 표적 있는 줄 %d (%.0f%%) — 사전 고정 기준: 과반이면 **무력**"
          % (len(ninfo), len(tgt), (100.0 * len(tgt) / len(ninfo)) if ninfo else 0))
    print("판정: %s" % ("무력(적용 범위 축소 후보)" if ninfo and len(tgt) * 2 > len(ninfo)
                       else "무력 아님 — 무정보의 과반이 **표적 없는 자리**(발화 자체가 범위 밖)"))

    print()
    print("## 무정보 줄의 gold 도구(그 자리에서 무엇을 해야 했나)")
    for r in sorted(ninfo, key=lambda x: (x["task"], x["turn"] or 0)):
        print("  %-9s %-6s turn=%-3s %-22s gold=%s"
              % (r["task"], r["tag"].split("_")[1], r["turn"], r["group"],
                 ", ".join(sorted(set(r["gold_tools"])))[:120]))


if __name__ == "__main__":
    main()
