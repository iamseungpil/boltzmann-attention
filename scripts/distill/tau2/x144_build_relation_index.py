# -*- coding: utf-8 -*-
"""x144 — **선행 관계 인덱스**를 A2 3선언에서 만들고, `prereq_map`과 **등가임을 증명**한다.

정본 = `A3_POLICY_ONTOLOGY_DESIGN_2026_08_08.md` §1d·§1d.1.
설계가 못박은 순서: ①A3 관계 생성(거동 변화 0) → **②등가 검정** → ③소비자 전환 → ④A2 키 삭제.
이 파일은 ①②다. **③ 전에 ②가 통과해야 한다** — 안 그러면 원천이 하나가 되기는커녕 넷이 된다
([[24]]가 2026-08-03에 실측한 그 버그 형태).

무엇을 만드나: `{도구: {"requires": [...], "gate_satisfiers": [...]}}` 인덱스.
출처는 **이미 있는 선언 셋**이고 새 A2 키는 0이다:
  · `gates[].satisfier_requires` · `require_tool_before` · `scaffold_get_tools[].requires_reads`
  \+ `graph_for`가 표적마다 즉석에서 합치던 **`applies_to ∋ target` 게이트의 satisfier**.
⇒ 이 셋+하나가 **한 자료구조**에 들어가면, *"이 행동 전에 확인할 것이 전부 무엇인가"* 가
   검색이 아니라 **조회**가 된다(§1d).

등가 검정(`--verify`): 모든 표적에 대해 **`graph_for(a2, t)` 와 인덱스로 재구성한 그래프가 같은가**.
같지 않으면 **0이 아닌 종료코드**로 죽는다 — 소비자 전환을 막는 것이 이 검정의 일이다.

⚠**기계 도출이라 정책 인용이 없다.** [[23]]이 요구하는 인용은 **정책 상수**에 붙는 것이고, 선행은
  도구 스키마·기존 선언에서 그대로 나온다(§1d의 "출처 규율은 관계마다 다르다") ⇒ opex 0.
⚠상수(`referrer_tenure_days` 등)의 `applies_to` 결속은 **여기서 하지 않는다** — 어느 결정점이
  그 축을 소비하는가는 결속 판단이라 별도 단계로 둔다(§1c).
⚠**분석·빌드 도구이지 런타임 엔진이 아니다**([[59]]).

usage: x144_build_relation_index.py --domain banking_knowledge [--out a3_relations.json] [--verify]
"""

import argparse
import io
import json
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import t2_precedence as prec                      # noqa: E402  — 술어를 복사하지 않는다
from gate_interpreter import load_domain_a2       # noqa: E402


def build(a2):
    """A2 3선언 → 인덱스. 술어를 **다시 구현하지 않는다**(두 벌이면 갈린다).

    ⚠`_raw_declarations`를 **직접** 부른다 — `declarations()`는 인덱스가 있으면 그것을 읽으므로,
    재생성 때 자기 출력을 입력으로 삼는 순환이 된다. 생성기는 언제나 **원천에서** 만든다.
    """
    raw = prec._raw_declarations(a2)
    edges = {}
    for d in raw:
        dep = prec._fam(d["dep"])
        if not dep:
            continue
        cur = edges.setdefault(dep, [])
        for r in d["reads"]:
            r = prec._fam(r)
            if r and r != dep and r not in cur:
                cur.append(r)
    tools = set(edges)
    gate_sat = {}
    for g in ((a2 or {}).get("gates") or []):
        skip = set((g.get("applies_when") or {}).get("not_in") or ())
        sats = sorted((g.get("satisfiers") or {}).keys())
        for t in (g.get("applies_to") or ()):
            if t in skip:
                continue
            cur = gate_sat.setdefault(t, [])
            for s in sats:
                s = prec._fam(s)
                if s and s != prec._fam(t) and s not in cur:
                    cur.append(s)
            tools.add(t)

    by_tool = {}
    for t in sorted(tools):
        by_tool[t] = {"requires": list(edges.get(prec._fam(t)) or []),
                      "gate_satisfiers": list(gate_sat.get(t) or [])}
    return {"_note": "선행 관계 인덱스 — A2 3선언 + applies_to 게이트 satisfier의 기계 도출. "
                     "정본 설계 = A3_POLICY_ONTOLOGY_DESIGN_2026_08_08 §1d·§1d.1. "
                     "declarations는 출처·순서·다중성을 그대로 진다(refcount가 개수를 센다).",
            "declarations": raw,
            "edges": {k: list(v) for k, v in edges.items()},
            "by_tool": by_tool}


def graph_from_index(index, target):
    """인덱스만으로 `graph_for`와 같은 그래프를 재구성한다(소비자가 할 일의 축소판)."""
    edges = {k: list(v) for k, v in (index.get("edges") or {}).items()}
    t = prec._fam(target)
    for s in ((index.get("by_tool") or {}).get(target, {}).get("gate_satisfiers") or []):
        if s and s != t and s not in edges.setdefault(t, []):
            edges[t].append(s)
    return edges


def verify(a2, index):
    """모든 표적에서 `graph_for(a2,·)` == `graph_from_index(index,·)` 인가 + 선언 경로 등가."""
    # ① 소비자 층 등가 — 같은 `declarations()`를 **원천 경로**와 **인덱스 경로**로 각각 태운다.
    piped = dict(a2)
    piped["relations"] = index
    for srcs in (None, (prec.SRC_REQUIRE_BEFORE,),
                 (prec.SRC_REQUIRE_BEFORE, prec.SRC_REQUIRES_READS)):
        want = prec.declarations(a2, srcs)
        got = prec.declarations(piped, srcs)
        tag = "전체" if srcs is None else "+".join(srcs)
        print("선언 등가(%s): 원천 %d쌍 · 인덱스 %d쌍 · %s"
              % (tag, len(want), len(got), "일치" if want == got else "**불일치**"))
        if want != got:
            return False

    # ② 그래프 층 등가
    targets = set(index.get("by_tool") or {})
    for g in ((a2 or {}).get("gates") or []):
        targets |= set(g.get("applies_to") or ())
    targets |= set((a2 or {}).get("action_tools") or ())
    bad = []
    for t in sorted(targets):
        want = prec.graph_for(a2, t)
        got = graph_from_index(index, t)
        if {k: sorted(v) for k, v in want.items()} != {k: sorted(v) for k, v in got.items()}:
            bad.append(t)
    print("등가 검정: 표적 %d개 · **일치 %d · 불일치 %d**"
          % (len(targets), len(targets) - len(bad), len(bad)))
    for t in bad[:15]:
        want, got = prec.graph_for(a2, t), graph_from_index(index, t)
        wk, gk = set(want), set(got)
        print("   ✗ %s" % t)
        print("      A2에만 있는 노드: %s" % sorted(wk - gk))
        print("      인덱스에만: %s" % sorted(gk - wk))
        for k in sorted(wk & gk):
            if sorted(want[k]) != sorted(got[k]):
                print("      %s: A2 %s ≠ 인덱스 %s" % (k, sorted(want[k]), sorted(got[k])))
    return not bad


def snapshot(a2, domain):
    """삭제 **전** 상태를 얼린다 — 4단계(A2 키 삭제)의 유일한 진짜 기준선.

    삭제하고 나면 `_raw_declarations`가 빈 것을 돌려주므로 *"원천 vs 인덱스"* 대조는 **공허하게
    참**이 된다(빈 것과 빈 것을 비교한다). 그래서 지우기 전에 **모든 표적의 그래프와 선언 전량**을
    파일로 굳혀 두고, 지운 뒤 그것과 맞댄다.
    """
    targets = set()
    for g in ((a2 or {}).get("gates") or []):
        targets |= set(g.get("applies_to") or ())
    targets |= set((a2 or {}).get("action_tools") or ())
    targets |= set(prec.prereq_map(a2))
    return {"_note": "A2 키 삭제 전 기준선 — `x144 --check-snapshot`이 이것과 맞댄다.",
            "domain": domain,
            "declarations": [list(x) for x in prec.declarations(a2)],
            "graphs": {t: {k: sorted(v) for k, v in prec.graph_for(a2, t).items()}
                       for t in sorted(targets)}}


def check_snapshot(a2, snap, domain):
    """삭제 후 상태가 기준선과 **같은 그래프·같은 선언**을 내는가."""
    if snap.get("domain") != domain:
        print("⚠기준선 도메인 불일치: %s ≠ %s" % (snap.get("domain"), domain))
        return False
    want_d = [tuple(x) for x in (snap.get("declarations") or [])]
    got_d = prec.declarations(a2)
    ok_d = want_d == got_d
    print("선언 %d쌍 · %s" % (len(want_d), "일치" if ok_d else "**불일치**"))
    if not ok_d:
        print("   기준선에만: %s" % [x for x in want_d if x not in got_d][:6])
        print("   현재에만  : %s" % [x for x in got_d if x not in want_d][:6])
    bad = []
    for t, want in sorted((snap.get("graphs") or {}).items()):
        got = {k: sorted(v) for k, v in prec.graph_for(a2, t).items()}
        if got != want:
            bad.append((t, want, got))
    print("그래프 표적 %d · 일치 %d · **불일치 %d**"
          % (len(snap.get("graphs") or {}), len(snap.get("graphs") or {}) - len(bad), len(bad)))
    for t, want, got in bad[:10]:
        print("   ✗ %s" % t)
        print("      기준선: %s" % want)
        print("      현재  : %s" % got)
    return ok_d and not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="banking_knowledge")
    ap.add_argument("--out", default="")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--snapshot", default="", help="삭제 전 기준선을 이 파일로 얼린다")
    ap.add_argument("--check-snapshot", default="", help="그 기준선과 현재를 맞댄다")
    a = ap.parse_args()

    a2 = load_domain_a2(a.domain)
    if not a2:
        print("A2 없음: %s" % a.domain)
        return 2

    if a.snapshot:
        snap = snapshot(a2, a.domain)
        io.open(a.snapshot, "w", encoding="utf-8").write(
            json.dumps(snap, ensure_ascii=False, indent=1))
        print("기준선 얼림: %s (선언 %d쌍 · 표적 %d)"
              % (a.snapshot, len(snap["declarations"]), len(snap["graphs"])))
        return 0
    if a.check_snapshot:
        with io.open(a.check_snapshot, encoding="utf-8") as f:
            snap = json.load(f)
        return 0 if check_snapshot(a2, snap, a.domain) else 1

    index = build(a2)
    n_req = sum(1 for v in index["by_tool"].values() if v["requires"])
    n_gs = sum(1 for v in index["by_tool"].values() if v["gate_satisfiers"])
    print("인덱스: 도구 %d · requires 있는 도구 %d · 게이트 satisfier 있는 도구 %d"
          % (len(index["by_tool"]), n_req, n_gs))

    ok = True
    if a.verify:
        ok = verify(a2, index)
    if a.out:
        io.open(a.out, "w", encoding="utf-8").write(
            json.dumps(index, ensure_ascii=False, indent=1))
        print("저장: %s" % a.out)
    if a.verify and not ok:
        print("⚠**등가 검정 실패 — 소비자 전환은 여기서 멈춘다**(설계서 §1d.1 순서).")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
