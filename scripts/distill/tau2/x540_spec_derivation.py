# -*- coding: utf-8 -*-
r"""x540 - env unlock 명세가 **오늘 손으로 적은 선언을 그대로 재현하는가** (무료·오프라인·2026-08-25)

## 왜 (사용자 질문 2026-08-25 밤)

*"태스크별로 특정한 방식을 만들지 않고 일반화로는 지금 문제를 해결 못하는 건가?"*

오늘 A2 에 손으로 적은 것은 셋이다 — `write_arg_enum` 의 **값 목록 6칸**, 같은 자리의
**불리언 이름 7개**, 그리고 (철회한) 이름 힌트. 셋 다 *"discoverable 도구는 agent 스키마
목록에 없다"* 는 이유로 베꼈다. 그런데 env 는 unlock 때 명세를 **고정 포맷으로 건네준다**:

    Parameters:
      - contacted_merchant: boolean (required) - Whether the user attempted to resolve …
      - dispute_reason: string (required) - … Must be one of: 'unauthorized_fraudulent_charge', …

⇒ 물음은 하나다: **그 블록에서 도출한 것이 손 선언과 같은가.** 같다면 손 선언은 증명 가능하게
불필요하고, 도메인-특화 순증은 0 으로 내려간다([[05]]). 다르다면 어디가 다른지가 곧 다음 일이다.

## 방법 (판단 0 · 코퍼스 실물만)

  ⑴ 지정한 런 태그들의 **tool 메시지 전량**에서 `Parameters:` 블록을 모은다.
     엔진 정본 `t2_gate_patch._declared_params` 를 그대로 부른다(사본 금지·[[67]]).
  ⑵ 그 블록의 설명문에서 `Must be one of: '…', '…'` 의 **작은따옴표 안 토큰**을 뽑는다.
     ⚠이것은 프로브다 — 엔진에 넣자는 제안이 아니라 *등가성 측정*이다. 엔진으로 옮길지는
       이 수치를 보고 사람이 정한다([[62]]·[[59]] 경계는 그때 다시 답한다).
  ⑶ A2 병합본의 `write_arg_enum` 선언과 **집합으로** 대조한다. gold 미접촉.

## 산출

  도구·인자별로 (선언 ↔ 도출) 이 같은가 · 어느 쪽에만 있나 · 불리언 이름 집합은 같은가.

사용: (로컬/리모트·cwd=scripts/distill/tau2) py -3 x540_spec_derivation.py
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

TAGS = ["bank_t7354_grpA1_20260825", "bank_t7354_grpA2_20260825",
        "bank_t7354_grpA3_20260825", "bank_t7354_grpA4_20260825",
        "bank_t7354_grpB1_20260825", "bank_t7354_grpB2_20260825"]
RE_TOOL = re.compile(r"^Tool:\s*(\S+)\s*$", re.M)
RE_PARAM = re.compile(r"^\s*-\s*(\w+):\s*(\w+)\s*\((required|optional)\)\s*-\s*(.*)$", re.M)
RE_ONEOF = re.compile(r"Must be one of:\s*(.+)$")
RE_QUOTED = re.compile(r"'([^']+)'")


class _M(object):
    def __init__(self, c):
        self.content = c


def harvest(tags):
    """런 궤적의 tool 메시지에서 (도구 → {인자: (타입, 열거값 리스트)}) 를 모은다."""
    import t2_forensic as F
    import t2_gate_patch as G
    out, seen_blocks = {}, 0
    for tag in tags:
        try:
            sims = F.sims(tag, ".results.json.gz")
        except Exception:
            continue
        for s in sims:
            for m in (s.get("messages") or []):
                c = str(m.get("content") or "")
                if "Parameters:" not in c:
                    continue
                seen_blocks += 1
                tm = RE_TOOL.search(c)
                tool = tm.group(1) if tm else "(unnamed)"
                # 타입은 **엔진 정본**이 읽는다(사본 금지).
                typed = G._declared_params([_M(c)])
                d = out.setdefault(tool, {})
                for name, typ, _req, desc in RE_PARAM.findall(c):
                    vals = []
                    hit = RE_ONEOF.search(desc)
                    if hit:
                        vals = RE_QUOTED.findall(hit.group(1))
                    t2 = (typed.get(name) or (typ, bool(hit)))[0]
                    prev = d.get(name)
                    if prev and prev[1] and not vals:
                        continue          # 이미 값을 본 칸을 빈 것으로 덮지 않는다
                    d[name] = (t2, vals)
    return out, seen_blocks


def main():
    from gate_interpreter import load_domain_a2
    derived, nblocks = harvest(TAGS)
    a2 = load_domain_a2("banking_knowledge")
    specs = [s for s in (a2.get("write_arg_enum") or []) if s.get("values") or s.get("booleans")]

    print("== env 명세 블록 %d개에서 도구 %d개를 읽었다 ==" % (nblocks, len(derived)))
    for tool in sorted(derived):
        n_enum = sum(1 for v in derived[tool].values() if v[1])
        n_bool = sum(1 for v in derived[tool].values() if v[0] == "boolean")
        print("  %-46s 인자 %2d · 열거 %d · 불리언 %d"
              % (tool, len(derived[tool]), n_enum, n_bool))

    print("\n== 손 선언 ↔ env 도출 대조 ==")
    same = diff = missing = 0
    for sp in specs:
        prefix = (sp.get("applies_when") or {}).get("prefix") or ""
        arg = sp.get("arg")
        tools = [t for t in derived if t.startswith(prefix)]
        if not tools:
            print("  [%s.%s] env 블록에 그 도구가 없다 — 대조 불가" % (prefix, arg))
            missing += 1
            continue
        tool = tools[0]
        dd = derived[tool]
        if sp.get("values"):
            got = dd.get(arg, ("", []))[1]
            ok = sorted(got) == sorted(str(x) for x in sp["values"])
            print("  [%s.%s] values 선언 %d ↔ 도출 %d  %s"
                  % (tool, arg, len(sp["values"]), len(got), "같다" if ok else "**다르다**"))
            if not ok:
                print("       선언에만: %s" % sorted(set(map(str, sp["values"])) - set(got)))
                print("       도출에만: %s" % sorted(set(got) - set(map(str, sp["values"]))))
            same += 1 if ok else 0
            diff += 0 if ok else 1
        if sp.get("booleans"):
            got = sorted(k for k, v in dd.items() if v[0] == "boolean")
            ok = got == sorted(sp["booleans"])
            print("  [%s.booleans] 선언 %d ↔ 도출 %d  %s"
                  % (tool, len(sp["booleans"]), len(got), "같다" if ok else "**다르다**"))
            if not ok:
                print("       선언에만: %s" % sorted(set(sp["booleans"]) - set(got)))
                print("       도출에만: %s" % sorted(set(got) - set(sp["booleans"])))
            same += 1 if ok else 0
            diff += 0 if ok else 1
    print("\n같다 %d · 다르다 %d · 대조 불가 %d" % (same, diff, missing))
    print("⇒ '다르다'와 '대조 불가'가 0 이면 손 선언은 **증명 가능하게 불필요**하다"
          "(도메인-특화 순증 0·[[05]]). 0 이 아니면 그 칸이 다음 일이다.")
    out = {"probe": "x540", "date": "2026-08-25", "blocks": nblocks,
           "tools": {t: {k: {"type": v[0], "values": v[1]} for k, v in d.items()}
                     for t, d in derived.items()},
           "same": same, "diff": diff, "uncomparable": missing}
    p = os.path.abspath(os.path.join(HERE, "..", "..", "..", "reports", "facet_rft_2026",
                                     "x540_spec_derivation_2026_08_25.json"))
    io.open(p, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, indent=1))
    print("->", p)


if __name__ == "__main__":
    main()
