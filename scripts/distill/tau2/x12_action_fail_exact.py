#!/usr/bin/env python3
"""X12 실패 action_check 엄격 분해 — tau2 채점 규약을 **소스에서 읽어** 재구성 (2026-07-30).

X11(C244)은 미해명 37.9%를 남겼고, 그 원인은 내가 **채점 규약을 추측**했기 때문이다.
`tau2-bench/src/tau2/evaluator/evaluator_action.py` + `data_model/tasks.py`를 직독한 결과:

    match(gold, pred) :=
        gold.name == pred.name                          # ★**외부** 도구명 (inner 아님)
        compare_args := gold.compare_args if not None else **pred.arguments.keys()**
        len(compare_args) == 0  →  True                 # ★예측 인자가 {} 면 **무조건 매치**
        {k:v for k,v in pred.arguments  if k in compare_args}
          == {k:v for k,v in gold.arguments if k in compare_args}

    gold action은 **어떤** 예측 호출과도 매치 안 될 때만 실패(1:1 배정 **없음** — 한 예측이
    여러 gold를 만족시킬 수 있다). 예측 호출 = assistant·user **양쪽**·**requestor 미비교**.

⇒ X11의 구조적 오류 4개가 여기서 확정된다:
  ① 내부 discoverable 이름으로 짝지었다 → 규약은 **외부 이름**을 본다
  ② 1:1 배정을 부과했다 → 규약에 없다(내 "교정 3"이 오히려 오류였다)
  ③ `requestor` 축을 만들었다 → 규약은 requestor를 **안 본다**
  ④ "gold 키가 예측에 없음(MISSING)"을 실패 원인으로 셌다 → 그 키는 **비교 대상에서 빠지므로**
     오히려 매치를 **쉽게** 한다. 실패 원인이 될 수 없다.

★규약에서 도출되는 **진짜** 실패 모드(상호배타·순서대로 판정):
  NAME_ABSENT      외부 도구명을 **한 번도** 호출 안 함
  PRED_EXTRA_KEY   예측이 gold에 **없는 키**를 가짐 → `action_args`에 그 키가 없어 **반드시 불일치**
  NESTED_SERIAL    유일한 차이가 중첩 `arguments` **문자열**이고 **파싱하면 동일** → **채점 아티팩트**
  NESTED_VALUE     중첩 파싱 dict가 실제로 다름 → 내부 키별로 재분류(REF/ENUM/NUM/…)
  TOP_VALUE        최상위 키 값이 다름(예: `agent_tool_name` 오지목)

용법: py -3 x12_action_fail_exact.py [--json out.json]
"""
import argparse
import glob
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = os.path.abspath(os.path.join(_HERE, "..", "..", "..",
                                    "reports", "facet_rft_2026", "sim_results"))
_ID_RE = re.compile(r"^(?:[a-z]{2,4}_)?[0-9a-f]{6,}$", re.I)
_NUM_RE = re.compile(r"^[-+]?\$?[\d,]+(?:\.\d+)?\s*(?:points?|%)?$", re.I)
_ENUM_KEYS = re.compile(r"reason|category|type|status|class|option|design|requested|_flag$", re.I)
_TEXT_KEYS = re.compile(r"summary|note|comment|description|message", re.I)


def preds(msgs):
    """규약대로: assistant·user 양쪽의 tool_calls 전부(requestor 미필터)."""
    out = []
    for m in msgs:
        if m.get("role") not in ("assistant", "user"):
            continue
        for tc in (m.get("tool_calls") or []):
            a = tc.get("arguments")
            if isinstance(a, str):
                try:
                    a = json.loads(a)
                except Exception:
                    a = {}
            out.append((tc.get("name"), a if isinstance(a, dict) else {}))
    return out


def matches(gold_name, gold_args, cmp_args, p_name, p_args):
    """`compare_with_tool_call` 축자 재구현."""
    if gold_name != p_name:
        return False
    ca = list(p_args.keys()) if cmp_args is None else list(cmp_args)
    if len(ca) == 0:
        return True
    t = {k: v for k, v in p_args.items() if k in ca}
    g = {k: v for k, v in gold_args.items() if k in ca}
    return t == g


def _pj(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return None
    return v if isinstance(v, dict) else None


def sub_kind(key, gv, pv):
    if _TEXT_KEYS.search(key):
        return "TEXT"
    sg, sp = str(gv), str(pv)
    if _ID_RE.match(sg) or _ID_RE.match(sp):
        return "REF"
    if _NUM_RE.match(sg) or _NUM_RE.match(sp):
        return "NUM"
    if _ENUM_KEYS.search(key):
        return "ENUM"
    return "OTHER"


def classify(gold, plist):
    """실패 gold 1건의 원인. 반환 (cls, detail)."""
    gname = gold.get("name")
    gargs = gold.get("arguments") or {}
    ca = gold.get("compare_args")
    same = [(n, a) for n, a in plist if n == gname]
    if not same:
        return "NAME_ABSENT", {"tool": gname}
    # 가장 가까운 예측: (extra 키 수, 값 불일치 수) 최소
    def cost(pa):
        c = list(pa.keys()) if ca is None else list(ca)
        extra = sum(1 for k in c if k not in gargs)
        diff = sum(1 for k in c if k in gargs and pa.get(k) != gargs.get(k))
        return (extra, diff)
    best = min((a for _, a in same), key=cost)
    c = list(best.keys()) if ca is None else list(ca)
    extra = [k for k in c if k not in gargs]
    if extra:
        return "PRED_EXTRA_KEY", {"tool": gname, "keys": extra}
    diffs = [k for k in c if k in gargs and best.get(k) != gargs.get(k)]
    if not diffs:
        return "UNEXPLAINED", {"tool": gname, "cmp": c}
    # 중첩 arguments 문자열만 다른가
    if diffs == ["arguments"]:
        pg, pp = _pj(gargs.get("arguments")), _pj(best.get("arguments"))
        if pg is not None and pp is not None:
            if pg == pp:
                return "NESTED_SERIAL", {"tool": gname}
            ik = sorted(set(pg) | set(pp))
            bad = [(k, sub_kind(k, pg.get(k), pp.get(k))) for k in ik
                   if pg.get(k) != pp.get(k)]
            return "NESTED_VALUE", {"tool": gname, "inner": bad}
        return "NESTED_UNPARSED", {"tool": gname}
    return "TOP_VALUE", {"tool": gname,
                         "keys": [(k, sub_kind(k, gargs.get(k), best.get(k))) for k in diffs]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cls = Counter()
    inner_kinds = Counter()
    top_kinds = Counter()
    by_tool = defaultdict(Counter)
    extra_keys = Counter()
    rows = []
    n_fail = n_verify_ok = n_verify_bad = 0
    for path in sorted(glob.glob(os.path.join(_SIM, "bank_day*front*.results.json.gz"))):
        try:
            data = json.load(gzip.open(path, "rt", encoding="utf-8"))
        except Exception:
            continue
        for sim in data.get("simulations") or []:
            ri = sim.get("reward_info") or {}
            acs = ri.get("action_checks") or []
            if not acs:
                continue
            plist = preds(sim.get("messages") or [])
            for ac in acs:
                gold = ac.get("action") or {}
                # ★규약 재구현 검증: 우리가 계산한 match와 저장된 action_match가 같은가
                mine = any(matches(gold.get("name"), gold.get("arguments") or {},
                                   gold.get("compare_args"), n, a) for n, a in plist)
                if bool(ac.get("action_match")) == mine:
                    n_verify_ok += 1
                else:
                    n_verify_bad += 1
                if ac.get("action_reward"):
                    continue
                n_fail += 1
                c, d = classify(gold, plist)
                cls[c] += 1
                by_tool[c][d.get("tool")] += 1
                for k in d.get("keys", []) or []:
                    if isinstance(k, tuple):
                        top_kinds[k[1]] += 1
                    else:
                        extra_keys[k] += 1
                for k, kd in d.get("inner", []) or []:
                    inner_kinds[kd] += 1
                rows.append({"file": os.path.basename(path), "sim": sim.get("id"),
                             "cls": c, **d})

    print("=" * 78)
    print("X12 실패 action_check 엄격 분해 (tau2 규약 소스 직독·축자 재구현)")
    print("=" * 78)
    tot_v = n_verify_ok + n_verify_bad
    print(f"★규약 재구현 검증: 저장된 `action_match`와 일치 {n_verify_ok}/{tot_v} = "
          f"{100.0 * n_verify_ok / tot_v if tot_v else 0:.1f}%  (불일치 {n_verify_bad})")
    if n_verify_bad:
        print("  ⚠불일치가 있으면 아래 분해도 그만큼 의심해야 한다(규약 재구현 미완).")
    print()
    print(f"실패 {n_fail}건 분해:")
    for k, v in cls.most_common():
        print(f"  {k:16s} {v:4d} ({100.0 * v / n_fail if n_fail else 0:5.1f}%)")
    print()
    if extra_keys:
        print(f"PRED_EXTRA_KEY 상위 키: {dict(extra_keys.most_common(8))}")
    if top_kinds:
        print(f"TOP_VALUE 종류: {dict(top_kinds)}")
    if inner_kinds:
        print(f"NESTED_VALUE 내부 종류: {dict(inner_kinds)}")
    print()
    for k in ("NAME_ABSENT", "PRED_EXTRA_KEY", "NESTED_VALUE", "NESTED_SERIAL"):
        if by_tool[k]:
            print(f"{k:16s} 상위 도구: {dict(by_tool[k].most_common(4))}")
    print()
    print("=" * 78)
    print("레버 귀속")
    print("=" * 78)
    lv = {"NAME_ABSENT": "미실행 축 — X10의 EXEC_GAP/DISCOVERY/DECISION 분해로 넘김",
          "PRED_EXTRA_KEY": "★**채점 규약 상호작용** — 예측이 여분 키를 넣으면 필패. "
                            "레버가 아니라 **인자 스키마 준수**(guided_json·검증기) 문제",
          "NESTED_SERIAL": "★**채점 아티팩트** — 의미 동일·직렬화만 다름. 레버 대상 아님",
          "NESTED_VALUE": "실제 operand 오류 — 내부 종류별(REF=F3/⋈ · NUM=compute · ENUM=접지)",
          "TOP_VALUE": "최상위 인자 오류(예: agent_tool_name 오지목)",
          "NESTED_UNPARSED": "중첩 인자 파싱 불가(형식 오류)",
          "UNEXPLAINED": "⚠재구현으로도 설명 안 됨 — 남으면 규약 재확인 필요"}
    for k, v in cls.most_common():
        print(f"  {k:16s} {v:4d} → {lv.get(k, '?')}")
    print()
    print("⚠[[08]]: 집계다. 상위 클래스 사례 2~3건 정독 후 인용할 것.")

    if args.json:
        json.dump({"cls": dict(cls), "inner": dict(inner_kinds), "top": dict(top_kinds),
                   "extra_keys": dict(extra_keys), "verify_ok": n_verify_ok,
                   "verify_bad": n_verify_bad,
                   "by_tool": {k: dict(v) for k, v in by_tool.items()}, "rows": rows},
                  open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
