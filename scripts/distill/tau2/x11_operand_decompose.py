#!/usr/bin/env python3
"""X11 OPERAND 분해 — 지배 조각(57.1%)의 내부 구성 (2026-07-30).

C243이 실패 원인 census에서 **OPERAND 397건(57.1%)**을 지배 조각으로 확정했다(EXEC_GAP 11.1%).
"도구는 맞게 골랐고 인자·참조가 틀렸다"는 것까지는 알지만 **어느 인자가 어떻게 틀렸는지**는
분해되지 않았다. 다음 레버가 그 분해에 달려 있다(참조 오선택이면 F3/REF_ISO 계열·계산이면
compute 계열·날조면 provenance 계열).

방법: 실패 `action_check`의 **기대 인자** vs 궤적의 **실제 호출 인자**를 키별로 diff.
  · 디스패처 경유면 `arguments.agent_tool_name`이 실 도구, `arguments.arguments`가 **JSON 문자열**
  · 같은 도구를 여러 번 호출했으면 **일치 인자가 가장 많은 호출**을 짝으로 잡는다(best-match)
  · `compare_args: null` = 전 인자 채점(실측 확인)

키별 오류 분류(전부 궤적 근거·[[08]] per-case 보존):
  REF_WRONG   id-형 인자가 다르고, 에이전트가 쓴 값이 **어떤 도구 출력에 실재** → 실재하는 **오참조**
  FABRICATED  id-형 인자가 다르고, 그 값이 **어디에도 없다** → 날조
  CALC_WRONG  수치-형 인자가 다름(금액·포인트 등) → 계산·집계 오류
  ENUM_WRONG  범주-형 인자가 다름(reason·category·type·status) → enum 오선택
  MISSING     기대 키가 실제 호출에 **없음**
  TEXT_DIFF   자유서술 인자 차이(summary 등) — 채점 아티팩트 가능성 있어 별도 계상

용법: py -3 x11_operand_decompose.py [--json out.json] [--exclude-tool NAME]
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
_SUF = re.compile(r"_\d{3,4}$")

# ★C244 자기정정: `unlock_`/`list_` 호출도 `agent_tool_name`을 갖는다. 그걸 "그 도구를 호출했다"로
#   세면 (a)X10에서 미호출을 OPERAND로 오분류 (b)X11에서 인자 없는 unlock을 짝지어 전 인자를
#   MISSING으로 오계상한다. **unlock/list는 실행이 아니다**(prekb `_effective_fams`·C159 교훈).
#   ⇒ **실행으로 세는 것은 dispatch(call_)·give_ 계열뿐**. 이름은 A2에서 읽는다(리터럴 0).
_A2_PATH = os.path.join(_HERE, "a2", "banking_knowledge.gate.json")


def _dispatch_cfg():
    try:
        ep = (json.load(open(_A2_PATH, encoding="utf-8")).get("eplan") or {})
    except Exception:
        return set(), set()
    exec_names = {ep.get("dispatch_tool")}
    nonexec = {ep.get("unlock_tool"), ep.get("list_tool")}
    return {x for x in exec_names if x}, {x for x in nonexec if x}


_EXEC_DISP, _NONEXEC_DISP = _dispatch_cfg()

_ID_RE = re.compile(r"^(?:[a-z]{2,4}_)?[0-9a-f]{8,}$|^[a-z]{2,4}_[0-9a-f]{6,}$", re.I)
_NUM_RE = re.compile(r"^[-+]?\$?[\d,]+(?:\.\d+)?\s*(?:points?|%)?$", re.I)
_ENUM_KEYS = re.compile(r"reason|category|type|status|class|option|design|requested|_flag$", re.I)
_TEXT_KEYS = re.compile(r"summary|note|comment|description|message", re.I)


def fam(n):
    return _SUF.sub("", str(n or ""))


def parse_args(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return {}
    return v if isinstance(v, dict) else {}


def flat_args(a):
    """디스패처 래핑을 벗겨 **실 도구의 인자 dict**와 실 도구명을 반환."""
    a = parse_args(a)
    inner_name = None
    for k in ("agent_tool_name", "discoverable_tool_name", "user_tool_name"):
        if a.get(k):
            inner_name = a[k]
            break
    if inner_name is not None:
        return fam(inner_name), parse_args(a.get("arguments"))
    return None, a


def all_calls(msgs):
    """궤적의 **실행** 호출 → [(fam, args_dict, requestor)]. ★C244: unlock/list 제외(실행 아님)."""
    out = []
    for m in msgs:
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name")
            if nm in _NONEXEC_DISP:
                continue
            inner, args = flat_args(tc.get("arguments"))
            out.append((inner or fam(nm), args, tc.get("requestor") or "assistant"))
    return out


def outputs_blob(msgs):
    return "\n".join(str(m.get("content") or "") for m in msgs
                     if m.get("role") == "tool" and not m.get("error"))


def kind_of(key, exp, act):
    if _TEXT_KEYS.search(key):
        return "TEXT_DIFF"
    if act is None:
        return "MISSING"
    se, sa = str(exp), str(act)
    if _ID_RE.match(se) or _ID_RE.match(sa):
        return "ID"
    if _NUM_RE.match(se) or _NUM_RE.match(sa):
        return "CALC_WRONG"
    if _ENUM_KEYS.search(key):
        return "ENUM_WRONG"
    return "OTHER"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--exclude-tool", default="",
                    help="쏠림 제거용(예: submit_cash_back_dispute)")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    cls = Counter()
    by_tool = defaultdict(Counter)
    by_key = defaultdict(Counter)
    rows = []
    n_operand = 0
    for path in sorted(glob.glob(os.path.join(_SIM, "bank_day*front*.results.json.gz"))):
        try:
            data = json.load(gzip.open(path, "rt", encoding="utf-8"))
        except Exception:
            continue
        for sim in data.get("simulations") or []:
            ri = sim.get("reward_info") or {}
            if (ri.get("reward") or 0) >= 1.0:
                continue
            msgs = sim.get("messages") or []
            calls = all_calls(msgs)
            blob = outputs_blob(msgs)
            # ★C244 4차 교정: **1:1 배정**. 초판은 기대 건마다 독립 best-match를 해서 기대 N건이
            #   실제 M<N건에 **중복 매칭**됐고, 인자가 우연히 맞는 기대 건들이 `NO_ARG_DIFF`로
            #   계상됐다(147건). 그 실체는 operand 오류가 아니라 **개수 부족=coverage(F4)**다.
            #   ⇒ 도구별로 실제 호출을 **소비**하며 배정하고, 남은 기대 건은 `COUNT_SHORT`.
            used = defaultdict(list)                # fam -> 소비된 인덱스
            pool = defaultdict(list)
            for _i, (_t, _a, _rq) in enumerate(calls):
                pool[_t].append((_a, _rq))
            for ac in (ri.get("action_checks") or []):
                if ac.get("action_reward"):
                    continue
                act = ac.get("action") or {}
                tgt, exp_args = flat_args(act.get("arguments"))
                tgt = tgt or fam(act.get("name"))
                if args.exclude_tool and tgt == args.exclude_tool:
                    continue
                avail = [x for j, x in enumerate(pool[fam(tgt)]) if j not in used[fam(tgt)]]
                if not pool[fam(tgt)]:
                    continue                       # 아예 미호출 — C243이 따로 분류
                if not avail:
                    cls["COUNT_SHORT"] += 1        # 호출은 했으나 **횟수 부족**(coverage)
                    by_tool["COUNT_SHORT"][fam(tgt)] += 1
                    rows.append({"file": os.path.basename(path), "sim": sim.get("id"),
                                 "tool": fam(tgt), "diffs": [], "note": "실제 호출 소진(개수 부족)"})
                    continue
                cands = avail
                # ★C244 3차 교정: 기대 action이 **give 단계**(도구 이름만·내부 인자 없음)면
                #   비교할 operand가 애초에 없다. 초판은 이를 "인자 차이 없음"으로 계상해
                #   `NO_ARG_DIFF` 181건(그중 169건이 이 계열)을 만들었다 = operand 오류 아님.
                #   실체는 **프로비저닝/user-실행 실패**(실측: give 23회 vs user 실행 2/4회).
                if not exp_args:
                    cls["GIVE_STEP"] += 1
                    by_tool["GIVE_STEP"][fam(tgt)] += 1
                    rows.append({"file": os.path.basename(path), "sim": sim.get("id"),
                                 "tool": fam(tgt), "diffs": [], "note": "give 단계(내부 인자 없음)"})
                    continue
                n_operand += 1
                # best-match: 일치 키 최다
                exp_rq = act.get("requestor") or "assistant"
                # ★C244 5차 교정: **requestor(누가 실행했나)**도 채점 축이다. 1:1 배정 후에도
                #   `NO_ARG_DIFF` 135건이 남은 이유가 이것 — 인자는 같은데 **실행 주체가 다르다**
                #   (기대 user vs 실제 assistant 등). operand도 coverage도 아닌 **채널 축**이고,
                #   우리 스택엔 그 레버가 따로 있다(give-도구 채널 교정·completion_guard).
                best, best_rq = max(cands, key=lambda x: (
                    sum(1 for k, v in exp_args.items() if str(x[0].get(k)) == str(v)),
                    x[1] == exp_rq))
                _bl = pool[fam(tgt)]
                used[fam(tgt)].append(next(j for j, x in enumerate(_bl)
                                           if x[0] is best and j not in used[fam(tgt)]))
                if best_rq != exp_rq:
                    cls["REQUESTOR"] += 1
                    by_tool["REQUESTOR"][fam(tgt)] += 1
                    rows.append({"file": os.path.basename(path), "sim": sim.get("id"),
                                 "tool": fam(tgt), "diffs": [],
                                 "note": f"실행 주체 불일치 기대={exp_rq} 실제={best_rq}"})
                    continue
                diffs = []
                for k, v in exp_args.items():
                    av = best.get(k)
                    if str(av) == str(v):
                        continue
                    kd = kind_of(k, v, av)
                    if kd == "ID":
                        kd = "REF_WRONG" if (av is not None and str(av) in blob) \
                            else "FABRICATED"
                    diffs.append({"key": k, "kind": kd, "expected": str(v)[:60],
                                  "actual": str(av)[:60]})
                    cls[kd] += 1
                    by_tool[kd][fam(tgt)] += 1
                    by_key[kd][k] += 1
                if not diffs:
                    cls["NO_ARG_DIFF"] += 1        # 인자 동일인데 불일치 = 1:1 배정·개수 축
                rows.append({"file": os.path.basename(path), "sim": sim.get("id"),
                             "tool": fam(tgt), "diffs": diffs})

    print("=" * 76)
    print("X11 OPERAND 분해" + (f" (제외: {args.exclude_tool})" if args.exclude_tool else ""))
    print("=" * 76)
    print(f"OPERAND action_check {n_operand}건 · 인자-차이 {sum(v for k, v in cls.items() if k != 'NO_ARG_DIFF')}개")
    print()
    tot = sum(cls.values())
    for k, v in cls.most_common():
        print(f"  {k:12s} {v:4d} ({100.0 * v / tot if tot else 0:5.1f}%)")
    print()
    for k in ("REF_WRONG", "FABRICATED", "CALC_WRONG", "ENUM_WRONG", "MISSING"):
        if by_key[k]:
            print(f"{k:11s} 상위 인자: {dict(by_key[k].most_common(5))}")
    print()
    for k in ("REF_WRONG", "FABRICATED", "CALC_WRONG", "ENUM_WRONG"):
        if by_tool[k]:
            print(f"{k:11s} 상위 도구: {dict(by_tool[k].most_common(4))}")
    print()
    print("=" * 76)
    print("레버 귀속")
    print("=" * 76)
    lever = {"REF_WRONG": "F3/⋈ · REF_ISO·reference_filter(참조 재선택)",
             "FABRICATED": "provenance/CLAIM_PROV(날조 차단)",
             "CALC_WRONG": "compute offload(t2_compute·scaffold_get)",
             "ENUM_WRONG": "enum 접지·KB 자격 문서(prekb)",
             "MISSING": "완결 게이트(단 write 강제 금지·§1.5 Q5)",
             "TEXT_DIFF": "(채점 아티팩트 가능·별도 판정)",
             "OTHER": "(미분류)", "NO_ARG_DIFF": "1:1 배정·개수 축(인자 동일)",
             "GIVE_STEP": "★operand 아님 — 프로비저닝/user-실행 실패(별 축)",
             "COUNT_SHORT": "★operand 아님 — **coverage/F4**(호출했으나 횟수 부족)",
             "REQUESTOR": "★operand 아님 — **채널 축**(실행 주체 불일치·give 채널·completion_guard)"}
    for k, v in cls.most_common():
        print(f"  {k:12s} {v:4d} → {lever.get(k, '?')}")
    print()
    print("⚠[[08]]: 집계다. 결론 전 상위 클래스 사례 2~3건 정독 필수.")
    print("⚠`TEXT_DIFF`·`NO_ARG_DIFF`는 채점 규약 아티팩트를 포함할 수 있어 레버 근거로 쓰지 말 것.")

    if args.json:
        json.dump({"cls": dict(cls), "by_key": {k: dict(v) for k, v in by_key.items()},
                   "by_tool": {k: dict(v) for k, v in by_tool.items()}, "rows": rows},
                  open(args.json, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"\n[saved] {args.json}")


if __name__ == "__main__":
    main()
