# -*- coding: utf-8 -*-
"""bank_perstep_decomp.py — per-step 다층 실패 분해 (2026-07-16·사용자 지시).

첫-원인 귀속 폐기([[08]]). 각 실패 sim의 gold plan *전체*를 per-step walk하며
각 스텝을 C92 연산-오분류 타입으로 분류하고, 한 sim의 *모든* 실패층을 센다.
목표 = "여러 단계 문제를 다 해결". reach/discovery/dispute가 같은 경계문제(연산)로
치환됨을 실증 → outer/inner loop을 *전 워크플로*에 균일 적용(사용자 통찰).

연산 타입 (C92·[[16]] GET/FIND/COMPUTE/ASK + coverage/over):
  FIND-discovery : 필요한 read(get/search) 미수행 → 발견 갭(reach)
  COVERAGE       : 필요한 write 미수행 → under-action
  GET-⋈(reference): write 수행했으나 엔티티 id 틀림 → 잘못된 참조
  COMPUTE        : write 필드 중 ABox-compute 값 틀림 → 계산
  GATHER/ASK     : write 필드 중 enum/reason/값 틀림 → 의미/비결정
  OVER-ACTION    : gold에 없는 write 수행 → 과행동

read/write = 도메인일반 이름-prefix 휴리스틱(get_/search_/list_/lookup_=read). ABox producers 비어 대체.
DB-basis 실패 sim만·infra(too_many_errors) 제외. action-check args-row 기준(proxy ~90% tight·C93).

사용: py bank_perstep_decomp.py
"""
import json, glob, re, sys, io, os
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
_HERE = os.path.dirname(os.path.abspath(__file__))
_ABOX = os.path.join(_HERE, "a2", "banking_knowledge.gate.json")


def _nd(x):
    try:
        v = json.loads(x) if isinstance(x, str) else x
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


_fam = lambda n: re.sub(r"_\d+$", "", str(n))
_READ_PREFIX = re.compile(r"^(get|search|list|lookup|find|retrieve|read|view|check)_", re.I)
# 절차/메타 도구(비-DB-write) — over-action 오탐 방지([[08]] 감사·2026-07-16).
# log/verification/KB검색/tool-unlock/escalation/shell = DB 상태 안 바꿈.
_PROCEDURAL = re.compile(
    r"(^log_|_verification$|^kb_search|^kb_|^search_|^shell$|discoverable|transfer_to_human|give_)", re.I)


def is_read(tool_family):
    return bool(_READ_PREFIX.match(tool_family))


def is_procedural(tool_family):
    return bool(_PROCEDURAL.search(tool_family))


_ID_FIELDS = {"transaction_id", "account_id", "card_id", "user_id", "order_id",
              "card_last_4_digits", "dispute_id", "report_id"}
_ENUM_HINT = re.compile(r"(reason|category|type|action|status|option|design|method|resolution|eligible|credit)", re.I)


def load_compute_fields(abox):
    ops = abox.get("compute_ops") or {}
    out = {}
    for key, fmap in ops.items():
        out[key] = set(fmap.keys())
    return out


def classify_field(field, tool_family, compute_map):
    """틀린 필드 → 연산 타입."""
    for k, fields in compute_map.items():
        if (k in tool_family or tool_family in k) and field in fields:
            return "COMPUTE"
    if field in _ID_FIELDS:
        return "GET-xmatch"          # 잘못된 엔티티 참조 = ⋈/GET
    if _ENUM_HINT.search(field):
        return "GATHER-ASK"          # enum/reason/eligibility = 의미/비결정
    return "GATHER-ASK"


def agent_calls_by_family(s):
    """agent 실호출: family → [arg dict, ...] (discoverable inner + 직접)."""
    calls = defaultdict(list)
    for m in (s.get("messages") or []):
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            nm = tc.get("name") or ""
            args = _nd(tc.get("arguments"))
            if nm == "call_discoverable_agent_tool":
                fm = _fam(args.get("agent_tool_name", ""))
                inner = _nd(args.get("arguments"))
                if fm:
                    calls[fm].append(inner)
            elif nm:
                calls[_fam(nm)].append(args)
    return calls


def best_match(gold_args, cand_list):
    """gold 액션과 같은 family의 agent 콜 중 필드-overlap 최대. 반환 (call, wrong_fields) or None."""
    best, best_ov = None, -1
    gkeys = [k for k in gold_args if k not in ("agent_tool_name",)]
    for c in cand_list:
        ov = sum(1 for k in gkeys if str(c.get(k)).strip().lower() == str(gold_args.get(k)).strip().lower())
        if ov > best_ov:
            best_ov, best = ov, c
    if best is None:
        return None
    wrong = {k for k in gkeys if k != "transaction_id"
             and str(best.get(k)).strip().lower() != str(gold_args.get(k)).strip().lower()}
    return best, wrong


def decompose_sim(s, abox, compute_map):
    """한 실패 sim의 per-step 다층 분해. 반환 layers(list of op-type) + meta."""
    ri = s.get("reward_info") or {}
    calls = agent_calls_by_family(s)
    layers = []                       # 이 sim의 모든 실패층 (op-type)
    gold_write_fams = Counter()
    n_gold_write = 0
    for ac in (ri.get("action_checks") or []):
        a = ac.get("action") or {}
        outer = _nd(a.get("arguments"))
        atn = outer.get("agent_tool_name", "")
        if not atn or "arguments" not in outer:
            continue
        tf = _fam(atn)
        gold_args = _nd(outer.get("arguments"))
        rd = is_read(tf)
        if not rd:
            n_gold_write += 1
            gold_write_fams[tf] += 1
        met = ac.get("action_reward")
        if met is None:
            met = 1.0 if ac.get("action_match") else 0.0
        if float(met) >= 1.0:
            continue
        # 미충족 gold 액션 → 연산 분류
        called = calls.get(tf, [])
        if not called:
            layers.append("FIND-discovery" if rd else "COVERAGE")
            continue
        mm = best_match(gold_args, called)
        if mm is None or not mm[1]:
            # 호출은 있으나 매칭 실패(다른 인스턴스) → coverage성
            layers.append("FIND-discovery" if rd else "COVERAGE")
            continue
        wrong = mm[1]
        # 틀린 필드들 → 각 연산타입 (dominant 하나만 층으로·중복 방지)
        optypes = Counter(classify_field(f, tf, compute_map) for f in wrong)
        # 우선순위: COMPUTE > GET-xmatch > GATHER-ASK (닫기 난이도 순 아님·표기용)
        for op in ("COMPUTE", "GET-xmatch", "GATHER-ASK"):
            if optypes.get(op):
                layers.append(op)
    # over-action: gold에 없는 *진짜 DB-write* family (read·절차 도구 제외·[[08]] 감사).
    # 신뢰 over-action rate = proxy X=1,Y=0=3.6%(C93 §3.5). 여기선 write-계열 근사만.
    over = 0
    for fm, cl in calls.items():
        if is_read(fm) or is_procedural(fm) or fm == "call_discoverable_agent_tool":
            continue
        if fm not in gold_write_fams:
            over += 1
    if over:
        layers.append("OVER-ACTION")
    return {
        "layers": layers,
        "n_layers": len(layers),
        "term": str(s.get("termination_reason")),
        "n_gold_write": n_gold_write,
    }


def main():
    abox = json.load(open(_ABOX, encoding="utf-8"))
    compute_map = load_compute_fields(abox)
    files = glob.glob("C:/tmp/traj/*_banking.json")

    op_total = Counter()              # 전 층 연산 빈도 (다층·첫-원인 아님)
    nlayer_dist = Counter()           # sim당 층수 분포
    cooc = Counter()                  # 연산 공존 쌍
    has_find_and_write = 0            # read-miss + 하류 write 문제 (연쇄 신호)
    term_by_nlayer = defaultdict(Counter)
    n = 0
    n_single = n_multi = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        for s in d.get("simulations", []):
            ri = s.get("reward_info") or {}
            if ri.get("reward") in (None, 1.0):
                continue
            if tuple(ri.get("reward_basis") or []) != ("DB",):
                continue
            if str(s.get("termination_reason")) == "too_many_errors":
                continue
            r = decompose_sim(s, abox, compute_map)
            if r["n_layers"] == 0:
                nlayer_dist["0(action-check 밖 실패)"] += 1
                n += 1
                continue
            n += 1
            nlayer_dist[min(r["n_layers"], 6)] += 1
            for op in r["layers"]:
                op_total[op] += 1
            uniq = sorted(set(r["layers"]))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    cooc[(uniq[i], uniq[j])] += 1
            if r["n_layers"] == 1:
                n_single += 1
            else:
                n_multi += 1
            if ("FIND-discovery" in r["layers"]) and any(op in r["layers"] for op in ("COVERAGE", "GET-xmatch", "GATHER-ASK", "COMPUTE")):
                has_find_and_write += 1
            term_by_nlayer[min(r["n_layers"], 6)][r["term"]] += 1

    print("=== per-step 다층 실패 분해 (DB-basis 실패 sim %d·infra제외·17궤적) ===" % n)
    print("\n[1] sim당 실패 층수 분포 (첫-원인 아님·모든 층):")
    tot = sum(nlayer_dist.values())
    for k in sorted(nlayer_dist, key=lambda x: (isinstance(x, str), x)):
        print("   층=%s : %5d (%.1f%%)" % (k, nlayer_dist[k], 100 * nlayer_dist[k] / max(tot, 1)))
    print("   → 단층(1) %d (%.1f%%) vs 다층(≥2) %d (%.1f%%)" % (
        n_single, 100 * n_single / max(n_single + n_multi, 1),
        n_multi, 100 * n_multi / max(n_single + n_multi, 1)))
    print("\n[2] 연산 타입 전체 빈도 (전 층·%d 층):" % sum(op_total.values()))
    ot = sum(op_total.values())
    for op, c in op_total.most_common():
        print("   %-16s %6d (%.1f%%)" % (op, c, 100 * c / max(ot, 1)))
    print("\n[3] 연산 공존 쌍 Top8 (다층 sim서 함께 실패):")
    for (a, b), c in cooc.most_common(8):
        print("   %-16s + %-16s %5d" % (a, b, c))
    print("\n[4] 연쇄 신호: read-miss(FIND) + 하류 write 문제 동시 = %d sim (%.1f%%)"
          % (has_find_and_write, 100 * has_find_and_write / max(n, 1)))
    print("\n[5] 종료거동 × 층수 (user_stop=under-action):")
    for k in sorted(term_by_nlayer, key=lambda x: (isinstance(x, str), x)):
        print("   층=%s : %s" % (k, dict(term_by_nlayer[k])))
    print("\n★해석: 다층 지배 → '다 해결' 필요. 모든 층이 {FIND/COVERAGE/GET-⋈/COMPUTE/GATHER-ASK/OVER}")
    print("  = 동일 경계문제(C92 연산-오분류)의 인스턴스. outer(coverage)/inner(operator) loop을")
    print("  전 워크플로 per-step에 균일 적용 = 사용자 통찰. read-miss→write 연쇄 = discovery 선행성.")


if __name__ == "__main__":
    main()
