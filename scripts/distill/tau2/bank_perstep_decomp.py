# -*- coding: utf-8 -*-
"""bank_perstep_decomp.py — per-step 다층 실패 분해 (2026-07-16 · 2026-08-15 전면 갱신).

첫-원인 귀속 폐기([[08]]). 각 실패 sim 의 gold plan *전체*를 per-step walk 하며 각 스텝을
연산 타입으로 분류하고, 한 sim 의 **모든** 실패층을 센다. 목표 = "여러 단계 문제를 다 해결".

## 2026-08-15 갱신 (사용자 지시: *"75 sim 최종 fail 정밀 분석·per step 포렌식"*)

⒜ **태그 인자화** — `C:/tmp/traj` 하드코딩을 걷어내고 `t2_forensic`(정본 로더)로 읽는다([[67]]).
⒝ ★**충족 판정을 다시 계산한다** — `action_match`/`action_reward` 를 **믿지 않는다**.
   하네스는 래퍼의 중첩 `arguments` 를 **문자열로** 비교하므로(`tasks.py:195`) 모델이 JSON 을
   들여쓰기해 내면 의미가 같아도 False 다(C486·오늘 x324 로 기전 확정: 7건 전부 성공 실행).
   그 값을 그대로 쓰면 **없는 실패층을 만든다**. ⇒ 여기서는 중첩을 풀어 **의미 비교**한다.
⒞ **스텝 위치 기록** — 층마다 그 층이 결정된 **메시지 색인**을 남기고, sim 별로
   *gold 가 마지막으로 움직인 스텝*(= 궤적이 경로를 떠난 자리)을 낸다. 처방은 그 뒤에 문다.

⚠판정(pass/fail) 자체는 `reward` 로만 한다 — 이 파일은 **원인 분해**용이지 채점기가 아니다.

연산 타입 (C92·[[16]] GET/FIND/COMPUTE/ASK + coverage/over):
  FIND-discovery : 필요한 read(get/search) 미수행 → 발견 갭(reach)
  COVERAGE       : 필요한 write 미수행 → under-action
  GET-xmatch     : write 수행했으나 엔티티 id 틀림 → 잘못된 참조(⋈)
  COMPUTE        : write 필드 중 ABox-compute 값 틀림 → 계산
  GATHER-ASK     : write 필드 중 enum/reason/값 틀림 → 의미/비결정
  OVER-ACTION    : gold 에 없는 write 수행 → 과행동

read/write = 도메인일반 이름-prefix 휴리스틱. DB-basis 실패 sim 만 · infra 제외.

사용: py bank_perstep_decomp.py <tag> [<tag>...] [--detail]
"""
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ABOX = os.path.join(_HERE, "a2", "banking_knowledge.gate.json")

import t2_forensic as F                                            # noqa: E402


def _nd(x):
    try:
        v = json.loads(x) if isinstance(x, str) else x
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


_fam = lambda n: re.sub(r"_\d+$", "", str(n))
_READ_PREFIX = re.compile(r"^(get|search|list|lookup|find|retrieve|read|view|check)_", re.I)
# 절차/메타 도구(비-DB-write) — over-action 오탐 방지([[08]] 감사·2026-07-16).
_PROCEDURAL = re.compile(
    r"(^log_|_verification$|^kb_search|^kb_|^search_|^shell$|discoverable|transfer_to_human|give_)", re.I)


def is_read(tool_family):
    return bool(_READ_PREFIX.match(tool_family))


def is_procedural(tool_family):
    return bool(_PROCEDURAL.search(tool_family))


_ID_FIELDS = {"transaction_id", "account_id", "card_id", "user_id", "order_id",
              "card_last_4_digits", "dispute_id", "report_id",
              "credit_card_account_id", "checking_account_id", "bank_account_id"}
_ENUM_HINT = re.compile(
    r"(reason|category|type|action|status|option|design|method|resolution|eligible|credit|class|level)", re.I)


def load_compute_fields(abox):
    out = {}
    for key, fmap in (abox.get("compute_ops") or {}).items():
        out[key] = set(fmap.keys())
    return out


def classify_field(field, tool_family, compute_map):
    """틀린 필드 → 연산 타입."""
    for k, fields in compute_map.items():
        if (k in tool_family or tool_family in k) and field in fields:
            return "COMPUTE"
    if field in _ID_FIELDS:
        return "GET-xmatch"
    if _ENUM_HINT.search(field):
        return "GATHER-ASK"
    return "GATHER-ASK"


def agent_calls_by_family(s):
    """agent 실호출: family → [(step, arg dict), ...] (discoverable inner + 직접).

    ★step 을 함께 든다 — *어디서* 갈라졌는지가 처방이 무는 자리다(2026-08-15)."""
    calls = defaultdict(list)
    for step, (m, tc) in enumerate(F.calls(s)):
        if m.get("role") != "assistant":
            continue
        nm = F.nameof(tc)
        args = F.argsof(tc)
        inner_tool = F.inner_name(args)
        if inner_tool:
            fm = _fam(inner_tool)
            calls[fm].append((step, _nd(args.get("arguments"))))
        elif nm:
            calls[_fam(nm)].append((step, args))
    return calls


def _norm(v):
    """비교용 정규화 — 중첩 문자열을 풀고 대소문자·여백을 죽인다."""
    if isinstance(v, str):
        try:
            j = json.loads(v)
            if isinstance(j, (dict, list)):
                return _norm(j)
        except Exception:
            pass
        return v.strip().lower()
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [_norm(x) for x in v]
    if isinstance(v, float) and v == int(v):
        return int(v)
    return v


def best_match(gold_args, cand_list):
    """같은 family 호출 중 필드-overlap 최대. 반환 (step, call, wrong_fields) or None."""
    best, best_ov, best_step = None, -1, None
    gkeys = [k for k in gold_args if k != "agent_tool_name"]
    for step, c in cand_list:
        ov = sum(1 for k in gkeys if _norm(c.get(k)) == _norm(gold_args.get(k)))
        if ov > best_ov:
            best_ov, best, best_step = ov, c, step
    if best is None:
        return None
    wrong = {k for k in gkeys if _norm(best.get(k)) != _norm(gold_args.get(k))}
    return best_step, best, wrong


def decompose_sim(s, compute_map):
    """한 실패 sim 의 per-step 다층 분해. layers = [(op, step, tool, 필드들)]."""
    ri = s.get("reward_info") or {}
    calls = agent_calls_by_family(s)
    layers = []
    gold_write_fams = Counter()
    n_gold_write = 0
    last_gold_step = -1                # gold 가 마지막으로 움직인 자리
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
        called = calls.get(tf, [])
        if not called:
            layers.append(("FIND-discovery" if rd else "COVERAGE", None, tf, ()))
            continue
        mm = best_match(gold_args, called)
        # ★충족 판정 = **여기서 다시 계산**한다(C486: 하네스 값은 표기로 무너진다).
        if mm is not None and not mm[2]:
            last_gold_step = max(last_gold_step, mm[0])
            continue
        if mm is None:
            layers.append(("FIND-discovery" if rd else "COVERAGE", None, tf, ()))
            continue
        step, _call, wrong = mm
        optypes = Counter(classify_field(f, tf, compute_map) for f in wrong)
        for op in ("COMPUTE", "GET-xmatch", "GATHER-ASK"):
            if optypes.get(op):
                layers.append((op, step, tf, tuple(sorted(wrong))))
    over = []
    for fm, cl in calls.items():
        if is_read(fm) or is_procedural(fm) or fm == "call_discoverable_agent_tool":
            continue
        if fm not in gold_write_fams:
            over.append((fm, cl[0][0]))
    for fm, st in over:
        layers.append(("OVER-ACTION", st, fm, ()))
    return {"layers": layers, "n_layers": len(layers), "term": F.term_reason(s),
            "n_gold_write": n_gold_write, "last_gold_step": last_gold_step,
            "n_steps": sum(1 for _ in F.calls(s))}


def main(argv):
    detail = "--detail" in argv
    tags = [a for a in argv if not a.startswith("--")] or \
        ["bank_t7295_a_20260815n", "bank_t7295_b_20260815n"]
    abox = json.load(open(_ABOX, encoding="utf-8"))
    compute_map = load_compute_fields(abox)

    op_total = Counter()
    nlayer_dist = Counter()
    cooc = Counter()
    by_task = defaultdict(Counter)
    has_find_and_write = 0
    term_by_nlayer = defaultdict(Counter)
    tail = Counter()
    n = n_single = n_multi = 0
    n_pass = n_infra = 0

    for tag in tags:
        for s in F.scored(tag):
            ri = s.get("reward_info") or {}
            if ri.get("reward") == 1.0:
                n_pass += 1
                continue
            if str(F.term_reason(s)) == "too_many_errors":
                n_infra += 1
                continue
            r = decompose_sim(s, compute_map)
            tid = F.task_id(s)
            n += 1
            if r["n_layers"] == 0:
                nlayer_dist["0(action-check 밖 실패)"] += 1
                by_task[tid]["0(밖)"] += 1
                continue
            nlayer_dist[min(r["n_layers"], 6)] += 1
            ops = [op for op, _st, _tf, _f in r["layers"]]
            for op in ops:
                op_total[op] += 1
                by_task[tid][op] += 1
            uniq = sorted(set(ops))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    cooc[(uniq[i], uniq[j])] += 1
            n_single += r["n_layers"] == 1
            n_multi += r["n_layers"] > 1
            if "FIND-discovery" in ops and any(
                    o in ops for o in ("COVERAGE", "GET-xmatch", "GATHER-ASK", "COMPUTE")):
                has_find_and_write += 1
            term_by_nlayer[min(r["n_layers"], 6)][r["term"]] += 1
            if r["last_gold_step"] >= 0 and r["n_steps"]:
                frac = r["last_gold_step"] / float(r["n_steps"])
                tail[("전반부(≤1/3)" if frac <= .33 else
                      "중반(≤2/3)" if frac <= .66 else "후반(>2/3)")] += 1
            else:
                tail["gold 한 번도 안 움직임"] += 1
            if detail:
                print("%-12s steps=%3d gold이탈=%s  %s" % (
                    F.sim_key(s), r["n_steps"],
                    r["last_gold_step"] if r["last_gold_step"] >= 0 else "없음",
                    " · ".join("%s@%s:%s%s" % (op, st, tf, ("/" + ",".join(f)) if f else "")
                               for op, st, tf, f in r["layers"])))

    print("\n=== per-step 다층 실패 분해 · %s ===" % ", ".join(tags))
    print("실패 sim %d (통과 %d 제외 · infra %d 제외)" % (n, n_pass, n_infra))
    print("\n[1] sim 당 실패 층수 분포 (첫-원인 아님·모든 층):")
    tot = sum(nlayer_dist.values())
    for k in sorted(nlayer_dist, key=lambda x: (isinstance(x, str), x)):
        print("   층=%-22s %4d (%.1f%%)" % (k, nlayer_dist[k], 100 * nlayer_dist[k] / max(tot, 1)))
    print("   → 단층(1) %d (%.1f%%) vs 다층(≥2) %d (%.1f%%)" % (
        n_single, 100 * n_single / max(n_single + n_multi, 1),
        n_multi, 100 * n_multi / max(n_single + n_multi, 1)))
    print("\n[2] 연산 타입 전체 빈도 (전 층·%d 층):" % sum(op_total.values()))
    ot = sum(op_total.values())
    for op, c in op_total.most_common():
        print("   %-16s %5d (%.1f%%)" % (op, c, 100 * c / max(ot, 1)))
    print("\n[3] 연산 공존 쌍 Top8:")
    for (a, b), c in cooc.most_common(8):
        print("   %-16s + %-16s %4d" % (a, b, c))
    print("\n[4] 연쇄 신호: read-miss(FIND) + 하류 write 문제 동시 = %d sim (%.1f%%)"
          % (has_find_and_write, 100 * has_find_and_write / max(n, 1)))
    print("\n[5] ★gold 가 마지막으로 움직인 자리 (= 궤적이 경로를 떠난 지점):")
    for k, c in tail.most_common():
        print("   %-22s %4d (%.1f%%)" % (k, c, 100 * c / max(n, 1)))
    print("\n[6] 태스크 × 연산 타입:")
    print("   %-10s %s" % ("task", "  ".join("%-14s" % o for o in
                                             ("FIND-discovery", "COVERAGE", "GET-xmatch",
                                              "COMPUTE", "GATHER-ASK", "OVER-ACTION"))))
    for t in sorted(by_task):
        print("   %-10s %s" % (t, "  ".join(
            "%-14d" % by_task[t][o] for o in ("FIND-discovery", "COVERAGE", "GET-xmatch",
                                              "COMPUTE", "GATHER-ASK", "OVER-ACTION"))))
    print("\n[7] 종료거동 × 층수:")
    for k in sorted(term_by_nlayer, key=lambda x: (isinstance(x, str), x)):
        print("   층=%s : %s" % (k, dict(term_by_nlayer[k])))


if __name__ == "__main__":
    main(sys.argv[1:])
