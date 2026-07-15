# -*- coding: utf-8 -*-
"""bank_tierA_grounding.py — Tier-A(GATHER-ASK) 세분: 결정론 그라운딩 vs ASK vs 경계 (2026-07-16).

Tier-A = write 필드 중 enum/reason/값이 틀린 층. "우리 매커니즘 극복 가능?"의 핵심 =
그 틀린 필드의 gold 값이 agent가 *본 맥락*(user 발화 + tool result)에 존재하나:
  - 존재(literal/near) → 결정론 그라운딩(GET/원문-치환)으로 닫힘 = 우리 레버 사정권 (info-was-there·under-utilization)
  - 부재 → ASK 필요(user 원천) 또는 경계.
enum은 NL→정규화라 literal 부재가 흔함 → **literal 존재 = 그라운딩-closable 하한(floor)**. 부재≠경계(정규화 필요분 포함).

DB-basis 실패·infra제외. bank_perstep_decomp 로직 재사용.
사용: py bank_tierA_grounding.py
"""
import json, glob, re, sys, io, os
from collections import Counter, defaultdict

import bank_perstep_decomp as P
_HERE = os.path.dirname(os.path.abspath(__file__))
_ABOX = os.path.join(_HERE, "a2", "banking_knowledge.gate.json")

_nd = P._nd
_fam = P._fam


def seen_text(s):
    """agent가 본 텍스트(user 발화 + tool result). 소문자."""
    buf = []
    for m in (s.get("messages") or []):
        if m.get("role") in ("user", "tool"):
            buf.append(str(m.get("content")))
    return " ".join(buf).lower()


def val_present(val, ctx):
    """gold 값이 맥락에 존재하나 (literal 또는 토큰-분해)."""
    v = str(val).strip().lower()
    if not v or v in ("none", "null", "true", "false"):
        return None                      # 판정 제외(bool/빈값)
    if v in ctx:
        return True
    # enum 언더스코어 분해: goods_services_not_received → 토큰 다수 존재?
    toks = [t for t in re.split(r"[_\s]+", v) if len(t) >= 4]
    if toks and sum(1 for t in toks if t in ctx) >= max(1, len(toks) - 1):
        return True
    return False


def main():
    abox = json.load(open(_ABOX, encoding="utf-8"))
    cmap = P.load_compute_fields(abox)
    files = sorted(glob.glob("C:/tmp/traj/*_banking.json"))

    field_present = Counter()     # (present?) per GATHER-ASK 틀린 필드
    field_kind = Counter()        # 어떤 필드가 GATHER-ASK 잔여인가
    present_by_field = defaultdict(lambda: [0, 0])   # field → [present, absent]
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
            ctx = seen_text(s)
            calls = P.agent_calls_by_family(s)
            for ac in (ri.get("action_checks") or []):
                a = ac.get("action") or {}
                outer = _nd(a.get("arguments"))
                atn = outer.get("agent_tool_name", "")
                if not atn or "arguments" not in outer:
                    continue
                tf = _fam(atn)
                if P.is_read(tf):
                    continue
                met = ac.get("action_reward")
                if met is None:
                    met = 1.0 if ac.get("action_match") else 0.0
                if float(met) >= 1.0:
                    continue
                gold_args = _nd(outer.get("arguments"))
                mm = P.best_match(gold_args, calls.get(tf, []))
                if mm is None or not mm[1]:
                    continue
                wrong = mm[1]
                for field in wrong:
                    op = P.classify_field(field, tf, cmap)
                    if op != "GATHER-ASK":
                        continue
                    field_kind[field] += 1
                    pres = val_present(gold_args.get(field), ctx)
                    if pres is None:
                        field_present["(bool/빈값·제외)"] += 1
                        continue
                    field_present["존재(그라운딩-closable)" if pres else "부재(ASK/경계)"] += 1
                    present_by_field[field][0 if pres else 1] += 1

    print("=== Tier-A(GATHER-ASK) 틀린 필드 그라운딩 세분 (DB-basis 실패·17모델) ===")
    tot = field_present["존재(그라운딩-closable)"] + field_present["부재(ASK/경계)"]
    for k in ("존재(그라운딩-closable)", "부재(ASK/경계)", "(bool/빈값·제외)"):
        base = tot if not k.startswith("(") else sum(field_present.values())
        print("  %-24s %6d (%.1f%%)" % (k, field_present[k], 100 * field_present[k] / max(base, 1)))
    print("\n  GATHER-ASK 잔여 필드 Top12 (present/absent):")
    for fld, c in field_kind.most_common(12):
        p, ab = present_by_field[fld]
        print("    %-30s tot=%4d  present=%d absent=%d" % (fld, c, p, ab))
    print("\n  ★해석: '존재'=info-was-there=결정론 그라운딩(GET/원문치환)=우리 레버 사정권(하한).")
    print("         '부재'=user-원천 ASK 또는 경계. enum 정규화분은 부재로 과다계상(floor 성격).")


if __name__ == "__main__":
    main()
